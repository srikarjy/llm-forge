"""
Enhanced model trainer for memory-efficient LLM fine-tuning.

This module provides functionality for training large language models
with QLoRA, memory optimization, and scientific dataset support.
"""

import gc
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datasets import Dataset

try:
    from ..utils.logger import get_logger
    from .trainer import ModelTrainer
    from .model_loader import ModelLoader, ModelInfo
    from .qlora_config import QLoRAConfig, TrainingConfig
    from ..data.text_processor import ScientificTextProcessor
    from ..data.scientific_dataset import ScientificDataModule
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger
    from models.trainer import ModelTrainer
    from models.model_loader import ModelLoader, ModelInfo
    from models.qlora_config import QLoRAConfig, TrainingConfig
    from data.text_processor import ScientificTextProcessor
    from data.scientific_dataset import ScientificDataModule

logger = get_logger(__name__)

# Import ML libraries with error handling
try:
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available")
    TRANSFORMERS_AVAILABLE = False
    Trainer = None
    TrainingArguments = None
    DataCollatorForLanguageModeling = None
    EarlyStoppingCallback = None
    PreTrainedModel = None
    PreTrainedTokenizer = None

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow library not available")
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("Datasets library not available")
    DATASETS_AVAILABLE = False
    Dataset = None


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    training_time_seconds: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class TrainingResults:
    """Complete training results."""
    model_info: ModelInfo
    training_config: TrainingConfig
    final_metrics: TrainingMetrics
    training_history: List[TrainingMetrics]
    checkpoint_paths: List[str]
    total_training_time: float
    best_model_path: Optional[str] = None


class EnhancedModelTrainer(ModelTrainer):
    """Enhanced trainer for memory-efficient LLM fine-tuning."""
    
    def __init__(
        self,
        config: TrainingConfig,
        output_dir: Optional[str] = None
    ):
        """Initialize the enhanced trainer.
        
        Args:
            config: Training configuration
            output_dir: Output directory for models and logs
        """
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.text_processor = ScientificTextProcessor()
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self.trainer = None
        self.training_history = []
        self.start_time = None
        
        # MLflow setup
        self._setup_mlflow()
        
        logger.info(f"Initialized EnhancedModelTrainer with output dir: {self.output_dir}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        if not MLFLOW_AVAILABLE or not self.config.mlflow["enabled"]:
            logger.info("MLflow tracking disabled or not available")
            return
        
        try:
            mlflow.set_tracking_uri(self.config.mlflow["tracking_uri"])
            mlflow.set_experiment(self.config.mlflow["experiment_name"])
            logger.info(f"MLflow experiment: {self.config.mlflow['experiment_name']}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
    
    def setup_model_and_data(
        self,
        data_file: Optional[str] = None,
        dataset: Optional[Dataset] = None
    ) -> Tuple[PreTrainedModel, Dataset, "Trainer"]:
        """Setup model and data for training.
        
        Args:
            data_file: Path to scientific papers data file
            dataset: Pre-processed dataset (alternative to data_file)
            
        Returns:
            Tuple of (model, dataset, trainer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for training")
        
        logger.info("Setting up model and data for training...")
        
        # 1. Load and setup model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model, self.tokenizer, self.model_info = self.model_loader.load_complete_model(
            model_name=self.config.model_name,
            config=self.config.qlora,
            attach_lora=True,
            optimize_memory=True
        )
        
        # Validate memory constraints
        fits, message = self.model_loader.validate_memory_constraints(
            self.model_info, 
            max_memory_gb=self.config.max_memory_usage * 16.0  # Assume 16GB base
        )
        
        if not fits:
            logger.warning(f"Memory constraint validation: {message}")
        else:
            logger.info(f"Memory validation passed: {message}")
        
        # 2. Prepare dataset
        if dataset is None:
            if data_file is None:
                data_file = self.config.scientific_data["data_file"]
            
            logger.info(f"Loading scientific papers from: {data_file}")
            data_module = ScientificDataModule(data_file)
            papers = data_module.load_papers(min_quality_score=70)
            
            # Process papers into training dataset
            dataset = self.text_processor.process_papers_for_training(
                papers, 
                format_type="causal_lm"  # Default to causal LM
            )
        
        logger.info(f"Training dataset size: {len(dataset)}")
        
        # 3. Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8  # For efficiency
        )
        
        # 4. Tokenize dataset
        def tokenize_function(examples):
            """Tokenize text examples."""
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Will be handled by data collator
                max_length=self.config.max_length,
                return_tensors=None
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # 5. Split dataset
        if len(tokenized_dataset) > 100:  # Only split if we have enough data
            split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # 6. Setup training arguments
        training_args = self._create_training_arguments()
        
        # 7. Create trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
            "tokenizer": self.tokenizer,
        }
        
        # Add early stopping if eval dataset exists
        if eval_dataset is not None:
            trainer_kwargs["callbacks"] = [
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        
        self.trainer = Trainer(**trainer_kwargs)
        
        logger.info("Model and data setup complete")
        return self.model, train_dataset, self.trainer
    
    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            
            # Training hyperparameters
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # Memory optimization
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Evaluation
            evaluation_strategy="steps" if self.config.eval_steps > 0 else "no",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Other settings
            remove_unused_columns=False,
            report_to=["mlflow"] if MLFLOW_AVAILABLE and self.config.mlflow["enabled"] else [],
            run_name=f"genomics-llm-{int(time.time())}",
        )
        
        return args
    
    def dynamic_batch_sizing(self, initial_batch_size: int = None) -> int:
        """Determine optimal batch size based on available GPU memory.
        
        Args:
            initial_batch_size: Starting batch size to test
            
        Returns:
            Optimal batch size
        """
        if initial_batch_size is None:
            initial_batch_size = self.config.batch_size
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using configured batch size")
            return initial_batch_size
        
        logger.info("Determining optimal batch size...")
        
        # Get available GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_memory_gb = gpu_memory_gb * self.config.max_memory_usage
        
        # Estimate memory per sample (rough approximation)
        if self.model_info:
            # Base memory + per-sample overhead
            base_memory = self.model_info.memory_usage_gb
            per_sample_memory_mb = (self.config.max_length * 4 * 2) / (1024**2)  # Rough estimate
            
            # Calculate max batch size
            available_for_batch = (available_memory_gb - base_memory) * 1024  # Convert to MB
            max_batch_size = max(1, int(available_for_batch / per_sample_memory_mb))
            
            optimal_batch_size = min(initial_batch_size, max_batch_size)
            
            logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB")
            logger.info(f"Available for training: {available_memory_gb:.1f} GB")
            logger.info(f"Estimated max batch size: {max_batch_size}")
            logger.info(f"Using batch size: {optimal_batch_size}")
            
            return optimal_batch_size
        
        return initial_batch_size
    
    def monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitor GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            memory_stats = {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": memory_total,
                "free_gb": memory_total - memory_reserved,
                "utilization_percent": (memory_reserved / memory_total) * 100
            }
        else:
            memory_stats = {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "free_gb": 0.0,
                "utilization_percent": 0.0
            }
        
        return memory_stats
    
    def save_training_checkpoint(
        self, 
        epoch: int, 
        metrics: TrainingMetrics
    ) -> str:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        if self.model and self.tokenizer:
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training metrics
        metrics_file = checkpoint_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save training config
        config_file = checkpoint_dir / "training_config.json"
        with open(config_file, 'w') as f:
            # Convert config to dict (handling dataclass)
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        return str(checkpoint_dir)
    
    def train_with_memory_optimization(
        self,
        data_file: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        validation_data: Optional[Dataset] = None
    ) -> TrainingResults:
        """Train model with memory optimization.
        
        Args:
            data_file: Path to scientific papers data file
            dataset: Pre-processed training dataset
            validation_data: Validation dataset
            
        Returns:
            Training results
        """
        logger.info("Starting memory-optimized training...")
        self.start_time = time.time()
        
        # Log training configuration to MLflow
        if MLFLOW_AVAILABLE and self.config.mlflow["enabled"]:
            with mlflow.start_run():
                self._log_config_to_mlflow()
                return self._train_with_mlflow_logging(data_file, dataset, validation_data)
        else:
            return self._train_without_mlflow(data_file, dataset, validation_data)
    
    def _train_with_mlflow_logging(
        self,
        data_file: Optional[str],
        dataset: Optional[Dataset],
        validation_data: Optional[Dataset]
    ) -> TrainingResults:
        """Train with MLflow logging."""
        # Setup model and data
        model, train_dataset, trainer = self.setup_model_and_data(data_file, dataset)
        
        # Log model info
        mlflow.log_params({
            "model_name": self.model_info.model_name,
            "total_params": self.model_info.total_params,
            "trainable_params": self.model_info.trainable_params,
            "quantized": self.model_info.quantized,
            "has_lora": self.model_info.has_lora,
        })
        
        # Optimize batch size
        optimal_batch_size = self.dynamic_batch_sizing()
        if optimal_batch_size != self.config.batch_size:
            logger.info(f"Adjusting batch size from {self.config.batch_size} to {optimal_batch_size}")
            trainer.args.per_device_train_batch_size = optimal_batch_size
            trainer.args.per_device_eval_batch_size = optimal_batch_size
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Log model to MLflow
        if self.config.mlflow.get("log_model", True):
            mlflow.pytorch.log_model(model, "model")
        
        # Create training results
        total_time = time.time() - self.start_time
        
        final_metrics = TrainingMetrics(
            epoch=int(train_result.epoch),
            step=train_result.global_step,
            train_loss=train_result.training_loss,
            training_time_seconds=total_time
        )
        
        results = TrainingResults(
            model_info=self.model_info,
            training_config=self.config,
            final_metrics=final_metrics,
            training_history=self.training_history,
            checkpoint_paths=[],  # Would be populated by callback
            total_training_time=total_time,
            best_model_path=str(final_model_path)
        )
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        return results
    
    def _train_without_mlflow(
        self,
        data_file: Optional[str],
        dataset: Optional[Dataset],
        validation_data: Optional[Dataset]
    ) -> TrainingResults:
        """Train without MLflow logging."""
        # Setup model and data
        model, train_dataset, trainer = self.setup_model_and_data(data_file, dataset)
        
        # Optimize batch size
        optimal_batch_size = self.dynamic_batch_sizing()
        if optimal_batch_size != self.config.batch_size:
            logger.info(f"Adjusting batch size from {self.config.batch_size} to {optimal_batch_size}")
            trainer.args.per_device_train_batch_size = optimal_batch_size
            trainer.args.per_device_eval_batch_size = optimal_batch_size
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Create training results
        total_time = time.time() - self.start_time
        
        final_metrics = TrainingMetrics(
            epoch=int(train_result.epoch),
            step=train_result.global_step,
            train_loss=train_result.training_loss,
            training_time_seconds=total_time
        )
        
        results = TrainingResults(
            model_info=self.model_info,
            training_config=self.config,
            final_metrics=final_metrics,
            training_history=self.training_history,
            checkpoint_paths=[],
            total_training_time=total_time,
            best_model_path=str(final_model_path)
        )
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        return results
    
    def _log_config_to_mlflow(self) -> None:
        """Log training configuration to MLflow."""
        try:
            # Log basic config
            mlflow.log_params({
                "model_name": self.config.model_name,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "num_epochs": self.config.num_epochs,
                "max_length": self.config.max_length,
                "fp16": self.config.fp16,
                "gradient_checkpointing": self.config.gradient_checkpointing,
            })
            
            # Log QLoRA config
            mlflow.log_params({
                "lora_r": self.config.qlora.r,
                "lora_alpha": self.config.qlora.lora_alpha,
                "lora_dropout": self.config.qlora.lora_dropout,
                "load_in_4bit": self.config.qlora.load_in_4bit,
                "quantization_type": self.config.qlora.bnb_4bit_quant_type,
            })
            
        except Exception as e:
            logger.error(f"Failed to log config to MLflow: {e}")
    
    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cleanup completed")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics.
        
        Returns:
            Dictionary with training summary
        """
        if not self.model_info:
            return {"error": "No model loaded"}
        
        memory_stats = self.monitor_gpu_memory()
        
        summary = {
            "model_info": {
                "name": self.model_info.model_name,
                "total_params": self.model_info.total_params,
                "trainable_params": self.model_info.trainable_params,
                "trainable_percent": (self.model_info.trainable_params / self.model_info.total_params) * 100,
                "quantized": self.model_info.quantized,
                "has_lora": self.model_info.has_lora,
                "device": self.model_info.device,
            },
            "memory_usage": memory_stats,
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
                "max_length": self.config.max_length,
                "num_epochs": self.config.num_epochs,
            },
            "optimization": {
                "fp16": self.config.fp16,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "lora_enabled": self.config.qlora.r > 0,
                "quantization": "4-bit" if self.config.qlora.load_in_4bit else "8-bit" if self.config.qlora.load_in_8bit else "none",
            }
        }
        
        return summary
    
    # Legacy methods for compatibility with base ModelTrainer
    def train(self) -> None:
        """Legacy train method for compatibility."""
        logger.warning("Using legacy train method. Consider using train_with_memory_optimization()")
        results = self.train_with_memory_optimization()
        logger.info(f"Training completed. Final loss: {results.final_metrics.train_loss:.4f}")
    
    def save_model(self) -> None:
        """Legacy save model method for compatibility."""
        if self.trainer:
            save_path = self.output_dir / "legacy_model"
            self.trainer.save_model(str(save_path))
            logger.info(f"Model saved to {save_path}")
        else:
            logger.error("No trainer available. Run training first.")