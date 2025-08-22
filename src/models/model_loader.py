"""
Model loader for LLM fine-tuning with quantization support.

This module provides functionality for loading pre-trained models
with QLoRA quantization and memory optimization for genomics fine-tuning.
"""

import gc
import torch
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from ..utils.logger import get_logger
    from .qlora_config import QLoRAConfig, TrainingConfig
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger
    from models.qlora_config import QLoRAConfig, TrainingConfig

logger = get_logger(__name__)

# Import transformers and related libraries with error handling
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available")
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PreTrainedModel = None
    PreTrainedTokenizer = None

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning("PEFT library not available")
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_name: str
    model_type: str
    total_params: int
    trainable_params: int
    memory_usage_gb: float
    quantized: bool
    has_lora: bool
    device: str


class ModelLoader:
    """Model loader with quantization and LoRA support."""
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        "llama": {
            "model_names": [
                "meta-llama/Llama-2-7b-hf",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-hf",
                "huggyllama/llama-7b",
                "huggyllama/llama-13b"
            ],
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "llama"
        },
        "bert": {
            "model_names": [
                "bert-base-uncased",
                "bert-large-uncased",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                "dmis-lab/biobert-base-cased-v1.1"
            ],
            "target_modules": ["query", "value", "key", "dense"],
            "model_type": "bert"
        },
        "gpt": {
            "model_names": [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-large",
                "gpt2",
                "gpt2-medium",
                "gpt2-large"
            ],
            "target_modules": ["c_attn", "c_proj"],
            "model_type": "gpt"
        }
    }
    
    def __init__(self):
        """Initialize the model loader."""
        self.device = self._get_best_device()
        self.loaded_models = {}
        
        logger.info(f"Initialized ModelLoader with device: {self.device}")
        
        # Check library availability
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install: pip install transformers>=4.35.0")
        if not PEFT_AVAILABLE:
            logger.error("PEFT library not available. Please install: pip install peft>=0.6.0")
    
    def _get_best_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def validate_model_compatibility(self, model_name: str) -> bool:
        """Validate if model is supported.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        for model_family, config in self.SUPPORTED_MODELS.items():
            if any(supported_name in model_name.lower() for supported_name in config["model_names"]):
                return True
        
        # Check if it's a valid HuggingFace model path
        if "/" in model_name and len(model_name.split("/")) == 2:
            logger.warning(f"Model {model_name} not in supported list but may be compatible")
            return True
        
        return False
    
    def get_model_family(self, model_name: str) -> Optional[str]:
        """Get model family for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family name or None if not found
        """
        model_name_lower = model_name.lower()
        
        for family, config in self.SUPPORTED_MODELS.items():
            if any(supported_name.lower() in model_name_lower for supported_name in config["model_names"]):
                return family
            
            # Check by family name in model name
            if family in model_name_lower:
                return family
        
        return None
    
    def configure_target_modules(self, model_name: str) -> List[str]:
        """Configure target modules for LoRA based on model type.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of target module names
        """
        model_family = self.get_model_family(model_name)
        
        if model_family and model_family in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_family]["target_modules"]
        
        # Default target modules for unknown models
        logger.warning(f"Unknown model family for {model_name}, using default target modules")
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def setup_quantization_config(self, config: QLoRAConfig) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration.
        
        Args:
            config: QLoRA configuration
            
        Returns:
            BitsAndBytesConfig or None if quantization disabled
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for quantization")
        
        if not (config.load_in_4bit or config.load_in_8bit):
            return None
        
        try:
            compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            )
            
            logger.info(f"Created quantization config: {config.bnb_4bit_quant_type} quantization")
            return quantization_config
            
        except Exception as e:
            logger.error(f"Failed to create quantization config: {e}")
            raise
    
    def load_model_with_quantization(
        self, 
        model_name: str, 
        config: QLoRAConfig,
        trust_remote_code: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model with quantization support.
        
        Args:
            model_name: Name or path of the model
            config: QLoRA configuration
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for model loading")
        
        if not self.validate_model_compatibility(model_name):
            logger.warning(f"Model {model_name} may not be fully supported")
        
        logger.info(f"Loading model: {model_name}")
        
        # Setup quantization config
        quantization_config = self.setup_quantization_config(config)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                padding_side="left"  # Important for causal LM
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model
        logger.info("Loading model...")
        try:
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch.float16,  # Use float16 for memory efficiency
                "device_map": "auto" if self.device.startswith("cuda") else None,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using quantization for model loading")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if model_kwargs["device_map"] is None:
                model = model.to(self.device)
            
            logger.info(f"Successfully loaded model on device: {model.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Store model info
        self.loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "quantized": quantization_config is not None
        }
        
        return model, tokenizer
    
    def attach_lora_adapters(
        self, 
        model: PreTrainedModel, 
        config: QLoRAConfig,
        model_name: str
    ) -> PeftModel:
        """Attach LoRA adapters to the model.
        
        Args:
            model: Pre-trained model
            config: QLoRA configuration
            model_name: Name of the model (for target module configuration)
            
        Returns:
            Model with LoRA adapters attached
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA adapters")
        
        logger.info("Attaching LoRA adapters...")
        
        # Configure target modules based on model type
        target_modules = config.target_modules
        if not target_modules:
            target_modules = self.configure_target_modules(model_name)
        
        # Create LoRA config
        try:
            lora_config = LoraConfig(
                r=config.r,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=config.lora_dropout,
                bias=config.bias,
                task_type=TaskType.CAUSAL_LM,  # Default to causal LM
            )
            
            logger.info(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, targets={target_modules}")
            
        except Exception as e:
            logger.error(f"Failed to create LoRA config: {e}")
            raise
        
        # Apply LoRA to model
        try:
            peft_model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            trainable_params, total_params = self.count_trainable_parameters(peft_model)
            logger.info(
                f"LoRA adapters attached. Trainable params: {trainable_params:,} "
                f"({trainable_params/total_params*100:.2f}% of {total_params:,})"
            )
            
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to attach LoRA adapters: {e}")
            raise
    
    def count_trainable_parameters(self, model: PreTrainedModel) -> Tuple[int, int]:
        """Count trainable and total parameters.
        
        Args:
            model: Model to analyze
            
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = 0
        total_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, total_params
    
    def optimize_memory_usage(
        self, 
        model: PreTrainedModel, 
        config: QLoRAConfig
    ) -> PreTrainedModel:
        """Optimize model for memory usage.
        
        Args:
            model: Model to optimize
            config: QLoRA configuration
            
        Returns:
            Optimized model
        """
        logger.info("Optimizing model for memory usage...")
        
        # Enable gradient checkpointing if configured
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Set model to training mode for memory optimization
        model.train()
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return model
    
    def get_model_memory_usage(self, model: PreTrainedModel) -> Dict[str, float]:
        """Get detailed memory usage information.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with memory usage breakdown
        """
        memory_info = {}
        
        # Calculate model parameter memory
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_gb = param_memory_bytes / (1024**3)
        
        memory_info.update({
            "total_parameters": total_params,
            "parameter_memory_gb": param_memory_gb,
        })
        
        # GPU memory usage if available
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            memory_info.update({
                "gpu_memory_allocated_gb": gpu_memory_allocated,
                "gpu_memory_reserved_gb": gpu_memory_reserved,
                "gpu_memory_total_gb": gpu_memory_total,
                "gpu_memory_free_gb": gpu_memory_total - gpu_memory_reserved,
                "gpu_utilization_percent": (gpu_memory_reserved / gpu_memory_total) * 100
            })
        
        # Estimate additional memory for training
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
            # Rough estimate for activations and gradients
            activation_memory_gb = (total_params * 4 * 2) / (1024**3)  # Rough estimate
            memory_info["estimated_activation_memory_gb"] = activation_memory_gb
        
        return memory_info
    
    def create_model_info(
        self, 
        model_name: str, 
        model: PreTrainedModel, 
        config: QLoRAConfig,
        quantized: bool = False,
        has_lora: bool = False
    ) -> ModelInfo:
        """Create ModelInfo object with model details.
        
        Args:
            model_name: Name of the model
            model: Loaded model
            config: QLoRA configuration
            quantized: Whether model is quantized
            has_lora: Whether model has LoRA adapters
            
        Returns:
            ModelInfo object
        """
        trainable_params, total_params = self.count_trainable_parameters(model)
        memory_usage = self.get_model_memory_usage(model)
        
        return ModelInfo(
            model_name=model_name,
            model_type=self.get_model_family(model_name) or "unknown",
            total_params=total_params,
            trainable_params=trainable_params,
            memory_usage_gb=memory_usage.get("parameter_memory_gb", 0.0),
            quantized=quantized,
            has_lora=has_lora,
            device=str(model.device) if hasattr(model, 'device') else self.device
        )
    
    def load_complete_model(
        self, 
        model_name: str, 
        config: QLoRAConfig,
        attach_lora: bool = True,
        optimize_memory: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, ModelInfo]:
        """Load complete model with quantization and LoRA.
        
        Args:
            model_name: Name of the model to load
            config: QLoRA configuration
            attach_lora: Whether to attach LoRA adapters
            optimize_memory: Whether to optimize for memory usage
            
        Returns:
            Tuple of (model, tokenizer, model_info)
        """
        logger.info(f"Loading complete model: {model_name}")
        
        # Load base model with quantization
        model, tokenizer = self.load_model_with_quantization(model_name, config)
        
        # Attach LoRA adapters if requested
        if attach_lora:
            model = self.attach_lora_adapters(model, config, model_name)
        
        # Optimize memory usage if requested
        if optimize_memory:
            model = self.optimize_memory_usage(model, config)
        
        # Create model info
        model_info = self.create_model_info(
            model_name=model_name,
            model=model,
            config=config,
            quantized=config.load_in_4bit or config.load_in_8bit,
            has_lora=attach_lora
        )
        
        logger.info(f"Model loading complete. Memory usage: {model_info.memory_usage_gb:.2f} GB")
        
        return model, tokenizer, model_info
    
    def validate_memory_constraints(
        self, 
        model_info: ModelInfo, 
        max_memory_gb: float = 16.0
    ) -> Tuple[bool, str]:
        """Validate model fits within memory constraints.
        
        Args:
            model_info: Model information
            max_memory_gb: Maximum allowed memory in GB
            
        Returns:
            Tuple of (fits_constraint, message)
        """
        total_memory = model_info.memory_usage_gb
        
        # Add estimated training overhead (optimizer states, gradients, activations)
        if model_info.has_lora:
            # LoRA has much lower memory overhead
            training_overhead = model_info.trainable_params * 8 / (1024**3)  # Rough estimate
        else:
            # Full fine-tuning has higher overhead
            training_overhead = model_info.total_params * 12 / (1024**3)  # Rough estimate
        
        estimated_total = total_memory + training_overhead
        
        fits = estimated_total <= max_memory_gb
        
        message = (
            f"Model memory: {total_memory:.2f} GB, "
            f"Training overhead: {training_overhead:.2f} GB, "
            f"Total estimated: {estimated_total:.2f} GB, "
            f"Limit: {max_memory_gb} GB - "
            f"{'✓ FITS' if fits else '✗ EXCEEDS'}"
        )
        
        return fits, message
    
    def cleanup_model(self, model_name: str) -> None:
        """Clean up loaded model from memory.
        
        Args:
            model_name: Name of the model to cleanup
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        logger.info(f"Cleaned up model: {model_name}")
    
    def list_supported_models(self) -> Dict[str, List[str]]:
        """List all supported models by family.
        
        Returns:
            Dictionary mapping model families to model names
        """
        return {
            family: config["model_names"] 
            for family, config in self.SUPPORTED_MODELS.items()
        }