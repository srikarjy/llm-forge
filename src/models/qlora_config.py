"""
QLoRA configuration for parameter-efficient fine-tuning.

This module provides configuration classes for QLoRA (Quantized LoRA) fine-tuning,
including LoRA parameters, quantization settings, and model-specific configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import torch

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA parameter-efficient fine-tuning."""
    
    # LoRA parameters
    r: int = 16
    """LoRA rank - controls the dimensionality of the low-rank adaptation"""
    
    lora_alpha: int = 32
    """LoRA scaling parameter - controls the magnitude of LoRA updates"""
    
    lora_dropout: float = 0.1
    """Dropout probability for LoRA layers"""
    
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    """Target modules to apply LoRA to"""
    
    bias: str = "none"
    """Bias type for LoRA layers: 'none', 'all', or 'lora_only'"""
    
    task_type: str = "CAUSAL_LM"
    """Task type for PEFT configuration"""
    
    # Quantization settings
    load_in_4bit: bool = True
    """Enable 4-bit quantization"""
    
    load_in_8bit: bool = False
    """Enable 8-bit quantization (mutually exclusive with 4-bit)"""
    
    bnb_4bit_compute_dtype: str = "float16"
    """Compute dtype for 4-bit quantization"""
    
    bnb_4bit_use_double_quant: bool = True
    """Use double quantization for better accuracy"""
    
    bnb_4bit_quant_type: str = "nf4"
    """Quantization type: 'fp4' or 'nf4'"""
    
    def get_quantization_config(self) -> Optional["BitsAndBytesConfig"]:
        """Get BitsAndBytesConfig for quantization."""
        if not (self.load_in_4bit or self.load_in_8bit):
            return None
            
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError(
                "bitsandbytes is required for quantization. "
                "Install it with: pip install bitsandbytes>=0.41.0"
            )
            
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
        )
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT configuration dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class TrainingConfig:
    """Extended training configuration for LLM fine-tuning."""
    
    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    """Pre-trained model name or path"""
    
    model_type: str = "llama"
    """Model architecture type"""
    
    max_length: int = 2048
    """Maximum sequence length for scientific papers"""
    
    padding: str = "max_length"
    """Padding strategy"""
    
    truncation: bool = True
    """Enable truncation"""
    
    # QLoRA configuration
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    """QLoRA configuration"""
    
    # Memory optimization
    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save memory"""
    
    dataloader_pin_memory: bool = True
    """Pin memory for faster data loading"""
    
    dataloader_num_workers: int = 4
    """Number of workers for data loading"""
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    """Learning rate optimized for QLoRA"""
    
    batch_size: int = 1
    """Per-device batch size for memory efficiency"""
    
    gradient_accumulation_steps: int = 16
    """Gradient accumulation steps to simulate larger batch size"""
    
    num_epochs: int = 3
    """Number of training epochs"""
    
    warmup_steps: int = 100
    """Number of warmup steps"""
    
    weight_decay: float = 0.01
    """Weight decay for regularization"""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping"""
    
    # Memory management
    max_memory_usage: float = 0.9
    """Maximum GPU memory usage (0.0-1.0)"""
    
    fp16: bool = True
    """Enable mixed precision training"""
    
    bf16: bool = False
    """Enable bfloat16 precision (if supported)"""
    
    # Scientific data configuration
    scientific_data: Dict[str, Any] = field(default_factory=lambda: {
        "data_file": "data/high_quality_papers_demo.json",
        "text_fields": ["title", "abstract"],
        "preprocessing": {
            "remove_citations": True,
            "normalize_scientific_notation": True,
            "handle_special_tokens": True,
            "max_paper_length": 4096,
        }
    })
    """Scientific data processing configuration"""
    
    # MLflow tracking
    mlflow: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "experiment_name": "genomics-llm-finetuning",
        "tracking_uri": "file:./mlruns",
        "log_model": True,
    })
    """MLflow experiment tracking configuration"""
    
    # Distributed training
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "backend": "deepspeed",
        "deepspeed_config": "configs/deepspeed_config.json",
    })
    """Distributed training configuration"""
    
    # Output configuration
    output_dir: str = "outputs/llm-finetuning"
    """Output directory for model checkpoints"""
    
    save_steps: int = 500
    """Save checkpoint every N steps"""
    
    eval_steps: int = 500
    """Evaluate every N steps"""
    
    logging_steps: int = 10
    """Log metrics every N steps"""
    
    save_total_limit: int = 3
    """Maximum number of checkpoints to keep"""


# Model-specific default configurations
LLAMA2_7B_CONFIG = QLoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

BERT_CONFIG = QLoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value", "key", "dense"],
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Configuration presets
MODEL_CONFIGS = {
    "llama2-7b": LLAMA2_7B_CONFIG,
    "bert": BERT_CONFIG,
}


def get_model_config(model_name: str) -> QLoRAConfig:
    """Get QLoRA configuration for a specific model."""
    model_key = None
    
    if "llama" in model_name.lower():
        model_key = "llama2-7b"
    elif "bert" in model_name.lower():
        model_key = "bert"
    
    if model_key and model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]
    
    # Return default configuration
    return QLoRAConfig()


def estimate_memory_usage(
    model_name: str,
    config: QLoRAConfig,
    batch_size: int = 1,
    sequence_length: int = 2048
) -> Dict[str, float]:
    """Estimate GPU memory usage for QLoRA training."""
    
    # Rough estimates based on model size and configuration
    model_sizes = {
        "7b": 7_000_000_000,
        "13b": 13_000_000_000,
        "30b": 30_000_000_000,
    }
    
    # Determine model size
    model_size = 7_000_000_000  # Default to 7B
    for size_key, size_value in model_sizes.items():
        if size_key in model_name.lower():
            model_size = size_value
            break
    
    # Base model memory (4-bit quantization)
    base_memory_gb = (model_size * 0.5) / (1024**3)  # 4-bit = 0.5 bytes per parameter
    
    # LoRA parameters memory
    lora_params = config.r * 2 * len(config.target_modules) * (model_size // 1000)
    lora_memory_gb = (lora_params * 4) / (1024**3)  # float32 = 4 bytes per parameter
    
    # Activation memory
    activation_memory_gb = (batch_size * sequence_length * 4096 * 4) / (1024**3)
    
    # Optimizer states (AdamW)
    optimizer_memory_gb = lora_memory_gb * 2  # 2x for momentum and variance
    
    total_memory_gb = base_memory_gb + lora_memory_gb + activation_memory_gb + optimizer_memory_gb
    
    return {
        "base_model_gb": base_memory_gb,
        "lora_params_gb": lora_memory_gb,
        "activations_gb": activation_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "total_gb": total_memory_gb,
        "fits_16gb": total_memory_gb < 16.0,
    }