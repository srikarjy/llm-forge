#!/usr/bin/env python3
"""
Example script demonstrating QLoRA configuration usage.

This script shows how to use the QLoRA configuration classes
for setting up memory-efficient fine-tuning of large language models.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.qlora_config import (
    QLoRAConfig, 
    TrainingConfig, 
    get_model_config, 
    estimate_memory_usage,
    MODEL_CONFIGS
)


def main():
    """Demonstrate QLoRA configuration usage."""
    
    print("=== QLoRA Configuration Example ===\n")
    
    # 1. Create default QLoRA configuration
    print("1. Default QLoRA Configuration:")
    default_config = QLoRAConfig()
    print(f"   LoRA rank: {default_config.r}")
    print(f"   LoRA alpha: {default_config.lora_alpha}")
    print(f"   LoRA dropout: {default_config.lora_dropout}")
    print(f"   Target modules: {default_config.target_modules}")
    print(f"   4-bit quantization: {default_config.load_in_4bit}")
    print()
    
    # 2. Get model-specific configurations
    print("2. Model-specific Configurations:")
    
    models_to_test = [
        "meta-llama/Llama-2-7b-hf",
        "bert-base-uncased",
        "unknown-model"
    ]
    
    for model_name in models_to_test:
        config = get_model_config(model_name)
        print(f"   {model_name}:")
        print(f"     LoRA rank: {config.r}")
        print(f"     Target modules: {config.target_modules[:3]}...")  # Show first 3
        print()
    
    # 3. Memory usage estimation
    print("3. Memory Usage Estimation:")
    
    for model_name in ["llama-7b", "llama-13b"]:
        config = QLoRAConfig()
        memory_info = estimate_memory_usage(model_name, config)
        
        print(f"   {model_name.upper()} Model:")
        print(f"     Base model: {memory_info['base_model_gb']:.2f} GB")
        print(f"     LoRA params: {memory_info['lora_params_gb']:.2f} GB")
        print(f"     Activations: {memory_info['activations_gb']:.2f} GB")
        print(f"     Optimizer: {memory_info['optimizer_gb']:.2f} GB")
        print(f"     Total: {memory_info['total_gb']:.2f} GB")
        print(f"     Fits in 16GB: {'✓' if memory_info['fits_16gb'] else '✗'}")
        print()
    
    # 4. Training configuration
    print("4. Training Configuration:")
    training_config = TrainingConfig()
    print(f"   Model: {training_config.model_name}")
    print(f"   Max length: {training_config.max_length}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Gradient checkpointing: {training_config.gradient_checkpointing}")
    print()
    
    # 5. Scientific data configuration
    print("5. Scientific Data Configuration:")
    sci_data = training_config.scientific_data
    print(f"   Data file: {sci_data['data_file']}")
    print(f"   Text fields: {sci_data['text_fields']}")
    print(f"   Remove citations: {sci_data['preprocessing']['remove_citations']}")
    print(f"   Max paper length: {sci_data['preprocessing']['max_paper_length']}")
    print()
    
    # 6. MLflow configuration
    print("6. MLflow Configuration:")
    mlflow_config = training_config.mlflow
    print(f"   Enabled: {mlflow_config['enabled']}")
    print(f"   Experiment name: {mlflow_config['experiment_name']}")
    print(f"   Tracking URI: {mlflow_config['tracking_uri']}")
    print()
    
    # 7. PEFT configuration dictionary
    print("7. PEFT Configuration Dictionary:")
    peft_config = default_config.get_peft_config()
    for key, value in peft_config.items():
        print(f"   {key}: {value}")
    print()
    
    print("=== Configuration Complete ===")
    print("\nNext steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Update configs/training.yaml with your specific settings")
    print("3. Use the enhanced ModelTrainer for QLoRA fine-tuning")


if __name__ == "__main__":
    main()