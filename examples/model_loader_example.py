#!/usr/bin/env python3
"""
Example script demonstrating ModelLoader usage.

This script shows how to load models with quantization and LoRA
for memory-efficient fine-tuning of large language models.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_loader import ModelLoader
from models.qlora_config import QLoRAConfig, get_model_config, estimate_memory_usage


def main():
    """Demonstrate ModelLoader usage."""
    
    print("=== Model Loader Example ===\n")
    
    # 1. Initialize model loader
    print("1. Initializing Model Loader:")
    loader = ModelLoader()
    print(f"   ✅ Initialized ModelLoader")
    print(f"   Device: {loader.device}")
    print(f"   Transformers available: {hasattr(loader, 'TRANSFORMERS_AVAILABLE')}")
    print(f"   PEFT available: {hasattr(loader, 'PEFT_AVAILABLE')}")
    
    # 2. List supported models
    print("\n2. Supported Models:")
    supported_models = loader.list_supported_models()
    
    for family, models in supported_models.items():
        print(f"   {family.upper()} models:")
        for model in models[:3]:  # Show first 3 models
            print(f"     - {model}")
        if len(models) > 3:
            print(f"     ... and {len(models) - 3} more")
        print()
    
    # 3. Model compatibility validation
    print("3. Model Compatibility Validation:")
    
    test_models = [
        "meta-llama/Llama-2-7b-hf",
        "bert-base-uncased",
        "microsoft/DialoGPT-medium",
        "unknown/model",
        "organization/custom-model"
    ]
    
    for model_name in test_models:
        is_compatible = loader.validate_model_compatibility(model_name)
        family = loader.get_model_family(model_name)
        status = "✅ Compatible" if is_compatible else "❌ Not supported"
        print(f"   {model_name}: {status} (Family: {family})")
    
    # 4. Target modules configuration
    print("\n4. Target Modules Configuration:")
    
    model_examples = [
        "meta-llama/Llama-2-7b-hf",
        "bert-base-uncased",
        "microsoft/DialoGPT-medium"
    ]
    
    for model_name in model_examples:
        targets = loader.configure_target_modules(model_name)
        family = loader.get_model_family(model_name)
        print(f"   {family.upper()} ({model_name.split('/')[-1]}):")
        print(f"     Target modules: {targets}")
    
    # 5. QLoRA configuration examples
    print("\n5. QLoRA Configuration Examples:")
    
    # LLaMA-2 7B configuration
    llama_config = get_model_config("meta-llama/Llama-2-7b-hf")
    print(f"   LLaMA-2 7B QLoRA Config:")
    print(f"     LoRA rank: {llama_config.r}")
    print(f"     LoRA alpha: {llama_config.lora_alpha}")
    print(f"     Target modules: {llama_config.target_modules}")
    print(f"     4-bit quantization: {llama_config.load_in_4bit}")
    print(f"     Quantization type: {llama_config.bnb_4bit_quant_type}")
    
    # BERT configuration
    bert_config = get_model_config("bert-base-uncased")
    print(f"\n   BERT QLoRA Config:")
    print(f"     LoRA rank: {bert_config.r}")
    print(f"     LoRA alpha: {bert_config.lora_alpha}")
    print(f"     Target modules: {bert_config.target_modules}")
    
    # 6. Memory usage estimation
    print("\n6. Memory Usage Estimation:")
    
    models_to_estimate = [
        ("meta-llama/Llama-2-7b-hf", llama_config),
        ("bert-base-uncased", bert_config)
    ]
    
    for model_name, config in models_to_estimate:
        memory_info = estimate_memory_usage(model_name, config)
        model_size = model_name.split('/')[-1]
        
        print(f"   {model_size}:")
        print(f"     Base model: {memory_info['base_model_gb']:.2f} GB")
        print(f"     LoRA params: {memory_info['lora_params_gb']:.2f} GB")
        print(f"     Activations: {memory_info['activations_gb']:.2f} GB")
        print(f"     Optimizer: {memory_info['optimizer_gb']:.2f} GB")
        print(f"     Total estimated: {memory_info['total_gb']:.2f} GB")
        print(f"     Fits in 16GB: {'✅ Yes' if memory_info['fits_16gb'] else '❌ No'}")
        print()
    
    # 7. Quantization configuration
    print("7. Quantization Configuration:")
    
    # Test different quantization settings
    configs_to_test = [
        ("4-bit NF4", QLoRAConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")),
        ("4-bit FP4", QLoRAConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")),
        ("8-bit", QLoRAConfig(load_in_8bit=True, load_in_4bit=False)),
        ("No quantization", QLoRAConfig(load_in_4bit=False, load_in_8bit=False))
    ]
    
    for config_name, config in configs_to_test:
        try:
            quant_config = loader.setup_quantization_config(config)
            if quant_config is not None:
                print(f"   {config_name}: ✅ Configured")
            else:
                print(f"   {config_name}: ⚪ Disabled")
        except Exception as e:
            print(f"   {config_name}: ❌ Error - {e}")
    
    # 8. Memory constraint validation
    print("\n8. Memory Constraint Validation:")
    
    # Create mock model info for different scenarios
    from models.model_loader import ModelInfo
    
    scenarios = [
        {
            "name": "LLaMA-2 7B with QLoRA",
            "info": ModelInfo(
                model_name="meta-llama/Llama-2-7b-hf",
                model_type="llama",
                total_params=7_000_000_000,
                trainable_params=100_000_000,  # ~100M with LoRA
                memory_usage_gb=3.5,  # 4-bit quantized
                quantized=True,
                has_lora=True,
                device="cuda:0"
            )
        },
        {
            "name": "LLaMA-2 7B full fine-tuning",
            "info": ModelInfo(
                model_name="meta-llama/Llama-2-7b-hf",
                model_type="llama",
                total_params=7_000_000_000,
                trainable_params=7_000_000_000,  # All parameters
                memory_usage_gb=14.0,  # Full precision
                quantized=False,
                has_lora=False,
                device="cuda:0"
            )
        },
        {
            "name": "BERT-base with LoRA",
            "info": ModelInfo(
                model_name="bert-base-uncased",
                model_type="bert",
                total_params=110_000_000,
                trainable_params=10_000_000,  # ~10M with LoRA
                memory_usage_gb=0.5,  # Small model
                quantized=True,
                has_lora=True,
                device="cuda:0"
            )
        }
    ]
    
    for scenario in scenarios:
        fits, message = loader.validate_memory_constraints(scenario["info"], max_memory_gb=16.0)
        status = "✅ Fits" if fits else "❌ Exceeds"
        print(f"   {scenario['name']}: {status}")
        print(f"     {message}")
        print()
    
    # 9. Model loading simulation (without actual loading)
    print("9. Model Loading Simulation:")
    
    print("   Note: Actual model loading requires:")
    print("   - HuggingFace account with access to gated models (LLaMA-2)")
    print("   - Sufficient GPU memory")
    print("   - Internet connection for model download")
    print()
    
    print("   Example loading process:")
    print("   1. Validate model compatibility ✅")
    print("   2. Setup quantization config ✅")
    print("   3. Load tokenizer ⏳ (requires network)")
    print("   4. Load model with quantization ⏳ (requires network + GPU)")
    print("   5. Attach LoRA adapters ⏳ (requires PEFT)")
    print("   6. Optimize memory usage ⏳")
    print("   7. Validate memory constraints ✅")
    
    # 10. Best practices and recommendations
    print("\n10. Best Practices and Recommendations:")
    
    print("   Memory Optimization:")
    print("   - Use 4-bit quantization (NF4) for maximum memory savings")
    print("   - Enable gradient checkpointing")
    print("   - Use LoRA with rank 16-64 for good performance/memory trade-off")
    print("   - Set batch size to 1 with gradient accumulation")
    
    print("\n   Model Selection:")
    print("   - LLaMA-2 7B: Best for general genomics tasks")
    print("   - BioBERT: Good for biomedical text understanding")
    print("   - Start with smaller models for experimentation")
    
    print("\n   Training Configuration:")
    print("   - Use mixed precision (fp16) training")
    print("   - Monitor GPU memory usage during training")
    print("   - Save checkpoints frequently")
    print("   - Use learning rate 2e-4 for QLoRA")
    
    print("\n=== Model Loader Example Complete ===")
    print("\nNext steps:")
    print("1. Install required dependencies: transformers, peft, bitsandbytes")
    print("2. Set up HuggingFace authentication for gated models")
    print("3. Test with a small model first (e.g., BERT-base)")
    print("4. Integrate with the enhanced ModelTrainer for actual training")


if __name__ == "__main__":
    main()