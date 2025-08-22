#!/usr/bin/env python3
"""
Example script demonstrating EnhancedModelTrainer usage.

This script shows how to use the enhanced trainer for memory-efficient
LLM fine-tuning with QLoRA on scientific papers.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.enhanced_trainer import EnhancedModelTrainer, TrainingMetrics
from models.qlora_config import QLoRAConfig, TrainingConfig
from data.scientific_dataset import ScientificDataModule
from data.text_processor import ScientificTextProcessor


def main():
    """Demonstrate EnhancedModelTrainer usage."""
    
    print("=== Enhanced Model Trainer Example ===\n")
    
    # 1. Setup training configuration
    print("1. Setting up Training Configuration:")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create QLoRA configuration for memory-efficient training
        qlora_config = QLoRAConfig(
            r=16,                    # LoRA rank
            lora_alpha=32,           # LoRA scaling
            lora_dropout=0.1,        # LoRA dropout
            load_in_4bit=True,       # 4-bit quantization
            bnb_4bit_quant_type="nf4"  # NF4 quantization
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            model_name="microsoft/DialoGPT-medium",  # Smaller model for demo
            qlora=qlora_config,
            learning_rate=2e-4,
            batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=1,  # Short training for demo
            max_length=512,  # Shorter sequences for demo
            output_dir=temp_dir,
            fp16=True,
            gradient_checkpointing=True,
            mlflow={"enabled": False},  # Disable MLflow for demo
            scientific_data={
                "data_file": "data/high_quality_papers_demo.json",
                "text_fields": ["title", "abstract"],
                "preprocessing": {
                    "remove_citations": True,
                    "normalize_scientific_notation": True,
                    "handle_special_tokens": True,
                    "max_paper_length": 1024,
                }
            }
        )
        
        print(f"   ✅ Model: {training_config.model_name}")
        print(f"   ✅ LoRA rank: {qlora_config.r}")
        print(f"   ✅ Quantization: {'4-bit' if qlora_config.load_in_4bit else 'None'}")
        print(f"   ✅ Learning rate: {training_config.learning_rate}")
        print(f"   ✅ Batch size: {training_config.batch_size}")
        print(f"   ✅ Gradient accumulation: {training_config.gradient_accumulation_steps}")
        print(f"   ✅ Output directory: {training_config.output_dir}")
        
        # 2. Initialize enhanced trainer
        print("\n2. Initializing Enhanced Trainer:")
        
        trainer = EnhancedModelTrainer(training_config)
        
        print(f"   ✅ Trainer initialized")
        print(f"   ✅ Output directory created: {trainer.output_dir.exists()}")
        print(f"   ✅ Model loader available: {trainer.model_loader is not None}")
        print(f"   ✅ Text processor available: {trainer.text_processor is not None}")
        
        # 3. Load and prepare scientific data
        print("\n3. Loading Scientific Data:")
        
        data_file = "data/high_quality_papers_demo.json"
        
        if Path(data_file).exists():
            print(f"   📁 Loading papers from: {data_file}")
            
            # Load papers using ScientificDataModule
            data_module = ScientificDataModule(data_file)
            papers = data_module.load_papers(min_quality_score=100)
            
            print(f"   ✅ Loaded {len(papers)} high-quality papers")
            
            # Show paper details
            for i, paper in enumerate(papers[:2]):  # Show first 2 papers
                print(f"   Paper {i+1}: {paper.title[:50]}...")
                print(f"     Score: {paper.score}, Tier: {paper.tier.value}")
                print(f"     Benchmarks: {', '.join(paper.benchmarks_used)}")
            
            # Process papers into training dataset
            text_processor = ScientificTextProcessor()
            dataset = text_processor.process_papers_for_training(papers, format_type="causal_lm")
            
            print(f"   ✅ Created training dataset with {len(dataset)} samples")
            
            # Show sample training text
            if len(dataset) > 0:
                sample_text = dataset[0]["text"]
                print(f"   Sample training text (first 200 chars):")
                print(f"   {sample_text[:200]}...")
        
        else:
            print(f"   ❌ Data file not found: {data_file}")
            print("   Creating mock dataset for demonstration...")
            
            # Create mock dataset for demo
            from datasets import Dataset
            dataset = Dataset.from_dict({
                "text": [
                    "This is a sample genomics paper about ENCODE datasets and transformer models.",
                    "Research on TCGA data using deep learning approaches for cancer genomics.",
                    "Analysis of GTEx expression data with attention mechanisms."
                ],
                "pmid": ["123", "456", "789"]
            })
            
            print(f"   ✅ Created mock dataset with {len(dataset)} samples")
        
        # 4. Memory usage analysis
        print("\n4. Memory Usage Analysis:")
        
        # Get initial memory stats
        memory_stats = trainer.monitor_gpu_memory()
        
        print(f"   GPU Memory Status:")
        print(f"     Total: {memory_stats['total_gb']:.1f} GB")
        print(f"     Allocated: {memory_stats['allocated_gb']:.1f} GB")
        print(f"     Reserved: {memory_stats['reserved_gb']:.1f} GB")
        print(f"     Free: {memory_stats['free_gb']:.1f} GB")
        print(f"     Utilization: {memory_stats['utilization_percent']:.1f}%")
        
        # Test dynamic batch sizing
        optimal_batch_size = trainer.dynamic_batch_sizing(initial_batch_size=4)
        print(f"   Optimal batch size: {optimal_batch_size}")
        
        # 5. Training summary before training
        print("\n5. Training Summary (Before Training):")
        
        summary = trainer.get_training_summary()
        
        if "error" in summary:
            print(f"   ⚠️  {summary['error']}")
            print("   (This is expected before model loading)")
        else:
            print(f"   Model: {summary['model_info']['name']}")
            print(f"   Total parameters: {summary['model_info']['total_params']:,}")
            print(f"   Trainable parameters: {summary['model_info']['trainable_params']:,}")
            print(f"   Trainable percentage: {summary['model_info']['trainable_percent']:.2f}%")
        
        # 6. Checkpoint management demonstration
        print("\n6. Checkpoint Management:")
        
        # Create sample training metrics
        sample_metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            eval_loss=0.6,
            learning_rate=2e-4,
            gpu_memory_used_gb=memory_stats['allocated_gb'],
            gpu_memory_percent=memory_stats['utilization_percent'],
            training_time_seconds=3600.0,
            samples_per_second=10.5
        )
        
        # Save a sample checkpoint
        checkpoint_path = trainer.save_training_checkpoint(1, sample_metrics)
        
        print(f"   ✅ Saved checkpoint to: {checkpoint_path}")
        
        # Verify checkpoint contents
        checkpoint_dir = Path(checkpoint_path)
        files = list(checkpoint_dir.glob("*"))
        print(f"   Checkpoint contains {len(files)} files:")
        for file in files:
            print(f"     - {file.name}")
        
        # 7. Training simulation (without actual model loading)
        print("\n7. Training Pipeline Simulation:")
        
        print("   Training pipeline steps:")
        print("   1. ✅ Load and validate training configuration")
        print("   2. ✅ Initialize enhanced trainer")
        print("   3. ✅ Load and preprocess scientific papers")
        print("   4. ✅ Create training dataset")
        print("   5. ✅ Monitor GPU memory usage")
        print("   6. ✅ Optimize batch size")
        print("   7. ✅ Save training checkpoints")
        print("   8. ⏳ Load model with quantization (requires network/GPU)")
        print("   9. ⏳ Attach LoRA adapters (requires PEFT)")
        print("   10. ⏳ Run training loop (requires full ML stack)")
        print("   11. ⏳ Save final model")
        
        print("\n   Note: Steps 8-11 require:")
        print("   - Internet connection for model download")
        print("   - GPU with sufficient memory")
        print("   - Full ML libraries (transformers, peft, bitsandbytes)")
        
        # 8. Memory cleanup
        print("\n8. Memory Cleanup:")
        
        trainer.cleanup_memory()
        
        # Check memory after cleanup
        memory_after = trainer.monitor_gpu_memory()
        print(f"   Memory after cleanup:")
        print(f"     Allocated: {memory_after['allocated_gb']:.1f} GB")
        print(f"     Reserved: {memory_after['reserved_gb']:.1f} GB")
        print(f"     Utilization: {memory_after['utilization_percent']:.1f}%")
        
        # 9. Configuration recommendations
        print("\n9. Configuration Recommendations:")
        
        print("   For actual training:")
        print("   - Use LLaMA-2 7B for best genomics performance")
        print("   - Enable 4-bit quantization (NF4) for memory efficiency")
        print("   - Set LoRA rank 16-32 for good performance/memory trade-off")
        print("   - Use batch size 1 with gradient accumulation 16-32")
        print("   - Enable gradient checkpointing and mixed precision")
        print("   - Monitor GPU memory usage during training")
        print("   - Save checkpoints frequently")
        
        print("\n   Memory optimization tips:")
        print("   - Use smaller max_length for shorter papers")
        print("   - Reduce LoRA rank if memory is tight")
        print("   - Use gradient accumulation instead of larger batch sizes")
        print("   - Clear GPU cache between training runs")
        
        print("\n   Scientific data tips:")
        print("   - Filter papers by quality score (>100 recommended)")
        print("   - Use instruction format for better task performance")
        print("   - Include paper metadata in training text")
        print("   - Preprocess text to remove citations and normalize notation")
        
        # 10. Integration with other components
        print("\n10. Integration Summary:")
        
        print("   The EnhancedModelTrainer integrates with:")
        print("   ✅ QLoRAConfig (Task 1) - Memory-efficient training configuration")
        print("   ✅ ModelLoader (Task 3.1) - Model loading with quantization")
        print("   ✅ ScientificDataModule (Task 2.1) - High-quality paper loading")
        print("   ✅ ScientificTextProcessor (Task 2.2) - Text preprocessing")
        print("   ✅ MLflow - Experiment tracking and logging")
        print("   ✅ HuggingFace Transformers - Training infrastructure")
        print("   ✅ PEFT - Parameter-efficient fine-tuning")
        
        print("\n   Ready for:")
        print("   - Memory-efficient LLM fine-tuning")
        print("   - Scientific paper domain adaptation")
        print("   - Genomics-specific model training")
        print("   - Experiment tracking and reproducibility")
    
    print("\n=== Enhanced Trainer Example Complete ===")
    print("\nNext steps:")
    print("1. Install full ML stack: transformers, peft, bitsandbytes")
    print("2. Set up HuggingFace authentication for gated models")
    print("3. Configure MLflow for experiment tracking")
    print("4. Run actual training with train_with_memory_optimization()")


if __name__ == "__main__":
    main()