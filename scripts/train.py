#!/usr/bin/env python3
"""
Training script for ScientificLLM-Forge.

This script demonstrates how to use the platform for fine-tuning
scientific language models on custom datasets.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.trainer import ModelTrainer
from models.config import ModelConfig
from data.dataset_loader import DatasetLoader
from utils.config import ConfigManager
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a scientific LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("training", log_level)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Override config with command line arguments
        if args.model_name:
            config["training"]["model"]["name"] = args.model_name
        if args.epochs:
            config["training"]["hyperparameters"]["num_epochs"] = args.epochs
        if args.batch_size:
            config["training"]["hyperparameters"]["batch_size"] = args.batch_size
        if args.learning_rate:
            config["training"]["hyperparameters"]["learning_rate"] = args.learning_rate
            
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model configuration
        logger.info("Initializing model configuration...")
        model_config = ModelConfig.from_dict(config["training"])
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_loader = DatasetLoader(config["training"]["data"])
        train_dataset, val_dataset = dataset_loader.load_datasets()
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ModelTrainer(
            model_config=model_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=str(output_dir)
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 