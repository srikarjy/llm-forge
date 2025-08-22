"""
Model training and fine-tuning module for ScientificLLM-Forge.

This module provides tools for:
- Model initialization and configuration
- Fine-tuning pipelines for scientific LLMs
- Training loops with advanced techniques
- Model evaluation and metrics
- Checkpoint management
"""

from .trainer import ModelTrainer
from .config import ModelConfig
from .evaluator import ModelEvaluator
from .checkpoint_manager import CheckpointManager

__all__ = [
    "ModelTrainer",
    "ModelConfig",
    "ModelEvaluator", 
    "CheckpointManager"
] 