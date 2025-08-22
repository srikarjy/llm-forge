"""
ScientificLLM-Forge: An MLOps platform for fine-tuning scientific Large Language Models.

This package provides tools and utilities for:
- Data preprocessing and management for scientific datasets
- Model fine-tuning with advanced training techniques
- Model serving and deployment
- Utility functions for scientific ML workflows
"""

__version__ = "0.1.0"
__author__ = "ScientificLLM-Forge Team"
__email__ = "team@scientificllmforge.com"

# Import main modules
from . import data
from . import models
from . import serving
from . import utils

__all__ = ["data", "models", "serving", "utils"] 