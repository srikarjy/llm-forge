"""
Model configuration for ScientificLLM-Forge.

This module provides functionality for managing model configurations
and hyperparameters.
"""

from typing import Dict, Any


class ModelConfig:
    """Configuration class for model training."""
    
    def __init__(self, **kwargs):
        """Initialize model configuration.
        
        Args:
            **kwargs: Configuration parameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')} 