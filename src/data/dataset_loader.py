"""
Dataset loader for ScientificLLM-Forge.

This module provides functionality for loading and managing
scientific datasets in various formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class DatasetLoader:
    """Load and manage datasets for scientific LLM training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary containing dataset paths and settings
        """
        self.config = config
        self.text_column = config.get("text_column", "text")
        self.label_column = config.get("label_column", "label")
        
    def load_datasets(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        train_file = self.config.get("train_file")
        validation_file = self.config.get("validation_file")
        
        train_dataset = self._load_jsonl(train_file) if train_file else []
        val_dataset = self._load_jsonl(validation_file) if validation_file else []
        
        return train_dataset, val_dataset
    
    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries containing the data
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data 