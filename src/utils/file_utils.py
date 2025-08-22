"""
File utilities for ScientificLLM-Forge.

This module provides functionality for file operations and
data persistence.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def save_json(data: Any, file_path: str) -> None:
        """Save data to a JSON file.
        
        Args:
            data: Data to save
            file_path: Path to the file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load data from a JSON file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    @staticmethod
    def save_pickle(data: Any, file_path: str) -> None:
        """Save data to a pickle file.
        
        Args:
            data: Data to save
            file_path: Path to the file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """Load data from a pickle file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded data
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
            
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure a directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory 