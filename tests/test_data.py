"""
Unit tests for data processing module.
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset_loader import DatasetLoader
from data.preprocessor import DataPreprocessor
from data.validator import DataValidator


class TestDatasetLoader:
    """Test cases for DatasetLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def create_test_data(self):
        """Create test data files."""
        train_data = [
            {"text": "This is a scientific paper about machine learning.", "label": 1},
            {"text": "Another research paper on deep learning.", "label": 1},
            {"text": "A study about natural language processing.", "label": 0}
        ]
        
        val_data = [
            {"text": "Test scientific text.", "label": 1},
            {"text": "Another test text.", "label": 0}
        ]
        
        # Write test data
        with open(Path(self.temp_dir) / "train.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
                
        with open(Path(self.temp_dir) / "validation.jsonl", "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")
    
    def test_dataset_loader_initialization(self):
        """Test DatasetLoader initialization."""
        config = {
            "train_file": str(Path(self.temp_dir) / "train.jsonl"),
            "validation_file": str(Path(self.temp_dir) / "validation.jsonl"),
            "text_column": "text",
            "label_column": "label"
        }
        
        loader = DatasetLoader(config)
        assert loader.config == config
        assert loader.text_column == "text"
        assert loader.label_column == "label"
    
    def test_load_datasets(self):
        """Test loading datasets."""
        config = {
            "train_file": str(Path(self.temp_dir) / "train.jsonl"),
            "validation_file": str(Path(self.temp_dir) / "validation.jsonl"),
            "text_column": "text",
            "label_column": "label"
        }
        
        loader = DatasetLoader(config)
        train_dataset, val_dataset = loader.load_datasets()
        
        assert len(train_dataset) == 3
        assert len(val_dataset) == 2
        assert "text" in train_dataset[0]
        assert "label" in train_dataset[0]


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        config = {
            "max_length": 512,
            "padding": "max_length",
            "truncation": True
        }
        
        preprocessor = DataPreprocessor(config)
        assert preprocessor.max_length == 512
        assert preprocessor.padding == "max_length"
        assert preprocessor.truncation is True
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        config = {
            "max_length": 10,
            "padding": "max_length",
            "truncation": True
        }
        
        preprocessor = DataPreprocessor(config)
        text = "This is a very long text that should be truncated"
        
        result = preprocessor.preprocess_text(text)
        assert len(result) <= 10


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        assert validator is not None
    
    def test_validate_data_format(self):
        """Test data format validation."""
        validator = DataValidator()
        
        valid_data = [
            {"text": "Valid text", "label": 1},
            {"text": "Another valid text", "label": 0}
        ]
        
        invalid_data = [
            {"text": "Missing label"},
            {"label": 1}  # Missing text
        ]
        
        assert validator.validate_data_format(valid_data) is True
        assert validator.validate_data_format(invalid_data) is False


if __name__ == "__main__":
    pytest.main([__file__]) 