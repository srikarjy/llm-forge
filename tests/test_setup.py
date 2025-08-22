"""
Test setup verification for ScientificLLM-Forge.

This module contains tests to verify that the project setup
is working correctly.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all main modules can be imported."""
    # Test data module imports
    from data.dataset_loader import DatasetLoader
    from data.preprocessor import DataPreprocessor
    from data.validator import DataValidator
    from data.augmentor import DataAugmentor
    
    # Test models module imports
    from models.trainer import ModelTrainer
    from models.config import ModelConfig
    from models.evaluator import ModelEvaluator
    from models.checkpoint_manager import CheckpointManager
    
    # Test serving module imports
    from serving.server import ModelServer
    from serving.inference import InferenceEngine
    from serving.api import APIRouter
    from serving.deployment import DeploymentManager
    
    # Test utils module imports
    from utils.config import ConfigManager
    from utils.logger import setup_logger
    from utils.metrics import MetricsCollector
    from utils.file_utils import FileUtils
    
    assert True  # If we get here, all imports worked


def test_config_manager():
    """Test ConfigManager functionality."""
    from utils.config import ConfigManager
    
    config_manager = ConfigManager()
    assert config_manager is not None


def test_dataset_loader():
    """Test DatasetLoader functionality."""
    from data.dataset_loader import DatasetLoader
    
    config = {
        "train_file": "test_train.jsonl",
        "validation_file": "test_val.jsonl",
        "text_column": "text",
        "label_column": "label"
    }
    
    loader = DatasetLoader(config)
    assert loader.config == config
    assert loader.text_column == "text"
    assert loader.label_column == "label"


def test_model_config():
    """Test ModelConfig functionality."""
    from models.config import ModelConfig
    
    config_dict = {
        "model_name": "test-model",
        "learning_rate": 1e-4,
        "batch_size": 16
    }
    
    config = ModelConfig.from_dict(config_dict)
    assert config.model_name == "test-model"
    assert config.learning_rate == 1e-4
    assert config.batch_size == 16


def test_metrics_collector():
    """Test MetricsCollector functionality."""
    from utils.metrics import MetricsCollector
    
    collector = MetricsCollector()
    collector.add_metric("test_metric", 42)
    
    metrics = collector.get_metrics()
    assert "test_metric" in metrics
    assert metrics["test_metric"] == 42


def test_file_utils():
    """Test FileUtils functionality."""
    from utils.file_utils import FileUtils
    
    # Test directory creation
    test_dir = Path("test_dir")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    created_dir = FileUtils.ensure_directory("test_dir")
    assert created_dir.exists()
    assert created_dir.is_dir()
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    pytest.main([__file__]) 