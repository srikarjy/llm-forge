"""
Tests for enhanced trainer module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

from src.models.enhanced_trainer import (
    EnhancedModelTrainer,
    TrainingMetrics,
    TrainingResults
)
from src.models.qlora_config import QLoRAConfig, TrainingConfig
from src.models.model_loader import ModelInfo


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            eval_loss=0.6,
            learning_rate=2e-4,
            gpu_memory_used_gb=8.5,
            gpu_memory_percent=53.1,
            training_time_seconds=3600.0,
            samples_per_second=10.5
        )
        
        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.train_loss == 0.5
        assert metrics.eval_loss == 0.6
        assert metrics.learning_rate == 2e-4
        assert metrics.gpu_memory_used_gb == 8.5
        assert metrics.gpu_memory_percent == 53.1
        assert metrics.training_time_seconds == 3600.0
        assert metrics.samples_per_second == 10.5
    
    def test_training_metrics_optional_fields(self):
        """Test TrainingMetrics with optional fields."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5
        )
        
        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.train_loss == 0.5
        assert metrics.eval_loss is None
        assert metrics.learning_rate == 0.0
        assert metrics.gpu_memory_used_gb == 0.0


class TestTrainingResults:
    """Test TrainingResults dataclass."""
    
    @pytest.fixture
    def sample_model_info(self):
        """Create sample model info."""
        return ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=1000000,
            trainable_params=100000,
            memory_usage_gb=4.0,
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
    
    @pytest.fixture
    def sample_training_config(self):
        """Create sample training config."""
        return TrainingConfig(
            model_name="test-model",
            learning_rate=2e-4,
            batch_size=1,
            num_epochs=3
        )
    
    def test_training_results_creation(self, sample_model_info, sample_training_config):
        """Test creating TrainingResults."""
        metrics = TrainingMetrics(epoch=3, step=300, train_loss=0.3)
        
        results = TrainingResults(
            model_info=sample_model_info,
            training_config=sample_training_config,
            final_metrics=metrics,
            training_history=[metrics],
            checkpoint_paths=["/path/to/checkpoint"],
            total_training_time=7200.0,
            best_model_path="/path/to/best/model"
        )
        
        assert results.model_info == sample_model_info
        assert results.training_config == sample_training_config
        assert results.final_metrics == metrics
        assert len(results.training_history) == 1
        assert results.checkpoint_paths == ["/path/to/checkpoint"]
        assert results.total_training_time == 7200.0
        assert results.best_model_path == "/path/to/best/model"


class TestEnhancedModelTrainer:
    """Test EnhancedModelTrainer functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_config(self, temp_output_dir):
        """Create sample training config."""
        return TrainingConfig(
            model_name="test-model",
            learning_rate=2e-4,
            batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=1,  # Short for testing
            output_dir=temp_output_dir,
            max_length=512,
            fp16=True,
            gradient_checkpointing=True
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return Dataset.from_dict({
            "text": [
                "This is a sample scientific paper about genomics.",
                "Another paper discussing ENCODE datasets.",
                "Research on transformer models for DNA analysis."
            ],
            "pmid": ["123", "456", "789"],
            "title": ["Paper 1", "Paper 2", "Paper 3"]
        })
    
    def test_init(self, sample_config):
        """Test trainer initialization."""
        trainer = EnhancedModelTrainer(sample_config)
        
        assert trainer.config == sample_config
        assert trainer.output_dir == Path(sample_config.output_dir)
        assert trainer.model_loader is not None
        assert trainer.text_processor is not None
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.training_history == []
    
    def test_init_creates_output_dir(self, temp_output_dir):
        """Test that initialization creates output directory."""
        output_path = Path(temp_output_dir) / "new_output"
        config = TrainingConfig(output_dir=str(output_path))
        
        trainer = EnhancedModelTrainer(config)
        
        assert trainer.output_dir.exists()
        assert trainer.output_dir == output_path
    
    @patch('src.models.enhanced_trainer.MLFLOW_AVAILABLE', True)
    @patch('src.models.enhanced_trainer.mlflow')
    def test_setup_mlflow_enabled(self, mock_mlflow, sample_config):
        """Test MLflow setup when enabled."""
        sample_config.mlflow["enabled"] = True
        sample_config.mlflow["tracking_uri"] = "file:./test_mlruns"
        sample_config.mlflow["experiment_name"] = "test_experiment"
        
        trainer = EnhancedModelTrainer(sample_config)
        
        mock_mlflow.set_tracking_uri.assert_called_once_with("file:./test_mlruns")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    
    @patch('src.models.enhanced_trainer.MLFLOW_AVAILABLE', False)
    def test_setup_mlflow_disabled(self, sample_config):
        """Test MLflow setup when disabled."""
        trainer = EnhancedModelTrainer(sample_config)
        # Should not raise any errors
        assert trainer is not None
    
    def test_dynamic_batch_sizing_no_cuda(self, sample_config):
        """Test dynamic batch sizing when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = EnhancedModelTrainer(sample_config)
            batch_size = trainer.dynamic_batch_sizing(initial_batch_size=4)
            
            assert batch_size == 4  # Should return initial batch size
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('src.models.enhanced_trainer.ModelLoader')
    def test_dynamic_batch_sizing_with_cuda(self, mock_model_loader_class, mock_props, sample_config):
        """Test dynamic batch sizing with CUDA available."""
        # Mock GPU properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1024**3  # 16 GB
        mock_props.return_value = mock_device_props
        
        # Mock ModelLoader instance
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        # Create real config instead of using fixture (which might be mocked)
        real_config = TrainingConfig(
            model_name="test-model",
            max_memory_usage=0.9,
            max_length=512,
            batch_size=4
        )
        
        trainer = EnhancedModelTrainer(real_config)
        trainer.model_info = ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=1000000,
            trainable_params=100000,
            memory_usage_gb=4.0,
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
        
        batch_size = trainer.dynamic_batch_sizing(initial_batch_size=8)
        
        # Should return a reasonable batch size
        assert isinstance(batch_size, int)
        assert batch_size >= 1
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=4*1024**3)  # 4 GB
    @patch('torch.cuda.memory_reserved', return_value=6*1024**3)   # 6 GB
    @patch('torch.cuda.get_device_properties')
    @patch('src.models.enhanced_trainer.ModelLoader')
    def test_monitor_gpu_memory_with_cuda(self, mock_model_loader_class, mock_props, mock_reserved, mock_allocated, sample_config):
        """Test GPU memory monitoring with CUDA."""
        # Mock GPU properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1024**3  # 16 GB
        mock_props.return_value = mock_device_props
        
        # Mock ModelLoader instance
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        trainer = EnhancedModelTrainer(sample_config)
        memory_stats = trainer.monitor_gpu_memory()
        
        assert "allocated_gb" in memory_stats
        assert "reserved_gb" in memory_stats
        assert "total_gb" in memory_stats
        assert "free_gb" in memory_stats
        assert "utilization_percent" in memory_stats
        
        assert memory_stats["allocated_gb"] == 4.0
        assert memory_stats["reserved_gb"] == 6.0
        assert memory_stats["total_gb"] == 16.0
        assert memory_stats["free_gb"] == 10.0
        assert memory_stats["utilization_percent"] == 37.5  # 6/16 * 100
    
    def test_monitor_gpu_memory_no_cuda(self, sample_config):
        """Test GPU memory monitoring without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = EnhancedModelTrainer(sample_config)
            memory_stats = trainer.monitor_gpu_memory()
            
            assert memory_stats["allocated_gb"] == 0.0
            assert memory_stats["reserved_gb"] == 0.0
            assert memory_stats["total_gb"] == 0.0
            assert memory_stats["free_gb"] == 0.0
            assert memory_stats["utilization_percent"] == 0.0
    
    def test_save_training_checkpoint(self, sample_config):
        """Test saving training checkpoint."""
        trainer = EnhancedModelTrainer(sample_config)
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer
        
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            eval_loss=0.6
        )
        
        checkpoint_path = trainer.save_training_checkpoint(1, metrics)
        
        # Check that checkpoint directory was created
        checkpoint_dir = Path(checkpoint_path)
        assert checkpoint_dir.exists()
        assert checkpoint_dir.name == "checkpoint-epoch-1"
        
        # Check that model and tokenizer save methods were called
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        
        # Check that metrics file was created
        metrics_file = checkpoint_dir / "training_metrics.json"
        assert metrics_file.exists()
        
        # Check that config file was created
        config_file = checkpoint_dir / "training_config.json"
        assert config_file.exists()
        
        # Verify metrics content
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics["epoch"] == 1
        assert saved_metrics["step"] == 100
        assert saved_metrics["train_loss"] == 0.5
        assert saved_metrics["eval_loss"] == 0.6
    
    def test_cleanup_memory(self, sample_config):
        """Test memory cleanup."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect, \
             patch('src.models.enhanced_trainer.ModelLoader'):
            
            trainer = EnhancedModelTrainer(sample_config)
            trainer.cleanup_memory()
            
            mock_empty_cache.assert_called_once()
            mock_gc_collect.assert_called_once()
    
    def test_get_training_summary_no_model(self, sample_config):
        """Test getting training summary without loaded model."""
        trainer = EnhancedModelTrainer(sample_config)
        summary = trainer.get_training_summary()
        
        assert "error" in summary
        assert summary["error"] == "No model loaded"
    
    def test_get_training_summary_with_model(self, sample_config):
        """Test getting training summary with loaded model."""
        trainer = EnhancedModelTrainer(sample_config)
        
        # Mock model info
        trainer.model_info = ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=1000000,
            trainable_params=100000,
            memory_usage_gb=4.0,
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
        
        with patch.object(trainer, 'monitor_gpu_memory', return_value={
            "allocated_gb": 4.0,
            "reserved_gb": 6.0,
            "total_gb": 16.0,
            "free_gb": 10.0,
            "utilization_percent": 37.5
        }):
            summary = trainer.get_training_summary()
        
        assert "model_info" in summary
        assert "memory_usage" in summary
        assert "training_config" in summary
        assert "optimization" in summary
        
        # Check model info
        model_info = summary["model_info"]
        assert model_info["name"] == "test-model"
        assert model_info["total_params"] == 1000000
        assert model_info["trainable_params"] == 100000
        assert model_info["trainable_percent"] == 10.0  # 100k/1M * 100
        assert model_info["quantized"] is True
        assert model_info["has_lora"] is True
        
        # Check training config
        training_config = summary["training_config"]
        assert training_config["learning_rate"] == 2e-4
        assert training_config["batch_size"] == 1
        assert training_config["gradient_accumulation_steps"] == 4
        assert training_config["effective_batch_size"] == 4  # 1 * 4
        
        # Check optimization settings
        optimization = summary["optimization"]
        assert optimization["fp16"] is True
        assert optimization["gradient_checkpointing"] is True
        assert optimization["lora_enabled"] is True
        assert optimization["quantization"] == "4-bit"
    
    @patch('src.models.enhanced_trainer.TRANSFORMERS_AVAILABLE', False)
    def test_setup_model_and_data_no_transformers(self, sample_config):
        """Test setup when transformers not available."""
        trainer = EnhancedModelTrainer(sample_config)
        
        with pytest.raises(ImportError, match="Transformers library required"):
            trainer.setup_model_and_data()
    
    def test_legacy_train_method(self, sample_config):
        """Test legacy train method for compatibility."""
        trainer = EnhancedModelTrainer(sample_config)
        
        # Mock the train_with_memory_optimization method
        mock_results = TrainingResults(
            model_info=Mock(),
            training_config=sample_config,
            final_metrics=TrainingMetrics(epoch=1, step=100, train_loss=0.5),
            training_history=[],
            checkpoint_paths=[],
            total_training_time=3600.0
        )
        
        with patch.object(trainer, 'train_with_memory_optimization', return_value=mock_results):
            trainer.train()  # Should not raise any errors
    
    def test_legacy_save_model_no_trainer(self, sample_config):
        """Test legacy save model method without trainer."""
        trainer = EnhancedModelTrainer(sample_config)
        
        # Should handle case where no trainer is available
        trainer.save_model()  # Should not raise errors, just log warning
    
    def test_legacy_save_model_with_trainer(self, sample_config):
        """Test legacy save model method with trainer."""
        trainer = EnhancedModelTrainer(sample_config)
        
        # Mock trainer
        mock_trainer = Mock()
        trainer.trainer = mock_trainer
        
        trainer.save_model()
        
        # Should call save_model on the trainer
        mock_trainer.save_model.assert_called_once()


class TestEnhancedTrainerIntegration:
    """Integration tests for enhanced trainer."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def integration_config(self, temp_output_dir):
        """Create integration test config."""
        return TrainingConfig(
            model_name="test-model",
            learning_rate=2e-4,
            batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=1,
            output_dir=temp_output_dir,
            max_length=128,  # Small for testing
            fp16=False,  # Disable for testing
            gradient_checkpointing=False,  # Disable for testing
            mlflow={"enabled": False}  # Disable MLflow for testing
        )
    
    def test_trainer_initialization_and_cleanup(self, integration_config):
        """Test trainer initialization and cleanup."""
        trainer = EnhancedModelTrainer(integration_config)
        
        # Check initialization
        assert trainer.config == integration_config
        assert trainer.output_dir.exists()
        assert trainer.model_loader is not None
        assert trainer.text_processor is not None
        
        # Test cleanup
        trainer.cleanup_memory()  # Should not raise errors
    
    def test_training_summary_evolution(self, integration_config):
        """Test how training summary changes as model is loaded."""
        trainer = EnhancedModelTrainer(integration_config)
        
        # Initially no model
        summary1 = trainer.get_training_summary()
        assert "error" in summary1
        
        # After setting model info
        trainer.model_info = ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=1000000,
            trainable_params=50000,
            memory_usage_gb=2.0,
            quantized=True,
            has_lora=True,
            device="cpu"
        )
        
        summary2 = trainer.get_training_summary()
        assert "error" not in summary2
        assert summary2["model_info"]["trainable_percent"] == 5.0  # 50k/1M * 100


if __name__ == "__main__":
    pytest.main([__file__])