"""
Tests for QLoRA configuration module.
"""

import pytest
from src.models.qlora_config import (
    QLoRAConfig, 
    TrainingConfig, 
    get_model_config, 
    estimate_memory_usage,
    LLAMA2_7B_CONFIG,
    BERT_CONFIG
)


class TestQLoRAConfig:
    """Test QLoRA configuration functionality."""
    
    def test_default_qlora_config(self):
        """Test default QLoRA configuration."""
        config = QLoRAConfig()
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
    
    def test_quantization_config(self):
        """Test quantization configuration generation."""
        config = QLoRAConfig()
        
        # Test that we can create the config without bitsandbytes installed
        try:
            quant_config = config.get_quantization_config()
            if quant_config is not None:
                assert quant_config.load_in_4bit is True
                assert quant_config.bnb_4bit_use_double_quant is True
                assert quant_config.bnb_4bit_quant_type == "nf4"
        except ImportError as e:
            # Skip test if bitsandbytes is not installed
            pytest.skip(f"bitsandbytes not installed: {e}")
    
    def test_peft_config(self):
        """Test PEFT configuration generation."""
        config = QLoRAConfig()
        peft_config = config.get_peft_config()
        
        assert peft_config["r"] == 16
        assert peft_config["lora_alpha"] == 32
        assert peft_config["lora_dropout"] == 0.1
        assert peft_config["task_type"] == "CAUSAL_LM"
        assert "q_proj" in peft_config["target_modules"]
    
    def test_no_quantization_config(self):
        """Test configuration without quantization."""
        config = QLoRAConfig(load_in_4bit=False, load_in_8bit=False)
        quant_config = config.get_quantization_config()
        
        assert quant_config is None


class TestTrainingConfig:
    """Test training configuration functionality."""
    
    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.max_length == 2048
        assert config.learning_rate == 2e-4
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 16
        assert config.gradient_checkpointing is True
    
    def test_scientific_data_config(self):
        """Test scientific data configuration."""
        config = TrainingConfig()
        
        assert config.scientific_data["data_file"] == "data/high_quality_papers_demo.json"
        assert "title" in config.scientific_data["text_fields"]
        assert "abstract" in config.scientific_data["text_fields"]
        assert config.scientific_data["preprocessing"]["remove_citations"] is True
    
    def test_mlflow_config(self):
        """Test MLflow configuration."""
        config = TrainingConfig()
        
        assert config.mlflow["enabled"] is True
        assert config.mlflow["experiment_name"] == "genomics-llm-finetuning"
        assert config.mlflow["tracking_uri"] == "file:./mlruns"


class TestModelConfigs:
    """Test model-specific configurations."""
    
    def test_llama2_config(self):
        """Test LLaMA-2 specific configuration."""
        config = get_model_config("meta-llama/Llama-2-7b-hf")
        
        assert config.r == 16
        assert "gate_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules
    
    def test_bert_config(self):
        """Test BERT specific configuration."""
        config = get_model_config("bert-base-uncased")
        
        assert config.r == 8
        assert "query" in config.target_modules
        assert "value" in config.target_modules
        assert "key" in config.target_modules
    
    def test_unknown_model_config(self):
        """Test configuration for unknown model."""
        config = get_model_config("unknown-model")
        
        # Should return default configuration
        assert config.r == 16
        assert config.lora_alpha == 32


class TestMemoryEstimation:
    """Test memory usage estimation."""
    
    def test_memory_estimation_7b(self):
        """Test memory estimation for 7B model."""
        config = QLoRAConfig()
        memory_info = estimate_memory_usage("llama-7b", config)
        
        assert "total_gb" in memory_info
        assert "base_model_gb" in memory_info
        assert "lora_params_gb" in memory_info
        assert "fits_16gb" in memory_info
        
        # Should fit in 16GB with QLoRA
        assert memory_info["fits_16gb"] is True
        assert memory_info["total_gb"] < 16.0
    
    def test_memory_estimation_components(self):
        """Test individual memory components."""
        config = QLoRAConfig()
        memory_info = estimate_memory_usage("llama-7b", config)
        
        # All components should be positive
        assert memory_info["base_model_gb"] > 0
        assert memory_info["lora_params_gb"] > 0
        assert memory_info["activations_gb"] > 0
        assert memory_info["optimizer_gb"] > 0
        
        # Total should be sum of components
        expected_total = (
            memory_info["base_model_gb"] + 
            memory_info["lora_params_gb"] + 
            memory_info["activations_gb"] + 
            memory_info["optimizer_gb"]
        )
        assert abs(memory_info["total_gb"] - expected_total) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])