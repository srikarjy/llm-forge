"""
Tests for model loader module.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.models.model_loader import ModelLoader, ModelInfo
from src.models.qlora_config import QLoRAConfig


class TestModelLoader:
    """Test ModelLoader functionality."""
    
    @pytest.fixture
    def model_loader(self):
        """Create model loader for testing."""
        return ModelLoader()
    
    @pytest.fixture
    def sample_qlora_config(self):
        """Create sample QLoRA config for testing."""
        return QLoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
    
    def test_init(self, model_loader):
        """Test model loader initialization."""
        assert model_loader.device is not None
        assert isinstance(model_loader.loaded_models, dict)
        assert len(model_loader.SUPPORTED_MODELS) > 0
    
    def test_get_best_device(self, model_loader):
        """Test device selection."""
        device = model_loader._get_best_device()
        assert isinstance(device, str)
        assert device in ["cpu", "mps"] or device.startswith("cuda:")
    
    def test_validate_model_compatibility_supported(self, model_loader):
        """Test validation of supported models."""
        # Test LLaMA models
        assert model_loader.validate_model_compatibility("meta-llama/Llama-2-7b-hf") is True
        assert model_loader.validate_model_compatibility("huggyllama/llama-7b") is True
        
        # Test BERT models
        assert model_loader.validate_model_compatibility("bert-base-uncased") is True
        assert model_loader.validate_model_compatibility("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") is True
        
        # Test GPT models
        assert model_loader.validate_model_compatibility("microsoft/DialoGPT-medium") is True
        assert model_loader.validate_model_compatibility("gpt2") is True
    
    def test_validate_model_compatibility_unsupported(self, model_loader):
        """Test validation of unsupported models."""
        assert model_loader.validate_model_compatibility("invalid-model") is False
        assert model_loader.validate_model_compatibility("") is False
    
    def test_validate_model_compatibility_huggingface_format(self, model_loader):
        """Test validation of HuggingFace format models."""
        # Should return True for valid HF format even if not in supported list
        assert model_loader.validate_model_compatibility("organization/model-name") is True
        assert model_loader.validate_model_compatibility("user/custom-model") is True
    
    def test_get_model_family_llama(self, model_loader):
        """Test getting model family for LLaMA models."""
        assert model_loader.get_model_family("meta-llama/Llama-2-7b-hf") == "llama"
        assert model_loader.get_model_family("huggyllama/llama-7b") == "llama"
        assert model_loader.get_model_family("some-llama-model") == "llama"
    
    def test_get_model_family_bert(self, model_loader):
        """Test getting model family for BERT models."""
        assert model_loader.get_model_family("bert-base-uncased") == "bert"
        assert model_loader.get_model_family("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") == "bert"
    
    def test_get_model_family_gpt(self, model_loader):
        """Test getting model family for GPT models."""
        assert model_loader.get_model_family("microsoft/DialoGPT-medium") == "gpt"
        assert model_loader.get_model_family("gpt2") == "gpt"
    
    def test_get_model_family_unknown(self, model_loader):
        """Test getting model family for unknown models."""
        assert model_loader.get_model_family("unknown-model") is None
        assert model_loader.get_model_family("") is None
    
    def test_configure_target_modules_llama(self, model_loader):
        """Test configuring target modules for LLaMA models."""
        targets = model_loader.configure_target_modules("meta-llama/Llama-2-7b-hf")
        expected = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert targets == expected
    
    def test_configure_target_modules_bert(self, model_loader):
        """Test configuring target modules for BERT models."""
        targets = model_loader.configure_target_modules("bert-base-uncased")
        expected = ["query", "value", "key", "dense"]
        assert targets == expected
    
    def test_configure_target_modules_unknown(self, model_loader):
        """Test configuring target modules for unknown models."""
        targets = model_loader.configure_target_modules("unknown-model")
        expected = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Default
        assert targets == expected
    
    @patch('src.models.model_loader.TRANSFORMERS_AVAILABLE', True)
    @patch('src.models.model_loader.BitsAndBytesConfig')
    def test_setup_quantization_config_4bit(self, mock_bnb_config, model_loader, sample_qlora_config):
        """Test setting up 4-bit quantization config."""
        mock_config_instance = Mock()
        mock_bnb_config.return_value = mock_config_instance
        
        result = model_loader.setup_quantization_config(sample_qlora_config)
        
        assert result == mock_config_instance
        mock_bnb_config.assert_called_once()
        
        # Check the call arguments
        call_args = mock_bnb_config.call_args[1]
        assert call_args["load_in_4bit"] is True
        assert call_args["bnb_4bit_quant_type"] == "nf4"
    
    def test_setup_quantization_config_disabled(self, model_loader):
        """Test quantization config when disabled."""
        config = QLoRAConfig(load_in_4bit=False, load_in_8bit=False)
        result = model_loader.setup_quantization_config(config)
        assert result is None
    
    @patch('src.models.model_loader.TRANSFORMERS_AVAILABLE', False)
    def test_setup_quantization_config_no_transformers(self, model_loader, sample_qlora_config):
        """Test quantization config when transformers not available."""
        with pytest.raises(ImportError, match="Transformers library required"):
            model_loader.setup_quantization_config(sample_qlora_config)
    
    def test_count_trainable_parameters(self, model_loader):
        """Test counting trainable parameters."""
        # Create mock model with parameters
        mock_model = Mock()
        
        # Create mock parameters
        param1 = Mock()
        param1.numel.return_value = 1000
        param1.requires_grad = True
        
        param2 = Mock()
        param2.numel.return_value = 500
        param2.requires_grad = False
        
        param3 = Mock()
        param3.numel.return_value = 2000
        param3.requires_grad = True
        
        mock_model.parameters.return_value = [param1, param2, param3]
        
        trainable, total = model_loader.count_trainable_parameters(mock_model)
        
        assert total == 3500  # 1000 + 500 + 2000
        assert trainable == 3000  # 1000 + 2000 (only requires_grad=True)
    
    def test_get_model_memory_usage_basic(self, model_loader):
        """Test getting basic model memory usage."""
        # Create mock model
        mock_model = Mock()
        
        # Mock parameters
        param1 = Mock()
        param1.numel.return_value = 1000
        param1.element_size.return_value = 4  # 4 bytes per parameter
        
        param2 = Mock()
        param2.numel.return_value = 500
        param2.element_size.return_value = 4
        
        mock_model.parameters.return_value = [param1, param2]
        
        memory_info = model_loader.get_model_memory_usage(mock_model)
        
        assert "total_parameters" in memory_info
        assert "parameter_memory_gb" in memory_info
        assert memory_info["total_parameters"] == 1500
        # 1500 params * 4 bytes = 6000 bytes = 6000 / (1024^3) GB
        expected_gb = 6000 / (1024**3)
        assert abs(memory_info["parameter_memory_gb"] - expected_gb) < 1e-10
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)  # 1 GB
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)  # 2 GB
    @patch('torch.cuda.get_device_properties')
    def test_get_model_memory_usage_gpu(self, mock_props, mock_reserved, mock_allocated, mock_cuda_available, model_loader):
        """Test getting GPU memory usage."""
        # Mock GPU properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1024**3  # 16 GB
        mock_props.return_value = mock_device_props
        
        # Set device to CUDA
        model_loader.device = "cuda:0"
        
        # Create mock model
        mock_model = Mock()
        mock_model.parameters.return_value = []
        
        memory_info = model_loader.get_model_memory_usage(mock_model)
        
        assert "gpu_memory_allocated_gb" in memory_info
        assert "gpu_memory_reserved_gb" in memory_info
        assert "gpu_memory_total_gb" in memory_info
        assert "gpu_memory_free_gb" in memory_info
        assert "gpu_utilization_percent" in memory_info
        
        assert memory_info["gpu_memory_allocated_gb"] == 1.0
        assert memory_info["gpu_memory_reserved_gb"] == 2.0
        assert memory_info["gpu_memory_total_gb"] == 16.0
        assert memory_info["gpu_memory_free_gb"] == 14.0
        assert memory_info["gpu_utilization_percent"] == 12.5  # 2/16 * 100
    
    def test_create_model_info(self, model_loader, sample_qlora_config):
        """Test creating ModelInfo object."""
        # Create mock model
        mock_model = Mock()
        mock_model.device = "cuda:0"
        
        # Mock parameters for counting
        param = Mock()
        param.numel.return_value = 1000
        param.requires_grad = True
        param.element_size.return_value = 4
        mock_model.parameters.return_value = [param]
        
        model_info = model_loader.create_model_info(
            model_name="meta-llama/Llama-2-7b-hf",
            model=mock_model,
            config=sample_qlora_config,
            quantized=True,
            has_lora=True
        )
        
        assert isinstance(model_info, ModelInfo)
        assert model_info.model_name == "meta-llama/Llama-2-7b-hf"
        assert model_info.model_type == "llama"
        assert model_info.total_params == 1000
        assert model_info.trainable_params == 1000
        assert model_info.quantized is True
        assert model_info.has_lora is True
        assert model_info.device == "cuda:0"
    
    def test_validate_memory_constraints_fits(self, model_loader):
        """Test memory constraint validation when model fits."""
        model_info = ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=1000000,
            trainable_params=100000,
            memory_usage_gb=8.0,
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
        
        fits, message = model_loader.validate_memory_constraints(model_info, max_memory_gb=16.0)
        
        assert fits is True
        assert "✓ FITS" in message
        assert "8.00 GB" in message
    
    def test_validate_memory_constraints_exceeds(self, model_loader):
        """Test memory constraint validation when model exceeds limit."""
        model_info = ModelInfo(
            model_name="test-model",
            model_type="llama",
            total_params=100_000_000,  # 100M parameters
            trainable_params=100_000_000,  # All trainable (no LoRA)
            memory_usage_gb=12.0,
            quantized=False,  # Higher overhead without quantization
            has_lora=False,   # Higher overhead without LoRA
            device="cuda:0"
        )
        
        fits, message = model_loader.validate_memory_constraints(model_info, max_memory_gb=16.0)
        
        # With 100M params and no LoRA, training overhead should be significant
        # 100M params * 12 bytes / (1024^3) ≈ 1.1 GB overhead
        # Total: 12.0 + 1.1 = 13.1 GB, should still fit
        # Let's make it definitely exceed by using more parameters
        model_info.total_params = 1_000_000_000  # 1B parameters
        model_info.trainable_params = 1_000_000_000
        
        fits, message = model_loader.validate_memory_constraints(model_info, max_memory_gb=16.0)
        
        # Now it should definitely exceed: 12.0 + (1B * 12 / 1024^3) ≈ 12.0 + 11.2 = 23.2 GB
        assert fits is False
        assert "✗ EXCEEDS" in message
        assert "12.00 GB" in message
    
    def test_cleanup_model(self, model_loader):
        """Test model cleanup."""
        # Add a mock model to loaded_models
        model_loader.loaded_models["test-model"] = {
            "model": Mock(),
            "tokenizer": Mock(),
            "config": Mock()
        }
        
        assert "test-model" in model_loader.loaded_models
        
        model_loader.cleanup_model("test-model")
        
        assert "test-model" not in model_loader.loaded_models
    
    def test_list_supported_models(self, model_loader):
        """Test listing supported models."""
        supported = model_loader.list_supported_models()
        
        assert isinstance(supported, dict)
        assert "llama" in supported
        assert "bert" in supported
        assert "gpt" in supported
        
        # Check that each family has model names
        for family, models in supported.items():
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, str) for model in models)


class TestModelLoaderIntegration:
    """Integration tests for model loader."""
    
    @pytest.fixture
    def model_loader(self):
        """Create model loader for testing."""
        return ModelLoader()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return QLoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
    
    def test_model_family_detection_comprehensive(self, model_loader):
        """Test comprehensive model family detection."""
        test_cases = [
            ("meta-llama/Llama-2-7b-hf", "llama"),
            ("huggyllama/llama-13b", "llama"),
            ("bert-base-uncased", "bert"),
            ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bert"),
            ("microsoft/DialoGPT-medium", "gpt"),
            ("gpt2-large", "gpt"),
            ("unknown/model", None),
            ("", None)
        ]
        
        for model_name, expected_family in test_cases:
            actual_family = model_loader.get_model_family(model_name)
            assert actual_family == expected_family, f"Failed for {model_name}"
    
    def test_target_modules_configuration(self, model_loader):
        """Test target modules configuration for different model types."""
        # Test LLaMA target modules
        llama_targets = model_loader.configure_target_modules("meta-llama/Llama-2-7b-hf")
        assert "q_proj" in llama_targets
        assert "gate_proj" in llama_targets
        assert len(llama_targets) == 7
        
        # Test BERT target modules
        bert_targets = model_loader.configure_target_modules("bert-base-uncased")
        assert "query" in bert_targets
        assert "value" in bert_targets
        assert len(bert_targets) == 4
        
        # Test GPT target modules
        gpt_targets = model_loader.configure_target_modules("gpt2")
        assert "c_attn" in gpt_targets
        assert "c_proj" in gpt_targets
        assert len(gpt_targets) == 2
    
    def test_memory_estimation_accuracy(self, model_loader):
        """Test memory estimation accuracy."""
        # Create model info with known parameters
        model_info = ModelInfo(
            model_name="test-7b-model",
            model_type="llama",
            total_params=7_000_000_000,  # 7B parameters
            trainable_params=100_000_000,  # 100M trainable (with LoRA)
            memory_usage_gb=3.5,  # 4-bit quantized model
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
        
        fits, message = model_loader.validate_memory_constraints(model_info, max_memory_gb=16.0)
        
        # Should fit within 16GB with QLoRA
        assert fits is True
        
        # Test without LoRA (should use more memory)
        model_info.has_lora = False
        model_info.trainable_params = model_info.total_params
        
        fits_full, message_full = model_loader.validate_memory_constraints(model_info, max_memory_gb=16.0)
        
        # Full fine-tuning should use more memory
        assert "Training overhead" in message_full
    
    @patch('src.models.model_loader.TRANSFORMERS_AVAILABLE', False)
    def test_error_handling_no_transformers(self, model_loader, sample_config):
        """Test error handling when transformers not available."""
        with pytest.raises(ImportError, match="Transformers library required"):
            model_loader.load_model_with_quantization("test-model", sample_config)
    
    @patch('src.models.model_loader.PEFT_AVAILABLE', False)
    def test_error_handling_no_peft(self, model_loader, sample_config):
        """Test error handling when PEFT not available."""
        mock_model = Mock()
        
        with pytest.raises(ImportError, match="PEFT library required"):
            model_loader.attach_lora_adapters(mock_model, sample_config, "test-model")


if __name__ == "__main__":
    pytest.main([__file__])