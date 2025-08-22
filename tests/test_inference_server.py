"""
Tests for inference server module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Import the server components
from src.serving.inference_server import (
    ScientificLLMServer,
    model_state,
    performance_metrics
)

# Import Pydantic models if available
try:
    from src.serving.inference_server import (
        InferenceRequest,
        GenomicsQueryRequest,
        PaperAnalysisRequest,
        ModelLoadRequest,
        FASTAPI_AVAILABLE
    )
except ImportError:
    FASTAPI_AVAILABLE = False

from src.models.model_loader import ModelInfo


class TestScientificLLMServer:
    """Test ScientificLLMServer functionality."""
    
    @pytest.fixture
    def server(self):
        """Create server instance for testing."""
        return ScientificLLMServer()
    
    @pytest.fixture
    def mock_model_state(self):
        """Mock model state for testing."""
        original_state = model_state.copy()
        
        # Set up mock model state
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_info = ModelInfo(
            model_name="test-model",
            model_type="fine-tuned",
            total_params=1000000,
            trainable_params=100000,
            memory_usage_gb=4.0,
            quantized=True,
            has_lora=True,
            device="cuda:0"
        )
        
        model_state.update({
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "model_info": mock_model_info,
            "loaded_at": 1234567890,
            "request_count": 0,
            "total_inference_time": 0.0,
        })
        
        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "model_info": mock_model_info
        }
        
        # Restore original state
        model_state.clear()
        model_state.update(original_state)
    
    def test_init(self, server):
        """Test server initialization."""
        assert server.start_time > 0
        assert server.model_loader is not None
        assert server.text_processor is not None
        assert server.request_times == []
        assert server.error_count == 0
    
    @pytest.mark.asyncio
    async def test_load_model_file_not_found(self, server):
        """Test loading model when file doesn't exist."""
        with pytest.raises(Exception):  # Should raise HTTPException in real scenario
            await server.load_model("/nonexistent/path")
    
    @pytest.mark.asyncio
    @patch('src.serving.inference_server.TRANSFORMERS_AVAILABLE', True)
    @patch('transformers.AutoModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('src.models.qlora_config.QLoRAConfig.get_quantization_config', return_value=None)
    async def test_load_model_success(self, mock_quant_config, mock_cuda, mock_tokenizer_class, mock_model_class, server):
        """Test successful model loading."""
        # Create temporary model directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            
            # Create mock config file
            config_file = model_path / "training_config.json"
            with open(config_file, 'w') as f:
                json.dump({"model_name": "test-model"}, f)
            
            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.device = "cuda:0"
            mock_model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # Test loading
            result = await server.load_model(str(model_path))
            
            assert result["status"] == "success"
            assert "model_info" in result
            assert "load_time_seconds" in result
            assert model_state["model"] is not None
            assert model_state["tokenizer"] is not None
    
    @pytest.mark.asyncio
    async def test_generate_text_no_model(self, server):
        """Test text generation when no model is loaded."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        request = InferenceRequest(text="Test input")
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await server.generate_text(request)
    
    @pytest.mark.asyncio
    @patch('torch.no_grad')
    async def test_generate_text_success(self, mock_no_grad, server, mock_model_state):
        """Test successful text generation."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        # Mock tokenizer behavior
        mock_tokenizer = mock_model_state["tokenizer"]
        mock_input_ids = Mock()
        mock_input_ids.shape = [1, 10]
        mock_input_ids.__getitem__ = Mock(return_value=10)  # For shape[1] access
        mock_input_ids.to = Mock(return_value=mock_input_ids)  # For device movement
        
        mock_attention_mask = Mock()
        mock_attention_mask.to = Mock(return_value=mock_attention_mask)
        
        mock_inputs = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask
        }
        
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer.eos_token_id = 2
        
        # Mock model behavior
        mock_model = mock_model_state["model"]
        mock_model.device = "cuda:0"
        mock_output = Mock()
        mock_output.__getitem__ = Mock(return_value=list(range(15)))  # Mock token sequence
        mock_model.generate.return_value = [mock_output]
        
        # Mock text processor
        with patch.object(server.text_processor, 'preprocess_scientific_text', return_value="processed text"):
            request = InferenceRequest(text="Test genomics question")
            response = await server.generate_text(request)
            
            assert response.generated_text == ["Generated response"]
            assert response.input_text == "Test genomics question"
            assert response.inference_time_ms > 0
            assert model_state["request_count"] == 1
    
    @pytest.mark.asyncio
    async def test_process_genomics_query_no_model(self, server):
        """Test genomics query when no model is loaded."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        request = GenomicsQueryRequest(query="What is BRCA1?")
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await server.process_genomics_query(request)
    
    @pytest.mark.asyncio
    async def test_process_genomics_query_success(self, server, mock_model_state):
        """Test successful genomics query processing."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        # Mock the generate_text method
        mock_inference_response = Mock()
        mock_inference_response.generated_text = ["BRCA1 is a tumor suppressor gene involved in DNA repair."]
        
        with patch.object(server, 'generate_text', return_value=mock_inference_response):
            request = GenomicsQueryRequest(
                query="What is BRCA1?",
                query_type="gene_function"
            )
            
            response = await server.process_genomics_query(request)
            
            assert "BRCA1" in response.answer
            assert response.query_type == "gene_function"
            assert response.confidence > 0
            assert "gene" in response.relevant_concepts
            assert response.inference_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_analyze_paper_success(self, server, mock_model_state):
        """Test successful paper analysis."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        # Mock the generate_text method
        mock_inference_response = Mock()
        mock_inference_response.generated_text = [
            "This paper presents a novel approach to genomics analysis using ENCODE datasets. "
            "The methodology involves deep learning techniques. Key findings include improved accuracy."
        ]
        
        with patch.object(server, 'generate_text', return_value=mock_inference_response):
            request = PaperAnalysisRequest(
                title="Novel Genomics Analysis Method",
                abstract="This paper describes a new method for analyzing genomics data using ENCODE.",
                analysis_type="summary"
            )
            
            response = await server.analyze_paper(request)
            
            assert "genomics" in response.analysis.lower()
            assert response.analysis_type == "summary"
            assert len(response.key_points) > 0
            assert "ENCODE" in response.detected_benchmarks
            assert response.inference_time_ms > 0
    
    def test_get_health_status_no_model(self, server):
        """Test health status when no model is loaded."""
        # Clear model state
        model_state["model"] = None
        model_state["model_info"] = None
        
        health = server.get_health_status()
        
        assert health.status == "no_model_loaded"
        assert health.model_loaded is False
        assert health.model_info is None
        assert health.uptime_seconds > 0
        assert isinstance(health.performance_metrics, dict)
    
    def test_get_health_status_with_model(self, server, mock_model_state):
        """Test health status when model is loaded."""
        # Add some request history
        server.request_times = [100, 150, 120, 200, 180]
        server.error_count = 1
        model_state["request_count"] = 10
        
        health = server.get_health_status()
        
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.model_info is not None
        assert health.performance_metrics["average_response_time"] == 150.0  # Average of request times
        assert health.performance_metrics["error_rate"] == 10.0  # 1 error out of 10 requests


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_inference_request_valid(self):
        """Test valid inference request."""
        request = InferenceRequest(
            text="What is the function of BRCA1?",
            max_length=256,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request.text == "What is the function of BRCA1?"
        assert request.max_length == 256
        assert request.temperature == 0.7
        assert request.top_p == 0.9
    
    def test_inference_request_empty_text(self):
        """Test inference request with empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            InferenceRequest(text="   ")
    
    def test_inference_request_invalid_temperature(self):
        """Test inference request with invalid temperature."""
        with pytest.raises(ValueError):
            InferenceRequest(text="Test", temperature=3.0)  # Too high
    
    def test_genomics_query_request_valid(self):
        """Test valid genomics query request."""
        request = GenomicsQueryRequest(
            query="What is CRISPR?",
            context="Gene editing context",
            query_type="gene_function"
        )
        
        assert request.query == "What is CRISPR?"
        assert request.context == "Gene editing context"
        assert request.query_type == "gene_function"
    
    def test_genomics_query_request_invalid_type(self):
        """Test genomics query request with invalid type."""
        with pytest.raises(ValueError, match="Query type must be one of"):
            GenomicsQueryRequest(query="Test", query_type="invalid_type")
    
    def test_paper_analysis_request_valid(self):
        """Test valid paper analysis request."""
        request = PaperAnalysisRequest(
            title="Test Paper",
            abstract="Test abstract",
            analysis_type="summary"
        )
        
        assert request.title == "Test Paper"
        assert request.abstract == "Test abstract"
        assert request.analysis_type == "summary"
    
    def test_paper_analysis_request_invalid_type(self):
        """Test paper analysis request with invalid type."""
        with pytest.raises(ValueError, match="Analysis type must be one of"):
            PaperAnalysisRequest(title="Test", analysis_type="invalid_type")
    
    def test_model_load_request_valid(self):
        """Test valid model load request."""
        request = ModelLoadRequest(
            model_path="/path/to/model",
            model_name="test-model",
            use_quantization=True
        )
        
        assert request.model_path == "/path/to/model"
        assert request.model_name == "test-model"
        assert request.use_quantization is True


class TestServerIntegration:
    """Integration tests for the server."""
    
    @pytest.fixture
    def server(self):
        """Create server instance for testing."""
        return ScientificLLMServer()
    
    def test_server_initialization_and_state(self, server):
        """Test server initialization and state management."""
        # Check initial state
        assert server.start_time > 0
        assert server.request_times == []
        assert server.error_count == 0
        
        # Reset global state for this test
        model_state["model"] = None
        model_state["tokenizer"] = None
        model_state["request_count"] = 0
        
        # Check global state initialization
        assert model_state["model"] is None
        assert model_state["tokenizer"] is None
        assert model_state["request_count"] == 0
    
    def test_performance_metrics_tracking(self, server):
        """Test performance metrics tracking."""
        # Simulate some requests
        server.request_times = [100, 200, 150, 300, 250]
        server.error_count = 2
        model_state["request_count"] = 10
        
        health = server.get_health_status()
        
        # Check metrics calculation
        assert health.performance_metrics["average_response_time"] == 200.0
        assert health.performance_metrics["error_rate"] == 20.0  # 2/10 * 100
        assert health.uptime_seconds > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, server):
        """Test error handling and recovery."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        
        # Test with no model loaded
        request = InferenceRequest(text="Test")
        
        with pytest.raises(Exception):
            await server.generate_text(request)
        
        # Error count should not increase for expected errors
        initial_error_count = server.error_count
        
        # Test with invalid request (this would be handled by FastAPI validation)
        # Here we just verify the server can handle the state properly
        health = server.get_health_status()
        assert health.status == "no_model_loaded"
        assert server.error_count == initial_error_count


if __name__ == "__main__":
    pytest.main([__file__])