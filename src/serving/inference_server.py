"""
FastAPI inference server for fine-tuned scientific LLMs.

This module provides a production-ready REST API for serving fine-tuned
language models specialized for genomics and scientific text analysis.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

try:
    from ..utils.logger import get_logger
    from ..models.model_loader import ModelLoader, ModelInfo
    from ..models.qlora_config import QLoRAConfig
    from ..data.text_processor import ScientificTextProcessor, PreprocessingConfig
    from ..data.scientific_dataset import ScientificPaper, QualityTier
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger
    from models.model_loader import ModelLoader, ModelInfo
    from models.qlora_config import QLoRAConfig
    from data.text_processor import ScientificTextProcessor, PreprocessingConfig
    from data.scientific_dataset import ScientificPaper, QualityTier

logger = get_logger(__name__)

# Import FastAPI and related libraries
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not available")
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    BaseModel = None
    Field = None

# Import ML libraries
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available")
    TRANSFORMERS_AVAILABLE = False
    PreTrainedModel = None
    PreTrainedTokenizer = None

# Global model state
model_state = {
    "model": None,
    "tokenizer": None,
    "model_info": None,
    "model_loader": None,
    "text_processor": None,
    "loaded_at": None,
    "request_count": 0,
    "total_inference_time": 0.0,
}

# Performance metrics
performance_metrics = {
    "requests_per_minute": 0,
    "average_response_time": 0.0,
    "error_rate": 0.0,
    "memory_usage_gb": 0.0,
    "gpu_utilization": 0.0,
    "model_load_time": 0.0,
}


# Pydantic models for API
if FASTAPI_AVAILABLE:
    
    class InferenceRequest(BaseModel):
        """Request model for text inference."""
        text: str = Field(..., description="Input text for inference", max_length=4096)
        max_length: Optional[int] = Field(512, description="Maximum output length", ge=1, le=2048)
        temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
        top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.1, le=1.0)
        do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
        num_return_sequences: Optional[int] = Field(1, description="Number of sequences to return", ge=1, le=5)
        
        @field_validator('text')
        @classmethod
        def validate_text(cls, v):
            if not v.strip():
                raise ValueError("Text cannot be empty")
            return v.strip()
    
    class GenomicsQueryRequest(BaseModel):
        """Request model for genomics-specific queries."""
        query: str = Field(..., description="Genomics query", max_length=1024)
        context: Optional[str] = Field(None, description="Additional context", max_length=2048)
        query_type: Optional[str] = Field("general", description="Type of query")
        include_citations: Optional[bool] = Field(False, description="Include citations in response")
        
        @field_validator('query_type')
        @classmethod
        def validate_query_type(cls, v):
            allowed_types = ["general", "gene_function", "pathway_analysis", "disease_association", "drug_discovery"]
            if v not in allowed_types:
                raise ValueError(f"Query type must be one of: {allowed_types}")
            return v
    
    class PaperAnalysisRequest(BaseModel):
        """Request model for scientific paper analysis."""
        title: str = Field(..., description="Paper title", max_length=512)
        abstract: Optional[str] = Field(None, description="Paper abstract", max_length=2048)
        full_text: Optional[str] = Field(None, description="Full paper text", max_length=8192)
        analysis_type: str = Field("summary", description="Type of analysis to perform")
        
        @field_validator('analysis_type')
        @classmethod
        def validate_analysis_type(cls, v):
            allowed_types = ["summary", "key_findings", "methodology", "benchmarks", "limitations"]
            if v not in allowed_types:
                raise ValueError(f"Analysis type must be one of: {allowed_types}")
            return v
    
    class InferenceResponse(BaseModel):
        """Response model for inference results."""
        generated_text: List[str] = Field(..., description="Generated text sequences")
        input_text: str = Field(..., description="Original input text")
        model_info: Dict[str, Any] = Field(..., description="Model information")
        generation_params: Dict[str, Any] = Field(..., description="Generation parameters used")
        inference_time_ms: float = Field(..., description="Inference time in milliseconds")
        timestamp: str = Field(..., description="Response timestamp")
    
    class GenomicsQueryResponse(BaseModel):
        """Response model for genomics queries."""
        answer: str = Field(..., description="Generated answer")
        confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
        query_type: str = Field(..., description="Type of query processed")
        relevant_concepts: List[str] = Field(..., description="Relevant genomics concepts identified")
        citations: Optional[List[str]] = Field(None, description="Relevant citations if requested")
        inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    
    class PaperAnalysisResponse(BaseModel):
        """Response model for paper analysis."""
        analysis: str = Field(..., description="Generated analysis")
        analysis_type: str = Field(..., description="Type of analysis performed")
        key_points: List[str] = Field(..., description="Key points extracted")
        quality_score: Optional[float] = Field(None, description="Estimated quality score")
        detected_benchmarks: List[str] = Field(..., description="Detected benchmarks/datasets")
        inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    
    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(..., description="Service status")
        model_loaded: bool = Field(..., description="Whether model is loaded")
        model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")
        performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
        uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class ModelLoadRequest(BaseModel):
        """Request model for loading a model."""
        model_path: str = Field(..., description="Path to model checkpoint")
        model_name: Optional[str] = Field(None, description="Model name override")
        use_quantization: Optional[bool] = Field(True, description="Whether to use quantization")
        device: Optional[str] = Field("auto", description="Device to load model on")


class ScientificLLMServer:
    """Scientific LLM inference server."""
    
    def __init__(self):
        """Initialize the server."""
        self.start_time = time.time()
        self.model_loader = ModelLoader()
        self.text_processor = ScientificTextProcessor()
        
        # Performance tracking
        self.request_times = []
        self.error_count = 0
        
        logger.info("ScientificLLMServer initialized")
    
    async def load_model(
        self, 
        model_path: str, 
        model_name: Optional[str] = None,
        use_quantization: bool = True
    ) -> Dict[str, Any]:
        """Load a fine-tuned model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Optional model name override
            use_quantization: Whether to use quantization
            
        Returns:
            Dictionary with loading results
        """
        if not TRANSFORMERS_AVAILABLE:
            raise HTTPException(
                status_code=500, 
                detail="Transformers library not available"
            )
        
        logger.info(f"Loading model from: {model_path}")
        load_start = time.time()
        
        try:
            # Check if path exists
            checkpoint_path = Path(model_path)
            if not checkpoint_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model checkpoint not found: {model_path}"
                )
            
            # Load model configuration if available
            config_file = checkpoint_path / "training_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    training_config = json.load(f)
                    model_name = model_name or training_config.get("model_name", "unknown")
            
            # Create QLoRA config for loading
            qlora_config = QLoRAConfig(
                load_in_4bit=use_quantization,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model and tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            if use_quantization and torch.cuda.is_available():
                quantization_config = qlora_config.get_quantization_config()
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
            
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # Create model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = ModelInfo(
                model_name=model_name or "fine-tuned-model",
                model_type="fine-tuned",
                total_params=total_params,
                trainable_params=trainable_params,
                memory_usage_gb=0.0,  # Will be calculated
                quantized=use_quantization,
                has_lora=trainable_params < total_params,
                device=str(model.device) if hasattr(model, 'device') else "unknown"
            )
            
            # Update global state
            model_state.update({
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "model_loader": self.model_loader,
                "text_processor": self.text_processor,
                "loaded_at": time.time(),
            })
            
            load_time = time.time() - load_start
            performance_metrics["model_load_time"] = load_time
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            return {
                "status": "success",
                "model_info": asdict(model_info),
                "load_time_seconds": load_time,
                "checkpoint_path": str(checkpoint_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    async def generate_text(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using the loaded model.
        
        Args:
            request: Inference request
            
        Returns:
            Generated text response
        """
        if model_state["model"] is None or model_state["tokenizer"] is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        start_time = time.time()
        
        try:
            model = model_state["model"]
            tokenizer = model_state["tokenizer"]
            
            # Preprocess input text
            processed_text = self.text_processor.preprocess_scientific_text(request.text)
            
            # Tokenize input
            inputs = tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            # Move to model device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=min(inputs["input_ids"].shape[1] + request.max_length, 2048),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    num_return_sequences=request.num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_texts = []
            for output in outputs:
                # Skip the input tokens
                generated_tokens = output[inputs["input_ids"].shape[1]:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update metrics
            model_state["request_count"] += 1
            model_state["total_inference_time"] += inference_time / 1000
            self.request_times.append(inference_time)
            
            # Keep only last 100 request times for rolling average
            if len(self.request_times) > 100:
                self.request_times = self.request_times[-100:]
            
            return InferenceResponse(
                generated_text=generated_texts,
                input_text=request.text,
                model_info=asdict(model_state["model_info"]),
                generation_params={
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "do_sample": request.do_sample,
                    "num_return_sequences": request.num_return_sequences,
                },
                inference_time_ms=inference_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    async def process_genomics_query(self, request: GenomicsQueryRequest) -> GenomicsQueryResponse:
        """Process genomics-specific queries.
        
        Args:
            request: Genomics query request
            
        Returns:
            Genomics query response
        """
        if model_state["model"] is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        start_time = time.time()
        
        try:
            # Create genomics-specific prompt
            prompt_templates = {
                "general": f"Answer this genomics question: {request.query}",
                "gene_function": f"Explain the function of this gene or genetic element: {request.query}",
                "pathway_analysis": f"Analyze this biological pathway: {request.query}",
                "disease_association": f"Explain the disease association: {request.query}",
                "drug_discovery": f"Discuss drug discovery implications: {request.query}"
            }
            
            prompt = prompt_templates.get(request.query_type, prompt_templates["general"])
            
            if request.context:
                prompt += f"\n\nContext: {request.context}"
            
            # Generate response using the general inference method
            inference_request = InferenceRequest(
                text=prompt,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            
            response = await self.generate_text(inference_request)
            generated_answer = response.generated_text[0]
            
            # Extract relevant genomics concepts (simple keyword extraction)
            genomics_keywords = [
                "gene", "protein", "DNA", "RNA", "chromosome", "mutation", "expression",
                "pathway", "enzyme", "transcription", "translation", "genome", "allele",
                "phenotype", "genotype", "CRISPR", "sequencing", "variant", "SNP"
            ]
            
            relevant_concepts = [
                keyword for keyword in genomics_keywords 
                if keyword.lower() in generated_answer.lower()
            ]
            
            # Simple confidence scoring based on response length and concept coverage
            confidence = min(0.9, len(generated_answer) / 500 + len(relevant_concepts) / 10)
            
            inference_time = (time.time() - start_time) * 1000
            
            return GenomicsQueryResponse(
                answer=generated_answer,
                confidence=confidence,
                query_type=request.query_type,
                relevant_concepts=relevant_concepts,
                citations=None,  # Would be populated with actual citation extraction
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Genomics query error: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def analyze_paper(self, request: PaperAnalysisRequest) -> PaperAnalysisResponse:
        """Analyze a scientific paper.
        
        Args:
            request: Paper analysis request
            
        Returns:
            Paper analysis response
        """
        if model_state["model"] is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        start_time = time.time()
        
        try:
            # Combine paper content
            paper_content = f"Title: {request.title}"
            if request.abstract:
                paper_content += f"\n\nAbstract: {request.abstract}"
            if request.full_text:
                paper_content += f"\n\nContent: {request.full_text[:4000]}"  # Limit content
            
            # Create analysis-specific prompts
            analysis_prompts = {
                "summary": f"Summarize this scientific paper:\n\n{paper_content}",
                "key_findings": f"What are the key findings of this paper?\n\n{paper_content}",
                "methodology": f"Describe the methodology used in this paper:\n\n{paper_content}",
                "benchmarks": f"What benchmarks or datasets are used in this paper?\n\n{paper_content}",
                "limitations": f"What are the limitations of this research?\n\n{paper_content}"
            }
            
            prompt = analysis_prompts.get(request.analysis_type, analysis_prompts["summary"])
            
            # Generate analysis
            inference_request = InferenceRequest(
                text=prompt,
                max_length=512,
                temperature=0.5,  # Lower temperature for more focused analysis
                top_p=0.8,
                do_sample=True,
                num_return_sequences=1
            )
            
            response = await self.generate_text(inference_request)
            analysis = response.generated_text[0]
            
            # Extract key points (simple sentence splitting)
            sentences = analysis.split('. ')
            key_points = [s.strip() + '.' for s in sentences if len(s.strip()) > 20][:5]
            
            # Detect benchmarks/datasets
            benchmark_keywords = [
                "ENCODE", "TCGA", "GTEx", "1000 Genomes", "COSMIC", "ClinVar",
                "dbSNP", "GWAS", "UniProt", "PDB", "NCBI", "Ensembl"
            ]
            
            detected_benchmarks = [
                keyword for keyword in benchmark_keywords
                if keyword.lower() in paper_content.lower()
            ]
            
            # Simple quality scoring based on content length and benchmark usage
            quality_score = None
            if request.abstract and request.full_text:
                quality_score = min(100, len(request.abstract) / 10 + len(detected_benchmarks) * 20)
            
            inference_time = (time.time() - start_time) * 1000
            
            return PaperAnalysisResponse(
                analysis=analysis,
                analysis_type=request.analysis_type,
                key_points=key_points,
                quality_score=quality_score,
                detected_benchmarks=detected_benchmarks,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Paper analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Paper analysis failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get server health status.
        
        Returns:
            Health status response
        """
        uptime = time.time() - self.start_time
        
        # Update performance metrics
        if self.request_times:
            performance_metrics["average_response_time"] = sum(self.request_times) / len(self.request_times)
        
        if model_state["request_count"] > 0:
            performance_metrics["error_rate"] = (self.error_count / model_state["request_count"]) * 100
        
        # Calculate requests per minute
        if uptime > 60:
            performance_metrics["requests_per_minute"] = (model_state["request_count"] / uptime) * 60
        
        # Get memory usage if model is loaded
        if model_state["model"] is not None and torch.cuda.is_available():
            performance_metrics["memory_usage_gb"] = torch.cuda.memory_allocated() / (1024**3)
            performance_metrics["gpu_utilization"] = (torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory) * 100
        
        return HealthResponse(
            status="healthy" if model_state["model"] is not None else "no_model_loaded",
            model_loaded=model_state["model"] is not None,
            model_info=asdict(model_state["model_info"]) if model_state["model_info"] else None,
            performance_metrics=performance_metrics.copy(),
            uptime_seconds=uptime
        )


# Initialize server instance
server_instance = ScientificLLMServer()


# FastAPI app setup
if FASTAPI_AVAILABLE:
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifespan."""
        logger.info("Starting ScientificLLM Inference Server")
        yield
        logger.info("Shutting down ScientificLLM Inference Server")
    
    app = FastAPI(
        title="ScientificLLM Inference Server",
        description="Production-ready REST API for serving fine-tuned scientific language models",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    @app.post("/api/v1/load-model", response_model=Dict[str, Any])
    async def load_model_endpoint(request: ModelLoadRequest):
        """Load a fine-tuned model from checkpoint."""
        return await server_instance.load_model(
            model_path=request.model_path,
            model_name=request.model_name,
            use_quantization=request.use_quantization
        )
    
    
    @app.post("/api/v1/generate", response_model=InferenceResponse)
    async def generate_text_endpoint(request: InferenceRequest):
        """Generate text using the loaded model."""
        return await server_instance.generate_text(request)
    
    
    @app.post("/api/v1/genomics/query", response_model=GenomicsQueryResponse)
    async def genomics_query_endpoint(request: GenomicsQueryRequest):
        """Process genomics-specific queries."""
        return await server_instance.process_genomics_query(request)
    
    
    @app.post("/api/v1/papers/analyze", response_model=PaperAnalysisResponse)
    async def analyze_paper_endpoint(request: PaperAnalysisRequest):
        """Analyze a scientific paper."""
        return await server_instance.analyze_paper(request)
    
    
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        """Get server health status."""
        return server_instance.get_health_status()
    
    
    @app.get("/api/v1/metrics")
    async def get_metrics():
        """Get detailed performance metrics."""
        return {
            "model_state": {
                "loaded": model_state["model"] is not None,
                "loaded_at": model_state["loaded_at"],
                "request_count": model_state["request_count"],
                "total_inference_time": model_state["total_inference_time"],
            },
            "performance": performance_metrics.copy(),
            "server": {
                "uptime_seconds": time.time() - server_instance.start_time,
                "error_count": server_instance.error_count,
                "recent_response_times": server_instance.request_times[-10:],  # Last 10 requests
            }
        }
    
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "ScientificLLM Inference Server",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "load_model": "/api/v1/load-model",
                "generate": "/api/v1/generate",
                "genomics_query": "/api/v1/genomics/query",
                "paper_analysis": "/api/v1/papers/analyze",
                "health": "/api/v1/health",
                "metrics": "/api/v1/metrics",
            },
            "documentation": "/docs"
        }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False
):
    """Run the inference server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        reload: Enable auto-reload for development
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.serving.inference_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)