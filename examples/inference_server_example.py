#!/usr/bin/env python3
"""
Example script demonstrating the FastAPI inference server usage.

This script shows how to:
1. Start the inference server
2. Load a fine-tuned model
3. Make inference requests
4. Monitor server performance

Usage:
    python examples/inference_server_example.py
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

try:
    import requests
    import uvicorn
    from fastapi.testclient import TestClient
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests or uvicorn not available. Install with: pip install requests uvicorn")
    REQUESTS_AVAILABLE = False

# Import the inference server
try:
    from src.serving.inference_server import app, run_server, ScientificLLMServer
    from src.serving.inference_server import (
        InferenceRequest, 
        GenomicsQueryRequest, 
        PaperAnalysisRequest,
        ModelLoadRequest
    )
    SERVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import inference server: {e}")
    SERVER_AVAILABLE = False


def create_mock_checkpoint():
    """Create a mock checkpoint directory for testing."""
    checkpoint_dir = Path("examples/mock_checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create mock config file
    config_file = checkpoint_dir / "training_config.json"
    config_data = {
        "model_name": "genomics-llama-7b-fine-tuned",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "training_params": {
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 4
        },
        "dataset": "high_quality_genomics_papers",
        "fine_tuning_method": "QLoRA"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Created mock checkpoint at: {checkpoint_dir}")
    return str(checkpoint_dir)


def test_server_with_client():
    """Test the server using FastAPI TestClient."""
    if not SERVER_AVAILABLE:
        print("Server not available, skipping client tests")
        return
    
    print("\n=== Testing Server with TestClient ===")
    
    # Create test client
    client = TestClient(app)
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    response = client.get("/")
    print(f"Root response: {response.status_code}")
    if response.status_code == 200:
        print(f"API Info: {response.json()['message']}")
    
    # Test health check
    print("\n2. Testing health check...")
    response = client.get("/api/v1/health")
    print(f"Health check: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
    
    # Test metrics endpoint
    print("\n3. Testing metrics...")
    response = client.get("/api/v1/metrics")
    print(f"Metrics: {response.status_code}")
    if response.status_code == 200:
        metrics = response.json()
        print(f"Server uptime: {metrics['server']['uptime_seconds']:.2f}s")
    
    # Test model loading (will fail without actual model)
    print("\n4. Testing model loading...")
    mock_checkpoint = create_mock_checkpoint()
    load_request = {
        "model_path": mock_checkpoint,
        "model_name": "test-genomics-model",
        "use_quantization": False  # Disable for testing
    }
    
    response = client.post("/api/v1/load-model", json=load_request)
    print(f"Model loading: {response.status_code}")
    if response.status_code != 200:
        print(f"Expected failure (no actual model): {response.json()}")
    
    print("TestClient tests completed!")


def test_genomics_queries():
    """Test genomics-specific query examples."""
    if not SERVER_AVAILABLE:
        print("Server not available, skipping genomics tests")
        return
    
    print("\n=== Testing Genomics Query Examples ===")
    
    # Example genomics queries
    genomics_queries = [
        {
            "query": "What is the function of the BRCA1 gene?",
            "query_type": "gene_function",
            "context": "Cancer research context"
        },
        {
            "query": "Explain the p53 pathway in cancer",
            "query_type": "pathway_analysis",
            "context": "Tumor suppressor mechanisms"
        },
        {
            "query": "What are the implications of CRISPR-Cas9 for treating genetic diseases?",
            "query_type": "drug_discovery",
            "context": "Gene therapy applications"
        }
    ]
    
    client = TestClient(app)
    
    for i, query_data in enumerate(genomics_queries, 1):
        print(f"\n{i}. Testing query: {query_data['query'][:50]}...")
        response = client.post("/api/v1/genomics/query", json=query_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 503:
            print("Expected: No model loaded")
        elif response.status_code == 200:
            result = response.json()
            print(f"Query type: {result['query_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Relevant concepts: {result['relevant_concepts']}")


def test_paper_analysis():
    """Test scientific paper analysis examples."""
    if not SERVER_AVAILABLE:
        print("Server not available, skipping paper analysis tests")
        return
    
    print("\n=== Testing Paper Analysis Examples ===")
    
    # Example scientific papers
    papers = [
        {
            "title": "CRISPR-Cas9 mediated genome editing in human embryos",
            "abstract": "We report the use of CRISPR-Cas9 system to edit genes in human embryos. Our results show successful targeting of disease-causing mutations with high efficiency and specificity.",
            "analysis_type": "summary"
        },
        {
            "title": "Deep learning approaches for genomic variant classification",
            "abstract": "This study presents novel deep learning methods for classifying genomic variants. We achieved 95% accuracy on benchmark datasets including ClinVar and COSMIC.",
            "analysis_type": "methodology"
        },
        {
            "title": "Single-cell RNA sequencing reveals cellular heterogeneity in cancer",
            "abstract": "Using single-cell RNA-seq, we identified distinct cell populations in tumor samples. Our analysis reveals novel therapeutic targets and resistance mechanisms.",
            "analysis_type": "key_findings"
        }
    ]
    
    client = TestClient(app)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. Analyzing: {paper['title'][:50]}...")
        response = client.post("/api/v1/papers/analyze", json=paper)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 503:
            print("Expected: No model loaded")
        elif response.status_code == 200:
            result = response.json()
            print(f"Analysis type: {result['analysis_type']}")
            print(f"Key points: {len(result['key_points'])}")
            print(f"Detected benchmarks: {result['detected_benchmarks']}")


def demonstrate_server_features():
    """Demonstrate key server features."""
    print("=== ScientificLLM FastAPI Inference Server Demo ===")
    print()
    print("This example demonstrates the FastAPI inference server features:")
    print("1. Model loading from enhanced trainer checkpoints")
    print("2. Scientific text generation")
    print("3. Genomics-specific query processing")
    print("4. Scientific paper analysis")
    print("5. Performance monitoring and health checks")
    print()
    
    # Test server components
    test_server_with_client()
    test_genomics_queries()
    test_paper_analysis()
    
    print("\n=== Server Usage Instructions ===")
    print("To run the server in production:")
    print("1. python -m src.serving.inference_server")
    print("2. Or: uvicorn src.serving.inference_server:app --host 0.0.0.0 --port 8000")
    print()
    print("API Documentation available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/api/v1/health")
    print("Metrics: http://localhost:8000/api/v1/metrics")
    print()
    print("Example curl commands:")
    print("# Load model")
    print('curl -X POST "http://localhost:8000/api/v1/load-model" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"model_path": "/path/to/checkpoint", "use_quantization": true}\'')
    print()
    print("# Generate text")
    print('curl -X POST "http://localhost:8000/api/v1/generate" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"text": "Explain the role of p53 in cancer", "max_length": 256}\'')
    print()
    print("# Genomics query")
    print('curl -X POST "http://localhost:8000/api/v1/genomics/query" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "What is BRCA1?", "query_type": "gene_function"}\'')


async def run_server_example():
    """Example of running the server programmatically."""
    if not SERVER_AVAILABLE:
        print("Server not available")
        return
    
    print("\n=== Running Server Example ===")
    print("This would start the server on localhost:8000")
    print("Uncomment the line below to actually start the server:")
    # run_server(host="127.0.0.1", port=8000, reload=True)


def main():
    """Main example function."""
    print("FastAPI Inference Server Example")
    print("=" * 50)
    
    if not SERVER_AVAILABLE:
        print("Error: Inference server not available")
        print("Make sure all dependencies are installed:")
        print("pip install fastapi uvicorn pydantic")
        return
    
    # Demonstrate server features
    demonstrate_server_features()
    
    # Show how to run server
    print("\nTo start the server, run:")
    print("python examples/inference_server_example.py --run-server")
    print("Or directly: python -m src.serving.inference_server")


if __name__ == "__main__":
    import sys
    
    if "--run-server" in sys.argv:
        if SERVER_AVAILABLE:
            print("Starting FastAPI inference server...")
            run_server(host="127.0.0.1", port=8000, reload=True)
        else:
            print("Server not available")
    else:
        main()