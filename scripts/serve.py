#!/usr/bin/env python3
"""
Model serving script for ScientificLLM-Forge.

This script demonstrates how to deploy and serve fine-tuned
scientific language models using FastAPI.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from serving.server import ModelServer
from serving.inference import InferenceEngine
from utils.config import ConfigManager
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Serve a scientific LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/serving.yaml",
        help="Path to serving configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Override model path from config"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Override host from config"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override port from config"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override number of workers from config"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main serving function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("serving", log_level)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Override config with command line arguments
        if args.model_path:
            config["serving"]["model"]["model_path"] = args.model_path
        if args.host:
            config["serving"]["server"]["host"] = args.host
        if args.port:
            config["serving"]["server"]["port"] = args.port
        if args.workers:
            config["serving"]["server"]["workers"] = args.workers
        if args.reload:
            config["serving"]["server"]["reload"] = True
            
        # Initialize inference engine
        logger.info("Initializing inference engine...")
        inference_engine = InferenceEngine(config["serving"]["model"])
        
        # Initialize model server
        logger.info("Initializing model server...")
        server = ModelServer(
            inference_engine=inference_engine,
            config=config["serving"]
        )
        
        # Start server
        logger.info("Starting model server...")
        server.start()
        
    except Exception as e:
        logger.error(f"Serving failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 