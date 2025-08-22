"""
Main entry point for ScientificLLM-Forge.

This module provides the main CLI interface for the ScientificLLM-Forge
platform, allowing users to train, serve, and manage scientific LLMs.
"""

import argparse
import sys
from pathlib import Path

from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ScientificLLM-Forge: MLOps platform for fine-tuning scientific LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scientific-llm-forge train --config configs/training.yaml
  scientific-llm-forge serve --config configs/serving.yaml
  scientific-llm-forge evaluate --model-path outputs/best_model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a scientific LLM')
    train_parser.add_argument('--config', type=str, default='configs/training.yaml',
                             help='Path to training configuration file')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='Directory to save model outputs')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Enable verbose logging')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Serve a trained model')
    serve_parser.add_argument('--config', type=str, default='configs/serving.yaml',
                             help='Path to serving configuration file')
    serve_parser.add_argument('--model-path', type=str,
                             help='Path to the trained model')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0',
                             help='Host to bind the server to')
    serve_parser.add_argument('--port', type=int, default=8000,
                             help='Port to bind the server to')
    serve_parser.add_argument('--verbose', action='store_true',
                             help='Enable verbose logging')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to the trained model')
    eval_parser.add_argument('--test-data', type=str, required=True,
                            help='Path to test data file')
    eval_parser.add_argument('--output-file', type=str,
                            help='Path to save evaluation results')
    eval_parser.add_argument('--verbose', action='store_true',
                            help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main entry point for ScientificLLM-Forge."""
    args = parse_args()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        sys.exit(1)
    
    # Setup logging
    log_level = 'DEBUG' if getattr(args, 'verbose', False) else 'INFO'
    logger = setup_logger('scientific-llm-forge', level=log_level)
    
    try:
        if args.command == 'train':
            from scripts.train import main as train_main
            train_main()
        elif args.command == 'serve':
            from scripts.serve import main as serve_main
            serve_main()
        elif args.command == 'evaluate':
            logger.info("Evaluation functionality not yet implemented")
            # TODO: Implement evaluation functionality
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 