#!/bin/bash
# Development environment activation script for ScientificLLM-Forge
echo "Activating ScientificLLM-Forge development environment..."

# Activate virtual environment
source "venv/bin/activate"

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export SCIENTIFIC_LLM_FORGE_ENV="development"

echo "Development environment activated!"
echo "Run 'deactivate' to exit the environment."
