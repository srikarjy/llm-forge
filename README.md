# ScientificLLM-Forge

An MLOps platform for fine-tuning scientific Large Language Models.

## Overview

ScientificLLM-Forge is a comprehensive platform designed to streamline the process of fine-tuning language models for scientific applications. It provides tools for data processing, model training, evaluation, and deployment with a focus on scientific datasets and use cases.

## Features

- **Data Processing**: Load, preprocess, validate, and augment scientific datasets
- **Model Training**: Fine-tune language models with advanced training techniques
- **Model Serving**: Deploy models with FastAPI-based REST APIs
- **Configuration Management**: YAML-based configuration system
- **Logging & Monitoring**: Comprehensive logging and metrics collection
- **Testing**: Unit and integration test suites

## Project Structure

```
scientific-llm-forge/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and evaluation
│   ├── serving/           # Model serving and deployment
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Automation scripts
├── tests/                 # Test suites
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
└── .gitignore            # Git ignore patterns
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd scientific-llm-forge
   ```

2. **Run the setup script**:
   ```bash
   python scripts/setup_dev.py
   ```

   This will:
   - Create a Python virtual environment
   - Install all dependencies from `requirements.txt`
   - Set up pre-commit hooks for code quality
   - Create example configuration files
   - Set up logging configuration

3. **Activate the development environment**:
   ```bash
   # On macOS/Linux:
   source activate_dev.sh
   
   # On Windows:
   activate_dev.bat
   ```

### Basic Usage

#### Training a Model

```bash
# Using the training script
python scripts/train.py --config configs/training.yaml

# Using the main CLI
python src/main.py train --config configs/training.yaml
```

#### Serving a Model

```bash
# Using the serving script
python scripts/serve.py --config configs/serving.yaml

# Using the main CLI
python src/main.py serve --config configs/serving.yaml
```

#### Running Tests

```bash
pytest tests/
```

## Configuration

The platform uses YAML configuration files for different components:

- `configs/training.yaml`: Model training configuration
- `configs/serving.yaml`: Model serving configuration
- `configs/pubmed_api.yaml`: PubMed API configuration for data collection

### Example Training Configuration

```yaml
training:
  model:
    name: "microsoft/DialoGPT-medium"
    max_length: 512
    
  hyperparameters:
    learning_rate: 5e-5
    batch_size: 8
    num_epochs: 3
    
  data:
    train_file: "data/train.jsonl"
    validation_file: "data/validation.jsonl"
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Adding New Features

1. Create feature branch from `main`
2. Implement your changes
3. Add tests in the `tests/` directory
4. Update documentation
5. Run tests and code quality checks
6. Submit a pull request

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub or contact the development team.

## Roadmap

- [ ] PubMed data collection integration
- [ ] Advanced data augmentation techniques
- [ ] Distributed training support
- [ ] Model versioning and management
- [ ] Web-based training dashboard
- [ ] Kubernetes deployment support
