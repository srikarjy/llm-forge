#!/usr/bin/env python3
"""
Development environment setup script for ScientificLLM-Forge.

This script automates the setup of a development environment including:
- Python virtual environment creation
- Dependency installation
- Pre-commit hooks setup
- Logging configuration
- Example configuration files
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


def run_command(cmd: list, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result.
    
    Args:
        cmd: Command to run as a list
        cwd: Working directory for the command
        check: Whether to raise an exception on non-zero return code
        
    Returns:
        CompletedProcess object
    """
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        if check:
            raise
        return e


def check_python_version() -> bool:
    """Check if Python version meets requirements.
    
    Returns:
        True if Python version is compatible
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_virtual_environment(venv_path: Path, force: bool = False) -> bool:
    """Create a Python virtual environment.
    
    Args:
        venv_path: Path where to create the virtual environment
        force: Whether to recreate if it already exists
        
    Returns:
        True if successful
    """
    if venv_path.exists() and not force:
        print(f"Virtual environment already exists at {venv_path}")
        return True
    
    if venv_path.exists() and force:
        print(f"Removing existing virtual environment at {venv_path}")
        import shutil
        shutil.rmtree(venv_path)
    
    print(f"Creating virtual environment at {venv_path}")
    try:
        run_command([sys.executable, "-m", "venv", str(venv_path)])
        return True
    except subprocess.CalledProcessError:
        print("Failed to create virtual environment")
        return False


def get_venv_python(venv_path: Path) -> Path:
    """Get the Python executable path in the virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        
    Returns:
        Path to the Python executable
    """
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip(venv_path: Path) -> Path:
    """Get the pip executable path in the virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        
    Returns:
        Path to the pip executable
    """
    if sys.platform == "win32":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def install_requirements(venv_path: Path, requirements_file: Path) -> bool:
    """Install requirements from requirements.txt.
    
    Args:
        venv_path: Path to the virtual environment
        requirements_file: Path to requirements.txt
        
    Returns:
        True if successful
    """
    if not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}")
        return False
    
    pip_path = get_venv_pip(venv_path)
    print(f"Installing requirements from {requirements_file}")
    
    try:
        # Upgrade pip first
        run_command([str(pip_path), "install", "--upgrade", "pip"])
        
        # Install requirements
        run_command([str(pip_path), "install", "-r", str(requirements_file)])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install requirements")
        return False


def setup_pre_commit(venv_path: Path) -> bool:
    """Set up pre-commit hooks.
    
    Args:
        venv_path: Path to the virtual environment
        
    Returns:
        True if successful
    """
    python_path = get_venv_python(venv_path)
    
    try:
        # Install pre-commit
        run_command([str(python_path), "-m", "pip", "install", "pre-commit"])
        
        # Install pre-commit hooks
        run_command([str(python_path), "-m", "pre_commit", "install"])
        
        # Install pre-commit hooks for all files
        run_command([str(python_path), "-m", "pre_commit", "install", "--hook-type", "pre-push"])
        
        return True
    except subprocess.CalledProcessError:
        print("Failed to set up pre-commit hooks")
        return False


def create_pre_commit_config() -> bool:
    """Create .pre-commit-config.yaml file.
    
    Returns:
        True if successful
    """
    config_content = """# Pre-commit configuration for ScientificLLM-Forge
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
"""
    
    config_path = Path(".pre-commit-config.yaml")
    try:
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"Created {config_path}")
        return True
    except Exception as e:
        print(f"Failed to create pre-commit config: {e}")
        return False


def create_example_config() -> bool:
    """Create example configuration files.
    
    Returns:
        True if successful
    """
    # Create PubMed API configuration
    pubmed_config = """# PubMed API configuration for ScientificLLM-Forge
api:
  pubmed:
    base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email: "your-email@example.com"  # Required for NCBI API
    tool: "ScientificLLM-Forge"
    max_requests_per_second: 3  # NCBI rate limit
    max_requests_per_day: 10000  # NCBI daily limit
    
  # Optional: API key for higher rate limits
  # api_key: "your-api-key-here"

data:
  pubmed:
    # Search parameters
    search_terms: [
      "machine learning",
      "deep learning", 
      "artificial intelligence",
      "natural language processing"
    ]
    date_from: "2020-01-01"
    date_to: "2024-01-01"
    max_results: 1000
    
    # Filter options
    article_types: [
      "research-article",
      "review-article",
      "case-report"
    ]
    languages: ["eng"]
    
    # Output settings
    output_format: "jsonl"
    include_abstract: true
    include_keywords: true
    include_authors: true
    include_journal_info: true

processing:
  # Text preprocessing
  max_abstract_length: 1000
  min_abstract_length: 50
  remove_html_tags: true
  normalize_whitespace: true
  
  # Data cleaning
  remove_duplicates: true
  filter_by_language: true
  required_fields: ["title", "abstract"]

storage:
  # Local storage settings
  output_dir: "data/pubmed"
  backup_dir: "data/backup"
  
  # Database settings (optional)
  # database:
  #   type: "postgresql"
  #   host: "localhost"
  #   port: 5432
  #   database: "scientific_llm_forge"
  #   username: "user"
  #   password: "password"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pubmed_data.log"
  max_file_size: "10MB"
  backup_count: 5
"""
    
    config_path = Path("configs/pubmed_api.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, "w") as f:
            f.write(pubmed_config)
        print(f"Created {config_path}")
        return True
    except Exception as e:
        print(f"Failed to create PubMed config: {e}")
        return False


def create_logging_config() -> bool:
    """Create logging configuration.
    
    Returns:
        True if successful
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create .gitkeep to ensure logs directory is tracked
    gitkeep_file = logs_dir / ".gitkeep"
    gitkeep_file.touch(exist_ok=True)
    
    # Create basic logging configuration
    logging_config = """# Logging configuration for ScientificLLM-Forge
import logging
import logging.handlers
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.handlers.RotatingFileHandler(
            logs_dir / "scientific_llm_forge.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)

# Configure specific loggers
loggers = {
    'data': logging.getLogger('data'),
    'models': logging.getLogger('models'),
    'serving': logging.getLogger('serving'),
    'utils': logging.getLogger('utils'),
}

for logger_name, logger in loggers.items():
    logger.setLevel(logging.INFO)
"""
    
    config_path = Path("src/logging_config.py")
    try:
        with open(config_path, "w") as f:
            f.write(logging_config)
        print(f"Created {config_path}")
        return True
    except Exception as e:
        print(f"Failed to create logging config: {e}")
        return False


def create_development_script(venv_path: Path) -> bool:
    """Create a development activation script.
    
    Args:
        venv_path: Path to the virtual environment
        
    Returns:
        True if successful
    """
    if sys.platform == "win32":
        script_content = f"""@echo off
REM Development environment activation script for ScientificLLM-Forge
echo Activating ScientificLLM-Forge development environment...

REM Activate virtual environment
call "{venv_path}\\Scripts\\activate.bat"

REM Set environment variables
set PYTHONPATH=%cd%\\src;%PYTHONPATH%
set SCIENTIFIC_LLM_FORGE_ENV=development

echo Development environment activated!
echo Run 'deactivate' to exit the environment.
"""
        script_path = Path("activate_dev.bat")
    else:
        script_content = f"""#!/bin/bash
# Development environment activation script for ScientificLLM-Forge
echo "Activating ScientificLLM-Forge development environment..."

# Activate virtual environment
source "{venv_path}/bin/activate"

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export SCIENTIFIC_LLM_FORGE_ENV="development"

echo "Development environment activated!"
echo "Run 'deactivate' to exit the environment."
"""
        script_path = Path("activate_dev.sh")
    
    try:
        with open(script_path, "w") as f:
            f.write(script_content)
        
        if sys.platform != "win32":
            os.chmod(script_path, 0o755)
        
        print(f"Created {script_path}")
        return True
    except Exception as e:
        print(f"Failed to create development script: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up ScientificLLM-Forge development environment")
    parser.add_argument("--venv-path", type=str, default="venv", help="Path to virtual environment")
    parser.add_argument("--force", action="store_true", help="Force recreation of virtual environment")
    parser.add_argument("--skip-pre-commit", action="store_true", help="Skip pre-commit setup")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip requirements installation")
    
    args = parser.parse_args()
    
    print("Setting up ScientificLLM-Forge development environment...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Convert paths to Path objects
    venv_path = Path(args.venv_path)
    requirements_file = Path("requirements.txt")
    project_root = Path.cwd()
    
    # Create virtual environment
    if not create_virtual_environment(venv_path, args.force):
        sys.exit(1)
    
    # Install requirements
    if not args.skip_requirements:
        if not install_requirements(venv_path, requirements_file):
            sys.exit(1)
    
    # Set up pre-commit
    if not args.skip_pre_commit:
        if not create_pre_commit_config():
            sys.exit(1)
        if not setup_pre_commit(venv_path):
            sys.exit(1)
    
    # Create configuration files
    if not create_example_config():
        sys.exit(1)
    
    # Set up logging
    if not create_logging_config():
        sys.exit(1)
    
    # Create development script
    if not create_development_script(venv_path):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Development environment setup completed successfully!")
    print("="*60)
    
    if sys.platform == "win32":
        print(f"\nTo activate the development environment, run:")
        print(f"  activate_dev.bat")
    else:
        print(f"\nTo activate the development environment, run:")
        print(f"  source activate_dev.sh")
    
    print(f"\nTo run tests:")
    print(f"  {get_venv_python(venv_path)} -m pytest tests/")
    
    print(f"\nTo start training:")
    print(f"  {get_venv_python(venv_path)} scripts/train.py --help")
    
    print(f"\nTo start serving:")
    print(f"  {get_venv_python(venv_path)} scripts/serve.py --help")
    
    print(f"\nHappy coding! 🚀")


if __name__ == "__main__":
    main() 