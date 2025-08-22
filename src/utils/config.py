"""
Configuration management for ScientificLLM-Forge.

This module provides functionality for loading and managing
configuration files in various formats.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manage configuration files for ScientificLLM-Forge."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        pass
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return self._load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            return self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_json(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to a file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            self._save_yaml(config, config_path)
        elif config_path.suffix.lower() == '.json':
            self._save_json(config, config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def _save_yaml(self, config: Dict[str, Any], config_path: Path) -> None:
        """Save configuration as YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the YAML file
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def _save_json(self, config: Dict[str, Any], config_path: Path) -> None:
        """Save configuration as JSON file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the JSON file
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2) 