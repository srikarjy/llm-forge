"""
Data validator for ScientificLLM-Forge.

This module provides functionality for validating and quality-checking
scientific datasets.
"""

from typing import Dict, Any, List


class DataValidator:
    """Validate and quality-check datasets for scientific LLM training."""
    
    def __init__(self):
        """Initialize the data validator."""
        pass
        
    def validate_data_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validate the format of a dataset.
        
        Args:
            data: List of data samples to validate
            
        Returns:
            True if data format is valid, False otherwise
        """
        if not data:
            return False
            
        required_fields = ["text", "label"]
        
        for sample in data:
            if not isinstance(sample, dict):
                return False
            for field in required_fields:
                if field not in sample:
                    return False
                    
        return True
    
    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of a dataset.
        
        Args:
            data: List of data samples to validate
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        quality_report = {
            "total_samples": len(data),
            "valid_samples": 0,
            "invalid_samples": 0,
            "issues": []
        }
        
        for i, sample in enumerate(data):
            if self._is_sample_valid(sample):
                quality_report["valid_samples"] += 1
            else:
                quality_report["invalid_samples"] += 1
                quality_report["issues"].append(f"Sample {i}: Invalid format")
                
        return quality_report
    
    def _is_sample_valid(self, sample: Dict[str, Any]) -> bool:
        """Check if a single sample is valid.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            True if sample is valid, False otherwise
        """
        if not isinstance(sample, dict):
            return False
            
        if "text" not in sample or "label" not in sample:
            return False
            
        if not isinstance(sample["text"], str) or not sample["text"].strip():
            return False
            
        return True 