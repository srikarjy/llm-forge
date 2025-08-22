"""
Metrics collection for ScientificLLM-Forge.

This module provides functionality for collecting and managing
metrics during training and evaluation.
"""

from typing import Dict, Any, List
import time


class MetricsCollector:
    """Collect and manage metrics for training and evaluation."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {}
        self.start_time = None
        
    def start_timer(self, name: str) -> None:
        """Start a timer for a specific operation.
        
        Args:
            name: Name of the timer
        """
        self.start_time = time.time()
        
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time.
        
        Args:
            name: Name of the timer
            
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
            
        elapsed = time.time() - self.start_time
        self.metrics[f"{name}_time"] = elapsed
        self.start_time = None
        return elapsed
        
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
        
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear() 