"""
Metrics package for Neural.

This package provides functionality to collect real metrics during model training.
"""

from .metrics_collector import MetricsCollector
from .model_trainer import ModelTrainer

__all__ = ['MetricsCollector', 'ModelTrainer']
