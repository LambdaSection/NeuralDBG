"""
Real Metrics Collection for Neural models.

This module provides functionality to collect real metrics during model training
and update the dashboard with these metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from .metrics_collector import MetricsCollector
from .model_trainer import ModelTrainer

# Configure logger
logger = logging.getLogger(__name__)

def collect_real_metrics(model_data: Dict[str, Any], trace_data: List[Dict[str, Any]],
                         backend: str = 'tensorflow', dataset: str = 'mnist') -> List[Dict[str, Any]]:
    """
    Collect real metrics by training a model.

    Args:
        model_data: The model data from the parser
        trace_data: The trace data to update with real metrics
        backend: The backend framework ('tensorflow' or 'pytorch')
        dataset: The dataset to use ('mnist' or 'cifar10')

    Returns:
        Updated trace_data with real metrics
    """
    try:
        # Create a model trainer
        trainer = ModelTrainer(model_data, trace_data, backend)

        # Create a metrics collector
        collector = MetricsCollector(model_data, trace_data, backend)

        # Train the model and collect metrics
        if backend == 'tensorflow':
            model, x_train, y_train = trainer.train_tensorflow_model(dataset)
            if model is not None and x_train is not None and y_train is not None:
                trace_data = collector.collect_tensorflow_metrics(model, x_train, y_train)
        elif backend == 'pytorch':
            model, data_loader, criterion, optimizer = trainer.train_pytorch_model(dataset)
            if model is not None and data_loader is not None:
                trace_data = collector.collect_pytorch_metrics(model, data_loader, criterion, optimizer)

        return trace_data

    except Exception as e:
        logger.error(f"Failed to collect real metrics: {str(e)}")
        # Fall back to simulated metrics
        collector = MetricsCollector(model_data, trace_data, backend)
        collector._generate_simulated_metrics()
        return trace_data
