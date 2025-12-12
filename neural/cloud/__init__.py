"""
Neural DSL Cloud Integration Module.

This module provides comprehensive tools and utilities for running Neural DSL
in cloud environments including Kaggle, Google Colab, and AWS SageMaker.

Features
--------
- Automatic cloud environment detection
- Seamless model compilation and deployment
- GPU resource management
- Remote training and monitoring
- Integration with cloud-native ML services

Classes
-------
CloudExecutor
    Execute Neural DSL models in cloud environments
RemoteConnection
    Manage remote connections for distributed training
SageMakerHandler
    AWS SageMaker integration (when available)

Examples
--------
>>> from neural.cloud import CloudExecutor
>>> executor = CloudExecutor()  # Auto-detects environment
>>> code_path = executor.compile_model(dsl_code, backend='tensorflow')
>>> executor.train_model(code_path, epochs=10, batch_size=32)
"""

from .cloud_execution import CloudExecutor
from .remote_connection import RemoteConnection

# Import SageMaker integration if running in SageMaker environment
import os
if 'SM_MODEL_DIR' in os.environ:
    from .sagemaker_integration import SageMakerHandler, sagemaker_entry_point

__all__ = ['CloudExecutor', 'RemoteConnection']
if 'SM_MODEL_DIR' in os.environ:
    __all__.extend(['SageMakerHandler', 'sagemaker_entry_point'])
