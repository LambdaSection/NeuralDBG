"""
Neural API module.

This module provides a REST API server for Neural DSL using FastAPI.
It includes endpoints for model compilation, training jobs, experiment tracking,
and deployment management.
"""

from neural.api.main import app, create_app
from neural.api.models import (
    CompileRequest,
    CompileResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    ExperimentResponse,
    DeploymentRequest,
    DeploymentResponse,
)

__all__ = [
    'app',
    'create_app',
    'CompileRequest',
    'CompileResponse',
    'TrainingJobRequest',
    'TrainingJobResponse',
    'ExperimentResponse',
    'DeploymentRequest',
    'DeploymentResponse',
]
