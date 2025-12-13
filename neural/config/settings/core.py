"""
Core Neural DSL configuration settings.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from neural.config.base import BaseConfig, EnvironmentType


class CoreSettings(BaseConfig):
    """Core Neural DSL settings."""
    
    model_config = {"env_prefix": "NEURAL_"}
    
    # Application settings
    app_name: str = Field(default="Neural DSL", description="Application name")
    version: str = Field(default="0.3.0", description="Application version")
    environment: str = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Environment type"
    )
    debug: bool = Field(default=False, description="Debug mode")
    
    # Parser settings
    parser_cache_enabled: bool = Field(
        default=True,
        description="Enable parser caching"
    )
    parser_strict_mode: bool = Field(
        default=False,
        description="Enable strict parsing mode"
    )
    
    # Code generation settings
    default_backend: Literal["tensorflow", "pytorch", "onnx"] = Field(
        default="tensorflow",
        description="Default ML backend"
    )
    code_generation_optimize: bool = Field(
        default=True,
        description="Enable code generation optimizations"
    )
    
    # Execution settings
    max_execution_time: int = Field(
        default=3600,
        gt=0,
        description="Maximum execution time in seconds"
    )
    execution_timeout: Optional[int] = Field(
        default=None,
        description="Execution timeout in seconds"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Performance settings
    num_workers: int = Field(
        default=4,
        gt=0,
        description="Number of worker processes"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    
    # Shape propagation settings
    shape_validation_enabled: bool = Field(
        default=True,
        description="Enable shape validation"
    )
    shape_inference_enabled: bool = Field(
        default=True,
        description="Enable shape inference"
    )
    
    # Feature flags
    enable_experimental_features: bool = Field(
        default=False,
        description="Enable experimental features"
    )
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU support if available"
    )
    enable_distributed: bool = Field(
        default=False,
        description="Enable distributed execution"
    )
