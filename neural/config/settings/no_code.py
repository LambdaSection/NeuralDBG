"""
No-code interface configuration settings.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from neural.config.base import BaseConfig


class NoCodeSettings(BaseConfig):
    """No-code interface configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_NOCODE_"}
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(
        default=8051,
        gt=0,
        lt=65536,
        description="Server port"
    )
    debug: bool = Field(default=False, description="Debug mode")
    
    # Paths
    saved_models_dir: str = Field(
        default="./neural_nocode/saved_models",
        description="Saved models directory"
    )
    exported_models_dir: str = Field(
        default="./neural_nocode/exported_models",
        description="Exported models directory"
    )
    templates_dir: str = Field(
        default="./neural_nocode/templates",
        description="Templates directory"
    )
    
    # UI settings
    default_input_shape: List[Optional[int]] = Field(
        default=[None, 28, 28, 1],
        description="Default input shape for models"
    )
    default_optimizer_type: str = Field(
        default="Adam",
        description="Default optimizer type"
    )
    default_learning_rate: float = Field(
        default=0.001,
        gt=0,
        description="Default learning rate"
    )
    default_loss: str = Field(
        default="categorical_crossentropy",
        description="Default loss function"
    )
    
    # Validation settings
    max_layers: int = Field(
        default=100,
        gt=0,
        description="Maximum number of layers"
    )
    validation_timeout: int = Field(
        default=10,
        gt=0,
        description="Validation timeout in seconds"
    )
    
    # Code generation settings
    supported_backends: List[str] = Field(
        default=["tensorflow", "pytorch", "onnx"],
        description="Supported ML backends"
    )
    default_backend: str = Field(
        default="tensorflow",
        description="Default ML backend"
    )
    
    # Feature flags
    enable_tutorials: bool = Field(
        default=True,
        description="Enable tutorials"
    )
    enable_templates: bool = Field(
        default=True,
        description="Enable model templates"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable model validation"
    )
    enable_code_generation: bool = Field(
        default=True,
        description="Enable code generation"
    )
    enable_model_save: bool = Field(
        default=True,
        description="Enable model saving"
    )
    enable_model_export: bool = Field(
        default=True,
        description="Enable model export"
    )
    
    # Security settings
    max_file_size_mb: int = Field(
        default=10,
        gt=0,
        description="Maximum file size in MB"
    )
    allowed_extensions: List[str] = Field(
        default=[".json", ".neural"],
        description="Allowed file extensions"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Auto-save settings
    auto_save_enabled: bool = Field(
        default=True,
        description="Enable auto-save"
    )
    auto_save_interval_seconds: int = Field(
        default=30,
        gt=0,
        description="Auto-save interval in seconds"
    )
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
