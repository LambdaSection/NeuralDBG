"""
Storage configuration settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator

from neural.config.base import BaseConfig


class StorageSettings(BaseConfig):
    """Storage configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_STORAGE_"}
    
    # Base storage paths
    base_path: str = Field(
        default="./neural_storage",
        description="Base storage directory"
    )
    experiments_path: str = Field(
        default="./neural_experiments",
        description="Experiments storage directory"
    )
    models_path: str = Field(
        default="./neural_models",
        description="Models storage directory"
    )
    datasets_path: str = Field(
        default="./neural_datasets",
        description="Datasets storage directory"
    )
    cache_path: str = Field(
        default="./neural_cache",
        description="Cache storage directory"
    )
    logs_path: str = Field(
        default="./neural_logs",
        description="Logs storage directory"
    )
    checkpoints_path: str = Field(
        default="./neural_checkpoints",
        description="Checkpoints storage directory"
    )
    
    # Storage settings
    auto_create_dirs: bool = Field(
        default=True,
        description="Automatically create storage directories"
    )
    max_storage_gb: Optional[float] = Field(
        default=None,
        description="Maximum storage size in GB"
    )
    cleanup_enabled: bool = Field(
        default=True,
        description="Enable automatic cleanup"
    )
    cleanup_threshold_days: int = Field(
        default=30,
        gt=0,
        description="Days before cleanup"
    )
    
    # File handling
    max_file_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum file size in MB"
    )
    allowed_extensions: list[str] = Field(
        default=[".json", ".neural", ".yaml", ".yml", ".h5", ".pt", ".onnx"],
        description="Allowed file extensions"
    )
    
    # Compression settings
    compression_enabled: bool = Field(
        default=True,
        description="Enable compression for stored files"
    )
    compression_level: int = Field(
        default=6,
        ge=0,
        le=9,
        description="Compression level (0-9)"
    )
    
    # Cloud storage settings
    cloud_storage_enabled: bool = Field(
        default=False,
        description="Enable cloud storage"
    )
    cloud_storage_provider: Optional[Literal["s3", "gcs", "azure"]] = Field(
        default=None,
        description="Cloud storage provider"
    )
    cloud_storage_bucket: Optional[str] = Field(
        default=None,
        description="Cloud storage bucket/container name"
    )
    cloud_storage_prefix: str = Field(
        default="neural-dsl",
        description="Cloud storage key prefix"
    )
    
    # S3 settings
    s3_region: str = Field(
        default="us-east-1",
        description="AWS S3 region"
    )
    s3_access_key: Optional[str] = Field(
        default=None,
        description="AWS S3 access key"
    )
    s3_secret_key: Optional[str] = Field(
        default=None,
        description="AWS S3 secret key"
    )
    
    # GCS settings
    gcs_project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID"
    )
    gcs_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to GCS credentials JSON"
    )
    
    # Azure settings
    azure_storage_account: Optional[str] = Field(
        default=None,
        description="Azure storage account name"
    )
    azure_storage_key: Optional[str] = Field(
        default=None,
        description="Azure storage access key"
    )
    
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    def get_full_path(self, path_type: str) -> Path:
        """
        Get full path for a storage type.
        
        Args:
            path_type: Type of path (experiments, models, etc.)
            
        Returns:
            Full Path object
        """
        path_map = {
            "base": self.base_path,
            "experiments": self.experiments_path,
            "models": self.models_path,
            "datasets": self.datasets_path,
            "cache": self.cache_path,
            "logs": self.logs_path,
            "checkpoints": self.checkpoints_path,
        }
        
        path_str = path_map.get(path_type, self.base_path)
        path_obj = Path(path_str)
        
        if self.auto_create_dirs:
            path_obj.mkdir(parents=True, exist_ok=True)
        
        return path_obj
