"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class BackendType(str, Enum):
    """Supported backend types."""
    tensorflow = "tensorflow"
    pytorch = "pytorch"
    onnx = "onnx"


class JobStatus(str, Enum):
    """Job status types."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class DeploymentStatus(str, Enum):
    """Deployment status types."""
    deploying = "deploying"
    deployed = "deployed"
    failed = "failed"
    stopped = "stopped"


class CompileRequest(BaseModel):
    """Request model for compilation endpoint."""
    dsl_code: str = Field(..., description="Neural DSL code to compile")
    backend: BackendType = Field(default=BackendType.tensorflow, description="Target backend")
    dataset: str = Field(default="MNIST", description="Dataset name")
    auto_flatten_output: bool = Field(default=False, description="Auto-insert Flatten before Dense")
    enable_hpo: bool = Field(default=False, description="Enable hyperparameter optimization")


class CompileResponse(BaseModel):
    """Response model for compilation endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    compiled_code: Optional[str] = Field(None, description="Generated Python code")
    errors: Optional[List[str]] = Field(None, description="Compilation errors if any")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    validation_split: float = Field(default=0.2, ge=0, le=1, description="Validation split ratio")
    optimizer: str = Field(default="adam", description="Optimizer name")
    loss: str = Field(default="categorical_crossentropy", description="Loss function")
    metrics: List[str] = Field(default=["accuracy"], description="Metrics to track")


class TrainingJobRequest(BaseModel):
    """Request model for training job."""
    dsl_code: Optional[str] = Field(None, description="Neural DSL code")
    model_id: Optional[str] = Field(None, description="Pre-compiled model ID")
    backend: BackendType = Field(default=BackendType.tensorflow, description="Backend to use")
    dataset: str = Field(default="MNIST", description="Dataset name")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for notifications")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Additional hyperparameters")


class TrainingJobResponse(BaseModel):
    """Response model for training job."""
    job_id: str = Field(..., description="Unique job identifier")
    experiment_id: Optional[str] = Field(None, description="Experiment tracker ID")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Current metrics")


class ExperimentMetrics(BaseModel):
    """Experiment metrics."""
    latest: Dict[str, Any] = Field(default_factory=dict, description="Latest metrics")
    best: Dict[str, Any] = Field(default_factory=dict, description="Best metrics")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Metrics history")


class ExperimentResponse(BaseModel):
    """Response model for experiment details."""
    experiment_id: str = Field(..., description="Experiment ID")
    experiment_name: str = Field(..., description="Experiment name")
    status: str = Field(..., description="Experiment status")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: ExperimentMetrics = Field(default_factory=ExperimentMetrics)
    artifacts: List[str] = Field(default_factory=list, description="List of artifact names")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class DeploymentConfig(BaseModel):
    """Deployment configuration."""
    replicas: int = Field(default=1, ge=1, description="Number of replicas")
    memory_limit: str = Field(default="512Mi", description="Memory limit")
    cpu_limit: str = Field(default="500m", description="CPU limit")
    port: int = Field(default=8080, ge=1, le=65535, description="Service port")
    health_check_path: str = Field(default="/health", description="Health check endpoint")


class DeploymentRequest(BaseModel):
    """Request model for deployment."""
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_name: str = Field(..., description="Deployment name")
    backend: BackendType = Field(..., description="Backend type")
    config: DeploymentConfig = Field(default_factory=DeploymentConfig)
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")


class DeploymentResponse(BaseModel):
    """Response model for deployment."""
    deployment_id: str = Field(..., description="Deployment ID")
    deployment_name: str = Field(..., description="Deployment name")
    status: DeploymentStatus = Field(..., description="Deployment status")
    endpoint: Optional[str] = Field(None, description="Deployment endpoint URL")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobStatusResponse(BaseModel):
    """Response model for job status query."""
    job_id: str
    status: JobStatus
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WebhookPayload(BaseModel):
    """Webhook notification payload."""
    job_id: str
    event: str = Field(..., description="Event type (started, progress, completed, failed)")
    status: JobStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters."""
    skip: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of items to return")


class ExperimentListResponse(BaseModel):
    """Response model for listing experiments."""
    experiments: List[ExperimentResponse]
    total: int
    skip: int
    limit: int


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    name: str
    backend: BackendType
    size: int = Field(..., description="Model size in bytes")
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[ModelInfo]
    total: int
    skip: int
    limit: int
