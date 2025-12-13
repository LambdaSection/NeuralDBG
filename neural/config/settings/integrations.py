"""
ML platform integrations configuration settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from neural.config.base import BaseConfig


class IntegrationSettings(BaseConfig):
    """ML platform integrations configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_INTEGRATION_"}
    
    # General settings
    timeout_seconds: int = Field(
        default=300,
        gt=0,
        description="Default timeout for integration operations"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts"
    )
    retry_delay_seconds: int = Field(
        default=5,
        gt=0,
        description="Delay between retries in seconds"
    )
    
    # AWS SageMaker
    sagemaker_enabled: bool = Field(
        default=False,
        description="Enable AWS SageMaker integration"
    )
    sagemaker_region: str = Field(
        default="us-east-1",
        description="AWS SageMaker region"
    )
    sagemaker_role_arn: Optional[str] = Field(
        default=None,
        description="AWS SageMaker execution role ARN"
    )
    sagemaker_bucket: Optional[str] = Field(
        default=None,
        description="AWS S3 bucket for SageMaker"
    )
    
    # Google Cloud Vertex AI
    vertex_enabled: bool = Field(
        default=False,
        description="Enable Google Cloud Vertex AI integration"
    )
    vertex_project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID"
    )
    vertex_region: str = Field(
        default="us-central1",
        description="Google Cloud region"
    )
    vertex_staging_bucket: Optional[str] = Field(
        default=None,
        description="GCS bucket for staging"
    )
    vertex_service_account: Optional[str] = Field(
        default=None,
        description="Service account email"
    )
    
    # Azure ML
    azure_enabled: bool = Field(
        default=False,
        description="Enable Azure ML integration"
    )
    azure_subscription_id: Optional[str] = Field(
        default=None,
        description="Azure subscription ID"
    )
    azure_resource_group: Optional[str] = Field(
        default=None,
        description="Azure resource group"
    )
    azure_workspace_name: Optional[str] = Field(
        default=None,
        description="Azure ML workspace name"
    )
    azure_tenant_id: Optional[str] = Field(
        default=None,
        description="Azure tenant ID"
    )
    
    # Databricks
    databricks_enabled: bool = Field(
        default=False,
        description="Enable Databricks integration"
    )
    databricks_host: Optional[str] = Field(
        default=None,
        description="Databricks workspace URL"
    )
    databricks_token: Optional[str] = Field(
        default=None,
        description="Databricks access token"
    )
    databricks_cluster_id: Optional[str] = Field(
        default=None,
        description="Databricks cluster ID"
    )
    
    # Paperspace
    paperspace_enabled: bool = Field(
        default=False,
        description="Enable Paperspace integration"
    )
    paperspace_api_key: Optional[str] = Field(
        default=None,
        description="Paperspace API key"
    )
    paperspace_region: str = Field(
        default="East Coast (NY2)",
        description="Paperspace region"
    )
    paperspace_machine_type: str = Field(
        default="P4000",
        description="Paperspace machine type"
    )
    
    # Run:AI
    runai_enabled: bool = Field(
        default=False,
        description="Enable Run:AI integration"
    )
    runai_cluster_url: Optional[str] = Field(
        default=None,
        description="Run:AI cluster URL"
    )
    runai_project: Optional[str] = Field(
        default=None,
        description="Run:AI project name"
    )
    runai_token: Optional[str] = Field(
        default=None,
        description="Run:AI access token"
    )
    
    # MLflow
    mlflow_enabled: bool = Field(
        default=False,
        description="Enable MLflow tracking"
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="neural-dsl",
        description="MLflow experiment name"
    )
    
    # Weights & Biases
    wandb_enabled: bool = Field(
        default=False,
        description="Enable Weights & Biases tracking"
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="W&B API key"
    )
    wandb_project: str = Field(
        default="neural-dsl",
        description="W&B project name"
    )
    wandb_entity: Optional[str] = Field(
        default=None,
        description="W&B entity (username or team)"
    )
    
    # TensorBoard
    tensorboard_enabled: bool = Field(
        default=True,
        description="Enable TensorBoard logging"
    )
    tensorboard_log_dir: str = Field(
        default="./logs/tensorboard",
        description="TensorBoard log directory"
    )
    
    # Experiment tracking
    auto_log_enabled: bool = Field(
        default=True,
        description="Enable automatic experiment logging"
    )
    log_model_artifacts: bool = Field(
        default=True,
        description="Log model artifacts"
    )
    log_system_metrics: bool = Field(
        default=True,
        description="Log system metrics"
    )
