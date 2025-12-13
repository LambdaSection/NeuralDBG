"""
Hyperparameter optimization (HPO) configuration settings.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from neural.config.base import BaseConfig


class HPOSettings(BaseConfig):
    """Hyperparameter optimization configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_HPO_"}
    
    # Optuna settings
    study_name_prefix: str = Field(
        default="neural_hpo",
        description="Prefix for study names"
    )
    storage_url: Optional[str] = Field(
        default=None,
        description="Optuna storage URL (None for in-memory)"
    )
    load_if_exists: bool = Field(
        default=True,
        description="Load existing study if it exists"
    )
    
    # Optimization settings
    n_trials: int = Field(
        default=100,
        gt=0,
        description="Number of optimization trials"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        description="Optimization timeout in seconds"
    )
    n_jobs: int = Field(
        default=1,
        gt=0,
        description="Number of parallel jobs"
    )
    
    # Search strategy
    sampler: Literal["tpe", "random", "grid", "cmaes"] = Field(
        default="tpe",
        description="Sampling strategy"
    )
    pruner: Optional[Literal["median", "hyperband", "percentile", "successive_halving"]] = Field(
        default="median",
        description="Pruning strategy"
    )
    
    # TPE sampler settings
    tpe_n_startup_trials: int = Field(
        default=10,
        ge=0,
        description="Number of startup trials for TPE"
    )
    tpe_n_ei_candidates: int = Field(
        default=24,
        gt=0,
        description="Number of EI candidates for TPE"
    )
    
    # Pruner settings
    pruner_warmup_steps: int = Field(
        default=10,
        ge=0,
        description="Number of warmup steps before pruning"
    )
    pruner_startup_trials: int = Field(
        default=5,
        ge=0,
        description="Number of startup trials before pruning"
    )
    
    # Early stopping
    early_stopping_enabled: bool = Field(
        default=True,
        description="Enable early stopping"
    )
    early_stopping_patience: int = Field(
        default=10,
        gt=0,
        description="Early stopping patience"
    )
    
    # Metric settings
    optimization_direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction"
    )
    primary_metric: str = Field(
        default="val_loss",
        description="Primary metric to optimize"
    )
    secondary_metrics: List[str] = Field(
        default=["val_accuracy"],
        description="Secondary metrics to track"
    )
    
    # Logging and visualization
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging"
    )
    show_progress_bar: bool = Field(
        default=True,
        description="Show progress bar"
    )
    
    # Checkpointing
    checkpoint_enabled: bool = Field(
        default=True,
        description="Enable checkpointing"
    )
    checkpoint_interval: int = Field(
        default=10,
        gt=0,
        description="Checkpoint save interval"
    )
    
    # Parameter importance analysis
    enable_parameter_importance: bool = Field(
        default=True,
        description="Enable parameter importance analysis"
    )
    importance_evaluator: Literal["fanova", "permutation"] = Field(
        default="fanova",
        description="Parameter importance evaluator"
    )
    
    # Distributed settings
    distributed_enabled: bool = Field(
        default=False,
        description="Enable distributed optimization"
    )
    distributed_backend: Optional[Literal["ray", "dask"]] = Field(
        default=None,
        description="Distributed backend"
    )
    
    # Resource allocation
    max_memory_gb: Optional[float] = Field(
        default=None,
        description="Maximum memory per trial in GB"
    )
    max_cpu_per_trial: Optional[int] = Field(
        default=None,
        description="Maximum CPUs per trial"
    )
    max_gpu_per_trial: Optional[int] = Field(
        default=None,
        description="Maximum GPUs per trial"
    )
