"""
AutoML and Neural Architecture Search (NAS) configuration settings.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from neural.config.base import BaseConfig


class AutoMLSettings(BaseConfig):
    """AutoML and NAS configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_AUTOML_"}
    
    # Search strategy
    search_strategy: Literal["random", "bayesian", "evolutionary", "reinforcement", "darts"] = Field(
        default="bayesian",
        description="Architecture search strategy"
    )
    
    # Search space
    max_layers: int = Field(
        default=20,
        gt=0,
        description="Maximum number of layers"
    )
    min_layers: int = Field(
        default=2,
        gt=0,
        description="Minimum number of layers"
    )
    allowed_operations: List[str] = Field(
        default=[
            "conv2d", "maxpool2d", "avgpool2d", "dense",
            "batchnorm", "dropout", "residual", "attention"
        ],
        description="Allowed layer operations"
    )
    
    # Search settings
    n_architectures: int = Field(
        default=100,
        gt=0,
        description="Number of architectures to evaluate"
    )
    max_epochs_per_architecture: int = Field(
        default=50,
        gt=0,
        description="Maximum epochs per architecture"
    )
    timeout_hours: Optional[float] = Field(
        default=None,
        description="Search timeout in hours"
    )
    
    # Performance estimation
    use_early_stopping: bool = Field(
        default=True,
        description="Use early stopping for architecture evaluation"
    )
    early_stopping_patience: int = Field(
        default=5,
        gt=0,
        description="Early stopping patience"
    )
    use_performance_prediction: bool = Field(
        default=True,
        description="Use performance prediction to skip poor architectures"
    )
    
    # Weight sharing
    weight_sharing_enabled: bool = Field(
        default=True,
        description="Enable weight sharing across architectures"
    )
    supernet_training_epochs: int = Field(
        default=100,
        gt=0,
        description="Epochs for supernet training"
    )
    
    # Evolutionary algorithm settings
    population_size: int = Field(
        default=50,
        gt=0,
        description="Population size for evolutionary search"
    )
    mutation_rate: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Mutation rate"
    )
    crossover_rate: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Crossover rate"
    )
    
    # Reinforcement learning settings
    rl_controller_hidden_size: int = Field(
        default=100,
        gt=0,
        description="RL controller hidden size"
    )
    rl_learning_rate: float = Field(
        default=0.001,
        gt=0,
        description="RL controller learning rate"
    )
    
    # DARTS settings
    darts_alpha_learning_rate: float = Field(
        default=3e-4,
        gt=0,
        description="DARTS architecture parameter learning rate"
    )
    darts_weight_decay: float = Field(
        default=1e-3,
        ge=0,
        description="DARTS weight decay"
    )
    
    # Resource constraints
    max_params_millions: Optional[float] = Field(
        default=None,
        description="Maximum model parameters in millions"
    )
    max_flops_millions: Optional[float] = Field(
        default=None,
        description="Maximum FLOPs in millions"
    )
    target_latency_ms: Optional[float] = Field(
        default=None,
        description="Target inference latency in milliseconds"
    )
    
    # Multi-objective optimization
    multi_objective_enabled: bool = Field(
        default=False,
        description="Enable multi-objective optimization"
    )
    objectives: List[str] = Field(
        default=["accuracy", "params", "latency"],
        description="Optimization objectives"
    )
    objective_weights: List[float] = Field(
        default=[1.0, 0.5, 0.5],
        description="Objective weights"
    )
    
    # Distributed search
    distributed_enabled: bool = Field(
        default=False,
        description="Enable distributed architecture search"
    )
    distributed_backend: Optional[Literal["ray", "dask"]] = Field(
        default=None,
        description="Distributed backend"
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        description="Number of distributed workers"
    )
    
    # Checkpointing
    checkpoint_frequency: int = Field(
        default=10,
        gt=0,
        description="Checkpoint frequency (architectures)"
    )
    save_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of top architectures to save"
    )
    
    # Logging and visualization
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging"
    )
    visualize_architectures: bool = Field(
        default=True,
        description="Visualize discovered architectures"
    )
    log_architecture_details: bool = Field(
        default=True,
        description="Log detailed architecture information"
    )
