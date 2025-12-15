"""
Neural HPO - Enhanced Hyperparameter Optimization

This module provides advanced hyperparameter optimization capabilities including:
- Bayesian optimization with TPE and CMA-ES samplers
- Multi-objective optimization with Pareto front analysis
- Distributed HPO with Ray Tune integration
- Enhanced parameter importance analysis with multiple methods
- Rich visualizations for optimization results

Features
--------
- Automatic hyperparameter search spaces from DSL
- Multiple search strategies (TPE, Random, Grid)
- Distributed optimization support
- Parameter importance analysis
- Visualization of optimization progress

Functions
---------
get_data
    Load datasets for HPO experiments
create_dynamic_model
    Create models with trial hyperparameters
resolve_hpo_params
    Resolve HPO specifications to concrete values

Examples
--------
>>> from neural.hpo import create_dynamic_model
>>> import optuna
>>> trial = optuna.create_trial(...)
>>> model = create_dynamic_model(model_dict, trial, hpo_params)
"""

from . import utils
from .hpo import (
    DynamicPTModel,
    create_dynamic_model,
    get_data,
    objective,
    optimize_and_return,
    resolve_hpo_params,
    train_model,
)
from .parameter_importance import ParameterImportanceAnalyzer
from .strategies import (
    BaseStrategy,
    BayesianStrategy,
    EvolutionaryStrategy,
    PopulationBasedTraining,
    create_strategy,
)
from .visualization import (
    create_optimization_report,
    plot_3d_pareto,
    plot_contour,
    plot_convergence_comparison,
    plot_correlation_heatmap,
    plot_multi_objective_pareto,
    plot_optimization_history,
    plot_optimization_landscape,
    plot_parallel_coordinates,
    plot_param_importance,
    plot_slice,
)


__all__ = [
    # Core functions
    'optimize_and_return',
    'objective',
    'train_model',
    'create_dynamic_model',
    'get_data',
    'resolve_hpo_params',
    'DynamicPTModel',
    
    # Analysis
    'ParameterImportanceAnalyzer',
    
    # Strategies
    'BaseStrategy',
    'BayesianStrategy',
    'EvolutionaryStrategy',
    'PopulationBasedTraining',
    'create_strategy',
    
    # Visualization functions
    'plot_optimization_history',
    'plot_param_importance',
    'plot_parallel_coordinates',
    'plot_correlation_heatmap',
    'plot_contour',
    'plot_slice',
    'plot_multi_objective_pareto',
    'plot_3d_pareto',
    'plot_convergence_comparison',
    'plot_optimization_landscape',
    'create_optimization_report',
    
    # Utilities module
    'utils',
]

__version__ = '0.3.0'
