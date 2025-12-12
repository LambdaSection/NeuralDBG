"""
Neural HPO - Enhanced Hyperparameter Optimization

This module provides advanced hyperparameter optimization capabilities including:
- Bayesian optimization with TPE and CMA-ES samplers
- Multi-objective optimization with Pareto front analysis
- Distributed HPO with Ray Tune integration
- Enhanced parameter importance analysis with multiple methods
- Rich visualizations for optimization results
"""

from .hpo import (
    optimize_and_return,
    objective,
    train_model,
    create_dynamic_model,
    MultiObjectiveOptimizer,
    DistributedHPO,
    BayesianParameterImportance
)

from .parameter_importance import ParameterImportanceAnalyzer

from .visualization import (
    plot_optimization_history,
    plot_param_importance,
    plot_parallel_coordinates,
    plot_correlation_heatmap,
    plot_contour,
    plot_slice,
    plot_multi_objective_pareto,
    plot_3d_pareto,
    plot_convergence_comparison,
    plot_optimization_landscape,
    create_optimization_report
)

from .strategies import (
    BaseStrategy,
    BayesianStrategy,
    EvolutionaryStrategy,
    PopulationBasedTraining,
    create_strategy
)

from . import utils

__all__ = [
    # Core functions
    'optimize_and_return',
    'objective',
    'train_model',
    'create_dynamic_model',
    
    # Optimization classes
    'MultiObjectiveOptimizer',
    'DistributedHPO',
    'BayesianParameterImportance',
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
