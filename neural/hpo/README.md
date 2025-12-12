# Neural HPO - Enhanced Hyperparameter Optimization

This module provides state-of-the-art hyperparameter optimization capabilities for Neural DSL.

## Features

### 1. Bayesian Optimization
- **TPE Sampler**: Tree-structured Parzen Estimator for efficient hyperparameter search
- **CMA-ES Sampler**: Covariance Matrix Adaptation Evolution Strategy
- **Gaussian Process**: GP-based parameter importance analysis

### 2. Multi-Objective Optimization
- Optimize multiple objectives simultaneously (e.g., accuracy, inference time, model size)
- Pareto front visualization and analysis
- NSGA-II sampler for multi-objective optimization
- 2D and 3D Pareto front plotting

### 3. Distributed HPO with Ray Tune
- Scale HPO across multiple CPUs/GPUs
- ASHA scheduler for early stopping
- Population-Based Training (PBT)
- Asynchronous parallel trials

### 4. Parameter Importance Analysis
- Random Forest importance
- Gradient Boosting importance
- Permutation importance
- fANOVA (functional ANOVA)
- Gaussian Process-based importance
- Bootstrap uncertainty estimation

### 5. Rich Visualizations
- Optimization history plots
- Parameter importance plots with uncertainty
- Parallel coordinates plots
- Correlation heatmaps
- Contour plots for 2D parameter spaces
- Slice plots for individual parameters
- Marginal effects plots
- Parameter interaction heatmaps
- Convergence comparison across runs
- Comprehensive HTML reports

## Usage Examples

### Basic Bayesian Optimization

```python
from neural.hpo import optimize_and_return

# Define your model configuration
config = """
network MyModel {
    input: (28, 28, 1)
    
    Dense(units: hpo(categorical: [64, 128, 256])) -> relu
    Dropout(rate: hpo(range: [0.3, 0.7, step=0.1]))
    Dense(units: 10) -> softmax
    
    optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
    training: {
        batch_size: hpo(categorical: [16, 32, 64])
    }
}
"""

# Run Bayesian optimization with TPE sampler (default)
results = optimize_and_return(
    config=config,
    n_trials=50,
    dataset_name='MNIST',
    backend='pytorch',
    device='auto',
    sampler='tpe',  # Bayesian optimization
    enable_pruning=True
)

print(f"Best parameters: {results}")
```

### Multi-Objective Optimization

```python
from neural.hpo import MultiObjectiveOptimizer, objective

# Define objectives to optimize
objectives = ['loss', 'accuracy', 'precision']
directions = ['minimize', 'maximize', 'maximize']

# Create multi-objective optimizer
moo = MultiObjectiveOptimizer(objectives, directions)

# Run optimization
results = moo.optimize(
    objective_fn=objective,
    n_trials=100,
    sampler='nsgaii',  # NSGA-II for multi-objective
    config=config,
    dataset_name='MNIST',
    backend='pytorch',
    device='auto'
)

# Get Pareto front
pareto_front = moo.get_pareto_front()
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# Visualize Pareto front
fig = moo.plot_pareto_front(obj_x=0, obj_y=1)
fig.savefig('pareto_front.png')
```

### Distributed HPO with Ray Tune

```python
from neural.hpo import DistributedHPO

# Initialize distributed HPO
dist_hpo = DistributedHPO(use_ray=True)

# Define configuration space for Ray Tune
config_space = {
    'batch_size': dist_hpo.tune.choice([16, 32, 64]),
    'learning_rate': dist_hpo.tune.loguniform(1e-4, 1e-1),
    'dense_units': dist_hpo.tune.choice([64, 128, 256]),
    'dropout_rate': dist_hpo.tune.uniform(0.3, 0.7)
}

# Run distributed optimization
results = dist_hpo.optimize_with_ray(
    trainable_fn=my_training_function,
    config_space=config_space,
    n_trials=100,
    n_cpus=2,
    n_gpus=1,
    scheduler='asha',  # Asynchronous Successive Halving Algorithm
    search_alg='optuna',  # Optuna search algorithm
    metric='accuracy',
    mode='max'
)

print(f"Best config: {results['best_config']}")
print(f"Best result: {results['best_result']}")
```

### Advanced Parameter Importance Analysis

```python
from neural.hpo import ParameterImportanceAnalyzer, BayesianParameterImportance
from neural.hpo import plot_param_importance

# Standard importance analysis
analyzer = ParameterImportanceAnalyzer(method='random_forest')
importance = analyzer.analyze(trials, target_metric='accuracy')

# Plot importance with uncertainty (bootstrap)
fig = analyzer.plot_importance_with_std(
    trials=trials,
    target_metric='accuracy',
    n_iterations=20
)
fig.savefig('importance_with_uncertainty.png')

# Bayesian parameter importance with Gaussian Process
bayesian_analyzer = BayesianParameterImportance()
gp_importance = bayesian_analyzer.analyze_with_gp(trials, 'accuracy')
fig = bayesian_analyzer.plot_importance_with_uncertainty(trials, 'accuracy')
fig.savefig('gp_importance.png')

# Parameter interaction analysis
fig = analyzer.plot_interaction_heatmap(trials, 'accuracy')
fig.savefig('interactions.png')

# Marginal effects
fig = analyzer.plot_marginal_effects(trials, 'accuracy')
fig.savefig('marginal_effects.png')

# fANOVA importance
fanova_importance = analyzer.analyze_with_fanova(trials, 'accuracy')
print("fANOVA importances:", fanova_importance)
```

### Visualization Suite

```python
from neural.hpo.visualization import (
    plot_optimization_history,
    plot_parallel_coordinates,
    plot_correlation_heatmap,
    plot_contour,
    plot_multi_objective_pareto,
    plot_convergence_comparison,
    create_optimization_report
)

# Optimization history
fig = plot_optimization_history(trials, metric='accuracy')
fig.savefig('history.png')

# Parallel coordinates plot
fig = plot_parallel_coordinates(trials, metric='accuracy', top_n=10)
fig.savefig('parallel_coords.png')

# Correlation heatmap
fig = plot_correlation_heatmap(trials, metric='accuracy')
fig.savefig('correlations.png')

# 2D contour plot for two parameters
fig = plot_contour(trials, 'learning_rate', 'batch_size', metric='accuracy')
fig.savefig('contour.png')

# Multi-objective Pareto front
fig = plot_multi_objective_pareto(
    trials, 
    obj_x='loss', 
    obj_y='accuracy',
    highlight_pareto=True
)
fig.savefig('pareto.png')

# Compare convergence across different runs
trials_dict = {
    'TPE': tpe_trials,
    'Random': random_trials,
    'CMA-ES': cmaes_trials
}
fig = plot_convergence_comparison(trials_dict, metric='accuracy')
fig.savefig('convergence.png')

# Generate comprehensive HTML report
report_path = create_optimization_report(
    trials,
    metric='accuracy',
    output_path='hpo_report.html'
)
print(f"Report saved to: {report_path}")
```

### Using Different Samplers

```python
# TPE (Tree-structured Parzen Estimator) - Default Bayesian
results_tpe = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='tpe'
)

# CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
results_cmaes = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='cmaes'
)

# Random sampling (baseline)
results_random = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='random'
)

# NSGA-II for multi-objective
results_nsgaii = optimize_and_return(
    config=config,
    n_trials=100,
    objectives=['loss', 'accuracy'],
    sampler='nsgaii'
)
```

### Complete Example with All Features

```python
from neural.hpo import (
    optimize_and_return,
    ParameterImportanceAnalyzer,
    BayesianParameterImportance,
    create_optimization_report
)

# 1. Run optimization
results = optimize_and_return(
    config=config,
    n_trials=100,
    dataset_name='MNIST',
    backend='pytorch',
    device='cuda',
    sampler='tpe',
    enable_pruning=True,
    study_name='mnist_experiment'
)

# 2. Extract trial history
trials = results.get('_trials_history', [])
study = results.get('_study')

# 3. Analyze parameter importance
analyzer = ParameterImportanceAnalyzer(method='random_forest')
importance = analyzer.analyze(trials, target_metric='accuracy')
print("Parameter importances:", importance)

# 4. Create visualizations
fig1 = analyzer.plot_importance_with_std(trials, 'accuracy', n_iterations=20)
fig1.savefig('importance.png')

fig2 = analyzer.plot_interaction_heatmap(trials, 'accuracy')
fig2.savefig('interactions.png')

fig3 = analyzer.plot_marginal_effects(trials, 'accuracy')
fig3.savefig('marginal_effects.png')

# 5. Generate comprehensive report
report_path = create_optimization_report(
    trials,
    metric='accuracy',
    output_path='hpo_report.html'
)

print(f"Optimization complete!")
print(f"Best parameters: {results}")
print(f"Report saved to: {report_path}")
```

## API Reference

### Core Functions

- `optimize_and_return(config, n_trials, dataset_name, backend, device, sampler, objectives, use_ray, enable_pruning, study_name)`
  - Main optimization function with all features
  
- `objective(trial, config, dataset_name, backend, device)`
  - Objective function for a single trial

### Classes

- `MultiObjectiveOptimizer(objectives, directions)`
  - Multi-objective optimization with Pareto analysis
  
- `DistributedHPO(use_ray)`
  - Distributed optimization with Ray Tune
  
- `BayesianParameterImportance()`
  - Bayesian parameter importance analysis
  
- `ParameterImportanceAnalyzer(method)`
  - General parameter importance analysis

### Visualization Functions

- `plot_optimization_history(trials, metric, figsize)`
- `plot_param_importance(trials, metric, method, figsize)`
- `plot_parallel_coordinates(trials, metric, top_n, figsize)`
- `plot_correlation_heatmap(trials, metric, figsize)`
- `plot_contour(trials, param_x, param_y, metric, figsize)`
- `plot_slice(trials, param, metric, figsize)`
- `plot_multi_objective_pareto(trials, obj_x, obj_y, highlight_pareto, figsize)`
- `plot_3d_pareto(trials, obj_x, obj_y, obj_z, figsize)`
- `plot_convergence_comparison(trials_dict, metric, figsize)`
- `plot_optimization_landscape(trials, param, metric, n_bins, figsize)`
- `create_optimization_report(trials, metric, output_path)`

## Installation

The HPO module requires the following optional dependencies:

```bash
# Basic HPO with Optuna
pip install optuna

# For distributed HPO with Ray Tune
pip install "ray[tune]"

# For Bayesian optimization
pip install scikit-learn scipy

# For all features
pip install -e ".[full]"
```

## Performance Tips

1. **Start with fewer trials**: Use 10-20 trials for initial exploration, then scale up
2. **Enable pruning**: Set `enable_pruning=True` to stop unpromising trials early
3. **Use appropriate sampler**: 
   - TPE for general Bayesian optimization
   - CMA-ES for continuous parameters
   - Random for baseline comparison
   - NSGA-II for multi-objective
4. **Leverage Ray Tune**: Use distributed optimization for large search spaces
5. **Monitor parameter importance**: Focus search on important parameters

## Citation

If you use Neural HPO in your research, please cite:

```bibtex
@software{neural_dsl,
  title = {Neural DSL: A Domain-Specific Language for Neural Networks},
  author = {Neural DSL Team},
  year = {2024},
  url = {https://github.com/Lemniscate-SHA-256/Neural}
}
```
