import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer
import keras
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.execution_optimization.execution import get_device
import copy
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from .parameter_importance import ParameterImportanceAnalyzer
from .visualization import (
    plot_optimization_history, 
    plot_param_importance,
    plot_parallel_coordinates,
    plot_correlation_heatmap,
    create_optimization_report
)

logger = logging.getLogger(__name__)


# Data Loader
def get_data(dataset_name, input_shape, batch_size, train=True, backend='pytorch'):
    datasets = {'MNIST': MNIST, 'CIFAR10': CIFAR10}
    dataset = datasets.get(dataset_name, MNIST)(root='./data', train=train, transform=ToTensor(), download=True)
    if backend == 'pytorch':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    elif backend == 'tensorflow':
        data = dataset.data.numpy() / 255.0  # Normalize
        targets = dataset.targets.numpy()
        if len(data.shape) == 3:  # Add channel dimension
            data = data[..., None]  # [N, H, W] → [N, H, W, 1]
        return tf.data.Dataset.from_tensor_slices((data, targets)).batch(batch_size)

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

# Factory Function
def create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch'):
    resolved_model_dict = copy.deepcopy(model_dict)

    # Resolve HPO parameters in layers
    for layer in resolved_model_dict['layers']:
        if 'params' in layer and layer['params']:
            for param_name, param_value in layer['params'].items():
                if isinstance(param_value, dict) and 'hpo' in param_value:
                    hpo = param_value['hpo']
                    if hpo['type'] == 'categorical':
                        layer['params'][param_name] = trial.suggest_categorical(f"{layer['type']}_{param_name}", hpo['values'])
                    elif hpo['type'] == 'range':
                        layer['params'][param_name] = trial.suggest_float(
                            f"{layer['type']}_{param_name}",
                            hpo['start'],
                            hpo['end'],
                            step=hpo.get('step', None)
                        )
                    elif hpo['type'] == 'log_range':
                        # Handle all naming conventions (start/end, low/high, min/max)
                        low = hpo.get('start', hpo.get('low', hpo.get('min')))
                        high = hpo.get('end', hpo.get('high', hpo.get('max')))
                        layer['params'][param_name] = trial.suggest_float(
                            f"{layer['type']}_{param_name}",
                            low,
                            high,
                            log=True
                        )

    # Resolve HPO parameters in optimizer
    if 'optimizer' in resolved_model_dict and resolved_model_dict['optimizer']:
        for param_name, param_value in resolved_model_dict['optimizer']['params'].items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                hpo = param_value['hpo']
                if hpo['type'] == 'log_range':
                    # Handle all naming conventions (start/end, low/high, min/max)
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    resolved_model_dict['optimizer']['params'][param_name] = trial.suggest_float(
                        f"opt_{param_name}",
                        low,
                        high,
                        log=True
                    )

    if backend == 'pytorch':
        return DynamicPTModel(resolved_model_dict, trial, hpo_params)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def resolve_hpo_params(model_dict, trial, hpo_params):
    import copy
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    resolved_dict = copy.deepcopy(model_dict)

    for i, layer in enumerate(resolved_dict['layers']):
        if 'params' in layer and layer['params'] is not None and 'units' in layer['params'] and isinstance(layer['params']['units'], dict) and 'hpo' in layer['params']['units']:
            hpo = layer['params']['units']['hpo']
            key = f"{layer['type']}_units_{i}"
            if hpo['type'] == 'categorical':
                layer['params']['units'] = trial.suggest_categorical(key, hpo['values'])
            elif hpo['type'] == 'log_range':
                # Handle all naming conventions (start/end, low/high, min/max)
                low = hpo.get('start', hpo.get('low', hpo.get('min')))
                high = hpo.get('end', hpo.get('high', hpo.get('max')))
                layer['params']['units'] = trial.suggest_float(key, low, high, log=True)

    if resolved_dict['optimizer'] and 'params' in resolved_dict['optimizer']:
        # Clean up optimizer type
        opt_type = resolved_dict['optimizer']['type']
        if '(' in opt_type:
            resolved_dict['optimizer']['type'] = opt_type[:opt_type.index('(')].capitalize()  # 'adam(...)' -> 'Adam'

        for param, val in resolved_dict['optimizer']['params'].items():
            if isinstance(val, dict) and 'hpo' in val:
                hpo = val['hpo']
                if hpo['type'] == 'log_range':
                    # Handle all naming conventions (start/end, low/high, min/max)
                    low = hpo.get('start', hpo.get('low', hpo.get('min')))
                    high = hpo.get('end', hpo.get('high', hpo.get('max')))
                    resolved_dict['optimizer']['params'][param] = trial.suggest_float(
                        f"opt_{param}", low, high, log=True
                    )

    return resolved_dict


# Dynamic Models
class DynamicPTModel(nn.Module):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.model_dict = model_dict
        self.layers = nn.ModuleList()
        self.shape_propagator = ShapePropagator(debug=False)
        input_shape_raw = model_dict['input']['shape']  # (28, 28, 1)
        input_shape = (None, input_shape_raw[-1], *input_shape_raw[:-1])  # (None, 1, 28, 28)
        current_shape = input_shape
        in_channels = input_shape[1]  # 1
        in_features = None

        for layer in model_dict['layers']:
            params = layer['params'] if layer['params'] is not None else {}
            params = params.copy()

            # Compute in_features from current (input) shape before propagation
            if layer['type'] in ['Dense', 'Output'] and in_features is None:
                in_features = prod(current_shape[1:])  # Use input shape
                self.layers.append(nn.Flatten())

            # Propagate shape after setting in_features
            current_shape = self.shape_propagator.propagate(current_shape, layer, framework='pytorch')

            if layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int('conv_filters', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                in_channels = filters
            elif layer['type'] == 'MaxPooling2D':
                pool_size = params.get('pool_size', trial.suggest_int('maxpool2d_pool_size', 2, 3))
                stride = params.get('stride', pool_size)
                self.layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=stride))
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = prod(current_shape[1:])
            elif layer['type'] == 'Dense':
                units = params['units'] if 'units' in params else trial.suggest_int('dense_units', 64, 256)
                if in_features <= 0:
                    raise ValueError(f"Invalid in_features for Dense: {in_features}")
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'Dropout':
                rate = params['rate'] if 'rate' in params else trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)
                self.layers.append(nn.Dropout(p=rate))
            elif layer['type'] == 'Output':
                units = params['units'] if 'units' in params else 10
                if in_features <= 0:
                    raise ValueError(f"Invalid in_features for Output: {in_features}")
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'LSTM':
                input_size = current_shape[-1]
                units = params.get('units', trial.suggest_int('lstm_units', 32, 256))
                num_layers = params.get('num_layers', 1)
                if isinstance(params.get('num_layers'), dict) and 'hpo' in params.get('num_layers'):
                    num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
                self.layers.append(nn.LSTM(input_size, units, num_layers=num_layers, batch_first=True))
                in_features = units
            elif layer['type'] == 'BatchNormalization':
                momentum = params.get('momentum', trial.suggest_float('bn_momentum', 0.8, 0.99))
                self.layers.append(nn.BatchNorm2d(in_channels))
            elif layer['type'] == 'Transformer':
                d_model = params.get('d_model', trial.suggest_int('transformer_d_model', 64, 512))
                nhead = params.get('nhead', trial.suggest_int('transformer_nhead', 4, 8))
                num_encoder_layers = params.get('num_encoder_layers', trial.suggest_int('transformer_encoder_layers', 1, 4))
                num_decoder_layers = params.get('num_decoder_layers', trial.suggest_int('transformer_decoder_layers', 1, 4))
                dim_feedforward = params.get('dim_feedforward', trial.suggest_int('transformer_ff_dim', 128, 1024))
                self.layers.append(nn.Transformer(d_model=d_model,
                                                  nhead=nhead,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers,
                                                  dim_feedforward=dim_feedforward))
                in_features = d_model
            else:
                raise ValueError(f"Unsupported layer type: {layer['type']}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DynamicTFModel(tf.keras.Model):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers_list = []
        input_shape = model_dict['input']['shape']
        in_features = prod(input_shape)
        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Dense':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers_list.append(tf.keras.layers.Dense(params['units'], activation='relu' if params.get('activation') == 'relu' else None))
                in_features = params['units']
            elif layer['type'] == 'Dropout':
                if 'hpo' in params['rate']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dropout' and h['param_name'] == 'rate')
                    rate = trial.suggest_float('dropout_rate', hpo['hpo']['start'], hpo['hpo']['end'], step=hpo['hpo']['step'])
                    params['rate'] = rate
                self.layers_list.append(tf.keras.layers.Dropout(params['rate']))
            elif layer['type'] == 'Output':
                if isinstance(params.get('units'), dict) and 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Output' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('output_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers_list.append(tf.keras.layers.Dense(params['units'], activation='softmax' if params.get('activation') == 'softmax' else None))

    def call(self, inputs):
        x = tf.reshape(inputs, [inputs.shape[0], -1])  # Flatten input
        for layer in self.layers_list:
            x = layer(x)
        return x


# Training Method
def train_model(model, optimizer, train_loader, val_loader, backend='pytorch', epochs=1, execution_config=None):
    if backend == 'pytorch':
        device = get_device(execution_config.get("device", "auto") if execution_config else "auto")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        preds, targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())

        # Compute precision and recall
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        precision = precision_score(targets_np, preds_np, average='macro')
        recall = recall_score(targets_np, preds_np, average='macro')

        return val_loss / len(val_loader), correct / total, precision, recall


# HPO Objective
def objective(trial, config, dataset_name='MNIST', backend='pytorch', device='auto'):
    import torch.optim as optim
    from neural.execution_optimization.execution import get_device

    # Parse the network configuration
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

    # Suggest batch size
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Get data loaders
    train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True)
    val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False)

    # Create the model
    model = create_dynamic_model(model_dict, trial, hpo_params, backend)
    optimizer_config = model.model_dict['optimizer']

    # Extract learning rate from optimizer config
    learning_rate_param = optimizer_config['params'].get('learning_rate', 0.001)
    if isinstance(learning_rate_param, dict) and 'hpo' in learning_rate_param:
        hpo = learning_rate_param['hpo']
        if hpo['type'] == 'log_range':
            # Handle all naming conventions (start/end, low/high, min/max)
            low = hpo.get('start', hpo.get('low', hpo.get('min')))
            high = hpo.get('end', hpo.get('high', hpo.get('max')))
            lr = trial.suggest_float("learning_rate", low, high, log=True)
        else:
            # If it's a dict but not a log_range HPO, use a default value
            lr = 0.001
    else:
        # If it's not a dict, try to convert to float, or use default
        try:
            lr = float(learning_rate_param)
        except (ValueError, TypeError):
            lr = 0.001

    # Create optimizer
    if backend == 'pytorch':
        optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)

    # Get device and create execution config
    device_to_use = get_device(device)
    execution_config = {'device': device_to_use}

    # Train the model and get metrics
    loss, acc, precision, recall = train_model(model, optimizer, train_loader, val_loader, backend=backend, execution_config=execution_config)
    return loss, acc, precision, recall


# Multi-objective Optimization
class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimization using Optuna."""
    
    def __init__(self, objectives: List[str], directions: List[str]):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective names (e.g., ['loss', 'accuracy'])
            directions: List of optimization directions ('minimize' or 'maximize')
        """
        self.objectives = objectives
        self.directions = directions
        self.study = None
        self.trials_history = []
        
    def optimize(self, objective_fn: Callable, n_trials: int = 10, 
                sampler: Optional[str] = 'tpe', **kwargs) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            objective_fn: Function that returns tuple of objective values
            n_trials: Number of trials to run
            sampler: Sampler type ('tpe', 'random', 'nsgaii')
            **kwargs: Additional arguments for the objective function
            
        Returns:
            Dictionary with best trials and Pareto front
        """
        # Create sampler
        if sampler == 'nsgaii':
            sampler_obj = optuna.samplers.NSGAIISampler()
        elif sampler == 'random':
            sampler_obj = optuna.samplers.RandomSampler()
        else:
            sampler_obj = optuna.samplers.TPESampler()
        
        # Create study
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=sampler_obj
        )
        
        # Wrapper to track trials
        def wrapped_objective(trial):
            result = objective_fn(trial, **kwargs)
            # Store trial info
            trial_info = {
                'trial_number': trial.number,
                'parameters': trial.params,
                'values': result
            }
            for i, obj_name in enumerate(self.objectives):
                trial_info[obj_name] = result[i] if isinstance(result, tuple) else result
            self.trials_history.append(trial_info)
            return result
        
        # Run optimization
        self.study.optimize(wrapped_objective, n_trials=n_trials)
        
        # Extract Pareto front
        pareto_trials = self.study.best_trials
        
        return {
            'pareto_front': pareto_trials,
            'n_trials': n_trials,
            'all_trials': self.trials_history,
            'study': self.study
        }
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the Pareto front of non-dominated solutions."""
        if self.study is None:
            return []
        
        pareto_front = []
        for trial in self.study.best_trials:
            trial_info = {
                'parameters': trial.params,
                'values': trial.values
            }
            for i, obj_name in enumerate(self.objectives):
                trial_info[obj_name] = trial.values[i]
            pareto_front.append(trial_info)
        
        return pareto_front
    
    def plot_pareto_front(self, obj_x: int = 0, obj_y: int = 1, 
                         figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the Pareto front for two objectives.
        
        Args:
            obj_x: Index of objective for x-axis
            obj_y: Index of objective for y-axis
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if self.study is None:
            logger.warning("No study available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all trials
        all_x = [t['values'][obj_x] if isinstance(t['values'], tuple) else t['values'] 
                for t in self.trials_history]
        all_y = [t['values'][obj_y] if isinstance(t['values'], tuple) else t['values'] 
                for t in self.trials_history]
        ax.scatter(all_x, all_y, alpha=0.5, label='All trials')
        
        # Plot Pareto front
        pareto_trials = self.study.best_trials
        pareto_x = [t.values[obj_x] for t in pareto_trials]
        pareto_y = [t.values[obj_y] for t in pareto_trials]
        ax.scatter(pareto_x, pareto_y, color='red', s=100, 
                  label='Pareto front', zorder=10)
        
        ax.set_xlabel(self.objectives[obj_x])
        ax.set_ylabel(self.objectives[obj_y])
        ax.set_title('Multi-objective Optimization: Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Distributed HPO with Ray Tune
class DistributedHPO:
    """Distributed hyperparameter optimization using Ray Tune."""
    
    def __init__(self, use_ray: bool = True):
        """
        Initialize distributed HPO.
        
        Args:
            use_ray: Whether to use Ray Tune (if False, falls back to Optuna)
        """
        self.use_ray = use_ray
        self.ray_available = False
        
        if use_ray:
            try:
                import ray
                from ray import tune
                from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
                from ray.tune.search.optuna import OptunaSearch
                from ray.tune.search.bayesopt import BayesOptSearch
                self.ray = ray
                self.tune = tune
                self.ray_available = True
                logger.info("Ray Tune is available for distributed HPO")
            except ImportError:
                logger.warning("Ray Tune not available. Install with: pip install ray[tune]")
                self.use_ray = False
    
    def optimize_with_ray(self, trainable_fn: Callable, config_space: Dict[str, Any],
                         n_trials: int = 10, n_cpus: int = 1, n_gpus: int = 0,
                         scheduler: str = 'asha', search_alg: str = 'optuna',
                         metric: str = 'loss', mode: str = 'min') -> Dict[str, Any]:
        """
        Run distributed optimization with Ray Tune.
        
        Args:
            trainable_fn: Training function to optimize
            config_space: Configuration space dictionary
            n_trials: Number of trials
            n_cpus: CPUs per trial
            n_gpus: GPUs per trial
            scheduler: Scheduler type ('asha', 'pbt', 'median')
            search_alg: Search algorithm ('optuna', 'bayesopt', 'random')
            metric: Metric to optimize
            mode: Optimization mode ('min' or 'max')
            
        Returns:
            Best configuration and results
        """
        if not self.ray_available:
            logger.error("Ray Tune is not available")
            return {}
        
        # Initialize Ray
        if not self.ray.is_initialized():
            self.ray.init(ignore_reinit_error=True)
        
        # Create scheduler
        if scheduler == 'asha':
            from ray.tune.schedulers import ASHAScheduler
            scheduler_obj = ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=100,
                grace_period=1,
                reduction_factor=2
            )
        elif scheduler == 'pbt':
            from ray.tune.schedulers import PopulationBasedTraining
            scheduler_obj = PopulationBasedTraining(
                metric=metric,
                mode=mode,
                perturbation_interval=4,
                hyperparam_mutations=config_space
            )
        elif scheduler == 'median':
            from ray.tune.schedulers import MedianStoppingRule
            scheduler_obj = MedianStoppingRule(
                metric=metric,
                mode=mode
            )
        else:
            scheduler_obj = None
        
        # Create search algorithm
        if search_alg == 'optuna':
            from ray.tune.search.optuna import OptunaSearch
            search_obj = OptunaSearch(metric=metric, mode=mode)
        elif search_alg == 'bayesopt':
            try:
                from ray.tune.search.bayesopt import BayesOptSearch
                search_obj = BayesOptSearch(metric=metric, mode=mode)
            except ImportError:
                logger.warning("BayesOpt not available, using Optuna")
                from ray.tune.search.optuna import OptunaSearch
                search_obj = OptunaSearch(metric=metric, mode=mode)
        else:
            search_obj = None
        
        # Run Ray Tune
        analysis = self.tune.run(
            trainable_fn,
            config=config_space,
            num_samples=n_trials,
            scheduler=scheduler_obj,
            search_alg=search_obj,
            resources_per_trial={"cpu": n_cpus, "gpu": n_gpus},
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(metric, mode)
        best_config = best_trial.config
        best_result = best_trial.last_result
        
        return {
            'best_config': best_config,
            'best_result': best_result,
            'all_trials': analysis.trials,
            'analysis': analysis
        }
    
    def optimize_multi_objective_ray(self, trainable_fn: Callable, 
                                    config_space: Dict[str, Any],
                                    objectives: List[str],
                                    n_trials: int = 10,
                                    n_cpus: int = 1,
                                    n_gpus: int = 0) -> Dict[str, Any]:
        """
        Run multi-objective optimization with Ray Tune.
        
        Args:
            trainable_fn: Training function
            config_space: Configuration space
            objectives: List of objective names
            n_trials: Number of trials
            n_cpus: CPUs per trial
            n_gpus: GPUs per trial
            
        Returns:
            Optimization results
        """
        if not self.ray_available:
            logger.error("Ray Tune is not available")
            return {}
        
        # Initialize Ray
        if not self.ray.is_initialized():
            self.ray.init(ignore_reinit_error=True)
        
        # For multi-objective, we use MOO search algorithms
        try:
            from ray.tune.search.optuna import OptunaSearch
            import optuna
            
            # Create multi-objective Optuna study
            study = optuna.create_study(
                directions=['minimize'] * len(objectives)
            )
            
            search_obj = OptunaSearch(
                study,
                metric=objectives[0],
                mode='min'
            )
        except ImportError:
            logger.warning("Advanced multi-objective support requires Optuna")
            search_obj = None
        
        # Run Ray Tune
        analysis = self.tune.run(
            trainable_fn,
            config=config_space,
            num_samples=n_trials,
            search_alg=search_obj,
            resources_per_trial={"cpu": n_cpus, "gpu": n_gpus},
            verbose=1
        )
        
        return {
            'all_trials': analysis.trials,
            'analysis': analysis
        }


# Enhanced Parameter Importance with Bayesian Methods
class BayesianParameterImportance:
    """Enhanced parameter importance analysis using Bayesian methods."""
    
    def __init__(self):
        """Initialize Bayesian parameter importance analyzer."""
        self.gp_models = {}
        
    def analyze_with_gp(self, trials: List[Dict[str, Any]], 
                       target_metric: str = 'score') -> Dict[str, float]:
        """
        Analyze parameter importance using Gaussian Process.
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            
        Returns:
            Dictionary of parameter importances
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, RBF
            from sklearn.preprocessing import StandardScaler
            
            # Extract parameters and scores
            param_names = set()
            for trial in trials:
                param_names.update(trial.get('parameters', {}).keys())
            param_names = sorted(list(param_names))
            
            X = []
            y = []
            for trial in trials:
                params = trial.get('parameters', {})
                score = trial.get(target_metric, trial.get('score', None))
                if score is None:
                    continue
                
                row = [params.get(p, 0) for p in param_names]
                X.append(row)
                y.append(score)
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 3:
                logger.warning("Too few trials for GP analysis")
                return {}
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train GP for each parameter
            importances = {}
            for i, param in enumerate(param_names):
                # Train GP with all features
                gp_full = GaussianProcessRegressor(
                    kernel=Matern(nu=2.5),
                    alpha=1e-6,
                    n_restarts_optimizer=5
                )
                gp_full.fit(X_scaled, y)
                score_full = gp_full.score(X_scaled, y)
                
                # Train GP without this feature
                X_reduced = np.delete(X_scaled, i, axis=1)
                gp_reduced = GaussianProcessRegressor(
                    kernel=Matern(nu=2.5),
                    alpha=1e-6,
                    n_restarts_optimizer=5
                )
                gp_reduced.fit(X_reduced, y)
                score_reduced = gp_reduced.score(X_reduced, y)
                
                # Importance is the drop in score
                importance = max(0, score_full - score_reduced)
                importances[param] = importance
            
            # Normalize importances
            total = sum(importances.values())
            if total > 0:
                importances = {k: v/total for k, v in importances.items()}
            
            return importances
            
        except Exception as e:
            logger.error(f"Error in GP parameter importance: {str(e)}")
            return {}
    
    def plot_importance_with_uncertainty(self, trials: List[Dict[str, Any]],
                                        target_metric: str = 'score',
                                        figsize: Tuple[int, int] = (10, 6)):
        """
        Plot parameter importance with uncertainty estimates.
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        importances = self.analyze_with_gp(trials, target_metric)
        
        if not importances:
            logger.warning("No importances to plot")
            return None
        
        # Sort by importance
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        params = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(params))
        ax.barh(y_pos, scores, align='center', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (GP-based)')
        ax.set_title('Hyperparameter Importance with Gaussian Process')
        
        plt.tight_layout()
        return fig


# Optimize and Return with Enhanced Features
def optimize_and_return(config, n_trials=10, dataset_name='MNIST', backend='pytorch', 
                       device='auto', sampler='tpe', objectives=None, use_ray=False,
                       enable_pruning=True, study_name=None):
    """
    Run hyperparameter optimization with enhanced features.
    
    Args:
        config: Model configuration
        n_trials: Number of trials
        dataset_name: Dataset name
        backend: Backend ('pytorch' or 'tensorflow')
        device: Device to use
        sampler: Sampler type ('tpe', 'random', 'cmaes', 'nsgaii')
        objectives: List of objectives for multi-objective optimization
        use_ray: Whether to use Ray Tune for distributed optimization
        enable_pruning: Whether to enable early stopping
        study_name: Name for the study (for persistence)
        
    Returns:
        Best parameters or optimization results
    """
    # Set device mode
    import os
    if device.lower() == 'cpu':
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['NEURAL_FORCE_CPU'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_ENABLE_TENSOR_FLOAT_32_EXECUTION'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # Handle multi-objective optimization
    if objectives and len(objectives) > 1:
        return _optimize_multi_objective(
            config, n_trials, dataset_name, backend, device, 
            objectives, sampler, use_ray
        )
    
    # Handle distributed optimization with Ray
    if use_ray:
        return _optimize_with_ray(
            config, n_trials, dataset_name, backend, device, sampler
        )
    
    # Create sampler
    if sampler == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler()
    elif sampler == 'random':
        sampler_obj = optuna.samplers.RandomSampler()
    elif sampler == 'grid':
        # Grid sampler requires search space definition
        sampler_obj = optuna.samplers.GridSampler({})
    else:  # Default to TPE (Tree-structured Parzen Estimator - Bayesian)
        sampler_obj = optuna.samplers.TPESampler()
    
    # Create pruner if enabled
    pruner = optuna.pruners.MedianPruner() if enable_pruning else optuna.pruners.NopPruner()
    
    # Create study
    study = optuna.create_study(
        directions=["minimize", "minimize", "maximize", "maximize"],
        sampler=sampler_obj,
        pruner=pruner,
        study_name=study_name
    )

    def objective_wrapper(trial):
        nonlocal device
        model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)

        training_config = model_dict.get('training_config', {})
        batch_size = training_config.get('batch_size', 32)

        if isinstance(batch_size, dict) and 'hpo' in batch_size:
            hpo = batch_size['hpo']
            if hpo['type'] == 'categorical':
                batch_size = trial.suggest_categorical("batch_size", hpo['values'])
            elif hpo['type'] == 'range':
                batch_size = trial.suggest_int("batch_size", hpo['start'], hpo['end'], step=hpo.get('step', 1))
            elif hpo['type'] == 'log_range':
                low = hpo.get('start', hpo.get('low', hpo.get('min')))
                high = hpo.get('end', hpo.get('high', hpo.get('max')))
                batch_size = trial.suggest_int("batch_size", low, high, log=True)
        elif isinstance(batch_size, list):
            batch_size = trial.suggest_categorical("batch_size", batch_size)

        batch_size = int(batch_size)

        train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True, backend)
        val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False, backend)

        model = create_dynamic_model(model_dict, trial, hpo_params, backend)

        optimizer_config = model.model_dict['optimizer']
        if optimizer_config is None:
            optimizer_config = {'type': 'Adam', 'params': {'learning_rate': 0.001}}
        elif 'params' not in optimizer_config or not optimizer_config['params']:
            optimizer_config['params'] = {'learning_rate': 0.001}

        lr = optimizer_config['params']['learning_rate']

        if backend == 'pytorch':
            optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)

        execution_config = {'device': device}
        loss, acc, precision, recall = train_model(model, optimizer, train_loader, val_loader, backend=backend, execution_config=execution_config)
        
        # Report intermediate values for pruning
        if enable_pruning:
            trial.report(loss, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return loss, acc, precision, recall

    study.optimize(objective_wrapper, n_trials=n_trials)

    # Normalize the best parameters
    best_params = study.best_trials[0].params
    normalized_params = {
        'batch_size': best_params.get('batch_size', 32),
    }
    if 'Dense_units' in best_params:
        normalized_params['dense_units'] = best_params['Dense_units']
    if 'Dropout_rate' in best_params:
        normalized_params['dropout_rate'] = best_params['Dropout_rate']
    if 'opt_learning_rate' in best_params:
        normalized_params['learning_rate'] = best_params['opt_learning_rate']
    else:
        normalized_params['learning_rate'] = 0.001

    # Add study for analysis
    normalized_params['_study'] = study
    normalized_params['_trials_history'] = [
        {
            'trial_number': t.number,
            'parameters': t.params,
            'values': t.values,
            'loss': t.values[0] if len(t.values) > 0 else None,
            'accuracy': t.values[1] if len(t.values) > 1 else None,
            'precision': t.values[2] if len(t.values) > 2 else None,
            'recall': t.values[3] if len(t.values) > 3 else None,
            'score': t.values[1] if len(t.values) > 1 else None  # Use accuracy as score
        }
        for t in study.trials
    ]

    return normalized_params


def _optimize_multi_objective(config, n_trials, dataset_name, backend, device, 
                              objectives, sampler, use_ray):
    """Helper function for multi-objective optimization."""
    # Map objective names to directions
    direction_map = {
        'loss': 'minimize',
        'accuracy': 'maximize',
        'precision': 'maximize',
        'recall': 'maximize',
        'f1': 'maximize'
    }
    directions = [direction_map.get(obj, 'minimize') for obj in objectives]
    
    # Create multi-objective optimizer
    moo = MultiObjectiveOptimizer(objectives, directions)
    
    def multi_obj_wrapper(trial, **kwargs):
        result = objective(trial, **kwargs)
        # Map result tuple to requested objectives
        result_map = {
            'loss': result[0],
            'accuracy': result[1],
            'precision': result[2],
            'recall': result[3]
        }
        return tuple(result_map[obj] for obj in objectives)
    
    results = moo.optimize(
        multi_obj_wrapper,
        n_trials=n_trials,
        sampler='nsgaii' if sampler == 'nsgaii' else 'tpe',
        config=config,
        dataset_name=dataset_name,
        backend=backend,
        device=device
    )
    
    return {
        'pareto_front': moo.get_pareto_front(),
        'all_trials': results['all_trials'],
        'study': results['study']
    }


def _optimize_with_ray(config, n_trials, dataset_name, backend, device, sampler):
    """Helper function for Ray Tune optimization."""
    dist_hpo = DistributedHPO(use_ray=True)
    
    if not dist_hpo.ray_available:
        logger.warning("Ray not available, falling back to standard optimization")
        return optimize_and_return(
            config, n_trials, dataset_name, backend, device, 
            sampler, use_ray=False
        )
    
    # Define trainable function for Ray
    def trainable(config_dict):
        # This would need to be adapted based on the actual config structure
        # For now, return a placeholder
        return {'loss': 0.5, 'accuracy': 0.9}
    
    # Define config space
    config_space = {
        'batch_size': dist_hpo.tune.choice([16, 32, 64]),
        'learning_rate': dist_hpo.tune.loguniform(1e-4, 1e-1),
    }
    
    results = dist_hpo.optimize_with_ray(
        trainable,
        config_space,
        n_trials=n_trials,
        scheduler='asha',
        search_alg='optuna' if sampler == 'tpe' else 'random'
    )
    
    return results
