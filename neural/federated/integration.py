from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def integrate_with_neural_training(
    model_data: Dict[str, Any],
    training_config: Dict[str, Any],
    backend: str = 'tensorflow',
) -> Dict[str, Any]:
    federated_config = {
        'enabled': training_config.get('federated', {}).get('enabled', False),
        'num_clients': training_config.get('federated', {}).get('num_clients', 10),
        'num_rounds': training_config.get('federated', {}).get('num_rounds', 100),
        'scenario': training_config.get('federated', {}).get('scenario', 'cross_device'),
        'aggregation': training_config.get('federated', {}).get('aggregation', 'fedavg'),
        'privacy': training_config.get('federated', {}).get('privacy', None),
        'compression': training_config.get('federated', {}).get('compression', None),
        'local_epochs': training_config.get('federated', {}).get('local_epochs', 1),
        'batch_size': training_config.get('batch_size', 32),
        'learning_rate': training_config.get('learning_rate', 0.01),
    }
    
    return federated_config


def create_model_from_neural_dsl(
    model_data: Dict[str, Any],
    backend: str = 'tensorflow',
) -> Any:
    if backend == 'tensorflow':
        return _create_tensorflow_model(model_data)
    elif backend == 'pytorch':
        return _create_pytorch_model(model_data)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _create_tensorflow_model(model_data: Dict[str, Any]) -> Any:
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        input_shape = tuple(model_data['input']['shape'])
        layers_config = model_data['layers']
        
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        
        for layer_config in layers_config:
            layer_type = layer_config.get('type', '').lower()
            
            if layer_type == 'conv2d':
                model.add(keras.layers.Conv2D(
                    filters=layer_config.get('filters', 32),
                    kernel_size=layer_config.get('kernel_size', 3),
                    activation=layer_config.get('activation', 'relu'),
                    padding=layer_config.get('padding', 'same'),
                ))
            elif layer_type == 'maxpool2d' or layer_type == 'maxpooling2d':
                model.add(keras.layers.MaxPooling2D(
                    pool_size=layer_config.get('pool_size', 2),
                ))
            elif layer_type == 'flatten':
                model.add(keras.layers.Flatten())
            elif layer_type == 'dense':
                model.add(keras.layers.Dense(
                    units=layer_config.get('units', 128),
                    activation=layer_config.get('activation', 'relu'),
                ))
            elif layer_type == 'dropout':
                model.add(keras.layers.Dropout(
                    rate=layer_config.get('rate', 0.5),
                ))
            elif layer_type == 'output':
                model.add(keras.layers.Dense(
                    units=layer_config.get('units', 10),
                    activation=layer_config.get('activation', 'softmax'),
                ))
        
        return model
    
    except ImportError:
        logger.error("TensorFlow not available")
        raise


def _create_pytorch_model(model_data: Dict[str, Any]) -> Any:
    try:
        import torch
        import torch.nn as nn
        
        input_shape = tuple(model_data['input']['shape'])
        layers_config = model_data['layers']
        
        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                
                for layer_config in layers_config:
                    layer_type = layer_config.get('type', '').lower()
                    
                    if layer_type == 'conv2d':
                        self.layers.append(nn.Conv2d(
                            in_channels=layer_config.get('in_channels', input_shape[0]),
                            out_channels=layer_config.get('filters', 32),
                            kernel_size=layer_config.get('kernel_size', 3),
                            padding=layer_config.get('padding', 1),
                        ))
                        if layer_config.get('activation') == 'relu':
                            self.layers.append(nn.ReLU())
                    elif layer_type == 'maxpool2d':
                        self.layers.append(nn.MaxPool2d(
                            kernel_size=layer_config.get('pool_size', 2),
                        ))
                    elif layer_type == 'flatten':
                        self.layers.append(nn.Flatten())
                    elif layer_type == 'dense':
                        self.layers.append(nn.Linear(
                            in_features=layer_config.get('in_features', 128),
                            out_features=layer_config.get('units', 128),
                        ))
                        if layer_config.get('activation') == 'relu':
                            self.layers.append(nn.ReLU())
                    elif layer_type == 'dropout':
                        self.layers.append(nn.Dropout(
                            p=layer_config.get('rate', 0.5),
                        ))
                    elif layer_type == 'output':
                        self.layers.append(nn.Linear(
                            in_features=layer_config.get('in_features', 128),
                            out_features=layer_config.get('units', 10),
                        ))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return DynamicModel()
    
    except ImportError:
        logger.error("PyTorch not available")
        raise


def prepare_federated_data(
    data: Tuple[np.ndarray, np.ndarray],
    num_clients: int,
    distribution: str = 'iid',
    alpha: float = 0.5,
) -> list:
    from neural.federated.utils import split_data_iid, split_data_non_iid
    
    if distribution == 'iid':
        return split_data_iid(data, num_clients)
    elif distribution == 'non_iid':
        return split_data_non_iid(data, num_clients, alpha=alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def run_federated_training(
    model: Any,
    data: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Any],
    backend: str = 'tensorflow',
    test_data: Optional[Tuple] = None,
) -> Dict[str, Any]:
    from neural.federated.orchestrator import create_federated_orchestrator
    
    orchestrator = create_federated_orchestrator(
        model=model,
        scenario_type=config.get('scenario', 'cross_device'),
        backend=backend,
        **config
    )
    
    def model_fn():
        if backend == 'tensorflow':
            try:
                import tensorflow as tf
                from tensorflow import keras
                return keras.models.clone_model(model)
            except ImportError:
                return None
        elif backend == 'pytorch':
            try:
                import torch
                import copy
                return copy.deepcopy(model)
            except ImportError:
                return None
    
    orchestrator.setup_clients(
        model_fn=model_fn,
        data=data,
    )
    
    history = orchestrator.train(
        num_rounds=config.get('num_rounds', 100),
        evaluate_every=config.get('evaluate_every', 1),
        test_data=test_data,
        save_model_path=config.get('save_model_path'),
    )
    
    summary = orchestrator.get_metrics_summary()
    
    return {
        'history': history,
        'summary': summary,
        'global_model': orchestrator.get_global_model(),
    }


class FederatedTrainingCallback:
    def __init__(self):
        self.metrics = []
    
    def on_round_begin(self, round_num: int):
        logger.info(f"Round {round_num} started")
    
    def on_round_end(self, round_num: int, metrics: Dict[str, Any]):
        self.metrics.append(metrics)
        logger.info(f"Round {round_num} completed with metrics: {metrics}")
    
    def on_client_train_begin(self, client_id: str):
        logger.debug(f"Client {client_id} training started")
    
    def on_client_train_end(self, client_id: str, metrics: Dict[str, Any]):
        logger.debug(f"Client {client_id} training completed with metrics: {metrics}")
    
    def on_aggregation_begin(self):
        logger.debug("Aggregation started")
    
    def on_aggregation_end(self, aggregated_weights: list):
        logger.debug("Aggregation completed")


def federated_train_from_config(
    config_path: str,
    data: Tuple[np.ndarray, np.ndarray],
    test_data: Optional[Tuple] = None,
) -> Dict[str, Any]:
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    federated_config = config.get('federated', {})
    
    backend = config.get('backend', 'tensorflow')
    
    if 'model_data' in model_config:
        model = create_model_from_neural_dsl(model_config['model_data'], backend)
    else:
        raise ValueError("Model configuration not found in config")
    
    merged_config = {**training_config, **federated_config}
    
    return run_federated_training(
        model=model,
        data=data,
        config=merged_config,
        backend=backend,
        test_data=test_data,
    )
