from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FederatedTrainingAdapter:
    def __init__(self, base_training_config: Dict[str, Any]):
        self.base_config = base_training_config
        self.federated_config = base_training_config.get('federated', {})
        self.is_federated = self.federated_config.get('enabled', False)
    
    def should_use_federated(self) -> bool:
        return self.is_federated
    
    def get_federated_params(self) -> Dict[str, Any]:
        return {
            'scenario': self.federated_config.get('scenario', 'cross_device'),
            'num_rounds': self.federated_config.get('num_rounds', 100),
            'aggregation_strategy': self.federated_config.get('aggregation', 'fedavg'),
            'privacy_mechanism': self.federated_config.get('privacy', {}).get('mechanism') if self.federated_config.get('privacy', {}).get('enabled') else None,
            'compression_strategy': self.federated_config.get('compression', {}).get('type') if self.federated_config.get('compression', {}).get('enabled') else None,
            'local_epochs': self.federated_config.get('local_epochs', 1),
            'batch_size': self.base_config.get('batch_size', 32),
            'learning_rate': self.base_config.get('learning_rate', 0.01),
        }


def train_model_federated(
    model: Any,
    data: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Any],
    backend: str = 'tensorflow',
    test_data: Optional[Tuple] = None,
) -> Dict[str, Any]:
    adapter = FederatedTrainingAdapter(config)
    
    if not adapter.should_use_federated():
        logger.info("Federated learning not enabled, using standard training")
        return train_model_standard(model, data, config, backend, test_data)
    
    logger.info("Using federated learning")
    
    from neural.federated.integration import run_federated_training
    
    federated_params = adapter.get_federated_params()
    
    return run_federated_training(
        model=model,
        data=data,
        config=federated_params,
        backend=backend,
        test_data=test_data,
    )


def train_model_standard(
    model: Any,
    data: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Any],
    backend: str = 'tensorflow',
    test_data: Optional[Tuple] = None,
) -> Dict[str, Any]:
    X_train, y_train = data
    
    if backend == 'tensorflow':
        return _train_tensorflow(model, X_train, y_train, config, test_data)
    elif backend == 'pytorch':
        return _train_pytorch(model, X_train, y_train, config, test_data)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _train_tensorflow(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    test_data: Optional[Tuple],
) -> Dict[str, Any]:
    try:
        import tensorflow as tf
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.01)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        
        validation_data = test_data if test_data else None
        
        history = model.fit(
            X_train,
            y_train,
            batch_size=config.get('batch_size', 32),
            epochs=config.get('epochs', 10),
            validation_data=validation_data,
            verbose=1,
        )
        
        return {
            'history': history.history,
            'model': model,
        }
    
    except ImportError:
        logger.error("TensorFlow not available")
        raise


def _train_pytorch(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    test_data: Optional[Tuple],
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.01))
        criterion = nn.CrossEntropyLoss()
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(config.get('epochs', 10)):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
            
            history['loss'].append(epoch_loss / len(dataloader))
            history['accuracy'].append(correct / total)
            
            logger.info(f"Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}")
        
        return {
            'history': history,
            'model': model,
        }
    
    except ImportError:
        logger.error("PyTorch not available")
        raise


def create_federated_training_config(
    scenario: str = 'cross_device',
    num_clients: int = 100,
    num_rounds: int = 100,
    enable_privacy: bool = False,
    enable_compression: bool = False,
    **kwargs
) -> Dict[str, Any]:
    config = {
        'federated': {
            'enabled': True,
            'scenario': scenario,
            'num_rounds': num_rounds,
            'local_epochs': kwargs.get('local_epochs', 1),
            'aggregation': kwargs.get('aggregation', 'fedavg'),
        },
        'batch_size': kwargs.get('batch_size', 32),
        'learning_rate': kwargs.get('learning_rate', 0.01),
    }
    
    if scenario == 'cross_device':
        config['federated'].update({
            'num_devices': num_clients,
            'devices_per_round': kwargs.get('devices_per_round', num_clients // 10),
            'data_heterogeneity': kwargs.get('data_heterogeneity', 0.5),
        })
    elif scenario == 'cross_silo':
        config['federated'].update({
            'num_silos': num_clients,
            'silos_per_round': kwargs.get('silos_per_round', num_clients // 2),
        })
    
    if enable_privacy:
        config['federated']['privacy'] = {
            'enabled': True,
            'mechanism': kwargs.get('privacy_mechanism', 'gaussian'),
            'epsilon': kwargs.get('epsilon', 1.0),
            'delta': kwargs.get('delta', 1e-5),
            'clip_norm': kwargs.get('clip_norm', 1.0),
        }
    
    if enable_compression:
        config['federated']['compression'] = {
            'enabled': True,
            'type': kwargs.get('compression_type', 'quantization'),
            'num_bits': kwargs.get('quantization_bits', 8),
        }
    
    return config


def migrate_standard_to_federated(
    model_path: str,
    data_path: str,
    output_config_path: str,
    scenario: str = 'cross_device',
    num_clients: int = 100,
):
    import yaml
    
    config = create_federated_training_config(
        scenario=scenario,
        num_clients=num_clients,
    )
    
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Federated config created: {output_config_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data path: {data_path}")
    logger.info("\nTo train with federated learning:")
    logger.info(f"  python -m neural.federated.cli train --config {output_config_path} --data {data_path}")


def benchmark_federated_vs_standard(
    model: Any,
    data: Tuple[np.ndarray, np.ndarray],
    backend: str = 'tensorflow',
    num_rounds: int = 10,
) -> Dict[str, Any]:
    import time
    
    results = {}
    
    logger.info("Running standard training benchmark...")
    standard_config = {'epochs': num_rounds, 'batch_size': 32, 'learning_rate': 0.01}
    start_time = time.time()
    standard_results = train_model_standard(model, data, standard_config, backend)
    results['standard'] = {
        'time': time.time() - start_time,
        'final_loss': standard_results['history']['loss'][-1],
        'final_accuracy': standard_results['history']['accuracy'][-1],
    }
    
    logger.info("Running federated training benchmark...")
    federated_config = create_federated_training_config(
        scenario='cross_device',
        num_clients=10,
        num_rounds=num_rounds,
    )
    start_time = time.time()
    federated_results = train_model_federated(model, data, federated_config, backend)
    results['federated'] = {
        'time': time.time() - start_time,
        'final_loss': federated_results['summary']['final_train_loss'],
        'final_accuracy': federated_results['summary']['final_train_accuracy'],
    }
    
    logger.info("\nBenchmark Results:")
    logger.info(f"Standard  - Time: {results['standard']['time']:.2f}s, Accuracy: {results['standard']['final_accuracy']:.4f}")
    logger.info(f"Federated - Time: {results['federated']['time']:.2f}s, Accuracy: {results['federated']['final_accuracy']:.4f}")
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Federated Training Integration Module")
    logger.info("Use this module to integrate federated learning with Neural DSL training")
