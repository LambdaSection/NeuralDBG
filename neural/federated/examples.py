from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_federated_training():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: Basic Federated Training (TensorFlow)")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(5000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 5000)
        X_test = np.random.randn(1000, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, 1000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossDeviceScenario(num_devices=50, devices_per_round=5),
            aggregation_strategy='fedavg',
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
            learning_rate=0.01,
        )
        
        history = orchestrator.train(
            num_rounds=10,
            evaluate_every=2,
            test_data=(X_test, y_test),
        )
        
        summary = orchestrator.get_metrics_summary()
        logger.info(f"\nTraining Summary:")
        logger.info(f"  Final train accuracy: {summary['final_train_accuracy']:.4f}")
        logger.info(f"  Final eval accuracy: {summary['final_eval_accuracy']:.4f}")
        logger.info(f"  Best eval accuracy: {summary['best_eval_accuracy']:.4f}")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_differential_privacy():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: Federated Training with Differential Privacy")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(3000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 3000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossDeviceScenario(num_devices=30, devices_per_round=5),
            aggregation_strategy='fedavg',
            privacy_mechanism='gaussian',
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        summary = orchestrator.get_metrics_summary()
        logger.info(f"\nPrivate Training Summary:")
        logger.info(f"  Privacy budget spent: {summary.get('privacy_budget_spent', 'N/A')}")
        logger.info(f"  Final accuracy: {summary['final_train_accuracy']:.4f}")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_communication_compression():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossSiloScenario
        
        logger.info("=" * 60)
        logger.info("Example: Communication Compression")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(2000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 2000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossSiloScenario(num_silos=10, silos_per_round=5),
            aggregation_strategy='fedavg',
            compression_strategy='quantization',
            quantization_bits=4,
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=2,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        summary = orchestrator.get_metrics_summary()
        logger.info(f"\nCompression Summary:")
        logger.info(f"  Avg communication cost: {summary.get('avg_communication_cost', 'N/A')}")
        logger.info(f"  Final accuracy: {summary['final_train_accuracy']:.4f}")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_fedprox():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: FedProx for Heterogeneous Clients")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(4000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 4000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossDeviceScenario(
                num_devices=40,
                devices_per_round=8,
                data_heterogeneity=0.8,
            ),
            aggregation_strategy='fedprox',
            proximal_mu=0.01,
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=2,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        logger.info(f"\nFedProx training completed")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_pytorch_federated():
    try:
        import torch
        import torch.nn as nn
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: PyTorch Federated Learning")
        logger.info("=" * 60)
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, 10)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleModel()
        
        X_train = np.random.randn(3000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 3000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='pytorch',
            scenario=CrossDeviceScenario(num_devices=30, devices_per_round=6),
            aggregation_strategy='fedavg',
        )
        
        import copy
        def model_fn():
            return copy.deepcopy(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        logger.info(f"\nPyTorch training completed")
        
    except ImportError as e:
        logger.warning(f"PyTorch not available: {e}")


def example_hybrid_scenario():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, HybridScenario
        
        logger.info("=" * 60)
        logger.info("Example: Hybrid Cross-Silo/Cross-Device Scenario")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(5000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 5000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=HybridScenario(
                num_silos=5,
                devices_per_silo=10,
                silos_per_round=3,
                devices_per_silo_per_round=3,
            ),
            aggregation_strategy='fedavg',
            use_secure_aggregation=True,
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        logger.info(f"\nHybrid training completed")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_adaptive_aggregation():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: Adaptive Aggregation and Compression")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(3000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 3000)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossDeviceScenario(num_devices=30, devices_per_round=6),
            aggregation_strategy='adaptive',
            adaptive_alpha=0.7,
            compression_strategy='adaptive',
            target_compression=0.3,
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        history = orchestrator.train(num_rounds=10)
        
        logger.info(f"\nAdaptive training completed")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def example_non_iid_data():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        from neural.federated.utils import split_data_non_iid, compute_data_statistics
        
        logger.info("=" * 60)
        logger.info("Example: Non-IID Data Distribution")
        logger.info("=" * 60)
        
        X_train = np.random.randn(3000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 3000)
        
        client_data = split_data_non_iid(
            data=(X_train, y_train),
            num_clients=20,
            alpha=0.1,
            num_classes=10,
        )
        
        stats = compute_data_statistics(client_data)
        logger.info(f"\nData distribution statistics:")
        logger.info(f"  Number of clients: {stats['num_clients']}")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Avg samples per client: {stats['avg_samples_per_client']:.1f}")
        logger.info(f"  Std samples per client: {stats['std_samples_per_client']:.1f}")
        logger.info(f"  Min samples: {stats['min_samples_per_client']}")
        logger.info(f"  Max samples: {stats['max_samples_per_client']}")
        
    except ImportError as e:
        logger.warning(f"Dependencies not available: {e}")


def example_metrics_tracking():
    try:
        import tensorflow as tf
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        
        logger.info("=" * 60)
        logger.info("Example: Metrics Tracking and Visualization")
        logger.info("=" * 60)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        X_train = np.random.randn(2000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 2000)
        X_test = np.random.randn(500, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, 500)
        
        orchestrator = FederatedOrchestrator(
            model=model,
            backend='tensorflow',
            scenario=CrossDeviceScenario(num_devices=20, devices_per_round=4),
            aggregation_strategy='fedavg',
        )
        
        def model_fn():
            return tf.keras.models.clone_model(model)
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        history = orchestrator.train(
            num_rounds=15,
            evaluate_every=3,
            test_data=(X_test, y_test),
        )
        
        orchestrator.save_metrics('federated_metrics.json')
        
        logger.info(f"\nMetrics saved successfully")
        logger.info(f"  Rounds: {history['rounds']}")
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"  Final train accuracy: {history['train_accuracy'][-1]:.4f}")
        
    except ImportError as e:
        logger.warning(f"TensorFlow not available: {e}")


def run_all_examples():
    logger.info("\n" + "=" * 60)
    logger.info("Running All Federated Learning Examples")
    logger.info("=" * 60 + "\n")
    
    examples = [
        example_basic_federated_training,
        example_differential_privacy,
        example_communication_compression,
        example_fedprox,
        example_pytorch_federated,
        example_hybrid_scenario,
        example_adaptive_aggregation,
        example_non_iid_data,
        example_metrics_tracking,
    ]
    
    for example_fn in examples:
        try:
            example_fn()
            logger.info("\n")
        except Exception as e:
            logger.error(f"Example {example_fn.__name__} failed: {e}")
            logger.info("\n")


if __name__ == '__main__':
    run_all_examples()
