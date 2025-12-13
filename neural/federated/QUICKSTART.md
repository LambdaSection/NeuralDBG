# Federated Learning Quick Start

This guide will help you get started with federated learning in Neural DSL.

## Installation

```bash
# Install with federated learning support
pip install -e ".[federated,backends]"

# Or install full package
pip install -e ".[full]"
```

## Basic Example (5 minutes)

### 1. Prepare Your Data

```python
import numpy as np

# Create sample data
X_train = np.random.randn(10000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 10000)
X_test = np.random.randn(2000, 784).astype(np.float32)
y_test = np.random.randint(0, 10, 2000)

# Save for later use
np.savez('train_data.npz', X_train=X_train, y_train=y_train)
np.savez('test_data.npz', X_test=X_test, y_test=y_test)
```

### 2. Create a Model (TensorFlow)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 3. Setup Federated Learning

```python
from neural.federated import FederatedOrchestrator, CrossDeviceScenario

# Create orchestrator
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(
        num_devices=100,          # Total number of devices
        devices_per_round=10,     # Devices selected per round
        data_heterogeneity=0.5,   # 0=homogeneous, 1=heterogeneous
    ),
    aggregation_strategy='fedavg',
)
```

### 4. Setup Clients

```python
def model_fn():
    return tf.keras.models.clone_model(model)

orchestrator.setup_clients(
    model_fn=model_fn,
    data=(X_train, y_train),
    local_epochs=1,
    learning_rate=0.01,
)
```

### 5. Train

```python
history = orchestrator.train(
    num_rounds=100,
    evaluate_every=5,
    test_data=(X_test, y_test),
)

# Get results
summary = orchestrator.get_metrics_summary()
print(f"Final accuracy: {summary['final_eval_accuracy']:.4f}")
```

### 6. Save Model

```python
orchestrator.server.save_model('federated_model.h5')
```

## Advanced Features

### With Differential Privacy

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(num_devices=100),
    privacy_mechanism='gaussian',
    epsilon=1.0,        # Privacy budget
    delta=1e-5,         # Privacy parameter
    clip_norm=1.0,      # Gradient clipping
)
```

### With Communication Compression

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(num_devices=100),
    compression_strategy='quantization',
    quantization_bits=8,  # 8-bit quantization
)
```

### Cross-Silo Scenario

```python
from neural.federated import CrossSiloScenario

orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossSiloScenario(
        num_silos=10,
        silos_per_round=5,
    ),
    aggregation_strategy='fedprox',
    proximal_mu=0.01,  # FedProx regularization
)
```

## PyTorch Example

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = SimpleModel()

from neural.federated import FederatedOrchestrator, CrossDeviceScenario
import copy

orchestrator = FederatedOrchestrator(
    model=model,
    backend='pytorch',
    scenario=CrossDeviceScenario(num_devices=50),
)

def model_fn():
    return copy.deepcopy(model)

orchestrator.setup_clients(
    model_fn=model_fn,
    data=(X_train, y_train),
)

history = orchestrator.train(num_rounds=100)
```

## CLI Usage

### Generate Configuration

```bash
python -m neural.federated.cli generate-config \
    --scenario cross_device \
    --num-clients 100 \
    --output federated_config.yaml
```

### Prepare Data

```bash
python -m neural.federated.cli prepare-data \
    --data train_data.npz \
    --num-clients 100 \
    --distribution non_iid \
    --alpha 0.5 \
    --output client_data/
```

### Train

```bash
python -m neural.federated.cli train \
    --config federated_config.yaml \
    --data train_data.npz \
    --test-data test_data.npz \
    --backend tensorflow \
    --output federated_model
```

### Generate Report

```bash
python -m neural.federated.cli report \
    --metrics federated_metrics.json \
    --output report.txt
```

## Common Scenarios

### Mobile Device Training (Cross-Device)

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(
        num_devices=1000,
        devices_per_round=50,
        data_heterogeneity=0.8,
        device_availability=0.7,  # 70% availability
    ),
    privacy_mechanism='gaussian',
    epsilon=1.0,
    compression_strategy='quantization',
    quantization_bits=4,  # Aggressive compression
)
```

### Hospital Collaboration (Cross-Silo)

```python
from neural.federated import CrossSiloScenario

orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossSiloScenario(
        num_silos=5,        # 5 hospitals
        silos_per_round=5,  # All participate
    ),
    privacy_mechanism='gaussian',
    epsilon=0.5,            # Strict privacy
    use_secure_aggregation=True,
)
```

### Hybrid Enterprise Deployment

```python
from neural.federated import HybridScenario

orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=HybridScenario(
        num_silos=3,                      # 3 data centers
        devices_per_silo=100,             # 100 edge devices each
        silos_per_round=2,                # 2 data centers per round
        devices_per_silo_per_round=20,    # 20 devices per center
    ),
    aggregation_strategy='fedadam',
    server_lr=0.01,
)
```

## Troubleshooting

### Client Convergence Issues

Try FedProx instead of FedAvg:

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    aggregation_strategy='fedprox',
    proximal_mu=0.01,  # Increase for more regularization
)
```

### Privacy Budget Exceeded

Reduce privacy requirements or increase budget:

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    privacy_mechanism='gaussian',
    epsilon=10.0,  # Increase budget
    clip_norm=5.0,  # Increase clipping threshold
)
```

### Communication Overhead

Enable aggressive compression:

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    compression_strategy='adaptive',
    target_compression=0.1,  # 90% compression
    sparsity=0.9,
    quantization_bits=2,
)
```

## Next Steps

- Read the [full README](README.md) for detailed API documentation
- Check out [examples.py](examples.py) for more examples
- Explore different aggregation strategies: FedAvg, FedProx, FedAdam, FedYogi
- Experiment with privacy-accuracy trade-offs
- Optimize communication efficiency

## Resources

- **Paper**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Documentation**: See README.md for complete API reference
- **Examples**: Run `python neural/federated/examples.py`
