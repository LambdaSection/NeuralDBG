# Federated Learning Module

A comprehensive federated learning implementation supporting client-server architecture, differential privacy, secure aggregation, communication efficiency, and heterogeneous client scenarios.

## Features

### Core Architecture
- **Client-Server Model**: Scalable federated learning with centralized coordination
- **Multiple Aggregation Strategies**: FedAvg, FedProx, FedAdam, FedYogi, FedMA, Adaptive
- **Backend Support**: TensorFlow and PyTorch

### Privacy Mechanisms
- **Differential Privacy**: Gaussian and Laplacian noise mechanisms
- **Privacy Accounting**: Track privacy budget consumption with RDP
- **Local Differential Privacy**: Client-side privacy guarantees
- **Shuffle Protocol**: Privacy amplification through shuffling
- **Adaptive Privacy**: Dynamic privacy budget allocation

### Secure Aggregation
- **Cryptographic Protocols**: Secure multi-party computation
- **Noise Masking**: Protect individual updates during aggregation
- **Verification**: Hash-based integrity checking

### Communication Efficiency
- **Quantization**: 1-32 bit quantization with stochastic rounding
- **Sparsification**: Top-k, threshold, and random sparsification
- **Adaptive Compression**: Combined quantization and sparsification
- **Gradient Compression**: Error feedback for improved convergence
- **Sketch Compression**: Random projection techniques
- **Communication Scheduling**: Adaptive communication intervals

### Heterogeneous Clients
- **Resource Awareness**: Account for compute capability and bandwidth
- **Adaptive Selection**: Resource-based client selection
- **Local Epochs**: Variable local training per client
- **Batch Size**: Per-client batch size configuration

### Scenarios
- **Cross-Device**: 100s-1000s of mobile/edge devices
- **Cross-Silo**: 2-100 organizations/data centers
- **Hybrid**: Combined cross-device and cross-silo
- **Vertical FL**: Feature partitioning across parties

## Quick Start

### Basic Federated Training

```python
from neural.federated import FederatedOrchestrator, CrossDeviceScenario
import numpy as np

# Create model (TensorFlow example)
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Prepare data
X_train = np.random.randn(10000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 10000)

# Create orchestrator
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(num_devices=100, devices_per_round=10),
    aggregation_strategy='fedavg',
)

# Setup clients
def model_fn():
    return tf.keras.models.clone_model(model)

orchestrator.setup_clients(
    model_fn=model_fn,
    data=(X_train, y_train),
    local_epochs=1,
    learning_rate=0.01,
)

# Train
history = orchestrator.train(
    num_rounds=100,
    evaluate_every=5,
)

# Save model
orchestrator.server.save_model('federated_model.h5')
```

### With Differential Privacy

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossDeviceScenario(num_devices=100),
    aggregation_strategy='fedavg',
    privacy_mechanism='gaussian',
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0,
)
```

### With Communication Compression

```python
orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=CrossSiloScenario(num_silos=10),
    aggregation_strategy='fedavg',
    compression_strategy='quantization',
    quantization_bits=8,
)
```

### Cross-Silo Scenario

```python
from neural.federated import CrossSiloScenario

scenario = CrossSiloScenario(
    num_silos=10,
    silos_per_round=5,
    data_heterogeneity=0.3,
)

orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=scenario,
    aggregation_strategy='fedprox',
    proximal_mu=0.01,
)
```

### Hybrid Scenario

```python
from neural.federated import HybridScenario

scenario = HybridScenario(
    num_silos=5,
    devices_per_silo=20,
    silos_per_round=3,
    devices_per_silo_per_round=5,
)

orchestrator = FederatedOrchestrator(
    model=model,
    backend='tensorflow',
    scenario=scenario,
    aggregation_strategy='fedavg',
    use_secure_aggregation=True,
)
```

## Advanced Usage

### Custom Aggregation Strategy

```python
from neural.federated.aggregation import AggregationStrategy
import numpy as np

class CustomAggregation(AggregationStrategy):
    def aggregate(self, weights_list, num_samples_list, **kwargs):
        # Custom aggregation logic
        return aggregated_weights

orchestrator.aggregation_strategy = CustomAggregation()
```

### Privacy Budget Tracking

```python
from neural.federated.privacy import PrivacyAccountant

accountant = PrivacyAccountant(epsilon_total=1.0, delta_total=1e-5)

for round_num in range(num_rounds):
    # Training logic
    epsilon_spent, delta_spent = accountant.compute_privacy_spent(
        noise_multiplier=1.0,
        sample_rate=0.1,
        steps=1,
    )
    accountant.spend_privacy_budget(epsilon_spent, delta_spent, f"round_{round_num}")
    
    # Check remaining budget
    remaining_eps, remaining_delta = accountant.get_remaining_budget()
    print(f"Remaining budget: ε={remaining_eps:.4f}, δ={remaining_delta:.4e}")
```

### Secure Aggregation

```python
from neural.federated.aggregation import SecureAggregator

secure_agg = SecureAggregator(threshold=2)

# In training loop
aggregated_weights = secure_agg.secure_aggregate(
    weights_list=client_weights,
    num_samples_list=num_samples,
    aggregation_strategy=FedAvg(),
)
```

### Communication Efficiency

```python
from neural.federated.communication import (
    QuantizationCompressor,
    SparsificationCompressor,
    AdaptiveCompressor,
    GradientCompression,
)

# Quantization
compressor = QuantizationCompressor(num_bits=8, stochastic=True)
compressed, metadata = compressor.compress(weights)
decompressed = compressor.decompress(compressed, metadata)

# Sparsification
compressor = SparsificationCompressor(sparsity=0.9, method='topk')
compressed, metadata = compressor.compress(weights)

# Adaptive (combines both)
compressor = AdaptiveCompressor(
    target_compression=0.5,
    quantization_bits=8,
    sparsity=0.5,
)

# Gradient compression with error feedback
grad_compressor = GradientCompression(
    compression_strategy=compressor,
    error_feedback=True,
)
compressed, metadata = grad_compressor.compress_gradients(gradients)
```

### Non-IID Data Distribution

```python
from neural.federated.utils import split_data_non_iid

client_data = split_data_non_iid(
    data=(X_train, y_train),
    num_clients=100,
    alpha=0.5,  # Lower = more heterogeneous
    num_classes=10,
)
```

### Byzantine Detection

```python
from neural.federated.utils import filter_byzantine_clients

# Filter out malicious clients
filtered_results = filter_byzantine_clients(
    client_results=client_results,
    global_weights=global_weights,
    threshold=0.5,
)
```

## Configuration

### YAML Configuration

```yaml
backend: tensorflow

model:
  model_data:
    input:
      shape: [784]
    layers:
      - type: dense
        units: 128
        activation: relu
      - type: dropout
        rate: 0.5
      - type: output
        units: 10
        activation: softmax

training:
  batch_size: 32
  learning_rate: 0.01

federated:
  enabled: true
  scenario: cross_device
  num_devices: 100
  devices_per_round: 10
  num_rounds: 100
  local_epochs: 1
  aggregation: fedavg
  privacy:
    enabled: true
    mechanism: gaussian
    epsilon: 1.0
    delta: 1e-5
    clip_norm: 1.0
  compression:
    enabled: true
    type: quantization
    num_bits: 8
  secure_aggregation: true
```

## Integration with Neural DSL

```python
from neural.federated.integration import (
    integrate_with_neural_training,
    run_federated_training,
    federated_train_from_config,
)

# From config file
results = federated_train_from_config(
    config_path='federated_config.yaml',
    data=(X_train, y_train),
    test_data=(X_test, y_test),
)

# Programmatic
results = run_federated_training(
    model=model,
    data=(X_train, y_train),
    config={
        'scenario': 'cross_device',
        'num_devices': 100,
        'num_rounds': 100,
        'aggregation': 'fedavg',
    },
    backend='tensorflow',
    test_data=(X_test, y_test),
)
```

## Metrics and Monitoring

```python
# Get training history
history = orchestrator.metrics_history

# Get summary
summary = orchestrator.get_metrics_summary()
print(f"Final accuracy: {summary['final_eval_accuracy']:.4f}")
print(f"Best accuracy: {summary['best_eval_accuracy']:.4f}")

# Save metrics
orchestrator.save_metrics('federated_metrics.json')
```

## API Reference

### Main Classes

- `FederatedClient`: Local client for federated learning
- `FederatedServer`: Central server for coordination
- `FederatedOrchestrator`: High-level orchestration
- `FedAvg`, `FedProx`, `FedAdam`, `FedYogi`: Aggregation strategies
- `GaussianDP`, `LaplacianDP`: Differential privacy mechanisms
- `PrivacyAccountant`: Privacy budget tracking
- `QuantizationCompressor`, `SparsificationCompressor`: Compression
- `CrossDeviceScenario`, `CrossSiloScenario`, `HybridScenario`: Scenarios

### Key Parameters

- `num_rounds`: Number of federated training rounds
- `local_epochs`: Local training epochs per client
- `fraction_fit`: Fraction of clients selected per round
- `epsilon`, `delta`: Differential privacy parameters
- `clip_norm`: Gradient clipping threshold
- `proximal_mu`: FedProx regularization parameter
- `quantization_bits`: Bit precision for quantization
- `sparsity`: Sparsification ratio (0-1)

## References

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx)
- Reddi et al. "Adaptive Federated Optimization" (FedAdam, FedYogi)
- Wang et al. "Federated Learning with Matched Averaging" (FedMA)
- Abadi et al. "Deep Learning with Differential Privacy"
- Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
