# Neural DSL Cloud Integration Guide

## Overview

Neural DSL provides comprehensive cloud integration support, enabling you to develop, train, and deploy neural networks on popular cloud platforms including:

- **Kaggle** - Data science competitions and collaborative notebooks
- **Google Colab** - Free GPU/TPU access for research and development
- **AWS SageMaker** - Production-grade ML infrastructure
- **Azure ML** - Enterprise machine learning platform
- **AWS Lambda** - Serverless model inference

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Supported Platforms](#supported-platforms)
4. [Core Features](#core-features)
5. [CloudExecutor API](#cloudexecutor-api)
6. [Error Handling](#error-handling)
7. [Cloud Optimizations](#cloud-optimizations)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

---

## Quick Start

### Using CloudExecutor (Recommended)

```python
from neural.cloud.cloud_execution import CloudExecutor

# Initialize executor (auto-detects environment)
executor = CloudExecutor()

# Define your model
dsl_code = """
network MyCNN {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

# Compile and run
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST', epochs=5)

# Cleanup
executor.cleanup()
```

### Using Jupyter Notebooks

We provide ready-to-use notebook templates:

- [Kaggle Template](../neural/cloud/examples/neural_kaggle_example.ipynb)
- [Colab Template](../neural/cloud/examples/neural_colab_example.ipynb)
- [SageMaker Template](../neural/cloud/examples/neural_sagemaker_example.ipynb)

---

## Installation

### Option 1: Direct Installation

```bash
pip install git+https://github.com/Lemniscate-SHA-256/Neural.git
```

### Option 2: Installation Script

```python
# In Kaggle/Colab notebook
!curl -s https://raw.githubusercontent.com/Lemniscate-SHA-256/Neural/main/neural/cloud/install_neural.py | python
```

### Optional Dependencies

For full cloud functionality:

```bash
pip install pyngrok  # For ngrok tunneling
pip install boto3 sagemaker  # For AWS SageMaker
pip install azure-ai-ml  # For Azure ML
```

---

## Supported Platforms

### Kaggle

**Features:**
- Auto-detection of Kaggle environment
- GPU support (NVIDIA Tesla P100)
- Dataset integration
- Kernel execution

**Example:**
```python
executor = CloudExecutor(environment='kaggle')
print(f"GPU Available: {executor.is_gpu_available}")
```

### Google Colab

**Features:**
- Free GPU/TPU access (Tesla T4, P100, V100)
- Google Drive integration
- ngrok tunnel support for dashboards
- Longer training times (up to 12 hours)
- **NEW:** Full notebook interface support
- **NEW:** Interactive debugging with NeuralDbg
- **NEW:** Automatic environment initialization

**Quick Start:**
```python
# Option 1: Use CloudExecutor directly
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()  # Auto-detects Colab environment
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST', epochs=5)

# Option 2: Use the Notebook Interface (NEW!)
from neural.cloud.notebook_interface import start_notebook_interface

# This creates a pre-configured notebook with helper functions
start_notebook_interface('colab', port=8888)
```

**Pre-built Notebook Template:**

We provide a comprehensive Colab notebook with all features pre-configured:

📓 [Neural DSL Colab Quick Start](../docs/colab/neural_colab_quickstart.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lemniscate-world/Neural/blob/main/docs/colab/neural_colab_quickstart.ipynb)

**Example:**
```python
# Enable GPU: Runtime → Change runtime type → GPU
executor = CloudExecutor(environment='colab')
dashboard = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
```

### AWS SageMaker

**Features:**
- Production-grade infrastructure
- Distributed training support
- Model registry integration
- Endpoint deployment

**Example:**
```python
executor = CloudExecutor(environment='sagemaker')
model_path = executor.compile_model(dsl_code, backend='tensorflow')
```

### Azure ML

**Features:**
- Enterprise security
- MLOps integration
- Compute cluster management

**Example:**
```python
executor = CloudExecutor(environment='azure_ml')
```

### AWS Lambda

**Features:**
- Serverless execution
- Auto-scaling
- Cost-effective inference

**Example:**
```python
executor = CloudExecutor(environment='lambda')
```

---

## Core Features

### 1. Environment Auto-Detection

CloudExecutor automatically detects your environment:

```python
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
# Output: kaggle, colab, sagemaker, azure_ml, lambda, or unknown
```

### 2. GPU Detection and Configuration

Automatic GPU detection and optimization:

```python
executor = CloudExecutor()
if executor.is_gpu_available:
    print("GPU optimizations enabled")
    print(f"Optimization level: {executor.optimization_level}/3")
```

### 3. Error Handling and Retry Logic

Built-in retry mechanism with exponential backoff:

```python
executor = CloudExecutor(
    timeout=600,  # 10 minutes
    retry_attempts=3  # Retry failed operations
)
```

### 4. Cloud-Specific Optimizations

Automatic optimizations based on platform:

- **Memory management**: GPU memory growth enabled
- **Logging**: Reduced TensorFlow verbosity
- **Caching**: CUDA cache configuration
- **Environment variables**: Platform-specific settings

### 5. Remote Dashboard Access

Access NeuralDbg and No-Code interfaces via ngrok tunnels:

```python
# Start NeuralDbg dashboard
dashboard = executor.start_debug_dashboard(
    dsl_code,
    setup_tunnel=True,
    port=8050
)
print(f"Dashboard URL: {dashboard['tunnel_url']}")

# Start No-Code interface
nocode = executor.start_nocode_interface(
    setup_tunnel=True,
    port=8051
)
print(f"Interface URL: {nocode['tunnel_url']}")
```

---

## CloudExecutor API

### Initialization

```python
CloudExecutor(
    environment: str = None,      # Auto-detect if None
    timeout: int = 300,            # Default timeout in seconds
    retry_attempts: int = 3        # Number of retry attempts
)
```

### Methods

#### compile_model()

Compile Neural DSL to executable code.

```python
model_path = executor.compile_model(
    dsl_code: str,                 # Neural DSL code
    backend: str = 'tensorflow',   # 'tensorflow', 'pytorch', 'jax'
    output_file: str = None,       # Optional output path
    validate: bool = True          # Validate model structure
)
```

**Returns:** Path to compiled model file

**Raises:** `CloudCompilationError` on failure

#### run_model()

Execute a compiled model.

```python
results = executor.run_model(
    model_file: str,               # Path to compiled model
    dataset: str = 'MNIST',        # Dataset name
    epochs: int = 5,               # Training epochs
    batch_size: int = 32,          # Batch size
    timeout: int = None            # Override default timeout
)
```

**Returns:** Dictionary with execution results:
```python
{
    'success': bool,
    'stdout': str,
    'stderr': str,
    'return_code': int,
    'error': str,              # If failed
    'error_type': str          # 'timeout', 'execution_error', etc.
}
```

#### visualize_model()

Generate model visualization.

```python
viz_path = executor.visualize_model(
    dsl_code: str,                 # Neural DSL code
    output_format: str = 'png',    # 'png', 'svg', 'html'
    output_file: str = None        # Optional output path
)
```

**Returns:** Path to visualization file

#### setup_ngrok_tunnel()

Create ngrok tunnel for remote access.

```python
public_url = executor.setup_ngrok_tunnel(
    port: int = 8050,              # Local port to expose
    auth_token: str = None         # Optional ngrok auth token
)
```

**Returns:** Public URL or None if failed

#### start_debug_dashboard()

Launch NeuralDbg dashboard.

```python
dashboard_info = executor.start_debug_dashboard(
    dsl_code: str,                 # Neural DSL code
    backend: str = 'tensorflow',   # Target backend
    setup_tunnel: bool = True,     # Create ngrok tunnel
    port: int = 8050              # Dashboard port
)
```

**Returns:** Dictionary with dashboard information:
```python
{
    'session_id': str,
    'dashboard_url': str,
    'process_id': int,
    'tunnel_url': str,
    'status': str                  # 'running' or 'failed'
}
```

#### start_nocode_interface()

Launch No-Code visual interface.

```python
nocode_info = executor.start_nocode_interface(
    port: int = 8051,              # Interface port
    setup_tunnel: bool = True      # Create ngrok tunnel
)
```

**Returns:** Dictionary with interface information (same structure as dashboard)

#### get_environment_info()

Get comprehensive environment information.

```python
info = executor.get_environment_info()
```

**Returns:** Dictionary with environment details:
```python
{
    'environment': str,
    'gpu_available': bool,
    'optimization_level': int,
    'python_version': str,
    'temp_dir': str,
    'gpu_info': str              # If GPU available
}
```

#### cleanup()

Clean up temporary files and processes.

```python
executor.cleanup()
```

---

## Error Handling

### Exception Hierarchy

```python
CloudExecutionError           # Base exception
├── CloudConnectionError      # Connection failures
├── CloudCompilationError     # Compilation failures
└── CloudRuntimeError         # Runtime failures
```

### Handling Errors

```python
from neural.cloud.cloud_execution import (
    CloudExecutor,
    CloudCompilationError,
    CloudRuntimeError
)

executor = CloudExecutor()

try:
    model_path = executor.compile_model(dsl_code, backend='tensorflow')
    results = executor.run_model(model_path, dataset='MNIST')
    
    if not results['success']:
        print(f"Training failed: {results['error']}")
        print(f"Error type: {results['error_type']}")
        
except CloudCompilationError as e:
    print(f"Compilation error: {e}")
except CloudRuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    executor.cleanup()
```

### Error Types

- **timeout**: Operation exceeded time limit
- **execution_error**: Model execution failed
- **unexpected_error**: Unforeseen error occurred

---

## Cloud Optimizations

### Automatic Optimizations

Neural DSL applies platform-specific optimizations automatically:

#### Kaggle
```python
TF_CPP_MIN_LOG_LEVEL=2          # Reduce TensorFlow logging
PYTHONUNBUFFERED=1              # Unbuffered output
```

#### Google Colab
```python
TF_CPP_MIN_LOG_LEVEL=2          # Reduce TensorFlow logging
CUDA_CACHE_DISABLE=0            # Enable CUDA cache
TF_FORCE_GPU_ALLOW_GROWTH=true  # Dynamic GPU memory
```

#### AWS SageMaker
```python
SM_FRAMEWORK_PARAMS={}          # SageMaker parameters
```

#### All GPU Environments
```python
TF_FORCE_GPU_ALLOW_GROWTH=true           # TensorFlow GPU growth
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # PyTorch CUDA allocation
```

### Manual Optimizations

```python
import os

# Disable GPU for testing
os.environ['NEURAL_FORCE_CPU'] = '1'

# Enable debug logging
os.environ['NEURAL_DEBUG'] = '1'

# Set custom TensorFlow settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
```

---

## Best Practices

### 1. Resource Management

```python
# Always use try-finally for cleanup
executor = CloudExecutor()
try:
    # Your code here
    pass
finally:
    executor.cleanup()
```

### 2. Timeouts

Set appropriate timeouts based on task complexity:

```python
# Short tasks (< 5 minutes)
executor = CloudExecutor(timeout=300)

# Long training runs (30+ minutes)
executor = CloudExecutor(timeout=1800)

# Override per operation
results = executor.run_model(model_path, timeout=3600)
```

### 3. Error Handling

Always check results:

```python
results = executor.run_model(model_path, dataset='MNIST')
if results['success']:
    print("Success!")
else:
    print(f"Failed: {results['error']}")
    if results.get('stderr'):
        print(f"Details: {results['stderr']}")
```

### 4. GPU Memory

Monitor GPU usage and clear when needed:

```python
# TensorFlow
import tensorflow as tf
tf.keras.backend.clear_session()

# PyTorch
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. Model Validation

Validate before training:

```python
model_path = executor.compile_model(
    dsl_code,
    backend='tensorflow',
    validate=True  # Enable validation
)
```

### 6. Batch Sizes

Choose appropriate batch sizes:

- **CPU**: 16-32
- **GPU (Colab free)**: 64-128
- **GPU (Premium)**: 128-256
- **TPU**: 256-512

### 7. Checkpointing

Save checkpoints for long training:

```python
dsl_code = """
network MyModel {
    input: (224, 224, 3)
    layers: ...
    train {
        epochs: 100
        checkpoint_freq: 10  # Save every 10 epochs
    }
}
"""
```

### 8. ngrok Authentication

For production, use authenticated ngrok:

```python
# Set ngrok auth token
public_url = executor.setup_ngrok_tunnel(
    port=8050,
    auth_token='your_ngrok_token'
)
```

---

## Troubleshooting

### Issue: Environment Not Detected

**Symptom:** `executor.environment == 'unknown'`

**Solution:**
```python
# Manually specify environment
executor = CloudExecutor(environment='colab')
```

### Issue: GPU Not Available

**Symptom:** `executor.is_gpu_available == False`

**Solutions:**
1. **Colab**: Runtime → Change runtime type → GPU
2. **Kaggle**: Settings → Accelerator → GPU
3. **SageMaker**: Use GPU instance type (e.g., ml.p3.2xlarge)

### Issue: ngrok Tunnel Fails

**Symptom:** `setup_ngrok_tunnel()` returns None

**Solutions:**
1. Install pyngrok: `pip install pyngrok`
2. Set auth token: `executor.setup_ngrok_tunnel(port=8050, auth_token='your_token')`
3. Check network connectivity

### Issue: Model Compilation Fails

**Symptom:** `CloudCompilationError` raised

**Solutions:**
1. Validate DSL syntax
2. Check for missing dependencies
3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Issue: Out of Memory

**Symptom:** CUDA OOM or MemoryError

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Clear GPU cache:
   ```python
   tf.keras.backend.clear_session()
   torch.cuda.empty_cache()
   ```
4. Use mixed precision training

### Issue: Timeout Errors

**Symptom:** `error_type: 'timeout'`

**Solutions:**
1. Increase timeout:
   ```python
   executor = CloudExecutor(timeout=1800)
   ```
2. Use retry logic (built-in)
3. Reduce model complexity
4. Use fewer epochs for testing

---

## Examples

### Example 1: Basic Kaggle Workflow

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()

# Define model
dsl_code = """
network KaggleCNN {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

# Compile and run
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST', epochs=5)

if results['success']:
    print("Training completed!")

executor.cleanup()
```

### Example 2: Colab with Visualization

```python
from neural.cloud.cloud_execution import CloudExecutor
from IPython.display import Image, display

executor = CloudExecutor()

dsl_code = """
network ColabNet {
    input: (32, 32, 3)
    layers:
        Conv2D(64, (3, 3), "relu", padding="same")
        MaxPooling2D((2, 2))
        Conv2D(128, (3, 3), "relu", padding="same")
        GlobalAveragePooling2D()
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

# Visualize
viz_path = executor.visualize_model(dsl_code, output_format='png')
display(Image(filename=viz_path))

# Compile and train
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='CIFAR10', epochs=10)

executor.cleanup()
```

### Example 3: Remote Debugging

```python
from neural.cloud.cloud_execution import CloudExecutor
from IPython.display import HTML, display

executor = CloudExecutor()

dsl_code = """
network DebugNet {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        GlobalAveragePooling2D()
        Dense(1000, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

# Start dashboard with ngrok tunnel
dashboard = executor.start_debug_dashboard(
    dsl_code,
    setup_tunnel=True
)

if dashboard['status'] == 'running':
    print(f"Dashboard URL: {dashboard['tunnel_url']}")
    display(HTML(f"<a href='{dashboard['tunnel_url']}' target='_blank'>Open Dashboard</a>"))

executor.cleanup()
```

### Example 4: Multi-Backend Comparison

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()

dsl_code = """
network MultiBackendNet {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

backends = ['tensorflow', 'pytorch']
results = {}

for backend in backends:
    try:
        model_path = executor.compile_model(dsl_code, backend=backend)
        run_results = executor.run_model(model_path, dataset='MNIST', epochs=3)
        results[backend] = run_results['success']
        print(f"{backend}: {'✓ Success' if run_results['success'] else '✗ Failed'}")
    except Exception as e:
        print(f"{backend}: ✗ Error - {e}")

executor.cleanup()
```

### Example 5: Error Handling

```python
from neural.cloud.cloud_execution import (
    CloudExecutor,
    CloudCompilationError,
    CloudRuntimeError
)

executor = CloudExecutor(retry_attempts=3)

dsl_code = """
network RobustNet {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam()
}
"""

try:
    # Compile with validation
    model_path = executor.compile_model(
        dsl_code,
        backend='tensorflow',
        validate=True
    )
    
    # Run with timeout
    results = executor.run_model(
        model_path,
        dataset='MNIST',
        epochs=5,
        timeout=600
    )
    
    # Check results
    if results['success']:
        print("✓ Training successful!")
    else:
        print(f"✗ Training failed: {results['error']}")
        print(f"Error type: {results['error_type']}")
        
except CloudCompilationError as e:
    print(f"Compilation failed: {e}")
except CloudRuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    executor.cleanup()
```

---

## Additional Resources

- **GitHub Repository**: [Neural DSL](https://github.com/Lemniscate-SHA-256/Neural)
- **Documentation**: [Full Docs](https://github.com/Lemniscate-SHA-256/Neural/tree/main/docs)
- **Examples**: [Example Notebooks](https://github.com/Lemniscate-SHA-256/Neural/tree/main/neural/cloud/examples)
- **Issues**: [Report Issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
- **Discussions**: [Community Forum](https://github.com/Lemniscate-SHA-256/Neural/discussions)

---

## Support

For help and support:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [Examples](#examples) for common use cases
3. Search [existing issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
4. Ask in [Discussions](https://github.com/Lemniscate-SHA-256/Neural/discussions)
5. Create a new issue if needed

---

## License

Neural DSL is released under the MIT License. See [LICENSE](../LICENSE.md) for details.
