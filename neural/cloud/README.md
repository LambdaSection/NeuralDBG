# Neural DSL Cloud Integration

Advanced cloud integration for Neural DSL, enabling seamless model development, training, and deployment across popular cloud platforms.

## 🚀 Quick Start

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()  # Auto-detects environment
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST', epochs=5)
executor.cleanup()
```

## 📋 Features

### Core Capabilities

- ✅ **Environment Auto-Detection** - Automatically detect Kaggle, Colab, SageMaker, Azure ML, and Lambda
- ✅ **GPU/TPU Support** - Automatic detection and optimization for hardware accelerators
- ✅ **Error Handling** - Comprehensive error handling with retry logic and exponential backoff
- ✅ **Cloud Optimizations** - Platform-specific optimizations for memory, caching, and performance
- ✅ **Remote Dashboard Access** - Access NeuralDbg and No-Code interfaces via ngrok tunnels
- ✅ **Multi-Backend Support** - Compile to TensorFlow, PyTorch, or JAX
- ✅ **Timeout Management** - Configurable timeouts with graceful handling
- ✅ **Resource Cleanup** - Automatic cleanup of temporary files and processes

### Supported Platforms

| Platform | Status | Features |
|----------|--------|----------|
| **Kaggle** | ✅ Full Support | GPU detection, dataset integration, kernel execution |
| **Google Colab** | ✅ Full Support | GPU/TPU, Drive integration, ngrok tunneling |
| **AWS SageMaker** | ✅ Full Support | Distributed training, model registry, endpoints |
| **Azure ML** | ✅ Basic Support | Compute clusters, MLOps integration |
| **AWS Lambda** | ✅ Basic Support | Serverless inference, auto-scaling |

## 📦 Installation

### Quick Install

```bash
# In your cloud notebook
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git
```

### With Cloud Dependencies

```bash
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git pyngrok
```

### For AWS SageMaker

```bash
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git boto3 sagemaker
```

## 📓 Notebook Templates

Ready-to-use notebooks for different platforms:

1. **[Quick Start (5 min)](examples/quick_start.ipynb)** - Get started in 5 minutes
2. **[Kaggle Complete Guide](examples/neural_kaggle_example.ipynb)** - Comprehensive Kaggle tutorial
3. **[Colab Advanced Features](examples/neural_colab_example.ipynb)** - Full Colab capabilities
4. **[SageMaker Production](examples/neural_sagemaker_example.ipynb)** - Production deployment

## 🔧 CloudExecutor API

### Initialization

```python
executor = CloudExecutor(
    environment=None,      # Auto-detect or specify: 'kaggle', 'colab', 'sagemaker'
    timeout=300,          # Default operation timeout (seconds)
    retry_attempts=3      # Number of retry attempts for transient failures
)
```

### Key Methods

#### compile_model()
```python
model_path = executor.compile_model(
    dsl_code,                    # Neural DSL code
    backend='tensorflow',        # 'tensorflow', 'pytorch', 'jax'
    output_file=None,           # Optional custom output path
    validate=True               # Validate model structure
)
```

#### run_model()
```python
results = executor.run_model(
    model_file,                  # Path to compiled model
    dataset='MNIST',            # Dataset name
    epochs=5,                   # Training epochs
    batch_size=32,              # Batch size
    timeout=None                # Override default timeout
)
# Returns: {'success': bool, 'stdout': str, 'stderr': str, 'error': str}
```

#### visualize_model()
```python
viz_path = executor.visualize_model(
    dsl_code,                    # Neural DSL code
    output_format='png',        # 'png', 'svg', 'html'
    output_file=None            # Optional output path
)
```

#### start_debug_dashboard()
```python
dashboard = executor.start_debug_dashboard(
    dsl_code,                    # Neural DSL code
    backend='tensorflow',       # Target backend
    setup_tunnel=True,          # Create ngrok tunnel
    port=8050                   # Dashboard port
)
# Returns: {'status': str, 'tunnel_url': str, 'process_id': int}
```

#### get_environment_info()
```python
info = executor.get_environment_info()
# Returns: {'environment': str, 'gpu_available': bool, 'optimization_level': int}
```

## 🎯 Usage Examples

### Example 1: Basic Model Training

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()

dsl_code = """
network SimpleCNN {
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

model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST', epochs=5)

if results['success']:
    print("✓ Training successful!")
else:
    print(f"✗ Failed: {results['error']}")

executor.cleanup()
```

### Example 2: Multi-Backend Compilation

```python
backends = ['tensorflow', 'pytorch']
models = {}

for backend in backends:
    try:
        path = executor.compile_model(dsl_code, backend=backend)
        models[backend] = path
        print(f"✓ {backend}: {path}")
    except Exception as e:
        print(f"✗ {backend}: {e}")
```

### Example 3: Remote Debugging

```python
# Start NeuralDbg dashboard with ngrok tunnel
dashboard = executor.start_debug_dashboard(
    dsl_code,
    setup_tunnel=True
)

if dashboard['status'] == 'running':
    print(f"Dashboard: {dashboard['tunnel_url']}")
```

### Example 4: Error Handling

```python
from neural.cloud.cloud_execution import (
    CloudExecutor,
    CloudCompilationError,
    CloudRuntimeError
)

executor = CloudExecutor(retry_attempts=3)

try:
    model_path = executor.compile_model(dsl_code, validate=True)
    results = executor.run_model(model_path, timeout=600)
    
    if not results['success']:
        print(f"Error type: {results['error_type']}")
        print(f"Details: {results['error']}")
        
except CloudCompilationError as e:
    print(f"Compilation failed: {e}")
except CloudRuntimeError as e:
    print(f"Runtime error: {e}")
finally:
    executor.cleanup()
```

## 🔥 Cloud Optimizations

Neural DSL automatically applies platform-specific optimizations:

### Kaggle
- Reduced TensorFlow logging (`TF_CPP_MIN_LOG_LEVEL=2`)
- Unbuffered Python output (`PYTHONUNBUFFERED=1`)

### Google Colab
- GPU memory growth enabled (`TF_FORCE_GPU_ALLOW_GROWTH=true`)
- CUDA caching enabled (`CUDA_CACHE_DISABLE=0`)
- PyTorch memory optimization (`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`)

### AWS SageMaker
- SageMaker framework parameters configured
- Distributed training support
- Model artifact management

### All GPU Environments
- Dynamic GPU memory allocation
- Optimized CUDA settings
- Batch size recommendations

## 🛠️ Advanced Features

### Custom Timeout Configuration

```python
executor = CloudExecutor(timeout=1800)  # 30 minutes

# Or override per operation
results = executor.run_model(model_path, timeout=3600)  # 1 hour
```

### Retry Logic

Built-in exponential backoff for transient failures:

```python
executor = CloudExecutor(retry_attempts=5)  # Retry up to 5 times
```

### Environment Information

```python
info = executor.get_environment_info()
print(f"Environment: {info['environment']}")
print(f"GPU Available: {info['gpu_available']}")
print(f"Optimization Level: {info['optimization_level']}/3")
if 'gpu_info' in info:
    print(f"GPU Info: {info['gpu_info']}")
```

### ngrok Authentication

For production use:

```python
public_url = executor.setup_ngrok_tunnel(
    port=8050,
    auth_token='your_ngrok_auth_token'
)
```

## 🐛 Error Types and Handling

### Exception Hierarchy

```
CloudExecutionError (base)
├── CloudConnectionError    # Connection failures
├── CloudCompilationError   # Compilation failures
└── CloudRuntimeError       # Runtime failures
```

### Error Result Types

When `run_model()` fails, check `error_type`:

- `timeout`: Operation exceeded time limit
- `execution_error`: Model execution failed
- `unexpected_error`: Unforeseen error

```python
results = executor.run_model(model_path)
if not results['success']:
    if results['error_type'] == 'timeout':
        print("Try increasing timeout")
    elif results['error_type'] == 'execution_error':
        print(f"Check stderr: {results['stderr']}")
```

## 📚 Best Practices

### 1. Always Use Cleanup

```python
executor = CloudExecutor()
try:
    # Your code
    pass
finally:
    executor.cleanup()
```

### 2. Validate Before Training

```python
model_path = executor.compile_model(dsl_code, validate=True)
```

### 3. Set Appropriate Timeouts

```python
# Quick test
executor = CloudExecutor(timeout=300)

# Long training
executor = CloudExecutor(timeout=3600)
```

### 4. Handle Errors Gracefully

```python
if not results['success']:
    print(f"Error: {results['error']}")
    print(f"Type: {results['error_type']}")
```

### 5. Monitor GPU Memory

```python
# TensorFlow
import tensorflow as tf
tf.keras.backend.clear_session()

# PyTorch
import torch
torch.cuda.empty_cache()
```

## 🔍 Troubleshooting

### GPU Not Detected

**Colab**: Runtime → Change runtime type → GPU  
**Kaggle**: Settings → Accelerator → GPU

### ngrok Tunnel Fails

```bash
!pip install pyngrok
```

Set auth token:
```python
executor.setup_ngrok_tunnel(port=8050, auth_token='your_token')
```

### Out of Memory

1. Reduce batch size
2. Clear GPU cache
3. Use gradient checkpointing
4. Enable mixed precision

### Timeout Errors

1. Increase timeout: `CloudExecutor(timeout=1800)`
2. Reduce model complexity
3. Use fewer epochs for testing

## 📖 Documentation

- **[Complete Cloud Guide](../../docs/cloud.md)** - Comprehensive documentation
- **[API Reference](../../docs/cloud.md#cloudexecutor-api)** - Full API documentation
- **[Examples](examples/)** - Notebook templates and examples

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](../../LICENSE.md) for details.

## 🔗 Links

- **GitHub**: [Neural DSL](https://github.com/Lemniscate-SHA-256/Neural)
- **Documentation**: [Full Docs](https://github.com/Lemniscate-SHA-256/Neural/tree/main/docs)
- **Issues**: [Report Issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
- **Discussions**: [Community](https://github.com/Lemniscate-SHA-256/Neural/discussions)

---

**Made with ❤️ by the Neural DSL team**
