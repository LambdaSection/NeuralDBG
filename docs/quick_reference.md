# Neural DSL Quick Reference

This document consolidates essential quick-start information from across the repository.

## Installation

```bash
# Minimal installation
pip install neural-dsl

# Full installation with all features
pip install neural-dsl[full]

# Development installation
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
python -m venv .venv
.\.venv\Scripts\Activate   # Windows
# or source .venv/bin/activate  # Linux/macOS
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Create Your First Model

Create `mnist.neural`:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 10
    batch_size: 64
  }
}
```

### 2. Compile to Your Framework

```bash
# TensorFlow
neural compile mnist.neural --backend tensorflow --output mnist_tf.py

# PyTorch
neural compile mnist.neural --backend pytorch --output mnist_pt.py

# ONNX
neural compile mnist.neural --backend onnx --output model.onnx
```

### 3. Visualize Architecture

```bash
neural visualize mnist.neural --format html
neural visualize mnist.neural --format png
```

### 4. Debug Your Model

```bash
neural debug mnist.neural
# Open http://localhost:8050 for NeuralDbg dashboard
```

## Common Commands

```bash
# Compile model
neural compile <file> --backend <tensorflow|pytorch|onnx>

# Run model (compile + train)
neural run <file> --backend <backend>

# Visualize architecture
neural visualize <file> --format <html|png|svg>

# Debug with dashboard
neural debug <file>

# Export for deployment
neural export <file> --format <onnx|tflite|torchscript|savedmodel>

# Experiment tracking
neural track list
neural track show <experiment_id>
neural track compare <exp1> <exp2>

# Launch no-code GUI
neural --no_code

# Show help
neural --help
neural <command> --help
```

## NeuralDbg Dashboard Quick Start

Start the dashboard:

```bash
python neural/dashboard/dashboard.py
# Open http://localhost:8050
```

Send data from your code:

```python
from neural.dashboard import update_dashboard_data

trace_data = [{
    "layer": "Conv2D_1",
    "execution_time": 0.045,
    "memory": 256.5,
    "flops": 1000000,
    "mean_activation": 0.35,
    "grad_norm": 0.02
}]

update_dashboard_data(new_trace_data=trace_data)
```

## Deployment Quick Reference

```bash
# ONNX (cross-platform)
neural export model.neural --format onnx --optimize

# TensorFlow Lite (mobile/edge)
neural export model.neural --backend tensorflow --format tflite --quantize

# TorchScript (PyTorch production)
neural export model.neural --backend pytorch --format torchscript

# TensorFlow Serving
neural export model.neural --backend tensorflow --format savedmodel --deployment tfserving
```

## Platform Integrations

The following integrations are available (install with `pip install neural-dsl[integrations]`):

- **Databricks**: `neural.integrations.databricks`
- **AWS SageMaker**: `neural.integrations.sagemaker`
- **Google Vertex AI**: `neural.integrations.vertexai`
- **Azure ML**: `neural.integrations.azureml`
- **Paperspace**: `neural.integrations.paperspace`
- **Run:AI**: `neural.integrations.runai`

See [INTEGRATIONS.md](INTEGRATIONS.md) for detailed setup.

## Feature Installation Groups

| Group | Command | Includes |
|-------|---------|----------|
| Core | `pip install neural-dsl` | DSL parsing only |
| Backends | `[backends]` | TensorFlow, PyTorch, ONNX |
| HPO | `[hpo]` | Optuna, hyperparameter optimization |
| AutoML | `[automl]` | Neural Architecture Search |
| Dashboard | `[dashboard]` | NeuralDbg interface |
| Visualization | `[visualization]` | Charts and diagrams |
| Integrations | `[integrations]` | Cloud platform connectors |
| Full | `[full]` | All features (~2.5 GB) |

## Development Commands

```bash
# Lint
python -m ruff check .

# Type check
python -m mypy neural/code_generation neural/utils

# Run tests
python -m pytest tests/ -v

# Test with coverage
pytest tests/ -v --cov=neural --cov-report=html

# Security audit
python -m pip_audit -l
```

## Troubleshooting

**Command not found?**
```bash
python -m neural.cli compile model.neural
```

**Import errors?**
```bash
pip install neural-dsl[backends]
```

**Visualization fails?**
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows: Download from graphviz.org
```

## Getting Help

- **Documentation**: [docs/](.)
- **Examples**: [examples/](../examples/)
- **Discord**: [Join community](https://discord.gg/KFku4KvS)
- **GitHub Issues**: [Report bugs](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [Ask questions](https://github.com/Lemniscate-world/Neural/discussions)

## Additional Resources

- [DSL Language Reference](dsl.md) - Complete syntax guide
- [Deployment Guide](deployment.md) - Production export options
- [AI Integration Guide](ai_integration_guide.md) - Natural language model generation
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [AGENTS.md](../AGENTS.md) - Development workflow for contributors
