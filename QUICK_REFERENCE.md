# Neural DSL - Quick Reference

## Installation

```bash
# Core only (minimal)
pip install neural-dsl

# With ML frameworks
pip install neural-dsl[backends]

# Full installation
pip install neural-dsl[full]

# Development
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
pip install -r requirements-dev.txt
```

## Common Commands

```bash
# Compile DSL to Python
neural compile model.neural --backend pytorch

# Run training
neural run model.neural

# Visualize architecture
neural visualize model.neural

# Debug model
neural debug model.neural

# Export for deployment
neural export model.neural --format onnx

# Experiment tracking
neural track list
neural track show <id>
neural track compare exp1 exp2
```

## Development Workflow

```bash
# Lint
python -m ruff check .

# Type check
python -m mypy neural/ --ignore-missing-imports

# Run tests
python -m pytest tests/ -v

# Security scan
python -m bandit -r neural/ -ll
python -m pip_audit -l

# Run specific test
pytest tests/parser/ -v
```

## CI/CD Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **essential-ci.yml** | Push, PR, Nightly | Lint, test, security |
| **release.yml** | Version tags | PyPI & GitHub release |
| **codeql.yml** | Weekly, PR | Security analysis |
| **validate-examples.yml** | Daily, Changes | Example validation |

## Repository Structure

```
neural/
├── cli/              # Command-line interface
├── parser/           # DSL parser (Lark)
├── code_generation/  # TF/PyTorch/ONNX generators
├── shape_propagation/# Shape validation
├── dashboard/        # NeuralDbg debugger
├── hpo/             # Hyperparameter optimization
├── automl/          # Neural Architecture Search
├── integrations/    # Cloud platform connectors
├── tracking/        # Experiment tracking
└── no_code/         # No-code web interface

examples/            # DSL examples
docs/                # Documentation
tests/               # Test suite
```

## Essential Documentation

| File | Purpose |
|------|---------|
| README.md | Project overview |
| GETTING_STARTED.md | Quick start guide |
| INSTALL.md | Installation details |
| CONTRIBUTING.md | Contribution guidelines |
| AGENTS.md | Agent/automation guide |
| CHANGELOG.md | Version history |
| SECURITY.md | Security policy |
| LICENSE.md | MIT License |

## DSL Syntax Basics

```yaml
network ModelName {
  input: (height, width, channels)
  
  layers:
    Conv2D(filters, (kernel_h, kernel_w), activation)
    MaxPooling2D((pool_h, pool_w))
    Dense(units, activation)
    Dropout(rate)
    Output(units, activation)
  
  loss: "loss_function"
  optimizer: OptimizerName(learning_rate=value)
  
  train {
    epochs: N
    batch_size: M
  }
}
```

## Common Layer Types

- **Conv2D**(filters, kernel, activation)
- **MaxPooling2D**(pool_size)
- **Dense**(units, activation)
- **LSTM**(units, return_sequences)
- **Dropout**(rate)
- **BatchNormalization**()
- **Flatten**()
- **Output**(units, activation)

## Backends

- **tensorflow** - TensorFlow/Keras
- **pytorch** - PyTorch
- **onnx** - ONNX format

## Export Formats

- **onnx** - Cross-platform inference
- **tflite** - TensorFlow Lite (mobile/edge)
- **torchscript** - PyTorch production
- **savedmodel** - TensorFlow Serving

## Troubleshooting

### Shape Mismatch Errors
```bash
# Visualize to debug shapes
neural visualize model.neural --show-shapes
```

### Import Errors
```bash
# Check dependencies
pip list | grep -E 'torch|tensorflow|onnx'

# Reinstall with backends
pip install -e ".[backends]"
```

### Test Failures
```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test file
pytest tests/parser/test_parser.py -v
```

## Cleanup

If the repository needs cleanup:

```bash
# Run cleanup scripts
python run_cleanup.py

# Or individual scripts
python cleanup_redundant_files.py
python cleanup_workflows.py
```

## Getting Help

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Discord**: https://discord.gg/KFku4KvS
- **Issues**: https://github.com/Lemniscate-world/Neural/issues
- **Twitter**: [@NLang4438](https://x.com/NLang4438)

## Links

- **PyPI**: https://pypi.org/project/neural-dsl/
- **GitHub**: https://github.com/Lemniscate-world/Neural
- **Documentation**: See [docs/](docs/) directory
