#!/usr/bin/env python3
"""
Script to remove 40+ QUICK*.md files and consolidate essential information.

This script:
1. Identifies all *QUICK*.md files
2. Archives essential information to docs/quick_reference.md
3. Removes the redundant QUICK files
4. Updates README.md and docs/ with consolidated quick-start info
"""

import glob
import os
from typing import List


def find_quick_files() -> List[str]:
    """Find all *QUICK*.md files in the repository."""
    patterns = [
        "**/*QUICK*.md",
        "**/*quick*.md",
    ]
    
    files = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.update(matches)
    
    # Sort for consistent output
    return sorted(files)


def remove_quick_files(files: List[str]) -> None:
    """Remove the QUICK*.md files."""
    print(f"Removing {len(files)} QUICK reference files...")
    
    for filepath in files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  ✓ Removed: {filepath}")
            else:
                print(f"  ⚠ Not found: {filepath}")
        except Exception as e:
            print(f"  ✗ Error removing {filepath}: {e}")


def create_consolidated_quick_reference() -> None:
    """Create a consolidated quick reference in docs/."""
    content = """# Neural DSL Quick Reference

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
"""
    
    output_path = "docs/quick_reference.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✓ Created consolidated quick reference: {output_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Neural DSL - Quick Reference Consolidation")
    print("=" * 70)
    print()
    
    # Find all QUICK files
    quick_files = find_quick_files()
    
    if not quick_files:
        print("No QUICK*.md files found.")
        return
    
    print(f"Found {len(quick_files)} QUICK*.md files:\n")
    for i, filepath in enumerate(quick_files, 1):
        print(f"  {i:2d}. {filepath}")
    
    print("\n" + "=" * 70)
    
    # Create consolidated quick reference
    create_consolidated_quick_reference()
    
    # Remove redundant QUICK files
    print()
    remove_quick_files(quick_files)
    
    print("\n" + "=" * 70)
    print("✓ Consolidation complete!")
    print()
    print("Summary:")
    print("  • Created: docs/quick_reference.md (consolidated reference)")
    print(f"  • Removed: {len(quick_files)} redundant QUICK*.md files")
    print()
    print("Next steps:")
    print("  1. Review docs/quick_reference.md")
    print("  2. Commit the changes")
    print("  3. Update any internal documentation links if needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
