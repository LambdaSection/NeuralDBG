# Neural DSL - Agent Guide

## Overview
Neural DSL is a focused, specialized tool for defining neural networks with a declarative syntax and multi-backend compilation. This is NOT a Swiss Army knife - it does one thing exceptionally well: DSL-based neural network definition with shape validation and code generation.

## Setup
```bash
python -m venv .venv                    # Create venv
.\.venv\Scripts\Activate                # Windows activation
pip install -e .                        # Install core (DSL, parser, shape validation)
pip install -e ".[backends]"            # Add TensorFlow, PyTorch, ONNX support
pip install -e ".[visualization]"       # Add visualization capabilities
pip install -e ".[hpo]"                 # Add hyperparameter optimization
pip install -e ".[automl]"              # Add AutoML/NAS
pip install -e ".[full]"                # All features
pip install -r requirements-dev.txt     # Development tools
```

## Core Dependencies
- **Core**: click, lark, numpy, pyyaml (DSL parsing and validation)
- **Backends**: torch, tensorflow, onnx (code generation targets)
- **Visualization**: matplotlib, graphviz, networkx (network diagrams)
- **HPO**: optuna, scikit-learn (hyperparameter optimization)
- **AutoML**: optuna, scikit-learn, scipy (neural architecture search)
- **Dashboard**: dash, flask (debugging interface)
- **Dev**: pytest, ruff, pylint, mypy, pre-commit

## Commands
- **Build**: N/A (pure Python)
- **Lint**: `python -m ruff check .`
- **Test**: `python -m pytest tests/ -v`
- **Dev Server**: `python neural/dashboard/dashboard.py` (debugging dashboard on :8050)

## Tech Stack
- **Language**: Python 3.8+ with type hints
- **Core**: Lark (DSL parser), Click (CLI)
- **Backends**: TensorFlow, PyTorch, ONNX (all optional)
- **Tools**: pytest, ruff, mypy

## Architecture
Core modules (always relevant):
- `neural/parser/` - DSL parser and AST transformer
- `neural/code_generation/` - Multi-backend code generators
- `neural/shape_propagation/` - Shape validation and propagation
- `neural/cli/` - CLI commands
- `neural/visualization/` - Network visualization
- `neural/utils/` - Shared utilities

Optional modules (install as needed):
- `neural/hpo/` - Hyperparameter optimization
- `neural/automl/` - Neural Architecture Search
- `neural/dashboard/` - Debugging dashboard
- `neural/training/` - Training utilities
- `neural/metrics/` - Metric computation

## Code Style
- Follow PEP 8, 100-char line length
- Use type hints
- Numpy-style docstrings
- No comments unless complex logic requires context
- Functional over classes where reasonable

## Philosophy
Neural DSL is focused on doing ONE thing exceptionally well: providing a clean DSL for neural network definition with strong guarantees (shape validation) and flexibility (multi-backend compilation). All features support this core mission.
