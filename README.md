# NeuralDBG

A minimal debugger for deep learning training that provides **causal introspection** into neural network training dynamics. Understand *why* your model failed during training, not just *that* it failed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

NeuralDBG treats training as an **execution trace** rather than a black box. It captures the complete temporal evolution of your model's internal state, enabling researchers to:

- **Trace gradient flow** through network layers
- **Detect training instabilities** (vanishing/exploding gradients)
- **Inspect tensor transformations** over time
- **Query historical states** for causal analysis

Unlike traditional monitoring tools (TensorBoard, Weights & Biases), NeuralDBG focuses on **debugging training failures** rather than tracking experiments.

## Key Features

- **Event Tracing**: Captures forward/backward passes with tensor snapshots
- **Gradient Breakpoints**: Automatic detection of vanishing/exploding gradients
- **Temporal Inspection**: Query tensor states across training steps
- **Non-Invasive**: Wraps existing PyTorch training loops without code changes
- **Minimal API**: Focused on causal understanding, not visualization

## Quick Start

### Installation

```bash
pip install neuraldbg
```

### Basic Usage

```python
import torch
import torch.nn as nn
from neuraldbg import NeuralDbg

# Your existing model and training setup
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Wrap your training loop
with NeuralDbg(model) as dbg:
    for step, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Capture state after each training step
        dbg.capture_step(step)

        # Check for gradient issues
        if dbg.has_vanishing_gradients():
            print(f"Vanishing gradients detected at step {step}")
            # Inspect the problematic layer
            gradient_info = dbg.inspect_layer('linear1')
            break
```

### Inspection API

```python
# Query tensor state at specific step
tensor_state = dbg.inspect(layer_name='conv1', step=42)

# Get gradient history for a layer
gradients = dbg.get_gradient_history('fc1')

# Check gradient health
health_report = dbg.analyze_gradient_health()
```

## Architecture

### Core Components

- **Event System**: Records each computation step with input/output tensors and gradients
- **Tensor Versioning**: Maintains historical snapshots for temporal analysis
- **Breakpoint Engine**: Monitors for pathological training conditions
- **Query Interface**: Provides programmatic access to captured state

### Event Structure

Each captured event contains:
- Step index in training sequence
- Layer/parameter identifier
- Input tensor snapshot (detached clone)
- Output tensor snapshot (detached clone)
- Gradient snapshot (when available)

## Target Users

- **ML Researchers** investigating training failures
- **PhD Students** debugging novel architectures
- **Research Engineers** optimizing training pipelines

*Not intended for production monitoring or no-code users.*

## Limitations (MVP Scope)

- PyTorch only
- Single model architecture per session
- Focus on gradient-based failures
- Command-line interface only
- Memory-intensive for large models

## Contributing

This is an MVP focused on proving the concept of causal training debugging. Contributions should align with the core mission of understanding training failures.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Documentation

- [PLAN.md](PLAN.md) - Detailed MVP specification and design rationale
- [logic_graph.md](logic_graph.md) - System architecture and data flow

## Citation

If you use NeuralDBG in your research, please cite:

```bibtex
@misc{neuraldbg2025,
  title={NeuralDBG: A Minimal Debugger for Deep Learning Training},
  author={SENOUVO Jacques-Charles Gad},
  year={2025},
  url={https://github.com/Lemniscate-world/Neural}
}
```
