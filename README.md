# NeuralDBG

Training failed? NeuralDBG tells you **why**. It hooks into your PyTorch training loop, detects what went wrong (vanishing gradients, exploding gradients, data anomalies), and pinpoints the exact layer and step — so you fix it in seconds, not hours.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/neuraldbg.svg)](https://pypi.org/project/neuraldbg/)
[![CI](https://github.com/LambdaSection/NeuralDBG/actions/workflows/ci.yml/badge.svg)](https://github.com/LambdaSection/NeuralDBG/actions/workflows/ci.yml)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lambdasection.github.io/NeuralDBG/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
[![W&B](https://img.shields.io/badge/-W%26B-FFCC00?logo=weightsandbiases&logoColor=black)](https://wandb.ai)

<p align="center">
  <img src="outputs/neuraldbg_workflow.gif" alt="NeuralDBG Workflow" width="800"/>
</p>

## Overview

NeuralDBG treats training as a **semantic trace of learning dynamics** rather than a black box. It extracts meaningful events and provides causal hypotheses about training failures, enabling researchers to:

- **Identify gradient health transitions** (stable -> vanishing/saturated)
- **Detect activation regime shifts** (normal -> saturated/dead)
- **Detect optimizer instability** (loss plateaus, spikes, divergence)
- **Catch data anomalies** (NaN, Inf, distribution shifts)
- **Track propagation of instabilities** through network layers
- **Generate ranked causal explanations** for training failures

Unlike traditional monitoring tools (TensorBoard, Weights & Biases), NeuralDBG focuses on **causal inference** rather than metric tracking.

## Neural Suite

| Component | Availability | Role |
|-----------|--------------|------|
| **NeuralDBG** | Public ([PyPI](https://pypi.org/project/neuraldbg/)) | Causal diagnostics in your training loop |
| **Diagnostic Workspace** | Private beta | Visual causal graphs and hypothesis explorer |
| **Neural Agent** | Private beta | Auto-remediation from causal hypotheses |

Request early access: [open an issue](https://github.com/LambdaSection/NeuralDBG/issues/new?labels=suite-access) with label `suite-access`.

## Try it in 60 seconds (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)

Or locally: `python examples/quickstart_interactive.py`

## Public benchmark

Reproducible causal accuracy on synthetic failures ([details](benchmark_public/README.md)):

```bash
python -m benchmark_public.run
```

Latest results: [benchmark_public/results.json](benchmark_public/results.json)

<p align="center">
  <img src="outputs/vanishing_gradient_demo.gif" alt="Vanishing Gradient Detection" width="800"/>
</p>

## Why NeuralDBG?

| Feature | TensorBoard / W&B | NeuralDBG |
|---|---|---|
| **What it shows** | Graphs of loss/accuracy over time | **Why** the loss spiked or vanished |
| **Diagnosis** | Manual inspection of curves | **Automated causal hypotheses** |
| **Actionable?** | You guess the fix | Suggests root causes (LR, Init, Data) |
| **Integration** | Separate dashboard | **One line of code** in your loop |
| **Privacy** | Data sent to cloud | **100% Local** (unless you opt-in) |

> "TensorBoard tells you *when* it failed. NeuralDBG tells you *why*."

## Latest: v1.5.0 (July 5, 2026)

- **Tier 1 Black-Swan detection: 96%** — GNN 88%, MoE 100%, Diffusion 100% (104/108 bugs detected)
- **Tier 2 Black-Swan detection: 94%** — FlashAttention 100%, Neural ODE 100%, Quantized 83% (102/108)
- **200-architecture validation** — 99.2% detection (1,190/1,200 bug injections) across 6 families (MLP/CNN/RNN/Transformer/Hybrid/Black-Swan); canonical numbers in [paper_number_audit.md](docs/paper_number_audit.md)
- **NeuralPrune v0.1** — Non-destructive redundancy diagnostic: dead neurons, low-rank weights, quantization opportunities
- **LSTM/GRU support** — Forward hooks unwrap RNN output tuples. Per-gate gradient tracking (input/forget/cell/output)
- **Architecture fuzzer** — 0 crashes, 16/20 injected bugs detected (4 analyzed residuals: 3 weak-signal vanishing + 1 noise-masked NaN); deterministic (seed 42, audit §6)
- **Stress test suite** — 15/15 tests pass: 10x gradients, NaN/Inf, fp16, 100-layer depth, 1K token attention
- **GPU classifier** — Qwen2-0.5B + LoRA (v5, 108 examples, 6 families); changelog "93.7% accuracy" **withdrawn** — re-verification scored 13.9% (15/108) with category collapse ([audit §2.7](docs/paper_number_audit.md))
- **Aquarium web dashboard** — Zero-dependency HTML causal viewer. [Open Aquarium](https://lambdasection.github.io/NeuralDBG/docs/aquarium.html)
- **2 upstream diagnostic test PRs** — svdvals NaN (#188053) + gradient health tests (#188923); F.normalize retiré (comportement voulu confirmé par albanD)
- **100% detection** on DeepMLP (6/6 bugs) | **96% Tier 1** black-swans | **94% Tier 2** black-swans
- **2 upstream PRs** (open) to PyTorch | **CI benchmark workflow** on GitHub Actions
- **E2E RNN pipeline** — detect→diagnose→fix→validate on LSTM. 2/4 bugs auto-fixed.

## Key Features

- **Semantic Event Extraction**: Detects meaningful transitions in training dynamics
- **Causal Compression**: Identifies first occurrences and propagation patterns
- **Post-Mortem Reasoning**: Provides ranked hypotheses about failure causes
- **Optimizer Instability Detection**: Tracks loss plateaus, spikes, and divergence
- **Data Anomaly Detection**: Catches NaN, Inf, and distribution shifts in inputs
- **Event Collapsing**: Merges sequential events into summary traces
- **Compiler-Aware**: Operates at module boundaries to survive torch.compile
- **Non-Invasive**: Wraps existing PyTorch training loops without code changes
- **Minimal API**: Focused on explanations, not raw data dumps
- **Diagnostic package export**: JSON export for the Neural Suite visualizer (private beta)

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
        dbg.record_loss(loss.item())
        optimizer.step()

# After training failure (or anytime), query for explanations
explanations = dbg.explain_failure()
if explanations:
    print(explanations[0])  # "Gradient vanishing originated in layer 'linear1' at step 234..."
else:
    print("No failure detected — training looks healthy.")
```

### Inference API

```python
# Get ranked causal hypotheses for the failure
hypotheses = dbg.get_causal_hypotheses()

# Query specific causal chains
chain = dbg.trace_causal_chain('vanishing_gradients')

# Check for coupled failures
couplings = dbg.detect_coupled_failures()

# Export diagnostic package (JSON) for Neural Suite visualizer
dbg.export_aquarium_package('debug_session.json')
```

### Optimizer Instability Detection

```python
with NeuralDbg(model) as dbg:
    for step in range(num_steps):
        dbg.step = step
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# Detect loss plateaus, spikes, or divergence
hypotheses = dbg.explain_failure("optimizer_instability")
for h in hypotheses:
    print(h.description)
```

### Data Anomaly Detection

Data anomalies (NaN, Inf, distribution shifts) are detected automatically from layer inputs during the forward pass:

```python
with NeuralDbg(model) as dbg:
    # ... training loop ...
    pass

hypotheses = dbg.explain_failure("data_anomaly")
for h in hypotheses:
    print(h.description)  # "NaN values detected in input to layer 'linear1'..."
```

## Supported Architectures

Validated on **200 architectures, 1,200 bug injections** (combinatorial sweep, 2026-10-03 P2b run; canonical, artifact-verifiable figures per [paper_number_audit.md](docs/paper_number_audit.md)) plus 4 out-of-sample production architectures.

| Family | Configs | Detection | Status |
|--------|:-------:|:---------:|:------:|
| MLP (deep, residual, bottleneck) | 38 | **98.7%** (225/228) | ✅ |
| CNN (Conv2d, varying kernel/depth/norm) | 33 | **100%** (198/198) | ✅ |
| Transformer (encoder, varying heads/depth) | 33 | **100%** (198/198) | ✅ |
| RNN (LSTM/GRU, varying depth/width/bidir) | 33 | **100%** (198/198) | ✅ |
| Hybrid | 30 | **100%** (180/180) | ✅ |
| Black-Swan (GNN, MoE, Diffusion, RL, RAG, FlashAttn, NeuralODE) | 33 | **96.5%** (191/198) | ✅ |
| **Overall** | **200** | **99.2% (1,190/1,200)** | |

RNN reached 100% after fixing a builder artifact: `out[:, -1, :]` zeroed the reverse-direction `W_hh` gradient on bidirectional LSTM (last reverse step = first step from h0=0), so healthy bi-LSTMs looked "vanishing". Fix: temporal mean pooling + family-adaptive thresholds with inclusive comparison (audit §6). Hybrid reached 100% after fixing a generator bug (LSTM shape misuse crashed 18 RNN-composite configs at step 0). Remaining 10/1200 analyzed without tuning: 3 shallow-MLP weak-signal vanishing, 1 GNN off-by-one, 6 FlashAttn with noisy healthy baselines. The engine's saturation heuristic is now conditioned on bounded module outputs (Sigmoid/Tanh/Softmax), removing false-saturated baselines (P2b, audit §6).

### By Bug Type (200 configs each)

| Bug | Detection | Difficulty |
|-----|:---------:|:----------:|
| Optimizer divergence | **100%** (200/200) | Easy |
| Exploding gradients | **100%** (200/200) | Easy |
| Zero initialization | **100%** (200/200) | Moderate |
| Vanishing gradients | **98.5%** (197/200) | Moderate |
| NaN in data | **98.5%** (197/200) | Moderate |
| Dead activations (bias) | **98%** (196/200) | Moderate |

### Out-of-Sample (4 production architectures, never seen in calibration)

ResNet-18 **4/5** (vanishing miss: sigmoid-stage swap asymptomatic on synthetic data, 0 events at 60 steps) · ViT-Tiny **5/5** · EfficientNet-B0 **5/5** · Mamba-Mini **5/5** (arch-agnostic injectors, 0 crashes). Overall **19/20 injected bugs (95%)** — healthy baselines: ResNet-18 0 events, Mamba-Mini 0, EfficientNet-B0 5, ViT-Tiny 123 (noisy, known limitation). `python validate_oos.py`. History: an initial "Mamba 0/6 crash" verdict was a builder bug (`torch.silu`, removed in torch ≥2.13), fixed and re-run; ResNet-only injectors that were no-ops elsewhere were made arch-agnostic; see [paper_number_audit.md](docs/paper_number_audit.md) §2.5b–2.5c and §6.

Reproduce: `python validate_combinatorial.py --full` (200 configs, ~4 min on CPU).

Detailed results: [combinatorial_results.json](combinatorial_results.json) |

## Supported Failure Types

| Failure Type | Description |
|---|---|
| `vanishing_gradients` | Root cause + saturation coupling |
| `exploding_gradients` | First layer to explode |
| `dead_neurons` | Neuron death in activation layers |
| `saturated_activations` | Activation saturation patterns |
| `optimizer_instability` | Loss plateaus, spikes, divergence |
| `data_anomaly` | NaN/Inf/distribution shift in inputs |

## Architecture

### Core Components

- **Semantic Event Extractor**: Detects meaningful transitions in learning dynamics
- **Causal Compressor**: Identifies patterns and propagation in training failures
- **Post-Mortem Reasoner**: Generates ranked hypotheses about failure causes
- **Compiler-Aware Monitor**: Operates at safe boundaries for optimization compatibility

### Event Types

| Event Type | Source | Detects |
|---|---|---|
| `gradient_health_transition` | Backward hooks | Vanishing, exploding, saturated gradients |
| `activation_regime_shift` | Forward hooks | Dead neurons, saturated activations |
| `optimizer_instability` | `record_loss()` | Loss plateaus, spikes, divergence |
| `data_anomaly` | Forward hooks (inputs) | NaN, Inf, distribution shifts |

## Editions

| Edition | Package | License | Features |
|---|---|---|---|
| **Core** | `pip install neuraldbg` | MIT | Hooks, events, export JSON, basic heuristics |
| **Engine** | `pip install neuraldbg-engine` | Proprietary | Full causal inference, detailed hypotheses, coupling detection |

The Core edition works standalone with basic heuristic fallbacks. Install the Engine for advanced causal reasoning.

## Target Users

- **ML Researchers** seeking causal explanations for training failures
- **PhD Students** analyzing learning dynamics in novel architectures
- **Research Engineers** understanding optimization instabilities

## Limitations

- PyTorch only
- Focus on semantic events, not tensor inspection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Developer Setup

```bash
make bootstrap
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Release Methodology

**Versioning**: SemVer (`MAJOR.MINOR.PATCH`). Changelog in `CHANGELOG.md` (Keep a Changelog).
**Cadence**: Monthly minor releases; patches as needed for fixes/security.
**Process**: `git tag vX.Y.Z` → `.github/workflows/publish.yml` → PyPI (`neuraldbg`) + GitHub Release.
**Support**: latest minor (`1.5.x`) actively maintained; see `SECURITY.md` and `GOVERNANCE.md`.
**Governance**: `GOVERNANCE.md` (2 maintainers), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.github/CODEOWNERS`.

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Documentation

- [Landing page](https://lambdasection.github.io/NeuralDBG/) — branding and suite overview
- [PyTorch support matrix](docs/PYTORCH_SUPPORT.md)
- [CHANGELOG.md](CHANGELOG.md) - Version history and notable changes
- [logic_graph.md](logic_graph.md) - System architecture and data flow
- [docs/PHASE2_DOGFOODING.md](docs/PHASE2_DOGFOODING.md) - Detailed dogfooding scenarios

## Citation

If you use NeuralDBG in your research, please cite:

```bibtex
@misc{neuraldbg2026,
  title={NeuralDBG: A Causal Inference Engine for Deep Learning Training Dynamics},
  author={SENOUVO Jacques-Charles Gad},
  year={2026},
  url={https://github.com/LambdaSection/NeuralDBG}
}
```
