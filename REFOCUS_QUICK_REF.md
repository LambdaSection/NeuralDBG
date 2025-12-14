# Neural DSL Refocusing - Quick Reference

## TL;DR
Neural DSL v0.4.0 is now **focused** on one thing: **DSL-based neural network definition with shape validation and multi-backend compilation**. Removed 25+ feature directories that diluted focus.

## What Changed

### Core (Kept)
```
✅ DSL Parser (Lark)
✅ Shape Validation
✅ Code Generation (PyTorch/TensorFlow/ONNX)
✅ Visualization
✅ CLI (compile, validate, visualize)
```

### Optional (Kept)
```
✅ HPO (Optuna)
✅ AutoML/NAS
✅ Debugging Dashboard
```

### Removed
```
❌ Teams/Billing/Marketplace
❌ MLOps (registry, deployment, A/B)
❌ Cloud Integrations
❌ API Server
❌ Monitoring/Tracking
❌ Collaboration/Federated
❌ No-code GUI
❌ LLM/Chat Features
❌ 15+ other features
```

## Quick Migration

| If You Used... | Now Use... |
|----------------|------------|
| `neural cloud *` | boto3, google-cloud, azure |
| `neural track *` | MLflow, W&B |
| `neural marketplace *` | HuggingFace Hub |
| `neural mlops *` | MLflow, Kubeflow |
| `neural cost *` | Cloud cost dashboards |
| `neural aquarium` | MLflow UI |
| `neural no-code` | Neural DSL (it's simple!) |
| Teams/Billing | Separate service |

## Installation

```bash
# Before (v0.3.x) - 50+ dependencies
pip install neural-dsl[full]  # Everything

# After (v0.4.0) - 15 core dependencies
pip install neural-dsl              # Core
pip install neural-dsl[backends]    # + TF/PyTorch/ONNX
pip install neural-dsl[hpo]         # + Optuna
pip install neural-dsl[automl]      # + NAS
pip install neural-dsl[full]        # All focused features
```

## Core Workflow (Unchanged)

```bash
# Your DSL files still work exactly the same
neural compile model.neural --backend pytorch
neural validate model.neural
neural visualize model.neural --format png
```

## Why?

**Focus over features.** Neural DSL now does one thing exceptionally well instead of many things poorly. 

- 70% less code
- 70% fewer dependencies
- Clearer value proposition
- Easier to maintain
- Faster to use

## Philosophy

> "Do one thing and do it well" - Unix

Neural DSL is now a **specialized tool**, not a Swiss Army knife.

## More Details

- **Full rationale**: [REFOCUS.md](REFOCUS.md)
- **Implementation**: [REFOCUSING_IMPLEMENTATION_SUMMARY.md](REFOCUSING_IMPLEMENTATION_SUMMARY.md)
- **Cleanup guide**: [CLEANUP_REFOCUS.md](CLEANUP_REFOCUS.md)
- **CLI changes**: [CLI_REFOCUS.md](CLI_REFOCUS.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md) (v0.4.0 section)

## Questions?

**Q: Will removed features come back?**  
A: No. They diluted the project's identity.

**Q: What if I need those features?**  
A: Use specialized tools (MLflow, cloud SDKs, etc.)

**Q: Is this a breaking change?**  
A: For core DSL users: No. For removed feature users: Yes.

**Q: Can I still use v0.3.x?**  
A: Yes, but v0.4.0+ is the future direction.

## Support

- Documentation: [docs/](docs/)
- Discord: https://discord.gg/KFku4KvS
- Issues: https://github.com/Lemniscate-world/Neural/issues
