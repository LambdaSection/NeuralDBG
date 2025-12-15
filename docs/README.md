# Neural DSL Documentation

Welcome to the Neural DSL documentation directory. This guide helps you navigate the organized documentation structure.

## Quick Navigation

### Essential Reading
- [**Quick Reference**](quick_reference.md) - **Consolidated quick-start guide** ⭐
- [Getting Started](../README.md#installation) - Installation and first steps
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Changelog](../CHANGELOG.md) - Version history and changes

### Project Guidance
- [**FOCUS.md**](FOCUS.md) - **Start here!** Project scope, boundaries, and philosophy
- [TYPE_SAFETY.md](TYPE_SAFETY.md) - Type checking guidelines and standards

### Feature Documentation
- [DSL Language Reference](dsl.md) - Complete syntax guide
- [Deployment Guide](deployment.md) - Production export options
- [AI Integration Guide](ai_integration_guide.md) - Natural language model generation

## Directory Structure

```
docs/
├── README.md                    # This file
├── quick_reference.md           # Consolidated quick reference ⭐
├── FOCUS.md                     # Project scope and boundaries ⭐
├── TYPE_SAFETY.md               # Type checking guidelines
│
├── dsl.md                       # DSL language reference
├── deployment.md                # Deployment guide
├── installation.md              # Installation guide
├── migration.md                 # Migration guide
├── troubleshooting.md           # Troubleshooting guide
│
├── api/                         # API documentation
├── blog/                        # Blog posts
├── deployment/                  # Deployment-specific guides
├── examples/                    # Example guides
├── features/                    # Feature documentation
├── images/                      # Documentation images
├── mlops/                       # MLOps documentation
├── releases/                    # Release notes
├── social/                      # Social media content
└── tutorials/                   # User tutorials
```

## Documentation Philosophy

Our documentation follows these principles:

1. **Clarity First**: Clear, concise explanations over comprehensiveness
2. **Examples Driven**: Show, don't just tell
3. **Up-to-Date**: If it's documented, it should work
4. **Organized**: Easy to find what you need
5. **Honest**: Clear about limitations and features

## Core vs Peripheral

### Core Documentation (Priority 1)
These docs cover essential, actively maintained features:
- DSL syntax and parser
- Code generation (TensorFlow, PyTorch, ONNX)
- Shape propagation and validation
- CLI commands
- NeuralDbg dashboard

### Semi-Core Documentation (Priority 2)
Supported but not the primary focus:
- HPO (hyperparameter optimization)
- AutoML (simplified architecture search)
- Experiment tracking
- Visualization

### Experimental Features (Priority 3)
Features that are experimental or being evaluated:
- AI model generation
- Cloud integrations
- No-code interface

## Finding What You Need

### "I want to..."

**...get started quickly**
→ [Quick Reference](quick_reference.md) or [README](../README.md)

**...understand the DSL syntax**
→ [dsl.md](dsl.md)

**...compile my model to PyTorch/TensorFlow**
→ [Quick Reference](quick_reference.md#common-commands) or [deployment.md](deployment.md)

**...debug my model**
→ [Quick Reference](quick_reference.md#neuraldbg-dashboard-quick-start)

**...deploy to production**
→ [deployment.md](deployment.md)

**...contribute code**
→ [CONTRIBUTING.md](../CONTRIBUTING.md) + [TYPE_SAFETY.md](TYPE_SAFETY.md)

**...understand project scope**
→ [FOCUS.md](FOCUS.md) ⭐

**...optimize hyperparameters**
→ [examples/hpo_guide.md](examples/hpo_guide.md)

## Contributing to Documentation

### Guidelines

1. **Location**: Put docs in the appropriate subdirectory
2. **Format**: Use Markdown with clear headers
3. **Examples**: Include working code examples
4. **Links**: Use relative links within docs
5. **Updates**: Update this README when adding major docs

### Documentation PRs

When submitting documentation:
- Test all code examples
- Check links work
- Add entry to this README if appropriate
- Follow the existing style and tone
- Keep it concise

## Getting Help

If you can't find what you need:

1. **Search the docs**: Use GitHub's search or grep
2. **Check examples**: Look in `examples/` directory
3. **Ask on Discord**: [Join our Discord](https://discord.gg/KFku4KvS)
4. **Open a discussion**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
5. **Report missing docs**: Open an issue with "documentation" label

## Maintenance

This documentation is maintained by the Neural DSL team and community contributors. 

**Last major reorganization**: January 2025 (v0.4.0 cleanup)

**Next review scheduled**: Q2 2025

---

**Quick Links**:
[Home](../README.md) |
[Quick Reference](quick_reference.md) |
[Focus](FOCUS.md) |
[Contributing](../CONTRIBUTING.md) |
[Discord](https://discord.gg/KFku4KvS)
