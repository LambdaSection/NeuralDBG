# Migration Guide: Upgrading to Neural DSL v0.3.0

This guide provides comprehensive instructions for upgrading from Neural DSL v0.2.x to v0.3.0, including breaking changes, new features, and upgrade steps.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Breaking Changes](#breaking-changes)
- [New Features](#new-features)
- [Upgrade Steps](#upgrade-steps)
- [Feature Migration](#feature-migration)
- [Troubleshooting](#troubleshooting)
- [Common Scenarios](#common-scenarios)

---

## Overview

### What's New in v0.3.0

Neural DSL v0.3.0 introduces three major feature sets:

1. **🤖 AI-Powered Development**: Natural language to DSL conversion
2. **🚀 Production Deployment**: ONNX, TFLite, TorchScript export with serving
3. **🔄 Automation System**: Automated releases, testing, and maintenance

### Compatibility

✅ **Fully backward compatible** - All v0.2.x code works in v0.3.0 without modifications

### Supported Versions

- Python: 3.8, 3.9, 3.10, 3.11, 3.12
- TensorFlow: 2.6+
- PyTorch: 1.10+
- ONNX: 1.10+

---

## Quick Start

### For Regular Users

If you just want to upgrade and use the new features:

```bash
# Update to v0.3.0
pip install --upgrade neural-dsl

# Verify installation
neural --version

# Try new AI features (optional)
pip install neural-dsl[ai]
neural ai "Create a CNN for MNIST"

# Try new export features
neural export examples/mnist.neural --format onnx
```

### For Contributors/Developers

If you're contributing to Neural DSL:

```bash
# Update repository
git pull origin main

# Reinstall with dev dependencies
pip install -r requirements-dev.txt

# Verify tests pass
python -m pytest tests/ -v
```

---

## Breaking Changes

**None** - v0.3.0 maintains full backward compatibility with v0.2.x.

### Deprecated Features

No features have been deprecated in v0.3.0. All v0.2.x features remain available and fully supported.

---

## New Features

### 1. AI-Powered Natural Language Model Creation

Create models using natural language descriptions.

#### Before (v0.2.x)

```python
# Had to write DSL manually
dsl_code = """
network MNIST {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
"""
```

#### After (v0.3.0)

```python
from neural.ai.ai_assistant import NeuralAIAssistant

# Use natural language
assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST with 32 filters")
dsl_code = result['dsl_code']

# Or use CLI
# neural ai "Create a CNN for MNIST with 32 filters"
```

#### Migration Path

Your existing DSL files work as-is. The AI features are purely additive:

```python
# Old way still works
neural compile my_model.neural --backend tensorflow

# New way also available
neural ai "Create a CNN for MNIST" --output my_model.neural
neural compile my_model.neural --backend tensorflow
```

### 2. Production Model Deployment

Export models to production formats with optimization.

#### Before (v0.2.x)

Manual export with framework-specific code:

```python
# TensorFlow
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# PyTorch
import torch
model = torch.load('model.pt')
torch.jit.trace(model, example_input).save('model_traced.pt')
```

#### After (v0.3.0)

Unified export interface for all formats:

```bash
# ONNX with optimization
neural export model.neural --format onnx --optimize

# TFLite with quantization
neural export model.neural --backend tensorflow --format tflite \
    --quantize --quantization-type int8

# TorchScript
neural export model.neural --backend pytorch --format torchscript

# Complete deployment setup
neural export model.neural --backend tensorflow --format savedmodel \
    --deployment tfserving --model-name my_model
```

#### Migration Path

1. Replace manual export code with `neural export` command
2. Add optimization flags for better performance
3. Use deployment flags for complete serving setup

### 3. Automation System

Automate development workflows.

#### Before (v0.2.x)

Manual processes for:
- Version bumping in multiple files
- Changelog updates
- Release creation
- Blog post writing and publishing
- Example validation
- Test reporting

#### After (v0.3.0)

Automated with single commands:

```bash
# Automated release (maintainers)
python scripts/automation/master_automation.py --task release --version 0.3.0

# Automated blog posts
python scripts/automation/master_automation.py --task blog \
    --title "My Neural DSL Tutorial"

# Validate all examples
python scripts/automation/master_automation.py --task validate-examples

# Run tests with reporting
python scripts/automation/master_automation.py --task test
```

#### Migration Path

If you maintain Neural DSL or a fork:
1. Review `scripts/automation/` directory
2. Configure automation settings
3. Use master script for common tasks

---

## Upgrade Steps

### Step 1: Backup Current Setup

Before upgrading, backup your current environment:

```bash
# Save current packages
pip freeze > requirements_backup.txt

# Backup your Neural DSL projects
cp -r my_neural_projects my_neural_projects_backup
```

### Step 2: Upgrade Neural DSL

```bash
# For regular users
pip install --upgrade neural-dsl

# For full installation (all features)
pip install --upgrade neural-dsl[full]

# For contributors
git pull origin main
pip install -r requirements-dev.txt
```

### Step 3: Verify Installation

```bash
# Check version
neural --version
# Expected output: neural-dsl, version 0.3.0

# Test basic functionality
neural --help

# Test compilation
neural compile examples/mnist.neural --backend tensorflow
```

### Step 4: Install Optional Features

Install new features as needed:

```bash
# AI-powered development
pip install neural-dsl[ai]

# Or install specific components
pip install openai  # For OpenAI integration
pip install anthropic  # For Claude integration
# Or install Ollama for local LLM
```

### Step 5: Test Your Projects

```bash
# Test your existing DSL files
neural compile my_model.neural --backend tensorflow

# Test visualization still works
neural visualize my_model.neural

# Test debugging still works
neural debug my_model.neural
```

### Step 6: Try New Features

```bash
# Try AI features
neural ai "Create a CNN for MNIST"

# Try export features
neural export my_model.neural --format onnx --optimize

# Try TFLite export
neural export my_model.neural --backend tensorflow --format tflite --quantize
```

---

## Feature Migration

### Migrating to AI-Powered Development

#### Scenario 1: Interactive Model Creation

**Before:**
```bash
# Manual DSL writing in text editor
vim my_model.neural
```

**After:**
```bash
# Interactive AI assistant
neural ai --interactive
```

#### Scenario 2: Template-Based Models

**Before:**
```bash
# Copy and modify templates
cp templates/cnn_template.neural my_cnn.neural
# Edit manually
```

**After:**
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN like VGG16 for image classification")
with open('my_cnn.neural', 'w') as f:
    f.write(result['dsl_code'])
```

### Migrating to Deployment System

#### Scenario 1: ONNX Export

**Before:**
```python
# Framework-specific export
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
output_path = 'model.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

**After:**
```bash
# Single command
neural export model.neural --format onnx --optimize
```

#### Scenario 2: Mobile Deployment

**Before:**
```python
# Manual TFLite conversion
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**After:**
```bash
# Unified interface
neural export model.neural --backend tensorflow --format tflite \
    --quantize --quantization-type float16
```

#### Scenario 3: Production Serving

**Before:**
```bash
# Manual setup
mkdir -p serving/models/my_model/1
python -c "import tensorflow as tf; model = tf.keras.models.load_model('model.h5'); model.save('serving/models/my_model/1')"
# Write docker-compose.yml manually
# Write test client manually
# Configure TF Serving manually
```

**After:**
```bash
# Complete deployment setup
neural export model.neural --backend tensorflow --format savedmodel \
    --deployment tfserving --model-name my_model

# Generated:
# - deployment/tfserving/models/my_model/1/
# - deployment/tfserving/models.config
# - deployment/tfserving/docker-compose.yml
# - deployment/tfserving/test_client.py
```

### Migrating to Automation System

#### For Maintainers

**Before:**
```bash
# Manual release process
# 1. Update version in setup.py
# 2. Update version in __init__.py
# 3. Update CHANGELOG.md
# 4. Create git tag
# 5. Create GitHub release
# 6. Build and upload to PyPI
# 7. Write blog post
# 8. Post to social media
```

**After:**
```bash
# One command
python scripts/automation/master_automation.py --task release --version 0.3.0
```

#### For Contributors

**Before:**
```bash
# Manual testing
pytest tests/
# Manual validation
python examples/mnist.neural
# Manual reporting
```

**After:**
```bash
# Automated with reporting
python scripts/automation/master_automation.py --task test
python scripts/automation/master_automation.py --task validate-examples
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors for AI Features

**Error:**
```
ModuleNotFoundError: No module named 'neural.ai'
```

**Solution:**
```bash
pip install --upgrade neural-dsl
# AI features are built-in, no extra dependencies needed
# Optional: Install LLM providers
pip install openai  # For OpenAI
pip install anthropic  # For Claude
```

#### Issue 2: Export Command Not Found

**Error:**
```
Error: no such command "export"
```

**Solution:**
```bash
# Ensure you have v0.3.0 or later
neural --version

# If older version, upgrade
pip install --upgrade neural-dsl
```

#### Issue 3: ONNX Export Fails

**Error:**
```
ModuleNotFoundError: No module named 'onnx'
```

**Solution:**
```bash
# Install backends extra
pip install neural-dsl[backends]

# Or install ONNX directly
pip install onnx onnxruntime
```

#### Issue 4: TFLite Export Fails

**Error:**
```
ValueError: Quantization requires representative dataset
```

**Solution:**
```python
# Provide representative dataset for int8 quantization
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='tensorflow')
exporter.export_tflite(
    output_path='model.tflite',
    quantize=True,
    quantization_type='int8',
    representative_dataset=your_calibration_data
)
```

Or use dynamic quantization:
```bash
neural export model.neural --backend tensorflow --format tflite \
    --quantize --quantization-type dynamic
```

#### Issue 5: AI Features Not Working

**Error:**
```
AI features require LLM provider
```

**Solution:**
```python
# Use rule-based mode (no LLM required)
assistant = NeuralAIAssistant(use_llm=False)

# Or set up LLM provider
# Option 1: Ollama (free, local)
# Install from https://ollama.ai
# ollama pull llama2

# Option 2: OpenAI
# export OPENAI_API_KEY=your_key

# Option 3: Anthropic
# export ANTHROPIC_API_KEY=your_key
```

### Verification Checklist

After upgrading, verify:

- [ ] Version is 0.3.0: `neural --version`
- [ ] Basic compilation works: `neural compile examples/mnist.neural`
- [ ] Visualization works: `neural visualize examples/mnist.neural`
- [ ] AI features work: `neural ai "Create a simple model"`
- [ ] Export works: `neural export examples/mnist.neural --format onnx`
- [ ] Your existing projects work
- [ ] Tests pass: `python -m pytest tests/ -v` (for contributors)

---

## Common Scenarios

### Scenario 1: Data Scientist Using PyTorch

You primarily use PyTorch and want to try the new export features.

**Current Setup (v0.2.x):**
```bash
pip install neural-dsl
# Using basic DSL features
```

**Upgrade Path:**
```bash
# Update Neural DSL
pip install --upgrade neural-dsl

# Install backends if not already
pip install neural-dsl[backends]

# Try new export features
neural export my_model.neural --backend pytorch --format torchscript
neural export my_model.neural --format onnx --optimize
```

**What You Get:**
- TorchScript export for production
- ONNX export for framework interoperability
- Optimized inference models
- No changes to existing workflow

### Scenario 2: Student Learning Deep Learning

You're learning and want to quickly create models.

**Current Setup (v0.2.x):**
```bash
pip install neural-dsl
# Learning DSL syntax
```

**Upgrade Path:**
```bash
# Update Neural DSL
pip install --upgrade neural-dsl

# Start using AI features (no extra dependencies)
neural ai "Create a CNN for MNIST"
neural ai --interactive  # Interactive learning
```

**What You Get:**
- Natural language model creation
- Faster prototyping
- Learn by describing, not just coding
- No syntax errors to debug

### Scenario 3: ML Engineer Deploying Models

You need to deploy models to production.

**Current Setup (v0.2.x):**
```bash
pip install neural-dsl[full]
# Manual deployment setup
```

**Upgrade Path:**
```bash
# Update with full features
pip install --upgrade neural-dsl[full]

# Use new deployment features
neural export model.neural --backend tensorflow --format savedmodel \
    --deployment tfserving --model-name production_model

# Start serving
cd deployment/tfserving
docker-compose up
```

**What You Get:**
- Complete deployment configs
- Docker Compose setup
- Test client generation
- Optimized models
- Production-ready setup

### Scenario 4: Researcher With Complex Models

You build complex architectures and need flexibility.

**Current Setup (v0.2.x):**
```bash
pip install neural-dsl
# Writing complex DSL manually
```

**Upgrade Path:**
```bash
# Update Neural DSL
pip install --upgrade neural-dsl

# Still write DSL manually (nothing changes)
vim complex_model.neural

# But now export easily
neural export complex_model.neural --format onnx --optimize

# And optionally get AI suggestions
neural ai "Suggest improvements for a ResNet50 model"
```

**What You Get:**
- All existing features work
- Easy export to multiple formats
- Optional AI assistance
- Better optimization

### Scenario 5: Open Source Contributor

You contribute to Neural DSL development.

**Current Setup (v0.2.x):**
```bash
git clone https://github.com/Lemniscate-world/Neural.git
pip install -r requirements-dev.txt
```

**Upgrade Path:**
```bash
# Update repository
git pull origin main

# Reinstall dependencies
pip install -r requirements-dev.txt

# Review new features
cat docs/releases/v0.3.0.md
cat MIGRATION_v0.3.0.md

# Try automation features
python scripts/automation/master_automation.py --help
```

**What You Get:**
- Automated testing and validation
- Better CI/CD workflows
- Easier release process
- More contribution opportunities

---

## Rollback Instructions

If you encounter critical issues and need to rollback:

```bash
# Uninstall current version
pip uninstall neural-dsl

# Install previous version
pip install neural-dsl==0.2.9

# Verify
neural --version
```

**Note:** Rollback should not be necessary as v0.3.0 is backward compatible. If you encounter issues, please report them on [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues).

---

## Getting Help

### Resources

- **Release Notes**: [docs/releases/v0.3.0.md](docs/releases/v0.3.0.md)
- **AI Guide**: [docs/ai_integration_guide.md](docs/ai_integration_guide.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **Main README**: [README.md](README.md)

### Support Channels

- **GitHub Issues**: https://github.com/Lemniscate-world/Neural/issues
- **Discord**: https://discord.gg/KFku4KvS
- **Twitter**: [@NLang4438](https://x.com/NLang4438)

### Reporting Issues

When reporting migration issues, include:
- Current Neural DSL version: `neural --version`
- Previous version you upgraded from
- Python version: `python --version`
- Operating system
- Complete error message
- Steps to reproduce

---

## Summary

### Key Takeaways

✅ **Fully Backward Compatible**: All v0.2.x code works in v0.3.0  
✅ **No Breaking Changes**: Upgrade with confidence  
✅ **New Features Are Additive**: Use them when ready  
✅ **Easy Upgrade**: `pip install --upgrade neural-dsl`  

### What Changed

- ✨ **Added**: AI-powered model creation
- ✨ **Added**: Production deployment features
- ✨ **Added**: Automation system
- 📚 **Enhanced**: Documentation
- 🐛 **Fixed**: Various bugs and issues

### What Stayed The Same

- ✅ DSL syntax (unchanged)
- ✅ CLI commands (enhanced, not changed)
- ✅ API interfaces (extended, not changed)
- ✅ Configuration format (unchanged)
- ✅ Existing features (all work as before)

### Next Steps

1. Upgrade: `pip install --upgrade neural-dsl`
2. Verify: `neural --version`
3. Try AI: `neural ai "Create a simple model"`
4. Try export: `neural export examples/mnist.neural --format onnx`
5. Read docs: [docs/releases/v0.3.0.md](docs/releases/v0.3.0.md)

---

**Happy upgrading! 🚀**

If you encounter any issues during migration, please don't hesitate to reach out through our support channels. We're here to help!
