# CLI Commands Refocusing

## Commands to Keep (Core Functionality)

### Essential Commands
```bash
neural compile <file>       # Compile DSL to backend code
  --backend                 # pytorch|tensorflow|onnx
  --output                  # Output file path
  --dry-run                 # Validate without generating

neural validate <file>      # Validate DSL syntax and shapes

neural visualize <file>     # Visualize network architecture
  --format                  # png|svg|pdf
  --output                  # Output file path

neural help                 # Show help

neural version              # Show version info

neural clean                # Remove generated artifacts
  --yes                     # Actually delete (dry-run by default)
```

### Optional Commands (HPO/AutoML)
```bash
neural optimize <file>      # Run HPO (if hpo installed)
  --trials                  # Number of trials
  --strategy                # tpe|random|grid

neural search <file>        # Run NAS (if automl installed)
  --strategy                # nas|enas
  --trials                  # Number of trials

neural debug <file>         # Launch debugging dashboard (if dashboard installed)
  --port                    # Dashboard port (default: 8050)
  --backend                 # Backend to use
```

## Commands to Remove

### Removed Features
```bash
# Cloud integrations (use cloud provider SDKs)
neural cloud run
neural cloud connect
neural cloud execute

# Experiment tracking (use MLflow, W&B)
neural track init
neural track log
neural track list
neural track show
neural track plot
neural track compare

# Aquarium dashboard (use MLflow UI)
neural aquarium

# No-code interface (DSL is the interface)
neural no-code

# Marketplace (out of scope)
neural marketplace search
neural marketplace download
neural marketplace publish
neural marketplace info
neural marketplace list
neural marketplace web
neural marketplace hub-upload
neural marketplace hub-download

# Cost tracking (out of scope)
neural cost estimate
neural cost track
neural cost report
neural cost optimize

# Model explanation (use SHAP, LIME directly)
neural explain

# Documentation generation (use standard tools)
neural docs
```

## Simplified CLI Structure

```python
# Core commands only
@click.group()
def cli():
    """Neural DSL - Declarative neural network compiler"""
    pass

@cli.command()
def compile(...):
    """Compile DSL to backend code"""
    
@cli.command()
def validate(...):
    """Validate DSL syntax and shapes"""
    
@cli.command()
def visualize(...):
    """Visualize network architecture"""
    
@cli.command()
def optimize(...):
    """Optimize hyperparameters (requires hpo)"""
    
@cli.command()
def search(...):
    """Neural architecture search (requires automl)"""
    
@cli.command()
def debug(...):
    """Launch debugging dashboard (requires dashboard)"""
    
@cli.command()
def clean(...):
    """Remove generated artifacts"""
    
@cli.command()
def version(...):
    """Show version info"""
```

## Implementation Notes

### Files to Remove
- `neural/cli/aquarium.py` - Aquarium command (experiment tracking UI)
- Remove cloud/track/marketplace/cost command groups from `cli.py`
- Remove `neural docs` command
- Remove `neural explain` command
- Remove `neural no-code` command

### Files to Keep
- `neural/cli/cli.py` - Refactored to core commands only
- `neural/cli/cli_aesthetics.py` - UI helpers
- `neural/cli/cpu_mode.py` - CPU mode support
- `neural/cli/lazy_imports.py` - Lazy loading
- `neural/cli/version.py` - Version info
- `neural/cli/welcome_message.py` - Welcome banner

### Simplified Help Output

```
$ neural --help

Neural DSL v0.4.0 - Declarative neural network compiler

Commands:
  compile     Compile DSL to backend code (PyTorch/TensorFlow/ONNX)
  validate    Validate DSL syntax and shapes
  visualize   Visualize network architecture
  optimize    Optimize hyperparameters [requires: pip install neural-dsl[hpo]]
  search      Neural architecture search [requires: pip install neural-dsl[automl]]
  debug       Launch debugging dashboard [requires: pip install neural-dsl[dashboard]]
  clean       Remove generated artifacts
  version     Show version and dependencies

Use 'neural COMMAND --help' for command-specific help.

Examples:
  neural compile model.neural --backend pytorch
  neural validate model.neural
  neural visualize model.neural --format png
  neural optimize model.neural --trials 100
```

## Migration Guide for Users

### Removed Commands and Alternatives

| Removed Command | Alternative |
|----------------|-------------|
| `neural cloud *` | Use boto3, google-cloud, azure SDKs directly |
| `neural track *` | Use MLflow: `mlflow ui`, `mlflow log-param`, etc. |
| `neural aquarium` | Use MLflow UI: `mlflow ui` |
| `neural marketplace *` | Use HuggingFace Hub CLI directly |
| `neural cost *` | Use cloud provider cost dashboards |
| `neural explain` | Use SHAP: `shap.TreeExplainer(model)` |
| `neural docs` | Use Sphinx or standard doc tools |
| `neural no-code` | Use Neural DSL (it's already simple) |

### Core Workflow Unchanged

```bash
# Before (v0.3.x) - still works in v0.4.0
neural compile model.neural --backend pytorch
neural visualize model.neural --format png
neural debug model.neural

# After (v0.4.0) - same commands
neural compile model.neural --backend pytorch
neural visualize model.neural --format png
neural debug model.neural
```

## Benefits

### For Users
- **Clarity**: Clear what Neural DSL does (compile DSL to code)
- **Simplicity**: Fewer commands to learn
- **Focus**: Core commands are well-documented and tested
- **Performance**: Faster CLI startup (fewer imports)

### For Maintainers
- **Maintainability**: Smaller CLI codebase
- **Testing**: Focus tests on core commands
- **Documentation**: Better docs for core features
- **Velocity**: Faster iteration on core functionality

## Implementation Checklist

- [ ] Remove cloud command group from cli.py
- [ ] Remove track command group from cli.py
- [ ] Remove marketplace command group from cli.py
- [ ] Remove cost command group from cli.py
- [ ] Remove aquarium command from cli.py
- [ ] Remove no-code command from cli.py
- [ ] Remove docs command from cli.py
- [ ] Remove explain command from cli.py
- [ ] Remove aquarium.py file
- [ ] Update help text to reflect focused mission
- [ ] Update CLI tests to cover only core commands
- [ ] Update CLI documentation
- [ ] Remove lazy imports for removed features
- [ ] Simplify dependency checking in CLI
