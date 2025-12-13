# Configuration System Quick Start

Get started with Neural DSL's configuration system in 5 minutes.

## Installation

The configuration system is part of core dependencies and is automatically installed:

```bash
pip install -e .
```

Dependencies:
- `pydantic>=2.0.0` - Schema validation
- `pydantic-settings>=2.0.0` - Settings management
- `python-dotenv>=1.0.0` - .env file support

## Basic Usage

### 1. Import and Access

```python
from neural.config import get_config

# Get the global config manager
config = get_config()

# Access settings
print(config.core.version)      # "0.3.0"
print(config.api.port)           # 8000
print(config.storage.base_path)  # "./neural_storage"
```

### 2. Create .env File

Copy the example file and customize:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Core settings
NEURAL_DEBUG=true
NEURAL_ENVIRONMENT=development

# API settings
NEURAL_API_PORT=8080
NEURAL_API_SECRET_KEY=my-secret-key

# Storage
NEURAL_STORAGE_BASE_PATH=/data/neural
```

### 3. Use in Your Code

```python
from neural.config import get_config

def main():
    config = get_config()
    
    # Check environment
    if config.is_production():
        print("Production mode")
    
    # Use settings
    api_url = f"http://{config.api.host}:{config.api.port}"
    storage_path = config.storage.get_full_path("models")
    
    print(f"API: {api_url}")
    print(f"Models: {storage_path}")

if __name__ == "__main__":
    main()
```

## Common Patterns

### Different Environments

```bash
# Development
export NEURAL_ENVIRONMENT=development
export NEURAL_DEBUG=true

# Production
export NEURAL_ENVIRONMENT=production
export NEURAL_DEBUG=false
```

### Access Subsystem Settings

```python
from neural.config import get_config

config = get_config()

# Core settings
backend = config.core.default_backend  # "tensorflow"
debug = config.core.debug              # False

# API settings
port = config.api.port                 # 8000
secret = config.api.secret_key         # "change-me..."

# Storage settings
base = config.storage.base_path        # "./neural_storage"

# Dashboard settings
dash_port = config.dashboard.port      # 8050

# HPO settings
trials = config.hpo.n_trials           # 100

# AutoML settings
strategy = config.automl.search_strategy  # "bayesian"

# Integrations
mlflow = config.integrations.mlflow_enabled  # False

# Teams
quotas = config.teams.get_plan_quotas("professional")

# Monitoring
enabled = config.monitoring.enabled    # True
```

### Validation

```python
from neural.config import get_config
from neural.config.utils import validate_environment

config = get_config()

# Validate all settings
try:
    config.validate_all()
    print("✓ Configuration valid")
except Exception as e:
    print(f"✗ Error: {e}")

# Get validation report
results = validate_environment()
if results["warnings"]:
    for warning in results["warnings"]:
        print(f"⚠ {warning}")
```

## CLI Tools

### Show Configuration

```bash
# Show all settings (YAML, safe mode)
python -m neural.config.cli show

# Show in JSON
python -m neural.config.cli show --format json

# Show summary
python -m neural.config.cli show --format summary
```

### Validate Configuration

```bash
python -m neural.config.cli validate
```

### Generate Template

```bash
python -m neural.config.cli generate-template
# Creates .env.template with all options
```

### Get Specific Value

```bash
python -m neural.config.cli get core version
python -m neural.config.cli get api port
```

### Export Configuration

```bash
python -m neural.config.cli export --output config.yaml
```

## Real-World Examples

### API Server

```python
from neural.config import get_config
from flask import Flask

config = get_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = config.api.secret_key
app.config['DEBUG'] = config.api.debug

if __name__ == "__main__":
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug
    )
```

### Storage Manager

```python
from neural.config import get_config
from pathlib import Path

config = get_config()

class StorageManager:
    def __init__(self):
        self.models_path = config.storage.get_full_path("models")
        self.experiments_path = config.storage.get_full_path("experiments")
    
    def save_model(self, name: str, data):
        path = self.models_path / name
        # Save model...
```

### HPO Runner

```python
from neural.config import get_config
import optuna

config = get_config()

def optimize():
    study = optuna.create_study(
        study_name=f"{config.hpo.study_name_prefix}_experiment",
        direction=config.hpo.optimization_direction,
        storage=config.hpo.storage_url,
        load_if_exists=config.hpo.load_if_exists,
    )
    
    study.optimize(
        objective_function,
        n_trials=config.hpo.n_trials,
        timeout=config.hpo.timeout_seconds,
        n_jobs=config.hpo.n_jobs,
    )
```

### Integration Setup

```python
from neural.config import get_config

config = get_config()

def setup_integrations():
    integrations = []
    
    if config.integrations.mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(config.integrations.mlflow_tracking_uri)
        mlflow.set_experiment(config.integrations.mlflow_experiment_name)
        integrations.append("MLflow")
    
    if config.integrations.wandb_enabled:
        import wandb
        wandb.init(
            project=config.integrations.wandb_project,
            entity=config.integrations.wandb_entity,
        )
        integrations.append("W&B")
    
    return integrations
```

## Tips

1. **Always use `get_config()`** - Don't instantiate settings classes directly
2. **Set environment early** - Set `NEURAL_ENVIRONMENT` before importing
3. **Validate on startup** - Call `config.validate_all()` at application start
4. **Use safe exports** - Use `dump_config(safe=True)` when logging
5. **Keep .env local** - Never commit `.env` files with secrets

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples.py](examples.py) for more usage patterns
- See [neural/config/settings/](settings/) for all available settings
- Run `python -m neural.config.cli --help` for CLI help

## Troubleshooting

### Config not loading from .env

```python
from neural.config.base import get_env_path
print(get_env_path())  # Check if .env is found
```

### Import errors

```bash
# Install dependencies
pip install pydantic pydantic-settings python-dotenv
```

### Type checking issues

```python
# Use type hints for better IDE support
from neural.config import get_config

config = get_config()
port: int = config.api.port  # Type-safe access
```
