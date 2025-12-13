# Migration Guide to New Configuration System

Guide for migrating existing code to use the new unified configuration system.

## Overview

The new configuration system provides:
- Centralized configuration management
- Type-safe settings access
- Environment variable support
- Schema validation
- Consistent API across all modules

## Migration Strategy

### Phase 1: Gradual Adoption (Recommended)

Keep existing configurations working while gradually adopting the new system:

```python
# Existing code continues to work
from neural.api.config import settings
port = settings.port

# New code uses new system
from neural.config import get_config
config = get_config()
port = config.api.port
```

### Phase 2: Full Migration

Once comfortable, migrate all code to new system.

## Module-by-Module Migration

### 1. API Module Migration

#### Before (neural/api/server.py)
```python
from neural.api.config import settings

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers
    )
```

#### After
```python
from neural.config import get_config

config = get_config()
app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers
    )
```

#### Environment Variables
```bash
# Before
API_HOST=0.0.0.0
API_PORT=8000

# After (still works, or use new prefix)
NEURAL_API_HOST=0.0.0.0
NEURAL_API_PORT=8000
```

### 2. Dashboard Migration

#### Before (neural/dashboard/dashboard.py)
```python
import yaml

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

UPDATE_INTERVAL = config.get("websocket_interval", 1000)
```

#### After
```python
from neural.config import get_config

config = get_config()
UPDATE_INTERVAL = config.dashboard.websocket_interval
```

#### Configuration Files
```yaml
# Before: config.yaml
websocket_interval: 1000
auth:
  username: "admin"
  password: null

# After: .env
NEURAL_DASHBOARD_WEBSOCKET_INTERVAL=1000
NEURAL_DASHBOARD_AUTH_ENABLED=false
NEURAL_DASHBOARD_AUTH_USERNAME=admin
NEURAL_DASHBOARD_AUTH_PASSWORD=
```

### 3. No-Code Interface Migration

#### Before (neural/no_code/app.py)
```python
from neural.no_code.config import Config, get_config

config = get_config()
app.run(host=config.HOST, port=config.PORT)
```

#### After
```python
from neural.config import get_config

config = get_config()
app.run(host=config.no_code.host, port=config.no_code.port)
```

#### Environment Variables
```bash
# Before
NOCODE_HOST=127.0.0.1
NOCODE_PORT=8051

# After
NEURAL_NOCODE_HOST=127.0.0.1
NEURAL_NOCODE_PORT=8051
```

### 4. Teams Module Migration

#### Before (neural/teams/manager.py)
```python
from neural.teams.config import TeamConfig

quotas = TeamConfig.PLAN_QUOTAS['professional']
pricing = TeamConfig.PLAN_PRICING['professional']
```

#### After
```python
from neural.config import get_config

config = get_config()
quotas = config.teams.get_plan_quotas('professional')
pricing = config.teams.get_plan_pricing('professional')
```

#### Configuration
```python
# Before: neural/teams/config.py
class TeamConfig:
    PLAN_QUOTAS = {
        'professional': {
            'max_models': 200,
            ...
        }
    }

# After: .env
NEURAL_TEAMS_PROFESSIONAL_MAX_MODELS=200
# Or use defaults in neural/config/settings/teams.py
```

### 5. HPO Module Migration

#### Before (neural/hpo/hpo.py)
```python
def optimize(n_trials=100, sampler='tpe'):
    study = optuna.create_study(...)
    study.optimize(objective, n_trials=n_trials)
```

#### After
```python
from neural.config import get_config

def optimize():
    config = get_config()
    study = optuna.create_study(
        study_name=f"{config.hpo.study_name_prefix}_experiment",
        direction=config.hpo.optimization_direction,
        storage=config.hpo.storage_url,
    )
    study.optimize(
        objective,
        n_trials=config.hpo.n_trials,
        n_jobs=config.hpo.n_jobs,
    )
```

### 6. AutoML Module Migration

#### Before (neural/automl/engine.py)
```python
class AutoMLEngine:
    def __init__(self, search_strategy='bayesian', n_architectures=100):
        self.search_strategy = search_strategy
        self.n_architectures = n_architectures
```

#### After
```python
from neural.config import get_config

class AutoMLEngine:
    def __init__(self):
        config = get_config()
        self.search_strategy = config.automl.search_strategy
        self.n_architectures = config.automl.n_architectures
```

### 7. Integrations Migration

#### Before (neural/integrations/sagemaker.py)
```python
class SageMakerIntegration:
    def __init__(self, region='us-east-1', role_arn=None):
        self.region = region
        self.role_arn = role_arn or os.getenv('SAGEMAKER_ROLE_ARN')
```

#### After
```python
from neural.config import get_config

class SageMakerIntegration:
    def __init__(self):
        config = get_config()
        if not config.integrations.sagemaker_enabled:
            raise ValueError("SageMaker integration not enabled")
        self.region = config.integrations.sagemaker_region
        self.role_arn = config.integrations.sagemaker_role_arn
```

## Common Patterns

### Pattern 1: Configuration at Module Level

#### Before
```python
# config.py
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 8000))

# app.py
from .config import DEBUG, HOST, PORT
```

#### After
```python
# app.py
from neural.config import get_config

config = get_config()
# Access config.subsystem.setting
```

### Pattern 2: Configuration Classes

#### Before
```python
class Config:
    DEBUG = False
    HOST = '127.0.0.1'
    PORT = 8000

class DevelopmentConfig(Config):
    DEBUG = True

config = DevelopmentConfig()
```

#### After
```python
from neural.config import get_config

config = get_config()
# Set NEURAL_ENVIRONMENT=development
# Set NEURAL_DEBUG=true
```

### Pattern 3: Dictionary Configuration

#### Before
```python
CONFIG = {
    'server': {
        'host': '127.0.0.1',
        'port': 8000,
    },
    'database': {
        'url': 'sqlite:///db.sqlite',
    }
}
```

#### After
```python
from neural.config import get_config

config = get_config()
host = config.api.host
port = config.api.port
db_url = config.api.database_url
```

## Environment Variables Migration

### Old Variable Names → New Variable Names

| Old | New | Notes |
|-----|-----|-------|
| `DEBUG` | `NEURAL_DEBUG` | Global debug |
| `API_HOST` | `NEURAL_API_HOST` | API server host |
| `API_PORT` | `NEURAL_API_PORT` | API server port |
| `SECRET_KEY` | `NEURAL_API_SECRET_KEY` | JWT secret |
| `REDIS_HOST` | `NEURAL_API_REDIS_HOST` | Redis host |
| `DATABASE_URL` | `NEURAL_API_DATABASE_URL` | Database URL |
| `STORAGE_PATH` | `NEURAL_STORAGE_BASE_PATH` | Storage path |
| `NOCODE_HOST` | `NEURAL_NOCODE_HOST` | No-code host |
| `NOCODE_PORT` | `NEURAL_NOCODE_PORT` | No-code port |

### .env File Migration

#### Before
```env
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=my-secret
STORAGE_PATH=./storage
```

#### After
```env
NEURAL_DEBUG=true
NEURAL_API_HOST=0.0.0.0
NEURAL_API_PORT=8000
NEURAL_API_SECRET_KEY=my-secret
NEURAL_STORAGE_BASE_PATH=./storage
```

Or use the provided template:
```bash
cp .env.example .env
# Edit .env with your values
```

## Testing Migration

### Before
```python
def test_api_config():
    from neural.api.config import settings
    assert settings.port == 8000
```

### After
```python
import os
from neural.config import get_config, reset_config

def test_api_config():
    os.environ['NEURAL_API_PORT'] = '9000'
    reset_config()
    config = get_config()
    assert config.api.port == 9000
    del os.environ['NEURAL_API_PORT']
```

## Backward Compatibility

### Keep Old Config for Transition

```python
# Old config (keep for backward compatibility)
from neural.api.config import settings as old_settings

# New config (use in new code)
from neural.config import get_config
new_config = get_config()

# Ensure they match during migration
assert old_settings.port == new_config.api.port
```

### Deprecation Warnings

Add deprecation warnings to old config:

```python
import warnings

def get_old_config():
    warnings.warn(
        "This config module is deprecated. "
        "Use 'from neural.config import get_config' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return Config()
```

## Validation During Migration

Ensure configuration is valid after migration:

```python
from neural.config import get_config
from neural.config.utils import validate_environment

# Validate configuration
config = get_config()
config.validate_all()

# Get detailed validation results
results = validate_environment()
if not results['valid']:
    print("Errors:", results['errors'])
if results['warnings']:
    print("Warnings:", results['warnings'])
```

## Step-by-Step Migration Checklist

### 1. Preparation
- [ ] Read this migration guide
- [ ] Review new configuration system documentation
- [ ] Identify all configuration usage in your module
- [ ] Create backup of current configuration

### 2. Environment Setup
- [ ] Copy .env.example to .env
- [ ] Migrate environment variables to new names
- [ ] Test .env file loading

### 3. Code Migration
- [ ] Replace config imports with `from neural.config import get_config`
- [ ] Update config access patterns
- [ ] Test with new configuration
- [ ] Add validation checks

### 4. Testing
- [ ] Update tests to use new config
- [ ] Add environment variable override tests
- [ ] Verify backward compatibility
- [ ] Run full test suite

### 5. Documentation
- [ ] Update module documentation
- [ ] Update configuration examples
- [ ] Add migration notes to CHANGELOG

### 6. Cleanup (Optional)
- [ ] Remove old config files (if no longer needed)
- [ ] Remove deprecation warnings
- [ ] Update CI/CD pipelines

## Gradual Migration Example

Migrate one module at a time:

```python
# Week 1: Migrate API module
from neural.config import get_config
config = get_config()
# Use config.api.*

# Week 2: Migrate Dashboard
# Use config.dashboard.*

# Week 3: Migrate Storage
# Use config.storage.*

# Continue until all modules migrated
```

## Common Issues

### Issue 1: Environment Variables Not Loading

**Problem:** Configuration not reading from .env file

**Solution:** Ensure .env file is in project root
```python
from neural.config.base import get_env_path
print(get_env_path())  # Check if .env is found
```

### Issue 2: Import Errors

**Problem:** Cannot import pydantic or pydantic-settings

**Solution:** Install dependencies
```bash
pip install pydantic pydantic-settings python-dotenv
```

### Issue 3: Validation Errors

**Problem:** Configuration fails validation

**Solution:** Check validation details
```python
from neural.config import get_config
config = get_config()
try:
    config.validate_all()
except Exception as e:
    print(f"Validation error: {e}")
```

### Issue 4: Type Errors

**Problem:** Type hints don't match

**Solution:** Use correct types
```python
port: int = config.api.port  # Not str
debug: bool = config.core.debug  # Not int
```

## Best Practices During Migration

1. **Migrate Gradually** - One module at a time
2. **Test Thoroughly** - Add tests for new config
3. **Document Changes** - Update docs as you go
4. **Keep Backward Compatibility** - Don't break existing code
5. **Validate Configuration** - Check config is valid
6. **Use Type Hints** - Leverage type safety
7. **Review Settings** - Ensure all settings are covered

## Help and Support

If you encounter issues during migration:

1. Check the [README.md](README.md) for detailed documentation
2. Review [examples.py](examples.py) for usage patterns
3. Run [test_config.py](test_config.py) to verify setup
4. Use CLI tools: `python -m neural.config.cli validate`

## Summary

The new configuration system provides:
- ✅ Type-safe configuration
- ✅ Centralized management
- ✅ Environment variable support
- ✅ Schema validation
- ✅ Better organization
- ✅ Improved security

Migration is straightforward:
1. Install dependencies (already in CORE_DEPS)
2. Copy .env.example to .env
3. Update imports to use `get_config()`
4. Update config access patterns
5. Test and validate

The old configuration continues to work during migration period.
