# Configuration System Implementation

Complete implementation details for the Neural DSL configuration system.

## Overview

A comprehensive environment variable configuration system with:
- Schema validation using Pydantic v2
- .env file support with python-dotenv
- Configuration precedence (env vars > .env > defaults)
- Typed settings classes for each subsystem
- Centralized ConfigManager for consistent access
- CLI tools for configuration management
- Security features (sensitive field redaction)

## Architecture

### Components

```
neural/config/
├── __init__.py              # Public API exports
├── base.py                  # Base classes and utilities
├── manager.py               # ConfigManager implementation
├── utils.py                 # Utility functions
├── cli.py                   # CLI tools
├── examples.py              # Usage examples
├── test_config.py           # Basic tests
├── settings/                # Settings modules
│   ├── __init__.py
│   ├── core.py             # Core DSL settings
│   ├── api.py              # API server settings
│   ├── storage.py          # Storage settings
│   ├── dashboard.py        # Dashboard settings
│   ├── no_code.py          # No-code interface settings
│   ├── hpo.py              # HPO settings
│   ├── automl.py           # AutoML settings
│   ├── integrations.py     # ML platform integrations
│   ├── teams.py            # Team management settings
│   └── monitoring.py       # Monitoring settings
├── README.md               # Full documentation
├── QUICKSTART.md           # Quick start guide
└── IMPLEMENTATION.md       # This file
```

### Class Hierarchy

```
BaseSettings (pydantic_settings)
    └── BaseConfig (base.py)
            ├── CoreSettings (settings/core.py)
            ├── APISettings (settings/api.py)
            ├── StorageSettings (settings/storage.py)
            ├── DashboardSettings (settings/dashboard.py)
            ├── NoCodeSettings (settings/no_code.py)
            ├── HPOSettings (settings/hpo.py)
            ├── AutoMLSettings (settings/automl.py)
            ├── IntegrationSettings (settings/integrations.py)
            ├── TeamsSettings (settings/teams.py)
            └── MonitoringSettings (settings/monitoring.py)

ConfigManager (manager.py)
    - Singleton pattern
    - Lazy loading
    - Settings caching
```

## Key Features

### 1. Schema Validation

All settings use Pydantic v2 for automatic validation:

```python
from pydantic import Field

class CoreSettings(BaseConfig):
    max_execution_time: int = Field(
        default=3600,
        gt=0,  # Greater than 0
        description="Maximum execution time in seconds"
    )
```

### 2. Environment Variable Precedence

Configuration resolution order:
1. Direct environment variables (highest priority)
2. Values from .env file
3. Default values in code (lowest priority)

### 3. Typed Settings

Each subsystem has strongly-typed settings:

```python
from neural.config import get_config

config = get_config()
port: int = config.api.port  # Type-safe access
```

### 4. Centralized Access

Single ConfigManager provides consistent access:

```python
from neural.config import get_config

# All modules use the same instance
config = get_config()
```

### 5. Environment Prefixes

Each subsystem has a unique prefix to avoid collisions:

- `NEURAL_` - Core settings
- `NEURAL_API_` - API settings
- `NEURAL_STORAGE_` - Storage settings
- `NEURAL_DASHBOARD_` - Dashboard settings
- `NEURAL_NOCODE_` - No-code settings
- `NEURAL_HPO_` - HPO settings
- `NEURAL_AUTOML_` - AutoML settings
- `NEURAL_INTEGRATION_` - Integration settings
- `NEURAL_TEAMS_` - Teams settings
- `NEURAL_MONITORING_` - Monitoring settings

### 6. Security

Sensitive fields are automatically redacted in safe exports:

```python
config = get_config()

# Safe export (secrets redacted)
safe = config.dump_config(safe=True)
# {"api": {"secret_key": "***REDACTED***"}}

# Unsafe export (full values)
unsafe = config.dump_config(safe=False)
# {"api": {"secret_key": "actual-secret"}}
```

## Settings Modules

### Core Settings (settings/core.py)

Global Neural DSL configuration:
- Application metadata
- Parser settings
- Code generation settings
- Execution settings
- Logging configuration
- Performance settings
- Feature flags

**Environment prefix:** `NEURAL_`

### API Settings (settings/api.py)

REST API server configuration:
- Server settings (host, port, workers)
- Security (JWT, API keys)
- Rate limiting
- Redis/Celery configuration
- Database settings
- CORS configuration

**Environment prefix:** `NEURAL_API_`

### Storage Settings (settings/storage.py)

File storage configuration:
- Local storage paths
- Storage limits and cleanup
- File handling
- Compression settings
- Cloud storage (S3, GCS, Azure)

**Environment prefix:** `NEURAL_STORAGE_`

### Dashboard Settings (settings/dashboard.py)

NeuralDbg dashboard configuration:
- Server settings
- WebSocket configuration
- Visualization settings
- Performance monitoring
- Authentication

**Environment prefix:** `NEURAL_DASHBOARD_`

### No-Code Settings (settings/no_code.py)

No-code interface configuration:
- Server settings
- UI defaults
- Validation settings
- Code generation
- Feature flags

**Environment prefix:** `NEURAL_NOCODE_`

### HPO Settings (settings/hpo.py)

Hyperparameter optimization configuration:
- Optuna settings
- Search strategies
- Pruning configuration
- Metric settings
- Distributed optimization

**Environment prefix:** `NEURAL_HPO_`

### AutoML Settings (settings/automl.py)

Neural Architecture Search configuration:
- Search strategies (Random, Bayesian, Evolutionary, RL, DARTS)
- Search space definition
- Performance estimation
- Weight sharing
- Multi-objective optimization

**Environment prefix:** `NEURAL_AUTOML_`

### Integration Settings (settings/integrations.py)

ML platform integration configuration:
- AWS SageMaker
- Google Vertex AI
- Azure ML
- Databricks
- Paperspace
- Run:AI
- MLflow
- Weights & Biases
- TensorBoard

**Environment prefix:** `NEURAL_INTEGRATION_`

### Teams Settings (settings/teams.py)

Team management configuration:
- Plan quotas (free, starter, professional, enterprise)
- Pricing configuration
- Analytics settings
- Billing settings
- Security settings
- Feature flags

**Environment prefix:** `NEURAL_TEAMS_`

### Monitoring Settings (settings/monitoring.py)

Monitoring and metrics configuration:
- Prometheus export
- System metrics (CPU, memory, disk, GPU)
- Application metrics
- Model metrics
- Alerting configuration
- Distributed tracing

**Environment prefix:** `NEURAL_MONITORING_`

## ConfigManager API

### Properties

```python
config = get_config()

config.core          # CoreSettings
config.api           # APISettings
config.storage       # StorageSettings
config.dashboard     # DashboardSettings
config.no_code       # NoCodeSettings
config.hpo           # HPOSettings
config.automl        # AutoMLSettings
config.integrations  # IntegrationSettings
config.teams         # TeamsSettings
config.monitoring    # MonitoringSettings
```

### Methods

```python
# Get all settings
all_settings = config.get_all_settings()

# Reload configuration
config.reload()
config.reload(env_file=".env.production")

# Export configuration
config_dict = config.dump_config(safe=True)

# Validate all settings
config.validate_all()

# Environment checks
env = config.get_environment()
is_prod = config.is_production()
is_dev = config.is_development()
is_debug = config.is_debug()
```

## Utility Functions

### utils.py

```python
from neural.config.utils import (
    load_yaml_config,
    load_json_config,
    save_yaml_config,
    save_json_config,
    export_config_to_file,
    generate_env_template,
    validate_environment,
    get_config_summary,
    set_environment,
    merge_configs,
)

# Load from file
config = load_yaml_config("config.yaml")

# Export to file
export_config_to_file("config.yaml", format="yaml", safe=True)

# Generate template
generate_env_template(".env.template")

# Validate
results = validate_environment()

# Get summary
summary = get_config_summary()
```

## CLI Tools

### Available Commands

```bash
# Show configuration
python -m neural.config.cli show [--format yaml|json|summary] [--safe|--unsafe]

# List settings
python -m neural.config.cli list [subsystem]

# Generate template
python -m neural.config.cli generate-template [--output PATH]

# Export configuration
python -m neural.config.cli export --output PATH [--format yaml|json] [--safe|--unsafe]

# Validate configuration
python -m neural.config.cli validate

# Get specific value
python -m neural.config.cli get SUBSYSTEM KEY

# Show environment info
python -m neural.config.cli env

# Health check
python -m neural.config.cli check
```

## Integration Points

### Existing Modules

The configuration system is designed to be gradually adopted by existing modules:

1. **API Server** (neural/api/)
   - Already uses pydantic-settings
   - Can gradually migrate to new config system

2. **Dashboard** (neural/dashboard/)
   - Currently uses config.yaml
   - Can migrate to unified config

3. **No-Code** (neural/no_code/)
   - Currently uses custom Config class
   - Can migrate to NoCodeSettings

4. **Teams** (neural/teams/)
   - Currently uses TeamConfig class
   - Can migrate to TeamsSettings

### Migration Path

For each module:

1. Import config manager:
```python
from neural.config import get_config
```

2. Replace old config access:
```python
# Old
from neural.teams.config import TeamConfig
quotas = TeamConfig.PLAN_QUOTAS['professional']

# New
from neural.config import get_config
config = get_config()
quotas = config.teams.get_plan_quotas('professional')
```

3. Update tests to use new config

## Testing

### Basic Tests

Run included tests:
```bash
python neural/config/test_config.py
```

### Integration Tests

Create comprehensive tests in `tests/config/`:

```python
# tests/config/test_settings.py
def test_core_settings():
    from neural.config import get_config
    config = get_config()
    assert config.core.version == "0.3.0"

# tests/config/test_env_override.py
def test_env_override():
    import os
    from neural.config import get_config, reset_config
    
    os.environ["NEURAL_DEBUG"] = "true"
    reset_config()
    config = get_config()
    assert config.core.debug is True
```

## Best Practices

### 1. Use Environment Variables for Secrets

```bash
export NEURAL_API_SECRET_KEY="$(openssl rand -hex 32)"
export NEURAL_API_REDIS_PASSWORD="secure-password"
```

### 2. Different .env Files per Environment

```
.env.development
.env.staging
.env.production
```

Load appropriate file:
```python
env = os.getenv("NEURAL_ENV", "development")
config = get_config(env_file=f".env.{env}")
```

### 3. Validate on Startup

```python
from neural.config import get_config

def main():
    config = get_config()
    
    # Validate configuration
    try:
        config.validate_all()
    except Exception as e:
        print(f"Invalid configuration: {e}")
        sys.exit(1)
    
    # Continue with application...
```

### 4. Use Type Hints

```python
from neural.config import get_config

config = get_config()

# Type-safe access
port: int = config.api.port
debug: bool = config.core.debug
origins: list[str] = config.api.cors_origins
```

### 5. Don't Commit Secrets

Ensure .gitignore includes:
```
.env
.env.local
.env.production
.env.staging
*.key
*_token.txt
```

## Performance Considerations

### 1. Singleton Pattern

ConfigManager uses singleton pattern - only one instance exists:
```python
config1 = get_config()
config2 = get_config()
assert config1 is config2  # Same instance
```

### 2. Lazy Loading

Settings are loaded on first access:
```python
config = get_config()
# No settings loaded yet

config.core.version  # CoreSettings loaded now
config.api.port      # APISettings loaded now
```

### 3. Caching

Settings are cached after first load:
```python
config = get_config()
config.core.version  # Load from environment/defaults
config.core.version  # Return cached value
```

## Extensibility

### Adding New Settings

1. Create new settings class:

```python
# neural/config/settings/my_feature.py
from neural.config.base import BaseConfig
from pydantic import Field

class MyFeatureSettings(BaseConfig):
    model_config = {"env_prefix": "NEURAL_MY_FEATURE_"}
    
    enabled: bool = Field(default=True)
    timeout: int = Field(default=60, gt=0)
```

2. Add to ConfigManager:

```python
# neural/config/manager.py
from neural.config.settings.my_feature import MyFeatureSettings

class ConfigManager:
    @property
    def my_feature(self) -> MyFeatureSettings:
        return self._get_setting(MyFeatureSettings)
```

3. Export in __init__.py:

```python
# neural/config/__init__.py
from neural.config.settings.my_feature import MyFeatureSettings

__all__ = [..., "MyFeatureSettings"]
```

## Dependencies

### Core Dependencies

- `pydantic>=2.0.0` - Data validation and settings management
- `pydantic-settings>=2.0.0` - Settings management
- `python-dotenv>=1.0.0` - .env file support

Added to CORE_DEPS in setup.py.

### Optional Dependencies

- `pyyaml>=6.0.1` - YAML config file support (already in CORE_DEPS)

## Future Enhancements

Potential improvements:

1. **Config Validation Rules**
   - Custom validators for complex scenarios
   - Cross-field validation

2. **Dynamic Reloading**
   - Watch .env file for changes
   - Hot reload without restart

3. **Remote Configuration**
   - Load from remote config server
   - Support for etcd, Consul, etc.

4. **Config Profiles**
   - Named configuration profiles
   - Easy switching between profiles

5. **Environment Detection**
   - Auto-detect environment from deployment
   - Cloud platform integration

6. **Config Encryption**
   - Encrypted .env files
   - Decrypt at runtime

7. **Config Versioning**
   - Track config changes
   - Rollback support

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install pydantic pydantic-settings python-dotenv
   ```

2. **Validation Errors**
   ```python
   from neural.config import get_config
   config = get_config()
   config.validate_all()  # Shows validation errors
   ```

3. **.env Not Loading**
   ```python
   from neural.config.base import get_env_path
   print(get_env_path())  # Check if .env is found
   ```

4. **Environment Variables Not Working**
   ```python
   import os
   print(os.getenv("NEURAL_DEBUG"))  # Check if set
   ```

## Documentation

- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [examples.py](examples.py) - Usage examples
- [test_config.py](test_config.py) - Basic tests
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - This file

## Support

For issues or questions:
- Check documentation in neural/config/
- Run `python -m neural.config.cli --help`
- See examples in neural/config/examples.py
- Review tests in neural/config/test_config.py
