# Neural DSL Configuration System

Comprehensive environment variable configuration system with schema validation, .env file support, and centralized configuration management.

## Features

- **Schema Validation**: Uses Pydantic for type-safe configuration with automatic validation
- **Environment Variables**: Full support for environment variable overrides
- **.env File Support**: Automatic loading of `.env` files with precedence handling
- **Configuration Precedence**: Environment variables > .env file > defaults
- **Typed Settings**: Separate typed settings classes for each subsystem
- **Centralized Access**: Single `ConfigManager` for consistent configuration across all modules
- **CLI Tools**: Command-line interface for configuration management
- **Security**: Automatic redaction of sensitive fields in exports

## Quick Start

### Basic Usage

```python
from neural.config import get_config

# Get the global config manager
config = get_config()

# Access subsystem settings
print(config.core.version)
print(config.api.port)
print(config.storage.base_path)

# Check environment
if config.is_production():
    print("Running in production mode")
```

### Environment Variables

Set environment variables to override defaults:

```bash
# Core settings
export NEURAL_DEBUG=true
export NEURAL_ENVIRONMENT=production

# API settings
export NEURAL_API_PORT=8080
export NEURAL_API_SECRET_KEY=your-secret-key

# Storage settings
export NEURAL_STORAGE_BASE_PATH=/data/neural
```

### .env File

Create a `.env` file in your project root:

```env
# Core settings
NEURAL_DEBUG=false
NEURAL_ENVIRONMENT=production

# API settings
NEURAL_API_PORT=8000
NEURAL_API_SECRET_KEY=change-me-in-production

# Storage settings
NEURAL_STORAGE_BASE_PATH=./neural_storage
NEURAL_STORAGE_AUTO_CREATE_DIRS=true
```

## Available Settings

### Core Settings (`config.core`)

Global Neural DSL settings:

```python
config.core.app_name              # Application name
config.core.version               # Version
config.core.environment           # Environment (development/production)
config.core.debug                 # Debug mode
config.core.default_backend       # Default ML backend (tensorflow/pytorch/onnx)
config.core.log_level             # Logging level
config.core.enable_gpu            # Enable GPU support
```

### API Settings (`config.api`)

API server configuration:

```python
config.api.host                   # Server host
config.api.port                   # Server port
config.api.secret_key             # JWT secret key
config.api.rate_limit_enabled     # Enable rate limiting
config.api.database_url           # Database connection URL
config.api.redis_url              # Redis connection URL
config.api.cors_origins           # CORS allowed origins
```

### Storage Settings (`config.storage`)

File storage configuration:

```python
config.storage.base_path          # Base storage directory
config.storage.experiments_path   # Experiments directory
config.storage.models_path        # Models directory
config.storage.auto_create_dirs   # Auto-create directories
config.storage.max_storage_gb     # Maximum storage size
config.storage.cloud_storage_enabled  # Enable cloud storage
```

### Dashboard Settings (`config.dashboard`)

NeuralDbg dashboard configuration:

```python
config.dashboard.host             # Dashboard host
config.dashboard.port             # Dashboard port (default: 8050)
config.dashboard.websocket_interval  # Update interval (ms)
config.dashboard.profiling_enabled   # Enable profiling
config.dashboard.auth_enabled     # Enable authentication
```

### No-Code Settings (`config.no_code`)

No-code interface configuration:

```python
config.no_code.host               # Server host
config.no_code.port               # Server port (default: 8051)
config.no_code.max_layers         # Maximum layers
config.no_code.default_backend    # Default ML backend
config.no_code.enable_templates   # Enable model templates
```

### HPO Settings (`config.hpo`)

Hyperparameter optimization configuration:

```python
config.hpo.n_trials               # Number of trials
config.hpo.sampler                # Sampling strategy (tpe/random/grid)
config.hpo.pruner                 # Pruning strategy
config.hpo.optimization_direction # Minimize/maximize
config.hpo.distributed_enabled    # Enable distributed optimization
```

### AutoML Settings (`config.automl`)

Neural Architecture Search configuration:

```python
config.automl.search_strategy     # Search strategy
config.automl.n_architectures     # Number of architectures
config.automl.max_layers          # Maximum layers
config.automl.weight_sharing_enabled  # Enable weight sharing
config.automl.distributed_enabled     # Enable distributed search
```

### Integration Settings (`config.integrations`)

ML platform integration configuration:

```python
config.integrations.sagemaker_enabled    # AWS SageMaker
config.integrations.vertex_enabled       # Google Vertex AI
config.integrations.azure_enabled        # Azure ML
config.integrations.databricks_enabled   # Databricks
config.integrations.mlflow_enabled       # MLflow
config.integrations.wandb_enabled        # Weights & Biases
```

### Teams Settings (`config.teams`)

Team management configuration:

```python
config.teams.base_dir             # Base directory for team data
config.teams.get_plan_quotas('professional')  # Get plan quotas
config.teams.get_plan_pricing('professional') # Get plan pricing
config.teams.enable_stripe_integration  # Stripe integration
config.teams.enable_audit_logging      # Audit logging
```

### Monitoring Settings (`config.monitoring`)

Monitoring and metrics configuration:

```python
config.monitoring.enabled         # Enable monitoring
config.monitoring.prometheus_enabled  # Prometheus export
config.monitoring.collect_gpu_metrics # GPU metrics
config.monitoring.alerting_enabled    # Enable alerting
config.monitoring.tracing_enabled     # Distributed tracing
```

## CLI Tools

The configuration system includes a CLI for management tasks:

### Show Configuration

```bash
# Show all settings (YAML format, sensitive data redacted)
python -m neural.config.cli show

# Show in JSON format
python -m neural.config.cli show --format json

# Show without redacting sensitive data
python -m neural.config.cli show --unsafe

# Show summary
python -m neural.config.cli show --format summary
```

### List Settings

```bash
# List all subsystems
python -m neural.config.cli list

# List settings for specific subsystem
python -m neural.config.cli list core
python -m neural.config.cli list api
```

### Generate Template

```bash
# Generate .env template file
python -m neural.config.cli generate-template

# Custom output path
python -m neural.config.cli generate-template --output .env.example
```

### Export Configuration

```bash
# Export to YAML
python -m neural.config.cli export --output config.yaml

# Export to JSON
python -m neural.config.cli export --output config.json --format json
```

### Validate Configuration

```bash
# Validate current configuration
python -m neural.config.cli validate
```

### Get Specific Value

```bash
# Get a specific configuration value
python -m neural.config.cli get core version
python -m neural.config.cli get api port
```

### Environment Info

```bash
# Show environment information
python -m neural.config.cli env
```

### Health Check

```bash
# Run configuration health check
python -m neural.config.cli check
```

## Advanced Usage

### Programmatic Access

```python
from neural.config import get_config

config = get_config()

# Access nested settings
backend = config.core.default_backend
api_url = f"http://{config.api.host}:{config.api.port}"

# Get all settings
all_settings = config.get_all_settings()

# Dump configuration (safe mode redacts secrets)
config_dict = config.dump_config(safe=True)

# Validate all settings
config.validate_all()
```

### Custom .env File

```python
from neural.config import get_config

# Load from custom .env file
config = get_config(env_file="/path/to/custom.env")
```

### Reload Configuration

```python
from neural.config import get_config

config = get_config()

# Make changes to environment variables...

# Reload configuration
config.reload()
```

### Configuration in Different Modules

All modules can import and use the same config instance:

```python
# In neural/api/server.py
from neural.config import get_config

config = get_config()
app.run(host=config.api.host, port=config.api.port)

# In neural/storage/manager.py
from neural.config import get_config

config = get_config()
storage_path = config.storage.get_full_path("models")

# In neural/hpo/optimizer.py
from neural.config import get_config

config = get_config()
n_trials = config.hpo.n_trials
```

## Configuration Precedence

Settings are resolved in the following order (highest to lowest priority):

1. **Environment Variables**: Direct OS environment variables
2. **.env File**: Values from `.env` file
3. **Defaults**: Default values defined in settings classes

Example:

```python
# Default in code
class CoreSettings(BaseConfig):
    debug: bool = Field(default=False)

# Override in .env file
DEBUG=true

# Override with environment variable
export NEURAL_DEBUG=false  # This takes precedence
```

## Environment Prefixes

Each subsystem has its own environment variable prefix:

- Core: `NEURAL_`
- API: `NEURAL_API_`
- Storage: `NEURAL_STORAGE_`
- Dashboard: `NEURAL_DASHBOARD_`
- No-Code: `NEURAL_NOCODE_`
- HPO: `NEURAL_HPO_`
- AutoML: `NEURAL_AUTOML_`
- Integrations: `NEURAL_INTEGRATION_`
- Teams: `NEURAL_TEAMS_`
- Monitoring: `NEURAL_MONITORING_`

## Type Safety

All settings are type-checked using Pydantic:

```python
from neural.config import get_config

config = get_config()

# These are type-safe
port: int = config.api.port
debug: bool = config.core.debug
cors_origins: list[str] = config.api.cors_origins

# Pydantic validates types automatically
# Invalid values will raise ValidationError
```

## Security

Sensitive fields are automatically redacted when using safe export:

```python
config = get_config()

# Safe export (redacts secrets)
safe_config = config.dump_config(safe=True)
# Output: {"api": {"secret_key": "***REDACTED***"}}

# Unsafe export (includes secrets)
unsafe_config = config.dump_config(safe=False)
# Output: {"api": {"secret_key": "actual-secret-value"}}
```

## Testing

For testing, you can reset the configuration:

```python
from neural.config import reset_config, get_config

def test_custom_config():
    # Set test environment variables
    os.environ["NEURAL_DEBUG"] = "true"
    
    # Reset to force reload
    reset_config()
    
    # Get fresh config
    config = get_config()
    assert config.core.debug is True
```

## Best Practices

1. **Use Environment Variables for Secrets**: Never commit secrets to `.env` files
2. **Validate in Production**: Always run `config.validate_all()` on startup
3. **Use Safe Exports**: Use `dump_config(safe=True)` for logging/debugging
4. **Set Environment**: Always set `NEURAL_ENVIRONMENT` in deployment
5. **Type Hints**: Use type hints when accessing config for better IDE support
6. **Centralized Access**: Import from `neural.config` not individual settings modules

## Troubleshooting

### Configuration Not Loading

```python
from neural.config.base import get_env_path

# Check if .env file is found
env_path = get_env_path()
print(f".env file location: {env_path}")
```

### Validation Errors

```python
from neural.config import get_config

config = get_config()

try:
    config.validate_all()
except Exception as e:
    print(f"Validation error: {e}")
```

### Debug Configuration

```python
from neural.config.utils import get_config_summary

summary = get_config_summary()
print(summary)
```
