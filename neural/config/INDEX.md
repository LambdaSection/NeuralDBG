# Neural DSL Configuration System - Index

Complete index of all configuration system files and their purposes.

## Quick Navigation

- [Getting Started](#getting-started)
- [Core Files](#core-files)
- [Settings Modules](#settings-modules)
- [Documentation](#documentation)
- [Examples and Tests](#examples-and-tests)
- [Quick Reference](#quick-reference)

## Getting Started

Start here:
1. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start guide
2. **[README.md](README.md)** - Complete documentation
3. **[examples.py](examples.py)** - Usage examples

## Core Files

### __init__.py
**Purpose:** Public API exports  
**Size:** 1.2 KB  
**Exports:**
- `ConfigManager` - Main configuration manager class
- `get_config()` - Get global config instance
- All settings classes

**Usage:**
```python
from neural.config import get_config
config = get_config()
```

---

### base.py
**Purpose:** Base configuration classes and utilities  
**Size:** 4.5 KB  
**Contains:**
- `BaseConfig` - Base class for all settings
- `EnvironmentType` - Environment constants
- `ValidationError` - Configuration validation error
- Helper functions for environment variables

**Key Classes:**
- `BaseConfig(BaseSettings)` - Pydantic settings base with .env support
- `EnvironmentType` - Environment type constants

---

### manager.py
**Purpose:** ConfigManager implementation  
**Size:** 7.4 KB  
**Contains:**
- `ConfigManager` - Singleton configuration manager
- `get_config()` - Global config getter
- `reset_config()` - Reset for testing

**Key Features:**
- Singleton pattern
- Lazy loading
- Caching
- Unified access to all subsystems

**Usage:**
```python
from neural.config import get_config
config = get_config()
config.core.version  # Access settings
```

---

### utils.py
**Purpose:** Configuration utility functions  
**Size:** 7.4 KB  
**Contains:**
- File I/O functions (YAML, JSON)
- Export functions
- Template generation
- Validation helpers
- Configuration merging

**Key Functions:**
- `load_yaml_config()` - Load YAML config
- `export_config_to_file()` - Export configuration
- `generate_env_template()` - Generate .env template
- `validate_environment()` - Validate config
- `get_config_summary()` - Get config summary

---

### cli.py
**Purpose:** Command-line interface  
**Size:** 6.1 KB  
**Commands:**
- `show` - Display configuration
- `list` - List settings
- `generate-template` - Generate .env template
- `export` - Export configuration
- `validate` - Validate configuration
- `get` - Get specific value
- `env` - Show environment info
- `check` - Health check

**Usage:**
```bash
python -m neural.config.cli show
python -m neural.config.cli validate
```

---

### test_config.py
**Purpose:** Basic functionality tests  
**Size:** 7.4 KB  
**Tests:**
- Import verification
- ConfigManager functionality
- Settings access
- Environment variables
- Validation
- Export/dump
- Utility functions

**Usage:**
```bash
python neural/config/test_config.py
```

---

### examples.py
**Purpose:** Usage examples  
**Size:** 9.1 KB  
**Examples:**
- Basic usage
- Environment checks
- Storage paths
- API configuration
- HPO configuration
- Integration checks
- Teams quotas
- Monitoring setup
- Export/validation
- Dynamic reload

**Usage:**
```bash
python neural/config/examples.py
```

## Settings Modules

All settings modules are in `neural/config/settings/`

### settings/__init__.py
**Purpose:** Settings module exports  
**Size:** 0.9 KB  
**Exports:** All settings classes

---

### settings/core.py
**Purpose:** Core Neural DSL settings  
**Size:** 2.9 KB  
**Environment Prefix:** `NEURAL_`  
**Settings:**
- Application metadata
- Parser settings
- Code generation
- Execution settings
- Logging
- Performance
- Feature flags

**Key Settings:**
- `app_name`, `version`, `environment`
- `debug`, `log_level`
- `default_backend`
- `enable_gpu`, `enable_distributed`

---

### settings/api.py
**Purpose:** API server settings  
**Size:** 5.0 KB  
**Environment Prefix:** `NEURAL_API_`  
**Settings:**
- Server configuration
- Security (JWT, API keys)
- Rate limiting
- Redis/Celery
- Database
- CORS

**Key Settings:**
- `host`, `port`, `workers`
- `secret_key`, `api_key_header`
- `redis_url`, `database_url`
- `cors_origins`

**Properties:**
- `redis_url` - Computed Redis URL
- `broker_url` - Celery broker URL
- `result_backend` - Celery result backend

---

### settings/storage.py
**Purpose:** Storage and file management  
**Size:** 4.9 KB  
**Environment Prefix:** `NEURAL_STORAGE_`  
**Settings:**
- Storage paths
- File handling
- Compression
- Cloud storage (S3, GCS, Azure)

**Key Settings:**
- `base_path`, `models_path`, `experiments_path`
- `auto_create_dirs`, `max_storage_gb`
- `cloud_storage_enabled`, `cloud_storage_provider`

**Methods:**
- `get_full_path(type)` - Get full path for storage type

---

### settings/dashboard.py
**Purpose:** NeuralDbg dashboard settings  
**Size:** 3.5 KB  
**Environment Prefix:** `NEURAL_DASHBOARD_`  
**Settings:**
- Server configuration
- WebSocket settings
- Visualization
- Profiling
- Authentication

**Key Settings:**
- `host`, `port` (default: 8050)
- `websocket_interval`
- `profiling_enabled`
- `auth_enabled`, `auth_username`

---

### settings/no_code.py
**Purpose:** No-code interface settings  
**Size:** 3.7 KB  
**Environment Prefix:** `NEURAL_NOCODE_`  
**Settings:**
- Server configuration
- UI defaults
- Validation
- Code generation
- Feature flags

**Key Settings:**
- `host`, `port` (default: 8051)
- `max_layers`, `default_backend`
- `enable_templates`, `enable_validation`

**Properties:**
- `max_file_size_bytes` - File size in bytes

---

### settings/hpo.py
**Purpose:** Hyperparameter optimization  
**Size:** 4.3 KB  
**Environment Prefix:** `NEURAL_HPO_`  
**Settings:**
- Optuna configuration
- Search strategies
- Pruning
- Metrics
- Distributed optimization

**Key Settings:**
- `n_trials`, `timeout_seconds`
- `sampler`, `pruner`
- `optimization_direction`
- `distributed_enabled`

---

### settings/automl.py
**Purpose:** Neural Architecture Search  
**Size:** 5.2 KB  
**Environment Prefix:** `NEURAL_AUTOML_`  
**Settings:**
- Search strategies
- Search space
- Performance estimation
- Weight sharing
- Multi-objective optimization

**Key Settings:**
- `search_strategy` (random, bayesian, evolutionary, etc.)
- `n_architectures`, `max_layers`
- `weight_sharing_enabled`
- `multi_objective_enabled`

---

### settings/integrations.py
**Purpose:** ML platform integrations  
**Size:** 5.5 KB  
**Environment Prefix:** `NEURAL_INTEGRATION_`  
**Settings:**
- AWS SageMaker
- Google Vertex AI
- Azure ML
- Databricks
- Paperspace
- Run:AI
- MLflow
- Weights & Biases
- TensorBoard

**Key Settings:**
- `sagemaker_enabled`, `sagemaker_region`
- `vertex_enabled`, `vertex_project_id`
- `azure_enabled`, `azure_subscription_id`
- `mlflow_enabled`, `wandb_enabled`

---

### settings/teams.py
**Purpose:** Team management and billing  
**Size:** 8.9 KB  
**Environment Prefix:** `NEURAL_TEAMS_`  
**Settings:**
- Plan quotas (free, starter, professional, enterprise)
- Pricing
- Analytics
- Billing
- Security
- Notifications

**Key Settings:**
- `base_dir`
- Plan quotas and pricing
- `enable_stripe_integration`
- `enable_audit_logging`

**Methods:**
- `get_plan_quotas(plan)` - Get quotas for plan
- `get_plan_pricing(plan)` - Get pricing for plan

---

### settings/monitoring.py
**Purpose:** Monitoring and metrics  
**Size:** 4.8 KB  
**Environment Prefix:** `NEURAL_MONITORING_`  
**Settings:**
- Prometheus export
- System metrics
- Application metrics
- Model metrics
- Alerting
- Tracing

**Key Settings:**
- `enabled`, `prometheus_enabled`
- `collect_cpu_metrics`, `collect_gpu_metrics`
- `alerting_enabled`
- `tracing_enabled`

## Documentation

### README.md
**Purpose:** Complete documentation  
**Size:** 11.8 KB  
**Contents:**
- Features overview
- Quick start
- All available settings
- CLI tools
- Advanced usage
- Configuration precedence
- Security
- Testing
- Best practices
- Troubleshooting

**Read this for:** Complete API documentation

---

### QUICKSTART.md
**Purpose:** Quick start guide  
**Size:** 6.7 KB  
**Contents:**
- 5-minute quick start
- Installation
- Basic usage
- Common patterns
- CLI tools
- Real-world examples
- Tips

**Read this for:** Getting started quickly

---

### IMPLEMENTATION.md
**Purpose:** Implementation details  
**Size:** 15.2 KB  
**Contents:**
- Architecture overview
- Component breakdown
- Settings modules details
- ConfigManager API
- Utility functions
- CLI tools
- Integration points
- Migration path
- Testing
- Best practices
- Performance
- Extensibility

**Read this for:** Understanding internals

---

### MIGRATION_GUIDE.md
**Purpose:** Migration guide  
**Size:** 12.5 KB  
**Contents:**
- Migration strategy
- Module-by-module migration
- Common patterns
- Environment variable mapping
- Testing migration
- Backward compatibility
- Step-by-step checklist
- Common issues

**Read this for:** Migrating existing code

---

### SUMMARY.md
**Purpose:** Implementation summary  
**Size:** 8.5 KB  
**Contents:**
- Overview
- Files created
- Features
- Subsystems covered
- Usage examples
- Architecture
- Dependencies
- Testing
- Benefits

**Read this for:** High-level overview

---

### INDEX.md
**Purpose:** Complete file index  
**Size:** This file  
**Contents:**
- File descriptions
- Quick navigation
- Quick reference

**Read this for:** Finding specific information

## Quick Reference

### Import Patterns

```python
# Get config manager
from neural.config import get_config
config = get_config()

# Access settings
config.core.version
config.api.port
config.storage.base_path

# Import specific settings (rarely needed)
from neural.config.settings import CoreSettings, APISettings
```

### Environment Variables

```bash
# Core
NEURAL_DEBUG=true
NEURAL_ENVIRONMENT=production

# API
NEURAL_API_PORT=8080
NEURAL_API_SECRET_KEY=my-secret

# Storage
NEURAL_STORAGE_BASE_PATH=/data/neural
```

### CLI Commands

```bash
# View configuration
python -m neural.config.cli show

# Validate
python -m neural.config.cli validate

# Export
python -m neural.config.cli export --output config.yaml

# Generate template
python -m neural.config.cli generate-template
```

### Common Tasks

```python
# Validate configuration
from neural.config import get_config
config = get_config()
config.validate_all()

# Export configuration
config_dict = config.dump_config(safe=True)

# Check environment
if config.is_production():
    print("Production mode")

# Get all settings
all_settings = config.get_all_settings()
```

## File Statistics

- **Total Files:** 25
- **Total Size:** ~160 KB
- **Code Files:** 13
- **Settings Files:** 11
- **Documentation:** 6
- **Tests/Examples:** 2

## Module Size Breakdown

| Category | Files | Size (KB) |
|----------|-------|-----------|
| Core | 7 | 48.1 |
| Settings | 11 | 50.5 |
| Documentation | 6 | 73.1 |
| Tests/Examples | 2 | 16.4 |
| **Total** | **26** | **~188** |

## Recommended Reading Order

1. **First Time:** QUICKSTART.md → examples.py
2. **Deep Dive:** README.md → IMPLEMENTATION.md
3. **Migration:** MIGRATION_GUIDE.md → examples.py
4. **Reference:** INDEX.md (this file)

## Support Resources

- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Full Docs:** [README.md](README.md)
- **Examples:** [examples.py](examples.py)
- **Tests:** [test_config.py](test_config.py)
- **Migration:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **CLI Help:** `python -m neural.config.cli --help`

## Version

Configuration System Version: 1.0.0  
Neural DSL Version: 0.3.0  
Last Updated: 2024

---

**Navigation:**
- [↑ Back to Top](#neural-dsl-configuration-system---index)
- [→ Quick Start](QUICKSTART.md)
- [→ Full Documentation](README.md)
