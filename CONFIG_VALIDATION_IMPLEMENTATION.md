# Configuration Validation and Health Check System Implementation

## Overview

Implemented a comprehensive configuration validation and health check system for Neural DSL, including startup validation, health check endpoints for all services, Kubernetes readiness/liveness probes, configuration migration tools, and CLI commands for validation and health monitoring.

## Components Implemented

### 1. Configuration Validator (`neural/config/validator.py`)

**Features:**
- Validates required environment variables for all services
- Checks optional variables and provides default values
- Validates variable formats (ports, URLs, secret keys, etc.)
- Detects dangerous default values
- Checks for port conflicts between services
- Generates detailed validation reports
- Startup validation with exception raising

**Services Validated:**
- API (FastAPI server)
- Dashboard (NeuralDbg)
- Aquarium (Visual IDE)
- Marketplace (Model marketplace)
- Celery (Task queue)
- Redis (Cache/broker)

**Validation Rules:**
- Secret keys: Minimum 32 characters
- Database URLs: Valid connection string format
- Ports: Range 1024-65535 (services), 1-65535 (Redis)
- No port conflicts
- No dangerous defaults

### 2. Health Checker (`neural/config/health.py`)

**Features:**
- Health status enumeration (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
- Port availability checking
- HTTP endpoint health checking
- Redis connection testing
- Celery worker status checking
- Response time measurement
- Detailed health information
- Kubernetes probe support

**Health Endpoints:**
- `/health` - Standard health check with service status
- `/health/live` - Liveness probe (always returns 200 unless dead)
- `/health/ready` - Readiness probe (checks dependencies)
- `/health/detailed` - Detailed health with metrics (API only)

### 3. Configuration Migrator (`neural/config/migrator.py`)

**Features:**
- YAML to .env conversion
- .env to YAML conversion
- Template generation
- Automatic type inference
- Service grouping
- Comment preservation

**Supported Conversions:**
- Nested YAML structures to flat environment variables
- Boolean, integer, float type handling
- JSON serialization for complex types (lists, dicts)
- Bi-directional mapping

### 4. CLI Commands (`neural/cli/cli.py`)

**Commands Added:**

```bash
# Configuration validation
neural config validate                          # Validate all services
neural config validate -s api -s dashboard     # Validate specific services
neural config validate --report report.txt     # Export validation report
neural config validate --env-file .env.prod    # Validate custom .env file

# Health checks
neural config check-health                      # Check all services
neural config check-health -s api              # Check specific service
neural config check-health --detailed          # Show detailed information

# Configuration migration
neural config migrate config.yaml              # YAML to .env
neural config migrate .env --direction env-to-yaml  # .env to YAML
neural config migrate config.yaml -o .env.prod --overwrite

# Template generation
neural config template                         # Generate .env.example
neural config template -o .env.template        # Custom output
```

### 5. Service Integration

**API Service (`neural/api/main.py`):**
- Startup validation in lifespan manager
- Health check endpoint with service status
- Liveness probe endpoint
- Readiness probe endpoint
- Detailed health check endpoint

**Dashboard Service (`neural/dashboard/dashboard.py`):**
- Health check endpoint
- Liveness probe endpoint
- Readiness probe endpoint
- Flask route integration

**Aquarium Service (`neural/aquarium/aquarium.py`):**
- Health check endpoint
- Liveness probe endpoint
- Readiness probe endpoint
- Flask server integration

**Marketplace Service (`neural/marketplace/api.py`):**
- Health check endpoint
- Liveness probe endpoint
- Readiness probe endpoint
- Optional health checker integration

### 6. Kubernetes Manifests

**Created Deployments:**
- `kubernetes/neural-api-deployment.yaml` - API service with 3 replicas
- `kubernetes/neural-dashboard-deployment.yaml` - Dashboard with 2 replicas
- `kubernetes/neural-aquarium-deployment.yaml` - Aquarium with 2 replicas
- `kubernetes/neural-marketplace-deployment.yaml` - Marketplace with 2 replicas
- `kubernetes/neural-redis-deployment.yaml` - Redis with persistence
- `kubernetes/neural-secrets.yaml` - Secret management

**Probe Configuration:**

Liveness Probe:
- Initial delay: 30 seconds
- Period: 10 seconds
- Timeout: 5 seconds
- Failure threshold: 3

Readiness Probe:
- Initial delay: 10 seconds
- Period: 5 seconds
- Timeout: 3 seconds
- Failure threshold: 2

**Resource Allocation:**
- API: 256Mi-512Mi memory, 250m-500m CPU
- Dashboard: 512Mi-1Gi memory, 500m-1000m CPU
- Aquarium: 512Mi-1Gi memory, 500m-1000m CPU
- Marketplace: 256Mi-512Mi memory, 250m-500m CPU
- Redis: 128Mi-256Mi memory, 100m-200m CPU

### 7. Documentation

**Created:**
- `docs/CONFIGURATION_VALIDATION.md` - Complete user guide
- `kubernetes/README.md` - Kubernetes deployment guide

**Covers:**
- Configuration validation usage
- Health check system overview
- Startup validation
- Kubernetes integration
- Configuration migration
- CLI command reference
- Environment variable reference
- Best practices
- Troubleshooting

## File Structure

```
neural/
├── config/
│   ├── __init__.py              # Module exports
│   ├── validator.py             # Configuration validation
│   ├── health.py                # Health check system
│   └── migrator.py              # Configuration migration
├── api/
│   └── main.py                  # API with health endpoints
├── dashboard/
│   └── dashboard.py             # Dashboard with health endpoints
├── aquarium/
│   └── aquarium.py              # Aquarium with health endpoints
├── marketplace/
│   └── api.py                   # Marketplace with health endpoints
└── cli/
    └── cli.py                   # CLI commands

kubernetes/
├── neural-api-deployment.yaml
├── neural-dashboard-deployment.yaml
├── neural-aquarium-deployment.yaml
├── neural-marketplace-deployment.yaml
├── neural-redis-deployment.yaml
├── neural-secrets.yaml
└── README.md

docs/
└── CONFIGURATION_VALIDATION.md

.gitignore                       # Ignore validation reports
```

## Usage Examples

### Basic Validation

```bash
# Validate all services
neural config validate

# Validate specific services
neural config validate -s api -s celery

# Export validation report
neural config validate --report validation_report.txt
```

### Health Monitoring

```bash
# Check all service health
neural config check-health

# Check specific services
neural config check-health -s api -s redis

# Detailed health information
neural config check-health --detailed
```

### Configuration Migration

```bash
# YAML to .env
neural config migrate config.yaml

# .env to YAML
neural config migrate .env --direction env-to-yaml

# Generate template
neural config template
```

### Startup Validation

Services automatically validate configuration at startup:

```python
# In neural/api/main.py
from neural.config.validator import ConfigValidator

validator = ConfigValidator()
validator.validate_startup(services=['api', 'celery'])
```

### Health Check API

```bash
# Standard health check
curl http://localhost:8000/health

# Liveness probe
curl http://localhost:8000/health/live

# Readiness probe
curl http://localhost:8000/health/ready

# Detailed health
curl http://localhost:8000/health/detailed
```

### Kubernetes Deployment

```bash
# Create secrets
kubectl apply -f kubernetes/neural-secrets.yaml

# Deploy services
kubectl apply -f kubernetes/neural-redis-deployment.yaml
kubectl apply -f kubernetes/neural-api-deployment.yaml
kubectl apply -f kubernetes/neural-dashboard-deployment.yaml
kubectl apply -f kubernetes/neural-aquarium-deployment.yaml
kubectl apply -f kubernetes/neural-marketplace-deployment.yaml

# Check pod status
kubectl get pods
kubectl describe pod neural-api-xxxxx
```

## Validation Rules

### Required Variables

**All Services:**
- `SECRET_KEY` - Minimum 32 characters, not a dangerous default

**API Service:**
- `DATABASE_URL` - Valid database connection string

**Celery:**
- `REDIS_HOST` - Valid hostname or IP
- `CELERY_BROKER_URL` - Valid Redis URL
- `CELERY_RESULT_BACKEND` - Valid Redis URL

### Optional Variables with Defaults

- `API_HOST`: `0.0.0.0`
- `API_PORT`: `8000`
- `API_WORKERS`: `4`
- `DASHBOARD_PORT`: `8050`
- `AQUARIUM_PORT`: `8051`
- `MARKETPLACE_PORT`: `5000`
- `REDIS_PORT`: `6379`
- `DEBUG`: `false`
- `RATE_LIMIT_ENABLED`: `true`
- `RATE_LIMIT_REQUESTS`: `100`
- `RATE_LIMIT_PERIOD`: `60`

### Security Validations

Dangerous defaults that trigger errors:
- `change-me-in-production`
- `change-me-in-production-use-strong-random-key`
- `insecure-secret-key`
- `development`
- `test`

### Port Validations

- Service ports: 1024-65535
- Redis port: 1-65535
- No duplicate ports across services

## Health Status Levels

- **HEALTHY**: Service is fully operational
- **DEGRADED**: Service is operational but with issues
- **UNHEALTHY**: Service is not operational
- **UNKNOWN**: Health status cannot be determined

## Integration Points

### Python API

```python
from neural.config.validator import ConfigValidator
from neural.config.health import HealthChecker, HealthStatus

# Validate configuration
validator = ConfigValidator()
result = validator.validate(services=['api'])

if result.has_errors():
    raise RuntimeError("Configuration validation failed")

# Check service health
health_checker = HealthChecker()
api_health = health_checker.check_service('api')

if api_health.status != HealthStatus.HEALTHY:
    print(f"API is {api_health.status.value}")
```

### REST API

All services expose health endpoints:
- API: `http://localhost:8000/health`
- Dashboard: `http://localhost:8050/health`
- Aquarium: `http://localhost:8051/health`
- Marketplace: `http://localhost:5000/health`

### Kubernetes Probes

Liveness and readiness probes are configured in all deployments to ensure:
- Automatic pod restart on failure
- Traffic only routed to healthy pods
- Graceful scaling and updates

## Testing

The system can be tested with:

```bash
# Test validation
neural config validate

# Test health checks
neural config check-health

# Test migration
neural config template
neural config migrate .env.example --direction env-to-yaml

# Test health endpoints (with services running)
curl http://localhost:8000/health
curl http://localhost:8050/health
curl http://localhost:8051/health
curl http://localhost:5000/health
```

## Benefits

1. **Reliability**: Catch configuration errors before deployment
2. **Security**: Detect dangerous default values and insecure configurations
3. **Monitoring**: Real-time health status of all services
4. **Kubernetes Ready**: Native support for liveness and readiness probes
5. **Developer Friendly**: Clear error messages and suggestions
6. **Production Ready**: Comprehensive validation and health checking
7. **Migration Support**: Easy transition between configuration formats
8. **Documentation**: Extensive guides and examples

## Future Enhancements

Potential improvements:
- Database connection testing
- External service dependency checks
- Performance metrics collection
- Alert integration (PagerDuty, Slack)
- Configuration versioning
- Rollback support
- A/B testing configuration
- Dynamic configuration reloading

## Conclusion

The configuration validation and health check system provides a robust foundation for deploying and monitoring Neural DSL services in production environments. It ensures reliable startup, continuous health monitoring, and seamless Kubernetes integration.
