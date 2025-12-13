# Configuration Validation Quick Reference

## CLI Commands

### Validation
```bash
neural config validate                          # Validate all
neural config validate -s api -s dashboard     # Validate specific
neural config validate --report report.txt     # Export report
```

### Health Checks
```bash
neural config check-health                      # Check all
neural config check-health -s api              # Check specific
neural config check-health --detailed          # Detailed info
```

### Migration
```bash
neural config migrate config.yaml              # YAML → .env
neural config migrate .env --direction env-to-yaml  # .env → YAML
neural config template                         # Generate template
```

## Health Endpoints

| Service | Port | Endpoint |
|---------|------|----------|
| API | 8000 | `/health`, `/health/live`, `/health/ready`, `/health/detailed` |
| Dashboard | 8050 | `/health`, `/health/live`, `/health/ready` |
| Aquarium | 8051 | `/health`, `/health/live`, `/health/ready` |
| Marketplace | 5000 | `/health`, `/health/live`, `/health/ready` |

## Required Variables

```bash
# All Services
SECRET_KEY=<32+ chars>

# API Service
DATABASE_URL=<valid connection string>

# Celery
REDIS_HOST=<hostname>
CELERY_BROKER_URL=<redis url>
CELERY_RESULT_BACKEND=<redis url>
```

## Default Ports

```bash
API_PORT=8000
DASHBOARD_PORT=8050
AQUARIUM_PORT=8051
MARKETPLACE_PORT=5000
REDIS_PORT=6379
```

## Kubernetes Quick Deploy

```bash
kubectl apply -f kubernetes/neural-secrets.yaml
kubectl apply -f kubernetes/neural-redis-deployment.yaml
kubectl apply -f kubernetes/neural-api-deployment.yaml
kubectl apply -f kubernetes/neural-dashboard-deployment.yaml
kubectl apply -f kubernetes/neural-aquarium-deployment.yaml
kubectl apply -f kubernetes/neural-marketplace-deployment.yaml
```

## Python API

```python
from neural.config.validator import ConfigValidator
from neural.config.health import HealthChecker

# Validate
validator = ConfigValidator()
result = validator.validate(services=['api'])
if result.has_errors():
    raise RuntimeError("Validation failed")

# Health check
health_checker = HealthChecker()
health = health_checker.check_service('api')
print(f"Status: {health.status.value}")
```

## Common Issues

**Port conflict:**
```bash
# Change conflicting ports in .env
API_PORT=8000
DASHBOARD_PORT=8050
```

**Missing secret:**
```bash
# Generate secure key
python -c "import secrets; print(secrets.token_hex(32))"
```

**Health check fails:**
```bash
# Check logs
docker logs neural-api
kubectl logs deployment/neural-api
```

## Quick Setup

```bash
# 1. Generate template
neural config template

# 2. Edit .env
nano .env

# 3. Validate
neural config validate

# 4. Deploy
docker-compose up -d
# OR
kubectl apply -f kubernetes/
```

## See Also

- [Full Documentation](docs/CONFIGURATION_VALIDATION.md)
- [Kubernetes Guide](kubernetes/README.md)
- [Implementation Details](CONFIG_VALIDATION_IMPLEMENTATION.md)
