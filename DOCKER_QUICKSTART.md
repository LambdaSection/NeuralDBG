# Docker Quick Start Guide

Complete guide for deploying Neural DSL using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Profiles](#deployment-profiles)
- [Environment Configuration](#environment-configuration)
- [Common Usage Patterns](#common-usage-patterns)
- [Service Management](#service-management)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **System Requirements**:
  - 4GB RAM minimum (8GB recommended)
  - 10GB free disk space
  - Linux, macOS, or Windows with WSL2

### Installation

```bash
# Check Docker installation
docker --version
docker-compose --version

# Verify Docker is running
docker ps
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Environment Configuration section)
nano .env  # or vim, code, notepad, etc.
```

**Minimum required changes for production:**
- `SECRET_KEY`: Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `REDIS_PASSWORD`: Set a strong password
- `POSTGRES_PASSWORD`: Set a strong database password

### 3. Start Services

```bash
# Development mode (default profile)
docker-compose up -d

# Production mode
docker-compose --profile prod up -d

# Full stack with all services
docker-compose --profile full up -d
```

### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Test API health
curl http://localhost:8000/health
```

### 5. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8050 (with `--profile dashboard`)
- **No-Code Interface**: http://localhost:8051 (with `--profile nocode`)
- **Aquarium IDE**: http://localhost:8052 (with `--profile aquarium`)
- **Flower Monitoring**: http://localhost:5555 (with `--profile monitoring`)

## Deployment Profiles

The consolidated `docker-compose.yml` supports multiple profiles for different use cases:

### Default Profile (Development)

**Services**: redis, postgres, api, worker

```bash
docker-compose up -d
```

**Features**:
- Code hot-reload enabled
- Debug mode active
- Direct port exposure for debugging
- SQLite or PostgreSQL database
- Minimal resource requirements

**Use Case**: Local development, testing, debugging

### Production Profile

**Services**: redis, postgres, api, worker, nginx

```bash
docker-compose --profile prod up -d
```

**Features**:
- Security hardened
- Password-protected Redis
- Nginx reverse proxy
- HTTPS support (with SSL certificates)
- Resource limits enforced
- No code mounting
- Debug mode disabled
- Log rotation configured

**Use Case**: Production deployments, staging environments

### Minimal Profile

**Services**: redis, api

```bash
docker-compose up redis api
```

**Features**:
- Bare minimum services
- Fast startup time
- Low resource usage
- No database (SQLite used)

**Use Case**: Quick testing, CI/CD pipelines, demos

### Full Profile

**Services**: All services including dashboards and monitoring

```bash
docker-compose --profile full up -d
```

**Features**:
- Complete Neural DSL stack
- All UIs and tools available
- Full monitoring with Flower
- Nginx reverse proxy

**Use Case**: Full-featured development, demonstrations, workshops

### Individual Service Profiles

Start specific services as needed:

```bash
# Dashboard only
docker-compose --profile dashboard up dashboard

# No-code interface
docker-compose --profile nocode up nocode

# Aquarium IDE
docker-compose --profile aquarium up aquarium

# Monitoring with Flower
docker-compose --profile monitoring up flower
```

### Combining Profiles

Combine multiple profiles:

```bash
# API + Dashboard + Monitoring
docker-compose --profile dashboard --profile monitoring up -d

# Full stack with production settings
docker-compose --profile full --profile prod up -d
```

## Environment Configuration

### Essential Variables

Create a `.env` file with these minimum settings:

```bash
# Security (REQUIRED)
SECRET_KEY=your-secret-key-here-min-32-chars
REDIS_PASSWORD=your-redis-password
POSTGRES_PASSWORD=your-postgres-password

# Database
POSTGRES_DB=neural_db
POSTGRES_USER=neural

# Service Ports (optional, defaults shown)
API_PORT=8000
DASHBOARD_PORT=8050
NOCODE_PORT=8051
AQUARIUM_PORT=8052
FLOWER_PORT=5555
```

### Development Configuration

```bash
# Enable debug mode
DEBUG=true

# Enable code hot-reload
CODE_MOUNT=./neural

# Disable authentication (dev only)
REDIS_PASSWORD=

# Use SQLite instead of PostgreSQL (optional)
DATABASE_URL=sqlite:////app/data/neural_api.db

# Faster dashboard updates
UPDATE_INTERVAL=1000
```

### Production Configuration

```bash
# Disable debug mode
DEBUG=false

# Disable code mounting
CODE_MOUNT=

# Secure passwords
SECRET_KEY=<generate-with-secrets-token-hex>
REDIS_PASSWORD=<strong-password>
POSTGRES_PASSWORD=<strong-password>

# Use PostgreSQL
DATABASE_URL=postgresql://neural:${POSTGRES_PASSWORD}@postgres:5432/neural_db

# Restart policy
RESTART_POLICY=always

# Resource limits
API_WORKERS=4
CELERY_CONCURRENCY=4
WORKER_REPLICAS=2

# CORS configuration
CORS_ORIGINS=["https://yourdomain.com"]

# SSL/TLS
SSL_CERT_PATH=./ssl
```

### Advanced Configuration

```bash
# Docker image settings
IMAGE_TAG=v0.3.0
PYTHON_VERSION=3.10

# Network configuration
NETWORK_SUBNET=172.20.0.0/16

# Logging
LOG_MAX_SIZE=10m
LOG_MAX_FILES=3

# Monitoring
FLOWER_USER=admin
FLOWER_PASSWORD=secure-password

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

## Common Usage Patterns

### Start All Services

```bash
# With default profile
docker-compose up -d

# With logs in foreground
docker-compose up
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker

# Last 100 lines
docker-compose logs --tail=100 api
```

### Scale Workers

```bash
# Scale to 4 worker instances
docker-compose up -d --scale worker=4

# Or set in .env
WORKER_REPLICAS=4
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart api
docker-compose restart worker
```

### Update Services

```bash
# Pull latest images
docker-compose pull

# Rebuild images
docker-compose build

# Rebuild without cache
docker-compose build --no-cache

# Update and restart
docker-compose up -d --build
```

### Execute Commands

```bash
# Execute command in running container
docker-compose exec api python -c "print('Hello')"

# Open shell in container
docker-compose exec api bash

# Run one-off command
docker-compose run --rm api python manage.py migrate
```

### Database Operations

```bash
# Access PostgreSQL
docker-compose exec postgres psql -U neural -d neural_db

# Backup database
docker-compose exec postgres pg_dump -U neural neural_db > backup.sql

# Restore database
docker-compose exec -T postgres psql -U neural neural_db < backup.sql

# Access Redis CLI
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD}
```

### Volume Management

```bash
# List volumes
docker volume ls | grep neural

# Inspect volume
docker volume inspect neural_postgres-data

# Backup volume
docker run --rm -v neural_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data

# Remove unused volumes
docker volume prune
```

## Service Management

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health status
docker inspect --format='{{.State.Health.Status}}' neural-api
docker inspect --format='{{.State.Health.Status}}' neural-postgres
docker inspect --format='{{.State.Health.Status}}' neural-redis
```

### Resource Monitoring

```bash
# Monitor resource usage
docker stats

# Specific services
docker stats neural-api neural-worker neural-postgres neural-redis
```

### Service Dependencies

Services start in dependency order:
1. redis, postgres (infrastructure)
2. api (depends on redis, postgres)
3. worker (depends on redis)
4. dashboard, nocode, aquarium (independent or depend on api)
5. flower (depends on redis, worker)
6. nginx (depends on api and dashboards)

## Troubleshooting

### Services Won't Start

```bash
# Check logs for errors
docker-compose logs api
docker-compose logs postgres

# Verify environment variables
docker-compose config

# Check port conflicts
netstat -tuln | grep -E ':(8000|5432|6379)'

# Clean up and restart
docker-compose down -v
docker-compose up -d
```

### Database Connection Issues

```bash
# Verify PostgreSQL is healthy
docker-compose ps postgres
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U neural

# Check DATABASE_URL
echo $DATABASE_URL
```

### Redis Connection Issues

```bash
# Verify Redis is healthy
docker-compose ps redis
docker-compose logs redis

# Test connection
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} ping

# Check password configuration
echo $REDIS_PASSWORD
```

### Worker Not Processing Tasks

```bash
# Check worker logs
docker-compose logs worker

# Verify Redis connection
docker-compose exec worker celery -A neural.api.celery_app inspect ping

# Check active tasks
docker-compose exec worker celery -A neural.api.celery_app inspect active

# Restart workers
docker-compose restart worker
```

### API Health Check Fails

```bash
# Test API directly
curl http://localhost:8000/health

# Check API logs
docker-compose logs api

# Verify dependencies
docker-compose ps redis postgres

# Restart API
docker-compose restart api
```

### Permission Denied Errors

```bash
# Fix volume permissions
docker-compose run --rm --user root api chown -R neural:neural /app/data

# Check file ownership
docker-compose exec api ls -la /app/data
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up Docker resources
docker system prune -a --volumes

# Remove old images
docker image prune -a

# Remove stopped containers
docker container prune
```

### Memory Issues

```bash
# Check resource usage
docker stats

# Reduce worker replicas
WORKER_REPLICAS=1 docker-compose up -d

# Reduce concurrency
CELERY_CONCURRENCY=2 docker-compose up -d
```

### Network Issues

```bash
# Inspect network
docker network inspect neural_neural-network

# Recreate network
docker-compose down
docker network prune
docker-compose up -d

# Test inter-service communication
docker-compose exec api ping postgres
docker-compose exec api ping redis
```

### Build Failures

```bash
# Clear build cache
docker builder prune -a

# Build with verbose output
docker-compose build --progress=plain

# Build specific service
docker-compose build --no-cache api

# Check Dockerfile syntax
docker-compose config
```

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Set SSL path
SSL_CERT_PATH=./ssl docker-compose --profile prod up -d

# Verify certificate
openssl x509 -in ssl/cert.pem -text -noout
```

## Production Deployment Checklist

- [ ] Generate strong `SECRET_KEY`
- [ ] Set strong passwords for all services
- [ ] Disable debug mode (`DEBUG=false`)
- [ ] Remove code mounting (`CODE_MOUNT=`)
- [ ] Configure CORS for your domain
- [ ] Set up SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Enable log rotation
- [ ] Set up monitoring and alerts
- [ ] Configure backups for volumes
- [ ] Test health checks
- [ ] Document recovery procedures
- [ ] Set up automated backups
- [ ] Configure resource limits
- [ ] Enable security scanning
- [ ] Review and apply security best practices

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Neural DSL Documentation](./README.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Dockerfile Documentation](./dockerfiles/README.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Email: Lemniscate_zero@proton.me
- Documentation: See `docs/` directory
