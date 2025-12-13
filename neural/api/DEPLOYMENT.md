# Neural API Deployment Guide

Complete guide for deploying Neural API in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring & Logging](#monitoring--logging)

## Local Development

### Prerequisites

- Python 3.8+
- Redis server
- Git

### Setup

1. **Clone and install**:
```bash
git clone https://github.com/your-repo/neural.git
cd neural
pip install -e ".[api]"
```

2. **Start Redis**:
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally
# Ubuntu/Debian: sudo apt-get install redis-server
# macOS: brew install redis
redis-server
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Start services**:

**Option A: Using scripts**
```bash
# Linux/Mac
./scripts/run_api.sh

# Windows
.\scripts\run_api.ps1
```

**Option B: Manual**
```bash
# Terminal 1: Start Celery worker
celery -A neural.api.celery_app worker --loglevel=info

# Terminal 2: Start API server
python -m neural.api.main
# OR
uvicorn neural.api.main:app --reload
```

5. **Access API**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

The docker-compose setup includes:
- **api**: FastAPI server (port 8000)
- **worker**: Celery worker for async tasks
- **redis**: Message broker and result backend
- **flower**: Celery monitoring UI (port 5555)

### Custom Configuration

Create `docker-compose.override.yml`:
```yaml
version: '3.8'

services:
  api:
    environment:
      - API_WORKERS=8
      - RATE_LIMIT_REQUESTS=200
    ports:
      - "8080:8000"
```

## Production Deployment

### Using Docker Compose Production

1. **Set environment variables**:
```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -hex 16)
REDIS_PASSWORD=$(openssl rand -hex 16)
DEBUG=false
EOF
```

2. **Deploy with production config**:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

This includes:
- PostgreSQL database
- Redis with authentication
- Multiple Celery workers
- Nginx reverse proxy with SSL support

### SSL/TLS Setup

1. **Generate certificates**:
```bash
# Using Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com

# Or self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem
```

2. **Update nginx.conf**:
Uncomment the HTTPS server block and update paths to certificates.

### Database Migration

For PostgreSQL in production:

```bash
# Install alembic
pip install alembic

# Initialize migrations
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- Helm 3 (optional)

### Deployment Files

Create `k8s/namespace.yaml`:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neural-api
```

Create `k8s/redis.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: neural-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: neural-api
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

Create `k8s/api.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-api
  namespace: neural-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-api
  template:
    metadata:
      labels:
        app: neural-api
    spec:
      containers:
      - name: api
        image: your-registry/neural-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: neural-api
  namespace: neural-api
spec:
  type: LoadBalancer
  selector:
    app: neural-api
  ports:
  - port: 80
    targetPort: 8000
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets
kubectl create secret generic neural-secrets \
  --from-literal=secret-key=$(openssl rand -hex 32) \
  --namespace neural-api

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Deploy API
kubectl apply -f k8s/api.yaml

# Check status
kubectl get pods -n neural-api
kubectl get services -n neural-api
```

## Cloud Deployment

### AWS ECS

1. **Build and push image**:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t neural-api .
docker tag neural-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/neural-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/neural-api:latest
```

2. **Create task definition and service** via AWS Console or CLI

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/neural-api

# Deploy
gcloud run deploy neural-api \
  --image gcr.io/PROJECT_ID/neural-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name neural-api-rg --location eastus

# Deploy container
az container create \
  --resource-group neural-api-rg \
  --name neural-api \
  --image your-registry/neural-api:latest \
  --dns-name-label neural-api \
  --ports 8000
```

## Monitoring & Logging

### Application Metrics

The API includes built-in metrics collection. Access via:
```python
from neural.api.middleware import MetricsMiddleware
# Metrics automatically collected
```

### Celery Monitoring with Flower

Access Flower UI at http://localhost:5555 to monitor:
- Active/completed/failed tasks
- Worker status
- Task execution times
- Queue lengths

### Logging

Configure structured logging:
```python
# In production
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Health Checks

- Endpoint: `GET /health`
- Returns service status for API and Celery
- Use for load balancer health checks

### Prometheus Integration

Add to `neural/api/main.py`:
```python
from prometheus_client import make_asgi_app, Counter

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## Scaling

### Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up -d --scale worker=5
```

**Kubernetes:**
```bash
kubectl scale deployment neural-api --replicas=5 -n neural-api
```

### Vertical Scaling

Adjust resource limits in deployment configs:
- CPU: Based on computation intensity
- Memory: 512Mi minimum, 2Gi recommended for production
- Workers: 4-8 per instance

## Backup & Recovery

### Database Backup

**PostgreSQL:**
```bash
docker exec neural-postgres pg_dump -U neural neural_api > backup.sql
```

**Restore:**
```bash
docker exec -i neural-postgres psql -U neural neural_api < backup.sql
```

### Volume Backup

```bash
# Backup data volumes
docker run --rm -v neural_api-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/api-data-backup.tar.gz -C /data .

# Restore
docker run --rm -v neural_api-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/api-data-backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

**Redis Connection Failed:**
```bash
# Check Redis status
redis-cli ping
docker ps | grep redis

# Check logs
docker logs neural-redis
```

**Celery Worker Not Processing:**
```bash
# Check worker status
celery -A neural.api.celery_app inspect active

# Check logs
docker logs neural-worker
```

**API Server Errors:**
```bash
# Check logs
docker logs neural-api

# Access container
docker exec -it neural-api bash
```

### Performance Optimization

1. **Enable caching** for frequently accessed data
2. **Use connection pooling** for database
3. **Optimize Celery** concurrency based on workload
4. **Use CDN** for static assets
5. **Enable compression** in nginx

## Security Best Practices

1. **Use strong SECRET_KEY** (32+ random bytes)
2. **Enable HTTPS** in production
3. **Set up firewall** rules
4. **Use secrets management** (AWS Secrets Manager, Vault)
5. **Regular security updates** for dependencies
6. **Enable rate limiting**
7. **Implement proper authentication**
8. **Regular backups**
9. **Monitor for suspicious activity**
10. **Follow principle of least privilege**

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
- Email: support@your-domain.com
