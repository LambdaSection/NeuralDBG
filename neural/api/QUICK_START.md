# Neural API - Quick Start Guide

Get up and running with Neural API in 5 minutes!

## Prerequisites

- Python 3.8+
- Docker (optional, for Redis)

## Installation

### Option 1: Quick Install

```bash
# Install Neural DSL with API support
pip install -e ".[api]"
```

### Option 2: From Requirements

```bash
# Install from requirements file
pip install -r requirements-api.txt
pip install -e .
```

## Setup

### 1. Start Redis

**Using Docker (Recommended):**
```bash
docker run -d -p 6379:6379 --name neural-redis redis:7-alpine
```

**Or install Redis locally:**
- macOS: `brew install redis && redis-server`
- Ubuntu: `sudo apt-get install redis-server && redis-server`
- Windows: Download from https://redis.io/download

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults work for local development).

### 3. Start Services

**Option A: Using Scripts**

Linux/Mac:
```bash
chmod +x scripts/run_api.sh
./scripts/run_api.sh
```

Windows:
```powershell
.\scripts\run_api.ps1
```

**Option B: Manual Start**

Terminal 1 - Celery Worker:
```bash
celery -A neural.api.celery_app worker --loglevel=info
```

Terminal 2 - API Server:
```bash
python -m neural.api.main
```

### 4. Verify Installation

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "services": {
    "api": "healthy",
    "celery": "healthy"
  }
}
```

## First Request

### 1. Open API Documentation

Visit http://localhost:8000/docs in your browser.

### 2. Try the API

**Compile a Simple Model:**

```bash
curl -X POST "http://localhost:8000/compile/sync" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model SimpleMLP { Input(shape=[784]) Dense(units=128, activation=relu) Dense(units=10, activation=softmax) }",
    "backend": "tensorflow",
    "dataset": "MNIST"
  }'
```

**Submit Training Job:**

```bash
curl -X POST "http://localhost:8000/jobs/train" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model SimpleMLP { Input(shape=[784]) Dense(units=64, activation=relu) Dense(units=10, activation=softmax) }",
    "backend": "tensorflow",
    "dataset": "MNIST",
    "training_config": {
      "epochs": 5,
      "batch_size": 32
    }
  }'
```

## Using Python

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {
    "X-API-Key": "demo_api_key_12345",
    "Content-Type": "application/json"
}

# Compile model
response = requests.post(
    f"{BASE_URL}/compile/sync",
    headers=HEADERS,
    json={
        "dsl_code": """
        Model SimpleCNN {
            Input(shape=[28, 28, 1])
            Conv2D(filters=32, kernel_size=3, activation=relu)
            MaxPooling2D(pool_size=2)
            Flatten()
            Dense(units=10, activation=softmax)
        }
        """,
        "backend": "tensorflow"
    }
)

print(response.json()["compiled_code"])
```

## Run Tests

```bash
# Start the API server first
python neural/api/test_api.py
```

## Docker Deployment

For a complete setup with all services:

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

Services:
- API: http://localhost:8000
- Flower (Celery monitoring): http://localhost:5555

## Next Steps

- 📖 Read the [API Guide](API_GUIDE.md) for detailed endpoint documentation
- 🚀 Check the [Deployment Guide](DEPLOYMENT.md) for production deployment
- 💡 Explore examples in `neural/api/test_api.py`
- 📚 View API docs at http://localhost:8000/docs

## Troubleshooting

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If not, start Redis
docker start neural-redis
```

### Celery Worker Issues

```bash
# Check worker status
celery -A neural.api.celery_app inspect active

# Restart worker
pkill -f celery
celery -A neural.api.celery_app worker --loglevel=info
```

### API Server Issues

```bash
# Check if port 8000 is in use
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill existing process if needed
kill -9 <PID>
```

### Import Errors

```bash
# Reinstall with API dependencies
pip install -e ".[api]" --force-reinstall
```

## Common Commands

```bash
# Check API configuration
python -c "from neural.api.config import settings; print(settings.redis_url)"

# Test Redis connection
python -c "import redis; r = redis.Redis(); print(r.ping())"

# List running Docker containers
docker ps

# View API logs
tail -f logs/api.log

# Monitor Celery tasks
celery -A neural.api.celery_app flower
```

## Getting Help

- Check logs: `docker-compose logs api`
- View worker status: `celery -A neural.api.celery_app status`
- Test endpoint: `curl http://localhost:8000/health`
- GitHub Issues: [repository-url]/issues

## Example Workflows

### Complete Training Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "demo_api_key_12345"
HEADERS = {"X-API-Key": API_KEY}

# 1. Submit training job
response = requests.post(
    f"{BASE_URL}/jobs/train",
    headers=HEADERS,
    json={
        "dsl_code": "Model MyModel { ... }",
        "backend": "tensorflow",
        "dataset": "MNIST",
        "training_config": {"epochs": 10}
    }
)
job_id = response.json()["job_id"]

# 2. Poll for completion
while True:
    status_response = requests.get(
        f"{BASE_URL}/jobs/{job_id}",
        headers=HEADERS
    )
    status = status_response.json()["status"]
    
    if status in ["completed", "failed"]:
        break
    
    time.sleep(5)

# 3. Get experiment results
if status == "completed":
    exp_id = status_response.json()["result"]["experiment_id"]
    exp_response = requests.get(
        f"{BASE_URL}/experiments/{exp_id}",
        headers=HEADERS
    )
    print(exp_response.json())
```

Congratulations! You're now ready to use the Neural API! 🎉
