# Neural DSL REST API

A comprehensive REST API server for Neural DSL using FastAPI with async job processing, experiment tracking, and deployment management.

## Features

- **Model Compilation**: Compile Neural DSL code to TensorFlow, PyTorch, or ONNX
- **Training Jobs**: Submit asynchronous training jobs with real-time progress tracking
- **Experiment Tracking**: Track experiments with metrics, hyperparameters, and artifacts
- **Deployment Management**: Deploy trained models with configurable resources
- **Authentication**: API key and JWT token authentication
- **Rate Limiting**: Built-in rate limiting to protect server resources
- **Webhook Notifications**: Real-time notifications for job status updates
- **OpenAPI Documentation**: Interactive API documentation with Swagger UI
- **Async Job Queue**: Celery with Redis for background task processing

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -e ".[api]"
pip install fastapi uvicorn celery redis python-jose passlib pydantic-settings
```

2. **Start Redis** (required for Celery):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

3. **Start Celery worker**:
```bash
celery -A neural.api.celery_app worker --loglevel=info
```

4. **Start API server**:
```bash
python -m neural.api.main
# Or with uvicorn:
uvicorn neural.api.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access API documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### Docker Deployment

1. **Build and start all services**:
```bash
docker-compose up -d
```

This starts:
- `api`: FastAPI server on port 8000
- `worker`: Celery worker for async tasks
- `redis`: Redis for message broker
- `flower`: Celery monitoring UI on port 5555

2. **View logs**:
```bash
docker-compose logs -f api
docker-compose logs -f worker
```

3. **Stop services**:
```bash
docker-compose down
```

## Configuration

Configuration is managed through environment variables or `.env` file:

```env
# Server settings
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
API_KEY_HEADER=X-API-Key

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Redis/Celery
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Storage
STORAGE_PATH=./neural_storage
EXPERIMENTS_PATH=./neural_experiments
MODELS_PATH=./neural_models

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

## API Usage

### Authentication

Include your API key in the request header:
```bash
curl -H "X-API-Key: demo_api_key_12345" http://localhost:8000/experiments/
```

Default demo API key: `demo_api_key_12345`

### Compile Model

```bash
curl -X POST "http://localhost:8000/compile/" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model MyModel {\n  Input(shape=[28, 28, 1])\n  Conv2D(filters=32, kernel_size=3, activation=relu)\n  Dense(units=10, activation=softmax)\n}",
    "backend": "tensorflow",
    "dataset": "MNIST"
  }'
```

### Submit Training Job

```bash
curl -X POST "http://localhost:8000/jobs/train" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model MyModel {...}",
    "backend": "tensorflow",
    "dataset": "MNIST",
    "training_config": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "experiment_name": "my_experiment",
    "webhook_url": "https://your-webhook-url.com/notify"
  }'
```

### Check Job Status

```bash
curl "http://localhost:8000/jobs/{job_id}" \
  -H "X-API-Key: demo_api_key_12345"
```

### List Experiments

```bash
curl "http://localhost:8000/experiments/?skip=0&limit=10" \
  -H "X-API-Key: demo_api_key_12345"
```

### Get Experiment Details

```bash
curl "http://localhost:8000/experiments/{experiment_id}" \
  -H "X-API-Key: demo_api_key_12345"
```

## API Endpoints

### Compilation
- `POST /compile/` - Submit async compilation job
- `POST /compile/sync` - Synchronous compilation

### Jobs
- `POST /jobs/train` - Submit training job
- `GET /jobs/{job_id}` - Get job status
- `DELETE /jobs/{job_id}` - Cancel job

### Experiments
- `GET /experiments/` - List all experiments
- `GET /experiments/{experiment_id}` - Get experiment details
- `DELETE /experiments/{experiment_id}` - Delete experiment
- `GET /experiments/{experiment_id}/artifacts/{artifact_name}` - Download artifact
- `GET /experiments/{experiment_id}/compare` - Compare experiments

### Deployments
- `POST /deployments/` - Create deployment
- `GET /deployments/` - List deployments
- `GET /deployments/{deployment_id}` - Get deployment details
- `DELETE /deployments/{deployment_id}` - Delete deployment
- `POST /deployments/{deployment_id}/scale` - Scale deployment

### Models
- `GET /models/` - List compiled models
- `GET /models/{model_id}` - Get model details
- `GET /models/{model_id}/download` - Download model
- `DELETE /models/{model_id}` - Delete model

### System
- `GET /` - API information
- `GET /health` - Health check

## Webhooks

Async jobs support webhook notifications. Provide a `webhook_url` when creating jobs:

```json
{
  "webhook_url": "https://your-webhook-url.com/notify"
}
```

Webhook payload format:
```json
{
  "job_id": "abc123",
  "event": "completed",
  "status": "completed",
  "data": {
    "experiment_id": "xyz789",
    "final_metrics": {"accuracy": 0.95}
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Events: `started`, `progress`, `completed`, `failed`

## Rate Limiting

API implements rate limiting with configurable limits:
- Default: 100 requests per 60 seconds
- Per client (IP or API key)
- Response headers include rate limit info

## Monitoring

### Celery Flower

Monitor Celery tasks via Flower UI:
```bash
# Local
celery -A neural.api.celery_app flower --port=5555

# Docker
# Flower is included in docker-compose.yml
```

Access at: http://localhost:5555

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "api": "healthy",
    "celery": "healthy"
  }
}
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       v
┌─────────────┐     ┌─────────────┐
│  FastAPI    │────▶│   Redis     │
│   Server    │     │  (Broker)   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │                   v
       │            ┌─────────────┐
       │            │   Celery    │
       │            │   Worker    │
       │            └──────┬──────┘
       │                   │
       v                   v
┌─────────────────────────────┐
│      Storage/Database       │
│  - Experiments              │
│  - Models                   │
│  - Artifacts                │
└─────────────────────────────┘
```

## Security

- **API Keys**: Default demo key for development only
- **Production**: Set strong `SECRET_KEY` environment variable
- **HTTPS**: Use reverse proxy (nginx) with SSL/TLS in production
- **Rate Limiting**: Enabled by default
- **CORS**: Configure allowed origins

## Troubleshooting

### Redis Connection Error
```bash
# Check Redis is running
docker ps | grep redis
redis-cli ping
```

### Celery Worker Not Starting
```bash
# Check broker URL
celery -A neural.api.celery_app inspect ping
```

### API Server Error
```bash
# Check logs
docker-compose logs -f api

# Restart services
docker-compose restart api
```

## Development

### Running Tests
```bash
pytest tests/api/ -v
```

### Code Quality
```bash
# Linting
ruff check neural/api/

# Type checking
mypy neural/api/
```

## Production Deployment

For production deployment:

1. Set strong `SECRET_KEY`
2. Use PostgreSQL instead of SQLite
3. Configure proper CORS origins
4. Enable HTTPS with reverse proxy
5. Set up monitoring and logging
6. Configure backup for data volumes
7. Use environment-specific docker-compose files

Example production docker-compose:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## License

MIT License - see LICENSE.md for details
