# Neural DSL API - Complete Guide

Comprehensive guide for using the Neural DSL REST API.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Endpoints](#endpoints)
5. [Webhooks](#webhooks)
6. [Error Handling](#error-handling)
7. [Examples](#examples)
8. [SDK & Client Libraries](#sdk--client-libraries)

## Overview

The Neural DSL API provides programmatic access to:
- Model compilation (DSL to TensorFlow/PyTorch/ONNX)
- Asynchronous training jobs
- Experiment tracking and management
- Model deployment
- Job monitoring

**Base URL**: `http://localhost:8000` (development)  
**API Version**: 0.3.0  
**Documentation**: `/docs` (Swagger UI) or `/redoc` (ReDoc)

## Authentication

### API Key Authentication

Include your API key in the request header:

```bash
X-API-Key: your_api_key_here
```

**Example:**
```bash
curl -H "X-API-Key: demo_api_key_12345" \
  http://localhost:8000/experiments/
```

### Default Credentials

For development/testing:
- API Key: `demo_api_key_12345`
- User: `demo_user`

⚠️ **Production**: Generate and use secure API keys!

### Generating API Keys

```python
from neural.api.auth import generate_api_key

api_key = generate_api_key()
print(f"New API Key: {api_key}")
```

## Rate Limiting

API requests are rate-limited to prevent abuse:
- **Default**: 100 requests per 60 seconds
- **Per**: Client IP or API key
- **Configurable**: Via environment variables

### Rate Limit Headers

Response includes rate limit information:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 45
```

### Exceeding Limits

**HTTP 429 Too Many Requests:**
```json
{
  "error": "Rate limit exceeded. Try again in 45 seconds."
}
```

## Endpoints

### System Endpoints

#### Health Check
```http
GET /health
```

**Response:**
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

### Compilation Endpoints

#### Async Compilation
```http
POST /compile/
```

**Request:**
```json
{
  "dsl_code": "Model MyModel { ... }",
  "backend": "tensorflow",
  "dataset": "MNIST",
  "auto_flatten_output": false,
  "enable_hpo": false
}
```

**Response (202):**
```json
{
  "job_id": "abc123",
  "status": "pending",
  "message": "Compilation job submitted successfully",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Sync Compilation
```http
POST /compile/sync
```

**Request:** Same as async  
**Response (200):** Includes `compiled_code` field

### Job Endpoints

#### Submit Training Job
```http
POST /jobs/train
```

**Request:**
```json
{
  "dsl_code": "Model MyModel { ... }",
  "backend": "tensorflow",
  "dataset": "MNIST",
  "training_config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
  },
  "experiment_name": "my_experiment",
  "webhook_url": "https://webhook.site/unique-id",
  "hyperparameters": {
    "dropout_rate": 0.5
  }
}
```

**Response (202):**
```json
{
  "job_id": "xyz789",
  "status": "pending",
  "message": "Training job submitted successfully",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Get Job Status
```http
GET /jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "xyz789",
  "status": "running",
  "progress": 45.0,
  "result": {
    "current_epoch": 5,
    "experiment_id": "exp123"
  },
  "error": null,
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:10Z",
  "completed_at": null
}
```

#### Cancel Job
```http
DELETE /jobs/{job_id}
```

**Response (204):** No content

### Experiment Endpoints

#### List Experiments
```http
GET /experiments/?skip=0&limit=10&status_filter=completed
```

**Response:**
```json
{
  "experiments": [
    {
      "experiment_id": "exp123",
      "experiment_name": "my_experiment",
      "status": "completed",
      "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32
      },
      "metrics": {
        "latest": {
          "accuracy": 0.95,
          "loss": 0.15
        },
        "best": {
          "accuracy": {
            "value": 0.96,
            "step": 8
          }
        },
        "history": []
      },
      "artifacts": ["model.h5", "plot.png"],
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T13:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 10
}
```

#### Get Experiment
```http
GET /experiments/{experiment_id}
```

**Response:** Single experiment object with full details

#### Delete Experiment
```http
DELETE /experiments/{experiment_id}
```

**Response (204):** No content

#### Download Artifact
```http
GET /experiments/{experiment_id}/artifacts/{artifact_name}
```

**Response:** Binary file download

#### Compare Experiments
```http
GET /experiments/{experiment_id}/compare?compare_with=exp2&compare_with=exp3
```

**Response:**
```json
{
  "experiments": [
    {
      "experiment_id": "exp1",
      "experiment_name": "experiment_1",
      "metrics": [...]
    }
  ],
  "metrics_comparison": {},
  "hyperparameter_comparison": {}
}
```

### Deployment Endpoints

#### Create Deployment
```http
POST /deployments/
```

**Request:**
```json
{
  "model_id": "model123",
  "deployment_name": "my-model-v1",
  "backend": "tensorflow",
  "config": {
    "replicas": 2,
    "memory_limit": "1Gi",
    "cpu_limit": "1000m",
    "port": 8080,
    "health_check_path": "/health"
  },
  "environment": {
    "MODEL_VERSION": "v1.0"
  }
}
```

**Response (202):**
```json
{
  "deployment_id": "deploy123",
  "deployment_name": "my-model-v1",
  "status": "deploying",
  "endpoint": null,
  "message": "Deployment job submitted successfully",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### List Deployments
```http
GET /deployments/?skip=0&limit=10
```

#### Get Deployment
```http
GET /deployments/{deployment_id}
```

#### Delete Deployment
```http
DELETE /deployments/{deployment_id}
```

#### Scale Deployment
```http
POST /deployments/{deployment_id}/scale?replicas=5
```

### Model Endpoints

#### List Models
```http
GET /models/?skip=0&limit=10&backend_filter=tensorflow
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "model123",
      "name": "my_model_tensorflow.py",
      "backend": "tensorflow",
      "size": 15420,
      "created_at": "2024-01-01T12:00:00Z",
      "metadata": {}
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 10
}
```

#### Get Model
```http
GET /models/{model_id}
```

#### Download Model
```http
GET /models/{model_id}/download
```

**Response:** Python source code file

#### Delete Model
```http
DELETE /models/{model_id}
```

## Webhooks

### Configuration

Provide `webhook_url` when creating jobs:
```json
{
  "webhook_url": "https://your-webhook-url.com/notify"
}
```

### Payload Format

```json
{
  "job_id": "abc123",
  "event": "completed",
  "status": "completed",
  "data": {
    "experiment_id": "exp123",
    "final_metrics": {
      "accuracy": 0.95
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Event Types

- `started`: Job started processing
- `progress`: Job progress update (training)
- `completed`: Job completed successfully
- `failed`: Job failed with error

### Example Webhook Handler (Flask)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    payload = request.json
    
    job_id = payload['job_id']
    event = payload['event']
    status = payload['status']
    
    print(f"Job {job_id}: {event} - {status}")
    
    if event == 'completed':
        # Handle completion
        pass
    elif event == 'failed':
        # Handle failure
        pass
    
    return jsonify({'status': 'received'}), 200
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `202`: Accepted (async operation)
- `204`: No Content (successful deletion)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

### Error Response Format

```json
{
  "error": "Brief error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Errors

**Authentication Failed:**
```json
{
  "error": "API key required",
  "detail": null
}
```

**Rate Limit Exceeded:**
```json
{
  "error": "Rate limit exceeded. Try again in 45 seconds.",
  "detail": null
}
```

**Validation Error:**
```json
{
  "error": "Validation error",
  "detail": {
    "field": "dsl_code",
    "message": "Field required"
  }
}
```

## Examples

### Complete Training Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "demo_api_key_12345"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. Submit training job
response = requests.post(
    f"{BASE_URL}/jobs/train",
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
        "backend": "tensorflow",
        "dataset": "MNIST",
        "training_config": {
            "epochs": 10,
            "batch_size": 32
        },
        "experiment_name": "mnist_cnn"
    }
)

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# 2. Monitor progress
while True:
    response = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=HEADERS)
    data = response.json()
    
    status = data["status"]
    progress = data.get("progress", 0)
    
    print(f"Status: {status} - Progress: {progress}%")
    
    if status in ["completed", "failed"]:
        break
    
    time.sleep(5)

# 3. Get experiment details
if status == "completed":
    experiment_id = data["result"]["experiment_id"]
    response = requests.get(
        f"{BASE_URL}/experiments/{experiment_id}",
        headers=HEADERS
    )
    
    experiment = response.json()
    print(f"Final accuracy: {experiment['metrics']['latest']['accuracy']}")
```

### Using Webhooks

```python
# Submit job with webhook
response = requests.post(
    f"{BASE_URL}/jobs/train",
    headers=HEADERS,
    json={
        "dsl_code": "...",
        "backend": "tensorflow",
        "dataset": "MNIST",
        "training_config": {"epochs": 10},
        "webhook_url": "https://webhook.site/unique-id"
    }
)

# Your webhook endpoint will receive notifications automatically
```

## SDK & Client Libraries

### Python Client

```python
from neural.api.client import NeuralAPIClient

client = NeuralAPIClient(
    base_url="http://localhost:8000",
    api_key="demo_api_key_12345"
)

# Compile model
job = client.compile(
    dsl_code="Model MyModel { ... }",
    backend="tensorflow"
)

# Wait for completion
result = job.wait()
print(result.compiled_code)

# Train model
training_job = client.train(
    dsl_code="Model MyModel { ... }",
    backend="tensorflow",
    epochs=10
)

# List experiments
experiments = client.list_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.metrics}")
```

### JavaScript/TypeScript Client

```typescript
import { NeuralAPIClient } from 'neural-api-client';

const client = new NeuralAPIClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'demo_api_key_12345'
});

// Compile model
const job = await client.compile({
  dslCode: 'Model MyModel { ... }',
  backend: 'tensorflow'
});

// Get job status
const status = await client.getJobStatus(job.jobId);
console.log(status);
```

### cURL Examples

**Compile Model:**
```bash
curl -X POST "http://localhost:8000/compile/sync" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model MyModel { Input(shape=[784]) Dense(units=10, activation=softmax) }",
    "backend": "tensorflow"
  }'
```

**Train Model:**
```bash
curl -X POST "http://localhost:8000/jobs/train" \
  -H "X-API-Key: demo_api_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "dsl_code": "Model MyModel { ... }",
    "backend": "tensorflow",
    "dataset": "MNIST",
    "training_config": {
      "epochs": 10,
      "batch_size": 32
    }
  }'
```

**Check Status:**
```bash
curl "http://localhost:8000/jobs/{job_id}" \
  -H "X-API-Key: demo_api_key_12345"
```

## Best Practices

1. **Use async endpoints** for long-running operations
2. **Implement webhook handlers** for real-time updates
3. **Cache results** when appropriate
4. **Handle rate limits** with exponential backoff
5. **Validate DSL code** before submission
6. **Monitor experiments** regularly
7. **Clean up old experiments** and models
8. **Use meaningful experiment names**
9. **Set appropriate timeouts** for HTTP requests
10. **Implement proper error handling**

## Support & Resources

- **API Docs**: http://localhost:8000/docs
- **GitHub**: [repository-url]
- **Issues**: [repository-url]/issues
- **Email**: support@your-domain.com

## Changelog

### v0.3.0 (2024-01-01)
- Initial API release
- Compilation, training, and deployment endpoints
- Experiment tracking
- Webhook support
- Rate limiting
- Authentication
