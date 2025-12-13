# Integration Guide

This guide shows how to integrate Neural monitoring into your ML pipelines and applications.

## Table of Contents

- [Basic Integration](#basic-integration)
- [Flask/FastAPI Integration](#flaskfastapi-integration)
- [Batch Processing Integration](#batch-processing-integration)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Grafana Setup](#grafana-setup)
- [Alert Configuration](#alert-configuration)

## Basic Integration

### Step 1: Install Dependencies

```bash
pip install neural-dsl[monitoring]
# For full dashboard support
pip install neural-dsl[monitoring,dashboard,visualization]
```

### Step 2: Initialize Monitor

```python
from neural.monitoring import ModelMonitor

monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    storage_path="/path/to/monitoring/data",
    enable_prometheus=True,
    enable_alerting=True,
    enable_slo_tracking=True
)
```

### Step 3: Set Reference Data

```python
import numpy as np

# Load your training/validation data
reference_data = np.load("reference_data.npy")
reference_predictions = np.load("reference_predictions.npy")
reference_performance = {
    'accuracy': 0.95,
    'f1': 0.94
}

monitor.set_reference_data(
    data=reference_data,
    predictions=reference_predictions,
    performance=reference_performance
)
```

### Step 4: Log Predictions

```python
# In your prediction endpoint
def predict(input_data):
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log to monitor
    monitor.log_prediction(
        prediction_id=str(uuid.uuid4()),
        input_features=input_data,
        prediction=prediction,
        latency_ms=latency_ms
    )
    
    return prediction
```

## Flask/FastAPI Integration

### Flask Example

```python
from flask import Flask, request, jsonify
from neural.monitoring import ModelMonitor
import time

app = Flask(__name__)

# Initialize monitor
monitor = ModelMonitor(
    model_name="flask-model",
    model_version="1.0",
    enable_prometheus=True
)

# Add Prometheus metrics endpoint
from neural.monitoring.prometheus_exporter import PrometheusExporter
exporter = PrometheusExporter(registry=monitor.metrics_registry)

@app.route('/metrics')
def metrics():
    return exporter.get_metrics_text(), 200, {'Content-Type': 'text/plain'}

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    try:
        # Parse input
        data = request.get_json()
        
        # Validate and process
        # ... your prediction logic ...
        prediction = model.predict(data['features'])
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log prediction
        monitor.log_prediction(
            prediction_id=data.get('request_id', str(uuid.uuid4())),
            input_features=data['features'],
            prediction=prediction,
            latency_ms=latency_ms
        )
        
        return jsonify({
            'prediction': prediction,
            'request_id': data.get('request_id')
        })
        
    except Exception as e:
        monitor.record_error(type(e).__name__)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neural.monitoring import ModelMonitor
import time
import uuid

app = FastAPI()

# Initialize monitor
monitor = ModelMonitor(
    model_name="fastapi-model",
    model_version="1.0",
    enable_prometheus=True
)

# Add Prometheus metrics endpoint
from neural.monitoring.prometheus_exporter import PrometheusExporter
exporter = PrometheusExporter(registry=monitor.metrics_registry)

@app.get('/metrics')
def metrics():
    return exporter.get_metrics_text()

class PredictionRequest(BaseModel):
    features: dict
    request_id: str = None

class PredictionResponse(BaseModel):
    prediction: float
    request_id: str

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Make prediction
        # ... your prediction logic ...
        prediction = model.predict(request.features)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate request ID if not provided
        request_id = request.request_id or str(uuid.uuid4())
        
        # Log prediction
        monitor.log_prediction(
            prediction_id=request_id,
            input_features=request.features,
            prediction=prediction,
            latency_ms=latency_ms
        )
        
        return PredictionResponse(
            prediction=prediction,
            request_id=request_id
        )
        
    except Exception as e:
        monitor.record_error(type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    report = monitor.generate_health_report()
    return report
```

## Batch Processing Integration

```python
from neural.monitoring import ModelMonitor
import numpy as np
import time

class BatchPredictor:
    def __init__(self, model, monitor: ModelMonitor):
        self.model = model
        self.monitor = monitor
        
    def predict_batch(self, batch_data, batch_ids=None):
        """Process a batch of predictions with monitoring."""
        results = []
        
        for i, data in enumerate(batch_data):
            start_time = time.time()
            
            try:
                # Make prediction
                prediction = self.model.predict(data)
                latency_ms = (time.time() - start_time) * 1000
                
                # Log prediction
                pred_id = batch_ids[i] if batch_ids else f"batch_{i}"
                self.monitor.log_prediction(
                    prediction_id=pred_id,
                    input_features=data,
                    prediction=prediction,
                    latency_ms=latency_ms,
                    metadata={'batch': True}
                )
                
                results.append(prediction)
                
            except Exception as e:
                self.monitor.record_error(type(e).__name__)
                results.append(None)
        
        # Check drift periodically
        if len(batch_data) >= 100:
            drift_report = self.monitor.check_drift(
                data=np.array(batch_data)
            )
            if drift_report['is_drifting']:
                print(f"Warning: Drift detected with severity {drift_report['drift_severity']}")
        
        return results

# Usage
monitor = ModelMonitor(model_name="batch-model", model_version="1.0")
predictor = BatchPredictor(model, monitor)

batch_data = load_batch_data()
predictions = predictor.predict_batch(batch_data)
```

## Kubernetes Deployment

### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-model-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-model
  template:
    metadata:
      labels:
        app: neural-model
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: model-service
        image: your-registry/neural-model:latest
        ports:
        - containerPort: 5000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MONITORING_STORAGE_PATH
          value: "/data/monitoring"
        - name: SLACK_WEBHOOK_URL
          valueFrom:
            secretKeyRef:
              name: monitoring-secrets
              key: slack-webhook
        volumeMounts:
        - name: monitoring-data
          mountPath: /data/monitoring
      volumes:
      - name: monitoring-data
        persistentVolumeClaim:
          claimName: monitoring-pvc
```

### service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-model-service
spec:
  selector:
    app: neural-model
  ports:
  - name: http
    port: 80
    targetPort: 5000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### servicemonitor.yaml (for Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-model-monitor
spec:
  selector:
    matchLabels:
      app: neural-model
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Grafana Setup

### 1. Add Prometheus Data Source

1. Go to Configuration → Data Sources
2. Add Prometheus
3. URL: `http://prometheus:9090`
4. Save & Test

### 2. Import Dashboard

1. Go to Dashboards → Import
2. Upload `neural/monitoring/grafana_dashboard.json`
3. Select Prometheus data source
4. Import

### 3. Custom Dashboard

Create panels for specific metrics:

**Prediction Rate:**
```promql
rate(neural_predictions_total{model="your-model"}[5m])
```

**P95 Latency:**
```promql
histogram_quantile(0.95, rate(neural_prediction_latency_seconds_bucket{model="your-model"}[5m]))
```

**Accuracy:**
```promql
neural_model_accuracy{model="your-model"}
```

**Drift Score:**
```promql
neural_drift_score{model="your-model", drift_type="concept"}
```

### 4. Set Up Alerts

Create alert rules in Grafana:

**High Latency Alert:**
```yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(neural_prediction_latency_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High prediction latency"
    description: "P95 latency is {{ $value }}s"
```

**Model Drift Alert:**
```yaml
- alert: ModelDrift
  expr: neural_drift_score{drift_type="concept"} > 0.2
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Model drift detected"
    description: "Drift score is {{ $value }}"
```

## Alert Configuration

### Slack Integration

1. Create incoming webhook in Slack
2. Configure in monitor:

```python
alert_config = {
    'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
}

monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    alert_config=alert_config,
    enable_alerting=True
)
```

### Email Integration

```python
alert_config = {
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from_addr': 'your-email@gmail.com',
        'to_addrs': ['team@example.com', 'oncall@example.com']
    }
}
```

### Custom Alert Rules

```python
from neural.monitoring import AlertRule, AlertSeverity, AlertChannel

# Create custom rule
custom_rule = AlertRule(
    name="Custom Metric Alert",
    condition=lambda data: data.get('custom_metric', 0) > threshold,
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
    cooldown_seconds=300,
    message_template="Custom metric exceeded: {custom_metric}"
)

monitor.alert_manager.add_rule(custom_rule)
```

## Environment Variables

Set these environment variables for configuration:

```bash
# Storage
export NEURAL_MONITORING_PATH=/data/monitoring

# Prometheus
export NEURAL_METRICS_PORT=9090

# Alerts
export NEURAL_SLACK_WEBHOOK=https://hooks.slack.com/services/...
export NEURAL_ALERT_EMAIL=alerts@example.com

# SLO Targets
export NEURAL_SLO_AVAILABILITY=0.999
export NEURAL_SLO_LATENCY_MS=100
export NEURAL_SLO_ACCURACY=0.95
```

## Best Practices

1. **Initialize Early**: Set up monitoring before deploying to production
2. **Set Reference Data**: Always establish baseline metrics
3. **Monitor Gradually**: Start with basic logging, add features incrementally
4. **Tune Thresholds**: Adjust based on your model's behavior
5. **Regular Reviews**: Check monitoring dashboards regularly
6. **Automate Responses**: Set up automated actions for critical alerts
7. **Document Runbooks**: Create procedures for handling alerts
8. **Test Alerting**: Verify alert delivery before relying on it
9. **Monitor the Monitor**: Ensure monitoring system is healthy
10. **Version Control**: Track monitoring configuration changes

## Troubleshooting

### Metrics Not Appearing

1. Check Prometheus is scraping: `curl http://localhost:9090/metrics`
2. Verify service annotations in Kubernetes
3. Check firewall rules

### Alerts Not Sending

1. Test webhook URLs manually
2. Verify email credentials
3. Check network connectivity
4. Review cooldown periods

### High Memory Usage

1. Reduce batch size for prediction logging
2. Enable sampling for high-traffic models
3. Adjust storage retention policies

### Dashboard Not Loading

1. Check Dash dependencies: `pip install dash plotly`
2. Verify storage path exists
3. Check port availability

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See README.md
- Examples: See neural/monitoring/examples/
