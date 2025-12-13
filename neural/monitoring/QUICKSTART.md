# Quick Start Guide

Get started with Neural monitoring in 5 minutes.

## Installation

```bash
# Basic installation
pip install neural-dsl[monitoring]

# With dashboard support
pip install neural-dsl[monitoring,dashboard,visualization]
```

## 1-Minute Setup

```python
import numpy as np
from neural.monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0"
)

# Set reference data
reference_data = np.random.randn(1000, 10)
monitor.set_reference_data(data=reference_data)

# Log a prediction
monitor.log_prediction(
    prediction_id="pred_001",
    input_features={'feature_0': 0.5, 'feature_1': 1.2},
    prediction=1,
    latency_ms=25.0
)

# Check status
summary = monitor.get_monitoring_summary()
print(f"Total predictions: {summary['total_predictions']}")
```

## Common Use Cases

### 1. Monitor Predictions

```python
# In your prediction function
def predict(input_data):
    import time
    start = time.time()
    
    prediction = model.predict(input_data)
    
    monitor.log_prediction(
        prediction_id=generate_id(),
        input_features=input_data,
        prediction=prediction,
        latency_ms=(time.time() - start) * 1000
    )
    
    return prediction
```

### 2. Check for Drift

```python
# Periodically check drift
new_data = collect_recent_data()
drift_report = monitor.check_drift(data=new_data)

if drift_report['is_drifting']:
    print(f"⚠ Drift detected! Severity: {drift_report['drift_severity']}")
```

### 3. Monitor Data Quality

```python
# Check incoming data quality
quality_report = monitor.check_data_quality(new_data)

if not quality_report['is_healthy']:
    print(f"⚠ Data quality issues: {quality_report['warnings']}")
```

### 4. Track Performance

```python
# Update model performance metrics
monitor.update_performance_metrics({
    'accuracy': 0.95,
    'precision': 0.94,
    'recall': 0.96,
    'f1': 0.95
})
```

### 5. Set Up Alerts

```python
# Configure Slack alerts
monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    enable_alerting=True,
    alert_config={
        'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK'
    }
)

# Alerts will be sent automatically when issues are detected
```

## CLI Quick Reference

```bash
# Initialize monitoring
neural monitor init --model-name my-model --model-version 1.0

# Check status
neural monitor status

# View drift report
neural monitor drift --window 100

# View quality report
neural monitor quality --window 100

# View alerts
neural monitor alerts --hours 24

# Check SLO status
neural monitor slo

# Health check
neural monitor health

# Start dashboard
neural monitor dashboard --port 8052

# Start Prometheus metrics
neural monitor prometheus --port 9090
```

## Dashboard Access

After running `neural monitor dashboard`, access at:
- **URL**: http://localhost:8052
- **Features**: Real-time metrics, drift charts, quality trends, alerts

## Prometheus Integration

```python
# Start metrics server
monitor.start_prometheus_server(port=9090)

# Metrics available at: http://localhost:9090/metrics
```

## Next Steps

1. **Read the Full Documentation**: See [README.md](README.md)
2. **Integration Guide**: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
3. **Run Examples**: 
   ```bash
   python neural/monitoring/examples/basic_monitoring.py
   python neural/monitoring/examples/production_deployment.py
   python neural/monitoring/examples/drift_detection_demo.py
   ```
4. **Set Up Alerts**: Configure Slack/Email notifications
5. **Deploy Dashboard**: Make it accessible to your team
6. **Configure SLOs**: Define targets for your model

## Common Issues

**ImportError: prometheus_client**
```bash
pip install prometheus-client
```

**ImportError: dash**
```bash
pip install dash plotly dash-bootstrap-components
```

**No data in dashboard**
- Ensure you've logged some predictions first
- Check storage path is correct

## Support

- Documentation: [README.md](README.md)
- Examples: `neural/monitoring/examples/`
- Issues: https://github.com/Lemniscate-world/Neural/issues
