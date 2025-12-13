# Neural Monitoring & Observability

Comprehensive production monitoring and observability system for Neural DSL models.

## Features

### 🔍 Model Performance Drift Detection
- Multiple statistical methods (KS test, PSI, Wasserstein distance)
- Feature-level drift tracking
- Prediction distribution drift
- Performance metrics drift
- Concept drift detection with combined scoring

### 📊 Data Quality Monitoring
- Missing value detection
- Outlier detection using z-score
- Invalid value checks (type constraints)
- Schema validation
- Statistical drift from reference data
- Quality scoring and health checks

### 📝 Prediction Logging & Analysis
- Efficient batch logging
- Sampling support
- Latency tracking
- Confidence analysis
- Performance metrics calculation
- Anomaly detection in predictions

### 🚨 Alerting System
- Multiple channels: Slack, Email, Webhook, Log
- Configurable alert rules
- Severity levels (Info, Warning, Critical)
- Cooldown periods to prevent alert storms
- Predefined rules for common issues

### 📈 Prometheus/Grafana Integration
- Standard metrics export
- Counter, Gauge, Histogram support
- Custom metrics registry
- HTTP endpoint for metrics
- Fallback to simple collector when Prometheus unavailable

### 🎯 SLO/SLA Tracking
- Multiple SLO types: Availability, Latency, Accuracy, Error Rate, Throughput
- Time-windowed measurements
- Compliance rate calculation
- Error budget tracking
- Breach detection and duration tracking
- SLA report generation

### 💻 Dashboard UI
- Real-time monitoring visualization
- Drift charts
- Data quality trends
- Latency distribution
- SLO compliance visualization
- Recent alerts display
- Auto-refresh capability

## Installation

Install with monitoring dependencies:

```bash
pip install -e ".[full]"
# Or specific monitoring dependencies
pip install prometheus-client dash plotly dash-bootstrap-components
```

## Quick Start

### 1. Initialize Monitoring

```bash
neural monitor init \
  --model-name my-model \
  --model-version 1.0 \
  --enable-prometheus \
  --enable-alerting \
  --enable-slo \
  --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 2. Set Reference Data (Python)

```python
import numpy as np
from neural.monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    storage_path="monitoring_data",
    enable_prometheus=True,
    enable_alerting=True,
    enable_slo_tracking=True,
    alert_config={
        'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    }
)

# Set reference data for drift detection
reference_data = np.random.randn(1000, 10)
reference_predictions = np.random.randint(0, 2, 1000)
reference_performance = {'accuracy': 0.95, 'f1': 0.94}

monitor.set_reference_data(
    data=reference_data,
    predictions=reference_predictions,
    performance=reference_performance
)
```

### 3. Log Predictions

```python
# Log a prediction
monitor.log_prediction(
    prediction_id="pred_001",
    input_features={
        'feature_1': 0.5,
        'feature_2': 1.2,
        # ... more features
    },
    prediction=1,
    prediction_proba={'0': 0.3, '1': 0.7},
    ground_truth=1,  # If available
    latency_ms=25.5,
    metadata={'request_id': 'req_123'}
)
```

### 4. Check Drift

```python
# Check for drift with new data
new_data = np.random.randn(100, 10)
new_predictions = np.random.randint(0, 2, 100)
new_performance = {'accuracy': 0.92, 'f1': 0.91}

drift_report = monitor.check_drift(
    data=new_data,
    predictions=new_predictions,
    performance=new_performance
)

print(f"Drift detected: {drift_report['is_drifting']}")
print(f"Drift severity: {drift_report['drift_severity']}")
```

### 5. Check Data Quality

```python
# Check data quality
quality_report = monitor.check_data_quality(new_data)

print(f"Quality score: {quality_report['quality_score']:.3f}")
print(f"Is healthy: {quality_report['is_healthy']}")
print(f"Warnings: {quality_report['warnings']}")
```

### 6. Update Performance Metrics

```python
# Update model performance metrics
monitor.update_performance_metrics({
    'accuracy': 0.92,
    'precision': 0.91,
    'recall': 0.93,
    'f1': 0.92
})
```

### 7. View Status

```bash
# CLI commands
neural monitor status --format text
neural monitor drift --window 100
neural monitor quality --window 100
neural monitor alerts --hours 24
neural monitor slo
neural monitor health
```

### 8. Start Dashboard

```bash
# Start monitoring dashboard
neural monitor dashboard --port 8052

# Or start Prometheus metrics server
neural monitor prometheus --port 9090
```

## CLI Commands

### Initialize
```bash
neural monitor init [OPTIONS]
```
Options:
- `--model-name`: Model name (default: default)
- `--model-version`: Model version (default: 1.0)
- `--storage-path`: Storage path (default: monitoring_data)
- `--enable-prometheus`: Enable Prometheus metrics
- `--enable-alerting`: Enable alerting
- `--enable-slo`: Enable SLO tracking
- `--slack-webhook`: Slack webhook URL
- `--email-smtp`: SMTP server
- `--email-user`: Email username
- `--email-to`: Email recipients (multiple)

### Status
```bash
neural monitor status [OPTIONS]
```
Options:
- `--storage-path`: Storage path (default: monitoring_data)
- `--format`: Output format (text|json)

### Drift Report
```bash
neural monitor drift [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--window`: Number of samples to analyze (default: 100)
- `--format`: Output format (text|json)

### Quality Report
```bash
neural monitor quality [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--window`: Number of samples to analyze (default: 100)
- `--format`: Output format (text|json)

### Alert Summary
```bash
neural monitor alerts [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--hours`: Time window in hours (default: 24)
- `--severity`: Filter by severity (info|warning|critical)
- `--format`: Output format (text|json)

### SLO Status
```bash
neural monitor slo [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--name`: Specific SLO name
- `--format`: Output format (text|json)

### Health Check
```bash
neural monitor health [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--format`: Output format (text|json)

### Dashboard
```bash
neural monitor dashboard [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--port`: Dashboard port (default: 8052)
- `--host`: Dashboard host (default: localhost)

### Prometheus
```bash
neural monitor prometheus [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--port`: Metrics port (default: 9090)

## Architecture

```
neural/monitoring/
├── __init__.py                 # Module exports
├── monitor.py                  # Main ModelMonitor class
├── drift_detector.py           # Drift detection
├── data_quality.py             # Data quality monitoring
├── prediction_logger.py        # Prediction logging & analysis
├── alerting.py                 # Alert management
├── prometheus_exporter.py      # Prometheus integration
├── slo_tracker.py              # SLO/SLA tracking
├── cli_commands.py             # CLI commands
├── dashboard.py                # Dashboard UI
└── README.md                   # This file
```

## Alerting Configuration

### Slack Integration

```python
alert_config = {
    'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
}
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
        'to_addrs': ['recipient@example.com']
    }
}
```

### Custom Webhook

```python
alert_config = {
    'webhook_url': 'https://your-webhook-endpoint.com/alerts'
}
```

### Custom Alert Rules

```python
from neural.monitoring import AlertManager, AlertRule, AlertSeverity, AlertChannel

alert_manager = AlertManager(alert_config=alert_config)

# Add custom rule
custom_rule = AlertRule(
    name="High Error Rate",
    condition=lambda data: data.get('error_rate', 0) > 0.05,
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
    cooldown_seconds=600,
    message_template="Error rate is {error_rate:.2%}"
)

alert_manager.add_rule(custom_rule)
```

## SLO Configuration

### Predefined SLOs

```python
from neural.monitoring import (
    SLOTracker,
    create_availability_slo,
    create_latency_slo,
    create_accuracy_slo,
    create_error_rate_slo,
    create_throughput_slo
)

tracker = SLOTracker()

# Add SLOs
tracker.add_slo(create_availability_slo(target=0.999, window_hours=24))
tracker.add_slo(create_latency_slo(target_ms=100.0, window_hours=1))
tracker.add_slo(create_accuracy_slo(target=0.95, window_hours=24))
tracker.add_slo(create_error_rate_slo(target=0.01, window_hours=1))
tracker.add_slo(create_throughput_slo(target=100.0, window_hours=1))
```

### Custom SLO

```python
from neural.monitoring import SLO, SLOType

custom_slo = SLO(
    name="custom_metric",
    slo_type=SLOType.ACCURACY,
    target=0.98,
    window_seconds=3600,
    description="Custom accuracy SLO"
)

tracker.add_slo(custom_slo)
```

## Prometheus Integration

### Metrics Endpoint

The Prometheus exporter provides a `/metrics` endpoint with standard metrics:

- `neural_predictions_total`: Total predictions counter
- `neural_prediction_latency_seconds`: Prediction latency histogram
- `neural_model_accuracy`: Model accuracy gauge
- `neural_model_precision`: Model precision gauge
- `neural_model_recall`: Model recall gauge
- `neural_model_f1`: Model F1 score gauge
- `neural_drift_score`: Drift score gauge
- `neural_data_quality_score`: Data quality score gauge
- `neural_missing_values_rate`: Missing values rate gauge
- `neural_outlier_rate`: Outlier rate gauge
- `neural_errors_total`: Errors counter
- `neural_active_models`: Active models gauge

### Grafana Dashboard

Import the provided Grafana dashboard configuration to visualize metrics.

## Best Practices

1. **Set Reference Data**: Always set reference data before monitoring production traffic
2. **Monitor Gradually**: Start with logging, then add drift detection, then alerting
3. **Tune Thresholds**: Adjust drift and quality thresholds based on your model's behavior
4. **Alert Wisely**: Use cooldown periods to prevent alert fatigue
5. **Track SLOs**: Define and track SLOs that matter to your business
6. **Regular Reports**: Generate regular SLA reports for stakeholders
7. **Dashboard Monitoring**: Keep the dashboard visible for real-time monitoring

## Troubleshooting

### Missing Dependencies

If you get import errors:
```bash
pip install prometheus-client dash plotly dash-bootstrap-components
```

### Prometheus Not Available

The system falls back to a simple metrics collector if prometheus_client is not installed.

### Alert Delivery Issues

- Check webhook URLs are correct
- Verify email credentials
- Check network connectivity
- Review alert cooldown settings

## Examples

See `neural/monitoring/examples/` for complete examples:
- `basic_monitoring.py`: Basic monitoring setup
- `production_deployment.py`: Production deployment example
- `custom_alerts.py`: Custom alert rules
- `drift_detection_demo.py`: Drift detection demonstration

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE.md for details.
