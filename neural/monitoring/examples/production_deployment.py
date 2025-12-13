"""
Production deployment example with full monitoring setup.
"""

import time
import numpy as np
from neural.monitoring import (
    ModelMonitor,
    AlertSeverity,
)


def simulate_production_traffic(monitor, duration_seconds=60, requests_per_second=10):
    """
    Simulate production traffic with various scenarios.
    
    Parameters
    ----------
    monitor : ModelMonitor
        Monitor instance
    duration_seconds : int
        Simulation duration
    requests_per_second : int
        Target RPS
    """
    start_time = time.time()
    request_count = 0
    
    print(f"Simulating production traffic ({requests_per_second} RPS for {duration_seconds}s)...")
    
    while time.time() - start_time < duration_seconds:
        # Generate input
        input_features = {
            f"feature_{j}": float(np.random.randn())
            for j in range(10)
        }
        
        # Simulate model prediction
        prediction = np.random.randint(0, 2)
        prediction_proba = {
            '0': np.random.random(),
            '1': np.random.random()
        }
        total = sum(prediction_proba.values())
        prediction_proba = {k: v/total for k, v in prediction_proba.items()}
        
        # Simulate latency (with occasional spikes)
        base_latency = np.random.uniform(20, 50)
        if np.random.random() < 0.05:  # 5% spike rate
            latency = base_latency * np.random.uniform(5, 10)
        else:
            latency = base_latency
        
        # Simulate errors
        if np.random.random() < 0.001:  # 0.1% error rate
            monitor.record_error("prediction_error")
            continue
        
        # Log prediction
        monitor.log_prediction(
            prediction_id=f"prod_{request_count:06d}",
            input_features=input_features,
            prediction=prediction,
            prediction_proba=prediction_proba,
            ground_truth=prediction if np.random.random() > 0.05 else 1 - prediction,
            latency_ms=latency,
            metadata={'timestamp': time.time()}
        )
        
        request_count += 1
        
        # Progress update
        if request_count % 100 == 0:
            elapsed = time.time() - start_time
            current_rps = request_count / elapsed
            print(f"  Processed {request_count} requests ({current_rps:.1f} RPS)")
        
        # Sleep to maintain RPS
        time.sleep(1.0 / requests_per_second)
    
    elapsed = time.time() - start_time
    actual_rps = request_count / elapsed
    print(f"✓ Simulation complete: {request_count} requests in {elapsed:.1f}s ({actual_rps:.1f} RPS)")


def main():
    """Production deployment example."""
    
    print("=== Production Deployment Example ===\n")
    
    # Initialize monitor with production config
    print("1. Initializing production monitor...")
    
    # In production, you would configure real alert channels
    alert_config = {
        # 'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        # 'email_config': {
        #     'smtp_server': 'smtp.gmail.com',
        #     'smtp_port': 587,
        #     'username': 'your-email@gmail.com',
        #     'password': 'your-app-password',
        #     'from_addr': 'your-email@gmail.com',
        #     'to_addrs': ['alerts@example.com']
        # }
    }
    
    monitor = ModelMonitor(
        model_name="production-model",
        model_version="2.0",
        storage_path="monitoring_data_production",
        enable_prometheus=True,
        enable_alerting=True,
        enable_slo_tracking=True,
        alert_config=alert_config
    )
    print("   ✓ Production monitor initialized\n")
    
    # Set reference data from training/validation
    print("2. Loading reference data...")
    np.random.seed(42)
    reference_data = np.random.randn(5000, 10)
    reference_predictions = np.random.randint(0, 2, 5000)
    reference_performance = {
        'accuracy': 0.96,
        'precision': 0.95,
        'recall': 0.97,
        'f1': 0.96
    }
    
    monitor.set_reference_data(
        data=reference_data,
        predictions=reference_predictions,
        performance=reference_performance,
        feature_names=[f"feature_{i}" for i in range(10)]
    )
    print("   ✓ Reference data loaded\n")
    
    # Simulate initial traffic
    print("3. Phase 1: Normal operation (30s)...")
    simulate_production_traffic(monitor, duration_seconds=30, requests_per_second=10)
    print()
    
    # Check initial metrics
    print("4. Checking initial metrics...")
    summary = monitor.get_monitoring_summary()
    print(f"   Total predictions: {summary['total_predictions']}")
    print(f"   Error rate: {summary['error_rate']:.4f}")
    
    if summary['predictions']['status'] == 'ok' and 'latency' in summary['predictions']:
        latency = summary['predictions']['latency']
        print(f"   P95 latency: {latency['p95']:.1f}ms")
    print()
    
    # Simulate data drift
    print("5. Phase 2: Introducing data drift (15s)...")
    print("   (Shifting input distribution...)")
    
    # Temporarily modify traffic to introduce drift
    start_time = time.time()
    request_count = 0
    drift_data_batch = []
    
    while time.time() - start_time < 15:
        # Generate drifted input (shifted distribution)
        input_features = {
            f"feature_{j}": float(np.random.randn() + 0.5)  # Shifted mean
            for j in range(10)
        }
        
        drift_data_batch.append(list(input_features.values()))
        
        prediction = np.random.randint(0, 2)
        prediction_proba = {
            '0': np.random.random(),
            '1': np.random.random()
        }
        total = sum(prediction_proba.values())
        prediction_proba = {k: v/total for k, v in prediction_proba.items()}
        
        monitor.log_prediction(
            prediction_id=f"drift_{request_count:06d}",
            input_features=input_features,
            prediction=prediction,
            prediction_proba=prediction_proba,
            latency_ms=np.random.uniform(20, 50),
            metadata={'phase': 'drift'}
        )
        
        request_count += 1
        time.sleep(0.1)
    
    print(f"   Processed {request_count} requests with drift")
    
    # Check for drift
    drift_data = np.array(drift_data_batch)
    drift_report = monitor.check_drift(data=drift_data)
    print(f"   Drift detected: {drift_report['is_drifting']}")
    print(f"   Drift severity: {drift_report['drift_severity']}")
    print()
    
    # Generate health report
    print("6. Generating health report...")
    health = monitor.generate_health_report()
    
    print(f"   Overall status: {health['status'].upper()}")
    print(f"   Issues detected: {len(health['issues'])}")
    for i, issue in enumerate(health['issues'], 1):
        print(f"     {i}. {issue}")
    print()
    
    # Show SLO status
    print("7. SLO Compliance Status...")
    summary = monitor.get_monitoring_summary()
    if 'slos' in summary:
        for slo_name, status in summary['slos'].items():
            if status.get('status') == 'ok':
                icon = "✓" if status.get('is_meeting', True) else "✗"
                compliance = status.get('compliance_rate', 0)
                print(f"   {icon} {slo_name}: {compliance:.2%}")
    print()
    
    # Send custom alert
    print("8. Sending custom alert...")
    monitor.send_alert(
        title="Deployment Complete",
        message=f"Production monitoring active for model {monitor.model_name} v{monitor.model_version}",
        severity=AlertSeverity.INFO,
        metadata={'phase': 'deployment', 'requests_processed': summary['total_predictions']}
    )
    print("   ✓ Alert sent\n")
    
    print("=== Production Deployment Complete ===")
    print(f"\nMonitoring data: monitoring_data_production/")
    print("\nNext steps:")
    print("  1. View status: neural monitor status --storage-path monitoring_data_production")
    print("  2. Start dashboard: neural monitor dashboard --storage-path monitoring_data_production --port 8052")
    print("  3. Start Prometheus: neural monitor prometheus --storage-path monitoring_data_production --port 9090")
    print("\nDashboard URL: http://localhost:8052")
    print("Metrics URL: http://localhost:9090/metrics")


if __name__ == "__main__":
    main()
