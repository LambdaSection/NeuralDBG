"""
Basic monitoring example for Neural DSL models.
"""

import time
import numpy as np
from neural.monitoring import ModelMonitor


def main():
    """Basic monitoring example."""
    
    print("=== Basic Monitoring Example ===\n")
    
    # Initialize monitor
    print("1. Initializing monitor...")
    monitor = ModelMonitor(
        model_name="example-model",
        model_version="1.0",
        storage_path="monitoring_data_example",
        enable_prometheus=True,
        enable_alerting=True,
        enable_slo_tracking=True
    )
    print("   ✓ Monitor initialized\n")
    
    # Set reference data
    print("2. Setting reference data...")
    np.random.seed(42)
    reference_data = np.random.randn(1000, 10)
    reference_predictions = np.random.randint(0, 2, 1000)
    reference_performance = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'f1': 0.95
    }
    
    monitor.set_reference_data(
        data=reference_data,
        predictions=reference_predictions,
        performance=reference_performance,
        feature_names=[f"feature_{i}" for i in range(10)]
    )
    print("   ✓ Reference data set\n")
    
    # Log some predictions
    print("3. Logging predictions...")
    for i in range(100):
        input_features = {
            f"feature_{j}": float(np.random.randn())
            for j in range(10)
        }
        
        prediction = np.random.randint(0, 2)
        prediction_proba = {
            '0': np.random.random(),
            '1': np.random.random()
        }
        # Normalize probabilities
        total = sum(prediction_proba.values())
        prediction_proba = {k: v/total for k, v in prediction_proba.items()}
        
        monitor.log_prediction(
            prediction_id=f"pred_{i:04d}",
            input_features=input_features,
            prediction=prediction,
            prediction_proba=prediction_proba,
            ground_truth=prediction if np.random.random() > 0.1 else 1 - prediction,
            latency_ms=np.random.uniform(10, 100),
            metadata={'batch': i // 10}
        )
        
        if (i + 1) % 20 == 0:
            print(f"   Logged {i + 1} predictions...")
    
    print("   ✓ Predictions logged\n")
    
    # Check drift
    print("4. Checking for drift...")
    new_data = np.random.randn(100, 10) + 0.1  # Slight shift
    new_predictions = np.random.randint(0, 2, 100)
    new_performance = {
        'accuracy': 0.93,
        'precision': 0.92,
        'recall': 0.94,
        'f1': 0.93
    }
    
    drift_report = monitor.check_drift(
        data=new_data,
        predictions=new_predictions,
        performance=new_performance
    )
    
    print(f"   Drift detected: {drift_report['is_drifting']}")
    print(f"   Drift severity: {drift_report['drift_severity']}")
    print(f"   Prediction drift: {drift_report['prediction_drift']:.4f}")
    print(f"   Performance drift: {drift_report['performance_drift']:.4f}")
    print()
    
    # Check data quality
    print("5. Checking data quality...")
    quality_report = monitor.check_data_quality(new_data)
    
    print(f"   Quality score: {quality_report['quality_score']:.3f}")
    print(f"   Is healthy: {quality_report['is_healthy']}")
    print(f"   Missing rate: {quality_report['missing_rate']:.4f}")
    print(f"   Outlier rate: {quality_report['outlier_rate']:.4f}")
    print()
    
    # Update performance metrics
    print("6. Updating performance metrics...")
    monitor.update_performance_metrics(new_performance)
    print("   ✓ Performance metrics updated\n")
    
    # Get monitoring summary
    print("7. Getting monitoring summary...")
    summary = monitor.get_monitoring_summary()
    
    print(f"   Total predictions: {summary['total_predictions']}")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Error rate: {summary['error_rate']:.4f}")
    print(f"   Uptime: {summary['uptime_seconds']:.1f} seconds")
    print()
    
    # Generate health report
    print("8. Generating health report...")
    health = monitor.generate_health_report()
    
    print(f"   Status: {health['status']}")
    print(f"   Issues: {len(health['issues'])}")
    for issue in health['issues']:
        print(f"     - {issue}")
    print()
    
    print("=== Example Complete ===")
    print(f"\nMonitoring data stored in: monitoring_data_example/")
    print("View status with: neural monitor status --storage-path monitoring_data_example")


if __name__ == "__main__":
    main()
