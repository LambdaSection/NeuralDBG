"""
Drift detection demonstration.

This example shows how to detect different types of drift:
- Feature drift
- Prediction drift
- Performance drift
- Data distribution drift
"""

import numpy as np
import matplotlib.pyplot as plt
from neural.monitoring import DriftDetector, DriftMetrics


def generate_reference_data(n_samples=1000, n_features=5, seed=42):
    """Generate reference data."""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    predictions = (data[:, 0] + data[:, 1] > 0).astype(int)
    return data, predictions


def generate_drifted_data(n_samples=100, n_features=5, drift_type='none', drift_magnitude=0.5):
    """
    Generate data with different types of drift.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    drift_type : str
        Type of drift ('none', 'mean_shift', 'variance_change', 'distribution_change')
    drift_magnitude : float
        Magnitude of drift
    """
    data = np.random.randn(n_samples, n_features)
    
    if drift_type == 'mean_shift':
        # Shift mean of some features
        data[:, :2] += drift_magnitude
    elif drift_type == 'variance_change':
        # Change variance of some features
        data[:, :2] *= (1 + drift_magnitude)
    elif drift_type == 'distribution_change':
        # Change distribution completely
        data[:, :2] = np.random.exponential(drift_magnitude, (n_samples, 2))
    
    predictions = (data[:, 0] + data[:, 1] > 0).astype(int)
    return data, predictions


def visualize_drift(reference_data, drifted_data, drift_type, feature_idx=0):
    """Visualize data drift."""
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(reference_data[:, feature_idx], bins=30, alpha=0.5, label='Reference', density=True)
    plt.hist(drifted_data[:, feature_idx], bins=30, alpha=0.5, label='Current', density=True)
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Density')
    plt.title(f'Distribution Comparison - {drift_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(1, 2, 2)
    ref_sorted = np.sort(reference_data[:, feature_idx])
    curr_sorted = np.sort(drifted_data[:, feature_idx])
    
    # Interpolate to same length
    ref_quantiles = np.linspace(0, 1, len(ref_sorted))
    curr_quantiles = np.linspace(0, 1, len(curr_sorted))
    ref_interp = np.interp(curr_quantiles, ref_quantiles, ref_sorted)
    
    plt.scatter(ref_interp, curr_sorted, alpha=0.5)
    plt.plot([ref_interp.min(), ref_interp.max()], 
             [ref_interp.min(), ref_interp.max()], 
             'r--', label='No drift')
    plt.xlabel('Reference Quantiles')
    plt.ylabel('Current Quantiles')
    plt.title('Q-Q Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def main():
    """Drift detection demonstration."""
    
    print("=== Drift Detection Demonstration ===\n")
    
    # Initialize detector
    print("1. Initializing drift detector...")
    detector = DriftDetector(
        window_size=1000,
        drift_threshold=0.1,
        alert_threshold=0.2,
        storage_path="monitoring_data_drift"
    )
    print("   ✓ Detector initialized\n")
    
    # Generate reference data
    print("2. Generating reference data...")
    ref_data, ref_predictions = generate_reference_data(n_samples=1000, n_features=5)
    ref_performance = {'accuracy': 0.95, 'f1': 0.94}
    
    detector.set_reference(
        data=ref_data,
        predictions=ref_predictions,
        performance=ref_performance
    )
    print(f"   Reference data: {ref_data.shape}")
    print(f"   Reference accuracy: {ref_performance['accuracy']:.3f}\n")
    
    # Test different types of drift
    drift_scenarios = [
        ('none', 0.0, "No drift (baseline)"),
        ('mean_shift', 0.3, "Mean shift (gradual drift)"),
        ('mean_shift', 1.0, "Mean shift (sudden drift)"),
        ('variance_change', 0.5, "Variance change"),
        ('distribution_change', 2.0, "Distribution change"),
    ]
    
    results = []
    
    for drift_type, magnitude, description in drift_scenarios:
        print(f"3. Testing: {description}")
        print(f"   Drift type: {drift_type}, Magnitude: {magnitude}")
        
        # Generate drifted data
        curr_data, curr_predictions = generate_drifted_data(
            n_samples=100,
            n_features=5,
            drift_type=drift_type,
            drift_magnitude=magnitude
        )
        
        # Simulate performance degradation with drift
        if drift_type != 'none':
            performance_drop = min(0.15, magnitude * 0.1)
            curr_performance = {
                'accuracy': ref_performance['accuracy'] - performance_drop,
                'f1': ref_performance['f1'] - performance_drop
            }
        else:
            curr_performance = ref_performance
        
        # Detect drift
        metrics = detector.detect_drift(
            data=curr_data,
            predictions=curr_predictions,
            performance=curr_performance
        )
        
        results.append({
            'description': description,
            'drift_type': drift_type,
            'magnitude': magnitude,
            'metrics': metrics
        })
        
        # Print results
        print(f"   → Drift detected: {metrics.is_drifting}")
        print(f"   → Severity: {metrics.drift_severity}")
        print(f"   → Prediction drift: {metrics.prediction_drift:.4f}")
        print(f"   → Performance drift: {metrics.performance_drift:.4f}")
        print(f"   → Distribution drift: {metrics.data_distribution_drift:.4f}")
        print(f"   → Concept drift score: {metrics.concept_drift_score:.4f}")
        print()
        
        # Visualize if drift detected
        if metrics.is_drifting and drift_type != 'none':
            try:
                fig = visualize_drift(ref_data, curr_data, description)
                filename = f"drift_vis_{drift_type}_{magnitude}.png"
                fig.savefig(filename)
                print(f"   ✓ Visualization saved: {filename}")
                plt.close(fig)
            except Exception as e:
                print(f"   ⚠ Visualization failed: {e}")
        
        print("-" * 60)
    
    # Summary
    print("\n=== Drift Detection Summary ===\n")
    print(f"{'Scenario':<35} {'Detected':<10} {'Severity':<12} {'Score':<8}")
    print("-" * 70)
    
    for result in results:
        detected = "Yes" if result['metrics'].is_drifting else "No"
        severity = result['metrics'].drift_severity
        score = result['metrics'].concept_drift_score
        print(f"{result['description']:<35} {detected:<10} {severity:<12} {score:.4f}")
    
    print("\n" + "-" * 70)
    
    # Get drift report
    print("\n4. Generating drift report...")
    report = detector.get_drift_report(window=10)
    
    if report['status'] == 'ok':
        print(f"   Total samples analyzed: {report['total_samples']}")
        print(f"   Drift detected count: {report['drift_detected']}")
        print(f"   Drift rate: {report['drift_rate']:.2%}")
        print(f"   Avg prediction drift: {report['avg_prediction_drift']:.4f}")
        print(f"   Avg performance drift: {report['avg_performance_drift']:.4f}")
        print(f"   Avg distribution drift: {report['avg_distribution_drift']:.4f}")
        
        severity_dist = report['severity_distribution']
        print(f"\n   Severity distribution:")
        print(f"     Critical: {severity_dist['critical']}")
        print(f"     Warning: {severity_dist['warning']}")
        print(f"     None: {severity_dist['none']}")
    
    print("\n=== Demonstration Complete ===")
    print("\nKey Takeaways:")
    print("1. Different drift types have different signatures")
    print("2. Drift magnitude affects detection sensitivity")
    print("3. Multiple metrics help identify drift type")
    print("4. Regular monitoring is essential for catching drift early")
    print("\nMonitoring data stored in: monitoring_data_drift/")


if __name__ == "__main__":
    main()
