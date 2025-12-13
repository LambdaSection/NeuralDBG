"""
CLI commands for monitoring functionality.
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import numpy as np


@click.group()
def monitor():
    """Production monitoring and observability commands."""
    pass


@monitor.command('init')
@click.option('--model-name', default='default', help='Model name')
@click.option('--model-version', default='1.0', help='Model version')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--enable-prometheus', is_flag=True, default=True, help='Enable Prometheus metrics')
@click.option('--enable-alerting', is_flag=True, default=True, help='Enable alerting')
@click.option('--enable-slo', is_flag=True, default=True, help='Enable SLO tracking')
@click.option('--slack-webhook', help='Slack webhook URL for alerts')
@click.option('--email-smtp', help='SMTP server for email alerts')
@click.option('--email-user', help='Email username')
@click.option('--email-to', multiple=True, help='Email recipients')
def init_monitoring(
    model_name: str,
    model_version: str,
    storage_path: str,
    enable_prometheus: bool,
    enable_alerting: bool,
    enable_slo: bool,
    slack_webhook: str,
    email_smtp: str,
    email_user: str,
    email_to: tuple
):
    """Initialize monitoring for a model."""
    from neural.monitoring.monitor import ModelMonitor
    
    click.echo(f"Initializing monitoring for {model_name} v{model_version}")
    
    # Prepare alert config
    alert_config = {}
    if slack_webhook:
        alert_config['slack_webhook'] = slack_webhook
    
    if email_smtp and email_user:
        email_password = click.prompt('Email password', hide_input=True)
        alert_config['email_config'] = {
            'smtp_server': email_smtp,
            'smtp_port': 587,
            'username': email_user,
            'password': email_password,
            'from_addr': email_user,
            'to_addrs': list(email_to) if email_to else [email_user]
        }
    
    # Create monitor
    monitor = ModelMonitor(
        model_name=model_name,
        model_version=model_version,
        storage_path=storage_path,
        enable_prometheus=enable_prometheus,
        enable_alerting=enable_alerting,
        enable_slo_tracking=enable_slo,
        alert_config=alert_config
    )
    
    # Save config
    config = {
        'model_name': model_name,
        'model_version': model_version,
        'storage_path': storage_path,
        'enable_prometheus': enable_prometheus,
        'enable_alerting': enable_alerting,
        'enable_slo': enable_slo,
    }
    
    config_path = Path(storage_path) / 'monitor_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"✓ Monitoring initialized at {storage_path}")
    click.echo(f"✓ Config saved to {config_path}")


@monitor.command('status')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def monitoring_status(storage_path: str, format: str):
    """Get monitoring status."""
    from neural.monitoring.monitor import ModelMonitor
    
    # Load config
    config_path = Path(storage_path) / 'monitor_config.json'
    if not config_path.exists():
        click.echo(f"Error: Monitoring not initialized. Run 'neural monitor init' first.", err=True)
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create monitor
    monitor = ModelMonitor(
        model_name=config['model_name'],
        model_version=config['model_version'],
        storage_path=storage_path,
        enable_prometheus=config.get('enable_prometheus', True),
        enable_alerting=config.get('enable_alerting', True),
        enable_slo_tracking=config.get('enable_slo', True)
    )
    
    # Get summary
    summary = monitor.get_monitoring_summary()
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\n=== Monitoring Status for {config['model_name']} v{config['model_version']} ===\n")
        
        click.echo(f"Total Predictions: {summary['total_predictions']}")
        click.echo(f"Total Errors: {summary['total_errors']}")
        click.echo(f"Error Rate: {summary['error_rate']:.2%}")
        click.echo(f"Uptime: {summary['uptime_seconds']:.0f} seconds")
        
        # Drift
        if summary['drift']['status'] == 'ok':
            click.echo(f"\nDrift Detection:")
            click.echo(f"  Drift Rate: {summary['drift'].get('drift_rate', 0):.2%}")
            click.echo(f"  Avg Prediction Drift: {summary['drift'].get('avg_prediction_drift', 0):.4f}")
            click.echo(f"  Avg Performance Drift: {summary['drift'].get('avg_performance_drift', 0):.4f}")
        
        # Quality
        if summary['quality']['status'] == 'ok':
            click.echo(f"\nData Quality:")
            click.echo(f"  Healthy Rate: {summary['quality'].get('healthy_rate', 1.0):.2%}")
            click.echo(f"  Avg Quality Score: {summary['quality'].get('avg_quality_score', 1.0):.4f}")
            click.echo(f"  Avg Missing Rate: {summary['quality'].get('avg_missing_rate', 0):.2%}")
        
        # SLOs
        if 'slos' in summary:
            click.echo(f"\nSLO Status:")
            for slo_name, slo_status in summary['slos'].items():
                if slo_status.get('status') == 'ok':
                    status_icon = "✓" if slo_status.get('is_meeting', True) else "✗"
                    click.echo(f"  {status_icon} {slo_name}: {slo_status.get('compliance_rate', 0):.2%}")
        
        # Alerts
        if 'alerts' in summary and summary['alerts']['status'] == 'ok':
            click.echo(f"\nAlerts (24h):")
            click.echo(f"  Total: {summary['alerts']['total_alerts']}")
            by_severity = summary['alerts'].get('by_severity', {})
            click.echo(f"  Critical: {by_severity.get('critical', 0)}")
            click.echo(f"  Warning: {by_severity.get('warning', 0)}")
            click.echo(f"  Info: {by_severity.get('info', 0)}")


@monitor.command('drift')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--window', default=100, help='Number of recent samples to analyze')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def drift_report(storage_path: str, window: int, format: str):
    """Get drift detection report."""
    from neural.monitoring.drift_detector import DriftDetector
    
    detector = DriftDetector(storage_path=str(Path(storage_path) / 'drift'))
    report = detector.get_drift_report(window=window)
    
    if format == 'json':
        click.echo(json.dumps(report, indent=2))
    else:
        if report['status'] == 'no_data':
            click.echo("No drift data available")
            return
        
        click.echo(f"\n=== Drift Detection Report ===\n")
        click.echo(f"Total Samples: {report['total_samples']}")
        click.echo(f"Drift Detected: {report['drift_detected']} ({report['drift_rate']:.2%})")
        click.echo(f"Avg Prediction Drift: {report['avg_prediction_drift']:.4f}")
        click.echo(f"Avg Performance Drift: {report['avg_performance_drift']:.4f}")
        click.echo(f"Avg Distribution Drift: {report['avg_distribution_drift']:.4f}")
        
        severity_dist = report['severity_distribution']
        click.echo(f"\nSeverity Distribution:")
        click.echo(f"  Critical: {severity_dist['critical']}")
        click.echo(f"  Warning: {severity_dist['warning']}")
        click.echo(f"  None: {severity_dist['none']}")


@monitor.command('quality')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--window', default=100, help='Number of recent samples to analyze')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def quality_report(storage_path: str, window: int, format: str):
    """Get data quality report."""
    from neural.monitoring.data_quality import DataQualityMonitor
    
    monitor = DataQualityMonitor(storage_path=str(Path(storage_path) / 'quality'))
    summary = monitor.get_quality_summary(window=window)
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        if summary['status'] == 'no_data':
            click.echo("No quality data available")
            return
        
        click.echo(f"\n=== Data Quality Report ===\n")
        click.echo(f"Total Reports: {summary['total_reports']}")
        click.echo(f"Healthy Rate: {summary['healthy_rate']:.2%}")
        click.echo(f"Avg Quality Score: {summary['avg_quality_score']:.4f}")
        click.echo(f"Avg Missing Rate: {summary['avg_missing_rate']:.2%}")
        click.echo(f"Avg Outlier Rate: {summary['avg_outlier_rate']:.2%}")
        click.echo(f"Avg Invalid Rate: {summary['avg_invalid_rate']:.2%}")
        click.echo(f"Total Warnings: {summary['total_warnings']}")


@monitor.command('alerts')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--hours', default=24, help='Number of hours to show')
@click.option('--severity', type=click.Choice(['info', 'warning', 'critical']), help='Filter by severity')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def alert_summary(storage_path: str, hours: int, severity: str, format: str):
    """Get alert summary."""
    from neural.monitoring.alerting import AlertManager, AlertSeverity
    
    manager = AlertManager(storage_path=str(Path(storage_path) / 'alerts'))
    summary = manager.get_alert_summary(hours=hours)
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\n=== Alert Summary ({hours}h) ===\n")
        click.echo(f"Total Alerts: {summary['total_alerts']}")
        
        by_severity = summary.get('by_severity', {})
        click.echo(f"\nBy Severity:")
        click.echo(f"  Critical: {by_severity.get('critical', 0)}")
        click.echo(f"  Warning: {by_severity.get('warning', 0)}")
        click.echo(f"  Info: {by_severity.get('info', 0)}")
        
        if summary.get('recent_critical'):
            click.echo(f"\nRecent Critical Alerts:")
            for alert in summary['recent_critical'][-5:]:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))
                click.echo(f"  [{timestamp}] {alert['title']}")


@monitor.command('slo')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--name', help='SLO name to show (shows all if not specified)')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def slo_status(storage_path: str, name: str, format: str):
    """Get SLO status."""
    from neural.monitoring.slo_tracker import SLOTracker
    
    tracker = SLOTracker(storage_path=str(Path(storage_path) / 'slo'))
    
    if name:
        try:
            status = tracker.get_slo_status(name)
            if format == 'json':
                click.echo(json.dumps(status, indent=2))
            else:
                _print_slo_status(name, status)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        all_status = tracker.get_all_slo_status()
        if format == 'json':
            click.echo(json.dumps(all_status, indent=2))
        else:
            click.echo(f"\n=== SLO Status ===\n")
            for slo_name, status in all_status.items():
                _print_slo_status(slo_name, status)
                click.echo()


def _print_slo_status(name: str, status: dict):
    """Print SLO status in text format."""
    if status.get('status') == 'no_data':
        click.echo(f"{name}: No data available")
        return
    
    status_icon = "✓" if status.get('is_meeting', True) else "✗"
    click.echo(f"{status_icon} {name}")
    click.echo(f"  Target: {status['slo']['target']:.2%}")
    click.echo(f"  Compliance: {status.get('compliance_rate', 0):.2%}")
    click.echo(f"  Error Budget Remaining: {status.get('error_budget_remaining', 0):.2%}")
    click.echo(f"  Total Breaches: {status.get('total_breaches', 0)}")
    
    if status.get('current_breach'):
        click.echo(f"  ⚠ Currently in breach ({status.get('current_breach_duration', 0):.1f}s)")


@monitor.command('dashboard')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--port', default=8052, help='Dashboard port')
@click.option('--host', default='localhost', help='Dashboard host')
def monitoring_dashboard(storage_path: str, port: int, host: str):
    """Start monitoring dashboard."""
    click.echo(f"Starting monitoring dashboard on {host}:{port}")
    click.echo(f"Press Ctrl+C to stop")
    
    try:
        from neural.monitoring.dashboard import create_app
        app = create_app(storage_path=storage_path)
        app.run_server(debug=False, host=host, port=port)
    except ImportError as e:
        click.echo(f"Error: Dashboard dependencies not available: {e}", err=True)
        click.echo("Install with: pip install dash plotly", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped")


@monitor.command('prometheus')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--port', default=9090, help='Prometheus metrics port')
def start_prometheus(storage_path: str, port: int):
    """Start Prometheus metrics exporter."""
    from neural.monitoring.monitor import ModelMonitor
    
    # Load config
    config_path = Path(storage_path) / 'monitor_config.json'
    if not config_path.exists():
        click.echo(f"Error: Monitoring not initialized. Run 'neural monitor init' first.", err=True)
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    monitor = ModelMonitor(
        model_name=config['model_name'],
        model_version=config['model_version'],
        storage_path=storage_path,
        enable_prometheus=True
    )
    
    click.echo(f"Starting Prometheus metrics server on port {port}")
    click.echo(f"Metrics available at http://localhost:{port}/metrics")
    click.echo("Press Ctrl+C to stop")
    
    try:
        monitor.start_prometheus_server(port=port)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nMetrics server stopped")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@monitor.command('health')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def health_check(storage_path: str, format: str):
    """Get health report."""
    from neural.monitoring.monitor import ModelMonitor
    
    # Load config
    config_path = Path(storage_path) / 'monitor_config.json'
    if not config_path.exists():
        click.echo(f"Error: Monitoring not initialized. Run 'neural monitor init' first.", err=True)
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    monitor = ModelMonitor(
        model_name=config['model_name'],
        model_version=config['model_version'],
        storage_path=storage_path,
        enable_prometheus=config.get('enable_prometheus', True),
        enable_alerting=config.get('enable_alerting', True),
        enable_slo_tracking=config.get('enable_slo', True)
    )
    
    report = monitor.generate_health_report()
    
    if format == 'json':
        click.echo(json.dumps(report, indent=2))
    else:
        status_icon = {"healthy": "✓", "warning": "⚠", "critical": "✗"}.get(report['status'], "?")
        status_color = {"healthy": "green", "warning": "yellow", "critical": "red"}.get(report['status'], "white")
        
        click.echo(f"\n=== Health Report ===\n")
        click.echo(f"Model: {report['model']} v{report['version']}")
        click.echo(f"Status: {status_icon} {report['status'].upper()}")
        click.echo(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}")
        click.echo(f"\nIssues:")
        for issue in report['issues']:
            click.echo(f"  • {issue}")
        
        sys.exit(0 if report['status'] == 'healthy' else 1)


if __name__ == '__main__':
    monitor()
