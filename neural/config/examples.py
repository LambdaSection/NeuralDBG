"""
Configuration system usage examples.
"""

from __future__ import annotations

import os


def example_basic_usage():
    """Basic configuration usage example."""
    from neural.config import get_config
    
    # Get the global config manager
    config = get_config()
    
    # Access different subsystem settings
    print(f"Application: {config.core.app_name} v{config.core.version}")
    print(f"Environment: {config.core.environment}")
    print(f"Debug mode: {config.core.debug}")
    print(f"API running on: {config.api.host}:{config.api.port}")
    print(f"Dashboard on: {config.dashboard.host}:{config.dashboard.port}")


def example_environment_check():
    """Example of checking environment."""
    from neural.config import get_config
    
    config = get_config()
    
    if config.is_production():
        print("Running in PRODUCTION mode")
        print("Ensuring debug is disabled...")
        assert not config.core.debug, "Debug should be off in production!"
    elif config.is_development():
        print("Running in DEVELOPMENT mode")
        print("Debug features enabled")
    
    print(f"Current environment: {config.get_environment()}")


def example_storage_paths():
    """Example of using storage configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    # Get storage paths
    models_path = config.storage.get_full_path("models")
    experiments_path = config.storage.get_full_path("experiments")
    
    print(f"Models will be saved to: {models_path}")
    print(f"Experiments will be saved to: {experiments_path}")
    
    # Check cloud storage
    if config.storage.cloud_storage_enabled:
        print(f"Cloud storage enabled: {config.storage.cloud_storage_provider}")
        print(f"Bucket: {config.storage.cloud_storage_bucket}")


def example_api_configuration():
    """Example of using API configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    # Database configuration
    print(f"Database URL: {config.api.database_url}")
    
    # Redis configuration
    print(f"Redis URL: {config.api.redis_url}")
    
    # Celery configuration
    print(f"Celery broker: {config.api.broker_url}")
    print(f"Celery backend: {config.api.result_backend}")
    
    # CORS configuration
    print(f"CORS origins: {config.api.cors_origins}")


def example_hpo_configuration():
    """Example of using HPO configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    # Get HPO settings
    print(f"Running {config.hpo.n_trials} optimization trials")
    print(f"Using {config.hpo.sampler} sampler")
    print(f"Direction: {config.hpo.optimization_direction}")
    
    if config.hpo.distributed_enabled:
        print(f"Distributed optimization with {config.hpo.n_jobs} workers")


def example_integration_check():
    """Example of checking enabled integrations."""
    from neural.config import get_config
    
    config = get_config()
    
    integrations = config.integrations
    
    enabled = []
    if integrations.sagemaker_enabled:
        enabled.append("AWS SageMaker")
    if integrations.vertex_enabled:
        enabled.append("Google Vertex AI")
    if integrations.azure_enabled:
        enabled.append("Azure ML")
    if integrations.databricks_enabled:
        enabled.append("Databricks")
    if integrations.mlflow_enabled:
        enabled.append("MLflow")
    if integrations.wandb_enabled:
        enabled.append("Weights & Biases")
    
    if enabled:
        print("Enabled integrations:")
        for integration in enabled:
            print(f"  - {integration}")
    else:
        print("No integrations enabled")


def example_teams_quotas():
    """Example of using teams configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    # Get quotas for different plans
    for plan in ["free", "starter", "professional", "enterprise"]:
        quotas = config.teams.get_plan_quotas(plan)
        pricing = config.teams.get_plan_pricing(plan)
        
        print(f"\n{plan.upper()} Plan:")
        print(f"  Price: ${pricing['monthly']}/month")
        print(f"  Models: {quotas['max_models']}")
        print(f"  Storage: {quotas['max_storage_gb']} GB")
        print(f"  Team members: {quotas['max_team_members']}")


def example_monitoring_setup():
    """Example of monitoring configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    if config.monitoring.enabled:
        print("Monitoring enabled")
        
        if config.monitoring.prometheus_enabled:
            print(f"  Prometheus metrics on port {config.monitoring.prometheus_port}")
        
        metrics = []
        if config.monitoring.collect_cpu_metrics:
            metrics.append("CPU")
        if config.monitoring.collect_memory_metrics:
            metrics.append("Memory")
        if config.monitoring.collect_gpu_metrics:
            metrics.append("GPU")
        
        print(f"  Collecting: {', '.join(metrics)}")
        
        if config.monitoring.alerting_enabled:
            print(f"  Alerting enabled with {len(config.monitoring.alert_email_recipients)} recipients")


def example_export_config():
    """Example of exporting configuration."""
    from neural.config import get_config
    from neural.config.utils import export_config_to_file
    
    get_config()
    
    # Export to YAML (safe mode)
    export_config_to_file("config_export.yaml", format="yaml", safe=True)
    print("Exported configuration to config_export.yaml (secrets redacted)")
    
    # Export to JSON (unsafe mode for backup)
    export_config_to_file("config_backup.json", format="json", safe=False)
    print("Exported full configuration to config_backup.json")


def example_validate_config():
    """Example of configuration validation."""
    from neural.config import get_config
    from neural.config.utils import validate_environment
    
    config = get_config()
    
    # Validate all settings
    try:
        config.validate_all()
        print("✓ All settings are valid")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Get detailed validation results
    results = validate_environment()
    
    if not results["valid"]:
        print("Configuration errors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("Configuration warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")


def example_dynamic_reload():
    """Example of dynamically reloading configuration."""
    from neural.config import get_config
    
    config = get_config()
    
    print(f"Initial port: {config.api.port}")
    
    # Change environment variable
    os.environ["NEURAL_API_PORT"] = "9000"
    
    # Reload configuration
    config.reload()
    
    print(f"After reload: {config.api.port}")


def example_custom_env_file():
    """Example of using custom .env file."""
    from neural.config import get_config
    
    # Load from custom .env file
    config = get_config(env_file="/path/to/production.env")
    
    print("Loaded configuration from custom file")
    print(f"Environment: {config.get_environment()}")


def example_config_summary():
    """Example of getting configuration summary."""
    from neural.config.utils import get_config_summary
    
    summary = get_config_summary()
    
    print("Configuration Summary:")
    print(f"  Environment: {summary['environment']}")
    print(f"  Debug: {summary['debug']}")
    print(f"  Version: {summary['version']}")
    print("\nSubsystems:")
    for name, info in summary['subsystems'].items():
        print(f"  {name}: {info}")


def example_generate_template():
    """Example of generating .env template."""
    from neural.config.utils import generate_env_template
    
    # Generate template file
    generate_env_template(".env.template")
    print("Generated .env.template file")
    print("Copy this file to .env and customize the values")


def run_all_examples():
    """Run all examples."""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Environment Check", example_environment_check),
        ("Storage Paths", example_storage_paths),
        ("API Configuration", example_api_configuration),
        ("HPO Configuration", example_hpo_configuration),
        ("Integration Check", example_integration_check),
        ("Teams Quotas", example_teams_quotas),
        ("Monitoring Setup", example_monitoring_setup),
        ("Export Config", example_export_config),
        ("Validate Config", example_validate_config),
        ("Config Summary", example_config_summary),
    ]
    
    for title, func in examples:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}\n")
        try:
            func()
        except Exception as e:
            print(f"Error running example: {e}")


if __name__ == "__main__":
    run_all_examples()
