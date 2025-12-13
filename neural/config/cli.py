"""
CLI tool for configuration management.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click
import yaml

from neural.config.manager import get_config
from neural.config.utils import (
    export_config_to_file,
    generate_env_template,
    get_config_summary,
    validate_environment,
)


@click.group()
def config_cli():
    """Neural DSL configuration management."""
    pass


@config_cli.command()
@click.option(
    "--format",
    type=click.Choice(["json", "yaml", "summary"]),
    default="yaml",
    help="Output format",
)
@click.option(
    "--safe/--unsafe",
    default=True,
    help="Redact sensitive information",
)
def show(format: str, safe: bool):
    """Show current configuration."""
    config_manager = get_config()
    
    if format == "summary":
        summary = get_config_summary()
        click.echo(yaml.dump(summary, default_flow_style=False, sort_keys=False))
    else:
        config_dict = config_manager.dump_config(safe=safe)
        
        if format == "json":
            click.echo(json.dumps(config_dict, indent=2))
        else:
            click.echo(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


@config_cli.command()
@click.argument("subsystem", required=False)
def list_settings(subsystem: Optional[str]):
    """List available settings for a subsystem or all subsystems."""
    config_manager = get_config()
    
    if subsystem:
        all_settings = config_manager.get_all_settings()
        if subsystem not in all_settings:
            click.echo(f"Unknown subsystem: {subsystem}", err=True)
            click.echo(f"Available subsystems: {', '.join(all_settings.keys())}")
            sys.exit(1)
        
        settings = all_settings[subsystem]
        click.echo(f"{subsystem.upper()} Settings:")
        click.echo("")
        
        for field_name, field_info in settings.model_fields.items():
            description = field_info.description or "No description"
            default = field_info.default
            click.echo(f"  {field_name}:")
            click.echo(f"    Description: {description}")
            if default is not None and not callable(default):
                click.echo(f"    Default: {default}")
            click.echo("")
    else:
        all_settings = config_manager.get_all_settings()
        click.echo("Available subsystems:")
        for name in all_settings.keys():
            click.echo(f"  - {name}")
        click.echo("")
        click.echo("Use 'neural config list <subsystem>' to see settings for a specific subsystem")


@config_cli.command()
@click.option(
    "--output",
    "-o",
    default=".env.template",
    help="Output file path",
)
def generate_template(output: str):
    """Generate .env template file."""
    generate_env_template(output)
    click.echo(f"Generated template: {output}")


@config_cli.command()
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output file path",
)
@click.option(
    "--format",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
@click.option(
    "--safe/--unsafe",
    default=True,
    help="Redact sensitive information",
)
def export(output: str, format: str, safe: bool):
    """Export configuration to file."""
    export_config_to_file(output, format=format, safe=safe)
    click.echo(f"Exported configuration to: {output}")


@config_cli.command()
def validate():
    """Validate current configuration."""
    results = validate_environment()
    
    if results["valid"]:
        click.echo("✓ Configuration is valid", fg="green")
    else:
        click.echo("✗ Configuration has errors:", fg="red")
        for error in results["errors"]:
            click.echo(f"  - {error}", fg="red")
    
    if results["warnings"]:
        click.echo("")
        click.echo("Warnings:", fg="yellow")
        for warning in results["warnings"]:
            click.echo(f"  - {warning}", fg="yellow")
    
    sys.exit(0 if results["valid"] else 1)


@config_cli.command()
@click.argument("subsystem")
@click.argument("key")
def get(subsystem: str, key: str):
    """Get a specific configuration value."""
    config_manager = get_config()
    all_settings = config_manager.get_all_settings()
    
    if subsystem not in all_settings:
        click.echo(f"Unknown subsystem: {subsystem}", err=True)
        sys.exit(1)
    
    settings = all_settings[subsystem]
    
    if not hasattr(settings, key):
        click.echo(f"Unknown setting: {key}", err=True)
        sys.exit(1)
    
    value = getattr(settings, key)
    click.echo(value)


@config_cli.command()
def env():
    """Show current environment information."""
    config_manager = get_config()
    
    click.echo(f"Environment: {config_manager.get_environment()}")
    click.echo(f"Debug Mode: {config_manager.is_debug()}")
    click.echo(f"Version: {config_manager.core.version}")
    click.echo(f"Production: {config_manager.is_production()}")


@config_cli.command()
def check():
    """Run configuration health check."""
    config_manager = get_config()
    
    click.echo("Running configuration health check...")
    click.echo("")
    
    # Check basic connectivity
    click.echo("✓ Configuration manager initialized")
    
    # Validate all settings
    try:
        config_manager.validate_all()
        click.echo("✓ All settings are valid")
    except Exception as e:
        click.echo(f"✗ Validation error: {e}", fg="red")
        sys.exit(1)
    
    # Check environment
    if config_manager.is_production():
        click.echo("⚠ Running in PRODUCTION mode", fg="yellow")
        if config_manager.core.debug:
            click.echo("  ⚠ Debug mode is enabled!", fg="yellow")
    else:
        click.echo(f"✓ Running in {config_manager.get_environment()} mode")
    
    # Storage checks
    if config_manager.storage.auto_create_dirs:
        click.echo("✓ Auto-create directories enabled")
    
    # Summary
    click.echo("")
    click.echo("Configuration check complete!")


if __name__ == "__main__":
    config_cli()
