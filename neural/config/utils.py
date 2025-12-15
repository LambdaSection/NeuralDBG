"""
Configuration utility functions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml

from neural.config.manager import get_config


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save YAML file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json_config(config: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=indent)


def export_config_to_file(
    file_path: str,
    format: str = "yaml",
    safe: bool = True
) -> None:
    """
    Export current configuration to file.
    
    Args:
        file_path: Path to save configuration
        format: Output format (yaml or json)
        safe: If True, redact sensitive fields
    """
    config_manager = get_config()
    config_dict = config_manager.dump_config(safe=safe)
    
    if format.lower() == "json":
        save_json_config(config_dict, file_path)
    else:
        save_yaml_config(config_dict, file_path)


def generate_env_template(output_path: str = ".env.template") -> None:
    """
    Generate .env template file with all available configuration options.
    
    Args:
        output_path: Path to save template file
    """
    config_manager = get_config()
    
    lines = [
        "# Neural DSL Configuration Template",
        "# Copy this file to .env and customize the values",
        "",
    ]
    
    all_settings = config_manager.get_all_settings()
    
    for name, settings in all_settings.items():
        lines.append(f"# {name.upper()} SETTINGS")
        lines.append("")
        
        model_fields = settings.model_fields
        model_config = getattr(settings, "model_config", {})
        env_prefix = model_config.get("env_prefix", "")
        
        for field_name, field_info in model_fields.items():
            # Get environment variable name
            env_name = f"{env_prefix}{field_name}".upper()
            
            # Get description
            description = field_info.description or ""
            if description:
                lines.append(f"# {description}")
            
            # Get default value
            default = field_info.default
            if default is not None and not callable(default):
                lines.append(f"# Default: {default}")
            
            # Add template line
            lines.append(f"{env_name}=")
            lines.append("")
        
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validate_environment() -> Dict[str, Any]:
    """
    Validate current environment configuration.
    
    Returns:
        Dictionary with validation results
    """
    config_manager = get_config()
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    
    try:
        config_manager.validate_all()
    except Exception as e:
        results["valid"] = False
        results["errors"].append(str(e))
    
    # Check for sensitive defaults
    if config_manager.api.secret_key == "change-me-in-production":
        results["warnings"].append(
            "API secret_key is set to default value. Change it in production!"
        )
    
    # Check production settings
    if config_manager.is_production():
        if config_manager.core.debug:
            results["warnings"].append(
                "Debug mode is enabled in production environment"
            )
        
        if config_manager.api.debug:
            results["warnings"].append(
                "API debug mode is enabled in production environment"
            )
    
    return results


def get_config_summary() -> Dict[str, Any]:
    """
    Get summary of current configuration.
    
    Returns:
        Configuration summary dictionary
    """
    config_manager = get_config()
    
    return {
        "environment": config_manager.get_environment(),
        "debug": config_manager.is_debug(),
        "version": config_manager.core.version,
        "subsystems": {
            "api": {
                "enabled": True,
                "host": config_manager.api.host,
                "port": config_manager.api.port,
            },
            "dashboard": {
                "enabled": True,
                "host": config_manager.dashboard.host,
                "port": config_manager.dashboard.port,
            },
            "no_code": {
                "enabled": True,
                "host": config_manager.no_code.host,
                "port": config_manager.no_code.port,
            },
            "storage": {
                "base_path": config_manager.storage.base_path,
                "cloud_enabled": config_manager.storage.cloud_storage_enabled,
            },
            "monitoring": {
                "enabled": config_manager.monitoring.enabled,
                "prometheus": config_manager.monitoring.prometheus_enabled,
            },
            "integrations": {
                "sagemaker": config_manager.integrations.sagemaker_enabled,
                "vertex": config_manager.integrations.vertex_enabled,
                "azure": config_manager.integrations.azure_enabled,
                "databricks": config_manager.integrations.databricks_enabled,
            },
        },
    }


def set_environment(environment: str) -> None:
    """
    Set environment and reload configuration.
    
    Args:
        environment: Environment name (development, staging, production)
    """
    os.environ["NEURAL_ENVIRONMENT"] = environment
    config_manager = get_config()
    config_manager.reload()


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result
