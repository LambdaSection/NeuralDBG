"""
Neural DSL Configuration System.

Provides centralized configuration management with:
- Environment variable support
- .env file loading
- Schema validation using Pydantic
- Configuration precedence (env vars > .env > defaults)
- Typed settings for each subsystem
- Configuration validation and health checking
- Configuration migration tools
"""

from __future__ import annotations

from neural.config.manager import ConfigManager, get_config
from neural.config.settings.api import APISettings
from neural.config.settings.automl import AutoMLSettings
from neural.config.settings.core import CoreSettings
from neural.config.settings.dashboard import DashboardSettings
from neural.config.settings.hpo import HPOSettings
from neural.config.settings.integrations import IntegrationSettings
from neural.config.settings.monitoring import MonitoringSettings
from neural.config.settings.no_code import NoCodeSettings
from neural.config.settings.storage import StorageSettings
from neural.config.settings.teams import TeamsSettings


# Import validation, health checking, and migration tools if they exist
try:
    from neural.config.health import HealthChecker, HealthStatus, ServiceHealth
    from neural.config.migrator import ConfigMigrator
    from neural.config.validator import ConfigValidator, ValidationResult
    
    __all__ = [
        "ConfigManager",
        "get_config",
        "APISettings",
        "AutoMLSettings",
        "CoreSettings",
        "DashboardSettings",
        "HPOSettings",
        "IntegrationSettings",
        "MonitoringSettings",
        "NoCodeSettings",
        "StorageSettings",
        "TeamsSettings",
        "ConfigValidator",
        "ValidationResult",
        "HealthChecker",
        "HealthStatus",
        "ServiceHealth",
        "ConfigMigrator",
    ]
except ImportError:
    # Validation/health/migration tools not yet available
    __all__ = [
        "ConfigManager",
        "get_config",
        "APISettings",
        "AutoMLSettings",
        "CoreSettings",
        "DashboardSettings",
        "HPOSettings",
        "IntegrationSettings",
        "MonitoringSettings",
        "NoCodeSettings",
        "StorageSettings",
        "TeamsSettings",
    ]
