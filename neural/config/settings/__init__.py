"""
Settings modules for each Neural DSL subsystem.
"""

from __future__ import annotations

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

__all__ = [
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
