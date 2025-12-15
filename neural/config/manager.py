"""
Configuration manager for centralized access to all settings.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from neural.config.base import BaseConfig, load_env_file
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


T = TypeVar("T", bound=BaseConfig)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Provides lazy-loaded, cached access to all subsystem settings.
    Supports environment variable overrides and .env file loading.
    """
    
    _instance: Optional[ConfigManager] = None
    _initialized: bool = False
    
    def __new__(cls) -> ConfigManager:
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, env_file: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Optional path to .env file
        """
        if self._initialized:
            return
        
        # Load environment variables from .env file
        load_env_file(env_file)
        
        # Initialize cache for settings instances
        self._cache: Dict[str, BaseConfig] = {}
        self._env_file = env_file
        
        self._initialized = True
    
    def _get_setting(self, settings_class: Type[T]) -> T:
        """
        Get or create a settings instance.
        
        Args:
            settings_class: Settings class to instantiate
            
        Returns:
            Settings instance
        """
        class_name = settings_class.__name__
        
        if class_name not in self._cache:
            self._cache[class_name] = settings_class()
        
        return self._cache[class_name]  # type: ignore
    
    @property
    def core(self) -> CoreSettings:
        """Get core settings."""
        return self._get_setting(CoreSettings)
    
    @property
    def api(self) -> APISettings:
        """Get API settings."""
        return self._get_setting(APISettings)
    
    @property
    def storage(self) -> StorageSettings:
        """Get storage settings."""
        return self._get_setting(StorageSettings)
    
    @property
    def dashboard(self) -> DashboardSettings:
        """Get dashboard settings."""
        return self._get_setting(DashboardSettings)
    
    @property
    def no_code(self) -> NoCodeSettings:
        """Get no-code settings."""
        return self._get_setting(NoCodeSettings)
    
    @property
    def hpo(self) -> HPOSettings:
        """Get HPO settings."""
        return self._get_setting(HPOSettings)
    
    @property
    def automl(self) -> AutoMLSettings:
        """Get AutoML settings."""
        return self._get_setting(AutoMLSettings)
    
    @property
    def integrations(self) -> IntegrationSettings:
        """Get integrations settings."""
        return self._get_setting(IntegrationSettings)
    
    @property
    def teams(self) -> TeamsSettings:
        """Get teams settings."""
        return self._get_setting(TeamsSettings)
    
    @property
    def monitoring(self) -> MonitoringSettings:
        """Get monitoring settings."""
        return self._get_setting(MonitoringSettings)
    
    def get_all_settings(self) -> Dict[str, BaseConfig]:
        """
        Get all settings instances.
        
        Returns:
            Dictionary mapping setting names to instances
        """
        return {
            "core": self.core,
            "api": self.api,
            "storage": self.storage,
            "dashboard": self.dashboard,
            "no_code": self.no_code,
            "hpo": self.hpo,
            "automl": self.automl,
            "integrations": self.integrations,
            "teams": self.teams,
            "monitoring": self.monitoring,
        }
    
    def reload(self, env_file: Optional[str] = None) -> None:
        """
        Reload all settings from environment.
        
        Args:
            env_file: Optional path to .env file
        """
        if env_file:
            self._env_file = env_file
        
        # Clear cache to force reload
        self._cache.clear()
        
        # Reload environment variables
        load_env_file(self._env_file)
    
    def dump_config(self, safe: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Dump all configuration to dictionary.
        
        Args:
            safe: If True, redact sensitive fields
            
        Returns:
            Dictionary of all settings
        """
        all_settings = self.get_all_settings()
        
        if safe:
            return {
                name: settings.model_dump_safe()
                for name, settings in all_settings.items()
            }
        else:
            return {
                name: settings.model_dump()
                for name, settings in all_settings.items()
            }
    
    def validate_all(self) -> bool:
        """
        Validate all settings.
        
        Returns:
            True if all settings are valid
            
        Raises:
            ValidationError: If any setting is invalid
        """
        for settings in self.get_all_settings().values():
            # Accessing the settings triggers validation
            pass
        return True
    
    def get_environment(self) -> str:
        """
        Get current environment.
        
        Returns:
            Environment name (development, staging, production, etc.)
        """
        return self.core.environment
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == "development"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.core.debug


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config(env_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(env_file)
    elif env_file:
        _config_manager.reload(env_file)
    
    return _config_manager


def reset_config() -> None:
    """Reset the global configuration manager (useful for testing)."""
    global _config_manager
    _config_manager = None
    ConfigManager._instance = None
    ConfigManager._initialized = False
