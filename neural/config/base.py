"""
Base configuration classes and utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _USE_SETTINGS_CONFIG_DICT = True
except Exception:
    # Fallback for environments without pydantic-settings
    from pydantic import BaseModel as BaseSettings  # type: ignore
    _USE_SETTINGS_CONFIG_DICT = False


class BaseConfig(BaseSettings):
    """
    Base configuration class with common settings and behavior.
    
    All subsystem settings should inherit from this class.
    Provides automatic .env file loading and environment variable handling.
    """
    
    if _USE_SETTINGS_CONFIG_DICT:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
            validate_default=True,
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
    
    def model_dump_safe(self) -> Dict[str, Any]:
        """
        Dump model to dict, excluding sensitive fields.
        
        Returns:
            Dictionary representation with sensitive fields masked
        """
        data = self.model_dump()
        sensitive_keys = {
            'secret_key', 'api_key', 'password', 'token', 'credential',
            'redis_password', 'database_url', 'webhook_url'
        }
        
        for key in data:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if data[key]:
                    data[key] = "***REDACTED***"
        
        return data


class EnvironmentType(str):
    """Environment type constants."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


def get_env_path(env_file: Optional[str] = None) -> Optional[Path]:
    """
    Get path to .env file.
    
    Args:
        env_file: Optional custom .env file path
        
    Returns:
        Path to .env file if it exists, None otherwise
    """
    if env_file:
        path = Path(env_file)
        return path if path.exists() else None
    
    # Check common locations
    locations = [
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
        Path(__file__).parent.parent.parent / ".env",
    ]
    
    for location in locations:
        if location.exists():
            return location
    
    return None


def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Optional custom .env file path
    """
    try:
        from dotenv import load_dotenv
        
        env_path = get_env_path(env_file)
        if env_path:
            load_dotenv(env_path, override=False)
    except ImportError:
        # python-dotenv not installed, skip
        pass


class ValidationError(Exception):
    """Configuration validation error."""
    pass


def validate_path(path: str, create: bool = False) -> Path:
    """
    Validate and optionally create a directory path.
    
    Args:
        path: Path to validate
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    path_obj = Path(path)
    
    if create:
        path_obj.mkdir(parents=True, exist_ok=True)
    elif not path_obj.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    return path_obj


def get_bool_env(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_int_env(key: str, default: int) -> int:
    """
    Get integer value from environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float_env(key: str, default: float) -> float:
    """
    Get float value from environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
