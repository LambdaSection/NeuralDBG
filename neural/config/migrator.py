"""Configuration migration tool for Neural DSL."""

from __future__ import annotations

import os
from typing import Any, Dict


try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False


class ConfigMigrator:
    """Migrates configuration from YAML to environment variables."""
    
    # Mapping from YAML paths to environment variable names
    YAML_TO_ENV_MAP = {
        'api.host': 'API_HOST',
        'api.port': 'API_PORT',
        'api.workers': 'API_WORKERS',
        'api.debug': 'DEBUG',
        'api.secret_key': 'SECRET_KEY',
        'api.database_url': 'DATABASE_URL',
        'api.storage_path': 'STORAGE_PATH',
        'api.experiments_path': 'EXPERIMENTS_PATH',
        'api.models_path': 'MODELS_PATH',
        'api.rate_limit.enabled': 'RATE_LIMIT_ENABLED',
        'api.rate_limit.requests': 'RATE_LIMIT_REQUESTS',
        'api.rate_limit.period': 'RATE_LIMIT_PERIOD',
        'api.cors_origins': 'CORS_ORIGINS',
        'dashboard.host': 'DASHBOARD_HOST',
        'dashboard.port': 'DASHBOARD_PORT',
        'dashboard.debug': 'DEBUG',
        'dashboard.websocket_interval': 'DASHBOARD_WEBSOCKET_INTERVAL',
        'dashboard.auth.username': 'DASHBOARD_USERNAME',
        'dashboard.auth.password': 'DASHBOARD_PASSWORD',
        'redis.host': 'REDIS_HOST',
        'redis.port': 'REDIS_PORT',
        'redis.db': 'REDIS_DB',
        'redis.password': 'REDIS_PASSWORD',
        'celery.broker_url': 'CELERY_BROKER_URL',
        'celery.result_backend': 'CELERY_RESULT_BACKEND',
        'webhook.timeout': 'WEBHOOK_TIMEOUT',
        'webhook.retry_limit': 'WEBHOOK_RETRY_LIMIT',
    }
    
    def __init__(self):
        """Initialize configuration migrator."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is not installed. "
                "Install it with: pip install pyyaml"
            )
    
    def migrate_yaml_to_env(
        self,
        yaml_file: str,
        output_file: str = '.env',
        overwrite: bool = False
    ) -> Dict[str, str]:
        """Migrate configuration from YAML file to .env file.
        
        Parameters
        ----------
        yaml_file : str
            Path to YAML configuration file
        output_file : str
            Path to output .env file (default: .env)
        overwrite : bool
            Whether to overwrite existing .env file (default: False)
        
        Returns
        -------
        dict
            Dictionary of environment variables
        
        Raises
        ------
        FileNotFoundError
            If YAML file does not exist
        FileExistsError
            If output file exists and overwrite is False
        """
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")
        
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(
                f"Output file {output_file} already exists. "
                "Use overwrite=True to replace it."
            )
        
        # Load YAML configuration
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Convert to environment variables
        env_vars = self._yaml_to_env(config)
        
        # Write to .env file
        self._write_env_file(env_vars, output_file)
        
        return env_vars
    
    def _yaml_to_env(self, config: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
        """Convert YAML configuration to environment variables.
        
        Parameters
        ----------
        config : dict
            YAML configuration dictionary
        prefix : str
            Prefix for nested keys
        
        Returns
        -------
        dict
            Dictionary of environment variables
        """
        env_vars = {}
        
        for key, value in config.items():
            yaml_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                nested_vars = self._yaml_to_env(value, yaml_path)
                env_vars.update(nested_vars)
            else:
                # Convert to environment variable
                env_key = self.YAML_TO_ENV_MAP.get(yaml_path)
                
                if env_key:
                    # Convert value to string
                    if isinstance(value, bool):
                        env_value = 'true' if value else 'false'
                    elif isinstance(value, (list, dict)):
                        # JSON serialize complex types
                        import json
                        env_value = json.dumps(value)
                    elif value is None:
                        env_value = ''
                    else:
                        env_value = str(value)
                    
                    env_vars[env_key] = env_value
        
        return env_vars
    
    def _write_env_file(self, env_vars: Dict[str, str], output_file: str):
        """Write environment variables to .env file.
        
        Parameters
        ----------
        env_vars : dict
            Dictionary of environment variables
        output_file : str
            Path to output file
        """
        with open(output_file, 'w') as f:
            f.write("# Neural DSL Configuration\n")
            f.write("# Auto-generated from YAML configuration\n")
            f.write(f"# Generated at: {self._get_timestamp()}\n\n")
            
            # Group by service
            services = {}
            for key, value in env_vars.items():
                # Determine service from key prefix
                if key.startswith('API_'):
                    service = 'API'
                elif key.startswith('DASHBOARD_'):
                    service = 'Dashboard'
                elif key.startswith('REDIS_'):
                    service = 'Redis'
                elif key.startswith('CELERY_'):
                    service = 'Celery'
                elif key.startswith('WEBHOOK_'):
                    service = 'Webhooks'
                else:
                    service = 'General'
                
                if service not in services:
                    services[service] = []
                services[service].append((key, value))
            
            # Write grouped variables
            for service, vars_list in sorted(services.items()):
                f.write(f"# {service} Configuration\n")
                for key, value in sorted(vars_list):
                    # Quote values with spaces or special characters
                    if ' ' in value or any(c in value for c in ['#', '$', '"', "'"]):
                        value = f'"{value}"'
                    f.write(f"{key}={value}\n")
                f.write("\n")
    
    def migrate_env_to_yaml(
        self,
        env_file: str = '.env',
        output_file: str = 'config.yaml',
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Migrate configuration from .env file to YAML file.
        
        Parameters
        ----------
        env_file : str
            Path to .env file (default: .env)
        output_file : str
            Path to output YAML file
        overwrite : bool
            Whether to overwrite existing YAML file (default: False)
        
        Returns
        -------
        dict
            YAML configuration dictionary
        
        Raises
        ------
        FileNotFoundError
            If .env file does not exist
        FileExistsError
            If output file exists and overwrite is False
        """
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(
                f"Output file {output_file} already exists. "
                "Use overwrite=True to replace it."
            )
        
        # Load environment variables
        env_vars = self._load_env_file(env_file)
        
        # Convert to YAML structure
        config = self._env_to_yaml(env_vars)
        
        # Write to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config
    
    def _load_env_file(self, env_file: str) -> Dict[str, str]:
        """Load environment variables from .env file.
        
        Parameters
        ----------
        env_file : str
            Path to .env file
        
        Returns
        -------
        dict
            Dictionary of environment variables
        """
        env_vars = {}
        
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
        
        return env_vars
    
    def _env_to_yaml(self, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Convert environment variables to YAML structure.
        
        Parameters
        ----------
        env_vars : dict
            Dictionary of environment variables
        
        Returns
        -------
        dict
            YAML configuration dictionary
        """
        # Create reverse mapping
        env_to_yaml = {v: k for k, v in self.YAML_TO_ENV_MAP.items()}
        
        config = {}
        
        for env_key, env_value in env_vars.items():
            yaml_path = env_to_yaml.get(env_key)
            
            if yaml_path:
                # Parse the value
                value = self._parse_value(env_value)
                
                # Set nested dictionary value
                self._set_nested_value(config, yaml_path, value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a value in a nested dictionary using dot notation.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        path : str
            Dot-separated path (e.g., 'api.port')
        value : any
            Value to set
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.
        
        Parameters
        ----------
        value : str
            String value
        
        Returns
        -------
        any
            Parsed value
        """
        # Try to parse as boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try to parse as JSON (for lists/dicts)
        if value.startswith('[') or value.startswith('{'):
            try:
                import json
                return json.loads(value)
            except:
                pass
        
        # Return as string
        return value
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def generate_env_template(self, output_file: str = '.env.example'):
        """Generate a template .env file with all available options.
        
        Parameters
        ----------
        output_file : str
            Path to output file
        """
        template = """# Neural DSL Configuration Template
# Copy this file to .env and customize for your deployment

# =============================================================================
# SECURITY (REQUIRED)
# =============================================================================
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=change-me-in-production-use-strong-random-key

# =============================================================================
# API SERVER
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false

# Database configuration
# SQLite: sqlite:///./neural_api.db
# PostgreSQL: postgresql://user:pass@localhost:5432/neural_api
DATABASE_URL=sqlite:///./neural_api.db

# Storage paths
STORAGE_PATH=./neural_storage
EXPERIMENTS_PATH=./neural_experiments
MODELS_PATH=./neural_models

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# CORS origins (JSON array)
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# =============================================================================
# DASHBOARD (NeuralDbg)
# =============================================================================
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_WEBSOCKET_INTERVAL=1000
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=

# =============================================================================
# REDIS
# =============================================================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# =============================================================================
# CELERY (Async Task Queue)
# =============================================================================
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# =============================================================================
# WEBHOOKS
# =============================================================================
WEBHOOK_TIMEOUT=30
WEBHOOK_RETRY_LIMIT=3

# =============================================================================
# POST-RELEASE AUTOMATION (Optional)
# =============================================================================
# Netlify Build Hook URL
NETLIFY_BUILD_HOOK=

# Vercel Deploy Hook URL
VERCEL_DEPLOY_HOOK=

# Discord Webhook URL for notifications
DISCORD_WEBHOOK_URL=
"""
        
        with open(output_file, 'w') as f:
            f.write(template)
