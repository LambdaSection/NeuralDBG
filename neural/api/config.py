"""
Configuration for Neural API server.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API server settings."""
    
    app_name: str = "Neural DSL API"
    app_version: str = "0.3.0"
    debug: bool = Field(default=False, validation_alias="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    port: int = Field(default=8000, validation_alias="API_PORT")
    workers: int = Field(default=4, validation_alias="API_WORKERS")
    
    # Security settings
    secret_key: str = Field(default="change-me-in-production", validation_alias="SECRET_KEY")
    api_key_header: str = "X-API-Key"
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, validation_alias="RATE_LIMIT_PERIOD")
    
    # Redis settings for Celery
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, validation_alias="REDIS_PASSWORD")
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL for Celery."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Celery settings
    celery_broker_url: Optional[str] = Field(default=None, validation_alias="CELERY_BROKER_URL")
    celery_result_backend: Optional[str] = Field(default=None, validation_alias="CELERY_RESULT_BACKEND")
    
    @property
    def broker_url(self) -> str:
        """Get Celery broker URL."""
        return self.celery_broker_url or self.redis_url
    
    @property
    def result_backend(self) -> str:
        """Get Celery result backend URL."""
        return self.celery_result_backend or self.redis_url
    
    # Database settings
    database_url: str = Field(
        default="sqlite:///./neural_api.db",
        validation_alias="DATABASE_URL"
    )
    
    # Storage settings
    storage_path: str = Field(default="./neural_storage", validation_alias="STORAGE_PATH")
    experiments_path: str = Field(default="./neural_experiments", validation_alias="EXPERIMENTS_PATH")
    models_path: str = Field(default="./neural_models", validation_alias="MODELS_PATH")
    
    # Webhook settings
    webhook_timeout: int = Field(default=30, validation_alias="WEBHOOK_TIMEOUT")
    webhook_retry_limit: int = Field(default=3, validation_alias="WEBHOOK_RETRY_LIMIT")
    
    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        validation_alias="CORS_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
