"""
Team management configuration settings.
"""

from __future__ import annotations

from typing import Dict, Literal

from pydantic import Field

from neural.config.base import BaseConfig


class QuotaSettings(BaseConfig):
    """Resource quota settings for a plan."""
    
    model_config = {"extra": "allow"}
    
    max_models: int = Field(default=10, gt=0)
    max_experiments: int = Field(default=100, gt=0)
    max_storage_gb: float = Field(default=10.0, gt=0)
    max_compute_hours: float = Field(default=100.0, gt=0)
    max_team_members: int = Field(default=5, gt=0)
    max_api_calls_per_day: int = Field(default=10000, gt=0)
    max_concurrent_runs: int = Field(default=5, gt=0)


class PlanPricingSettings(BaseConfig):
    """Pricing settings for a plan."""
    
    monthly: float = Field(default=0.0, ge=0)
    annual: float = Field(default=0.0, ge=0)


class TeamsSettings(BaseConfig):
    """Team management configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_TEAMS_"}
    
    # Storage settings
    base_dir: str = Field(
        default="neural_organizations",
        description="Base directory for team data"
    )
    
    # Quota defaults for free plan
    free_max_models: int = Field(default=10, gt=0)
    free_max_experiments: int = Field(default=100, gt=0)
    free_max_storage_gb: float = Field(default=10.0, gt=0)
    free_max_compute_hours: float = Field(default=100.0, gt=0)
    free_max_team_members: int = Field(default=5, gt=0)
    free_max_api_calls_per_day: int = Field(default=10000, gt=0)
    free_max_concurrent_runs: int = Field(default=5, gt=0)
    
    # Quota defaults for starter plan
    starter_max_models: int = Field(default=50, gt=0)
    starter_max_experiments: int = Field(default=500, gt=0)
    starter_max_storage_gb: float = Field(default=100.0, gt=0)
    starter_max_compute_hours: float = Field(default=1000.0, gt=0)
    starter_max_team_members: int = Field(default=10, gt=0)
    starter_max_api_calls_per_day: int = Field(default=100000, gt=0)
    starter_max_concurrent_runs: int = Field(default=10, gt=0)
    
    # Quota defaults for professional plan
    professional_max_models: int = Field(default=200, gt=0)
    professional_max_experiments: int = Field(default=2000, gt=0)
    professional_max_storage_gb: float = Field(default=500.0, gt=0)
    professional_max_compute_hours: float = Field(default=5000.0, gt=0)
    professional_max_team_members: int = Field(default=50, gt=0)
    professional_max_api_calls_per_day: int = Field(default=1000000, gt=0)
    professional_max_concurrent_runs: int = Field(default=25, gt=0)
    
    # Quota defaults for enterprise plan
    enterprise_max_models: int = Field(default=999999, gt=0)
    enterprise_max_experiments: int = Field(default=999999, gt=0)
    enterprise_max_storage_gb: float = Field(default=99999.0, gt=0)
    enterprise_max_compute_hours: float = Field(default=99999.0, gt=0)
    enterprise_max_team_members: int = Field(default=999, gt=0)
    enterprise_max_api_calls_per_day: int = Field(default=99999999, gt=0)
    enterprise_max_concurrent_runs: int = Field(default=100, gt=0)
    
    # Pricing rates (in USD)
    compute_hour_rate: float = Field(
        default=0.50,
        ge=0,
        description="Cost per compute hour in USD"
    )
    storage_gb_month_rate: float = Field(
        default=0.10,
        ge=0,
        description="Cost per GB per month in USD"
    )
    api_call_1000_rate: float = Field(
        default=0.01,
        ge=0,
        description="Cost per 1000 API calls in USD"
    )
    
    # Plan pricing
    free_monthly: float = Field(default=0.0, ge=0)
    free_annual: float = Field(default=0.0, ge=0)
    starter_monthly: float = Field(default=29.0, ge=0)
    starter_annual: float = Field(default=290.0, ge=0)
    professional_monthly: float = Field(default=99.0, ge=0)
    professional_annual: float = Field(default=990.0, ge=0)
    enterprise_monthly: float = Field(default=499.0, ge=0)
    enterprise_annual: float = Field(default=4990.0, ge=0)
    
    # Analytics settings
    analytics_retention_days: int = Field(
        default=365,
        gt=0,
        description="Analytics data retention in days"
    )
    analytics_aggregation_interval: Literal["hourly", "daily", "weekly"] = Field(
        default="daily",
        description="Analytics aggregation interval"
    )
    
    # Billing settings
    invoice_due_days: int = Field(
        default=30,
        gt=0,
        description="Invoice due period in days"
    )
    payment_grace_period_days: int = Field(
        default=7,
        gt=0,
        description="Payment grace period in days"
    )
    
    # Security settings
    session_timeout_minutes: int = Field(
        default=60,
        gt=0,
        description="Session timeout in minutes"
    )
    max_login_attempts: int = Field(
        default=5,
        gt=0,
        description="Maximum login attempts before lockout"
    )
    password_min_length: int = Field(
        default=8,
        gt=0,
        description="Minimum password length"
    )
    require_mfa: bool = Field(
        default=False,
        description="Require multi-factor authentication"
    )
    
    # Feature flags
    enable_stripe_integration: bool = Field(
        default=True,
        description="Enable Stripe payment integration"
    )
    enable_usage_analytics: bool = Field(
        default=True,
        description="Enable usage analytics"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    enable_sso: bool = Field(
        default=False,
        description="Enable single sign-on"
    )
    
    # Notification settings
    enable_email_notifications: bool = Field(
        default=True,
        description="Enable email notifications"
    )
    enable_slack_notifications: bool = Field(
        default=False,
        description="Enable Slack notifications"
    )
    slack_webhook_url: str = Field(
        default="",
        description="Slack webhook URL"
    )
    
    def get_plan_quotas(self, plan: str) -> Dict[str, float]:
        """Get quota configuration for a billing plan."""
        quotas = {
            "free": {
                "max_models": self.free_max_models,
                "max_experiments": self.free_max_experiments,
                "max_storage_gb": self.free_max_storage_gb,
                "max_compute_hours": self.free_max_compute_hours,
                "max_team_members": self.free_max_team_members,
                "max_api_calls_per_day": self.free_max_api_calls_per_day,
                "max_concurrent_runs": self.free_max_concurrent_runs,
            },
            "starter": {
                "max_models": self.starter_max_models,
                "max_experiments": self.starter_max_experiments,
                "max_storage_gb": self.starter_max_storage_gb,
                "max_compute_hours": self.starter_max_compute_hours,
                "max_team_members": self.starter_max_team_members,
                "max_api_calls_per_day": self.starter_max_api_calls_per_day,
                "max_concurrent_runs": self.starter_max_concurrent_runs,
            },
            "professional": {
                "max_models": self.professional_max_models,
                "max_experiments": self.professional_max_experiments,
                "max_storage_gb": self.professional_max_storage_gb,
                "max_compute_hours": self.professional_max_compute_hours,
                "max_team_members": self.professional_max_team_members,
                "max_api_calls_per_day": self.professional_max_api_calls_per_day,
                "max_concurrent_runs": self.professional_max_concurrent_runs,
            },
            "enterprise": {
                "max_models": self.enterprise_max_models,
                "max_experiments": self.enterprise_max_experiments,
                "max_storage_gb": self.enterprise_max_storage_gb,
                "max_compute_hours": self.enterprise_max_compute_hours,
                "max_team_members": self.enterprise_max_team_members,
                "max_api_calls_per_day": self.enterprise_max_api_calls_per_day,
                "max_concurrent_runs": self.enterprise_max_concurrent_runs,
            },
        }
        return quotas.get(plan, quotas["free"])
    
    def get_plan_pricing(self, plan: str) -> Dict[str, float]:
        """Get pricing for a billing plan."""
        pricing = {
            "free": {"monthly": self.free_monthly, "annual": self.free_annual},
            "starter": {"monthly": self.starter_monthly, "annual": self.starter_annual},
            "professional": {"monthly": self.professional_monthly, "annual": self.professional_annual},
            "enterprise": {"monthly": self.enterprise_monthly, "annual": self.enterprise_annual},
        }
        return pricing.get(plan, pricing["free"])
