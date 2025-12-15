"""
Basic tests for configuration system.

This is a minimal test file to verify the configuration system works.
For comprehensive testing, add tests to tests/config/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_import():
    """Test that all configuration modules can be imported."""
    
    print("✓ All imports successful")


def test_config_manager():
    """Test basic ConfigManager functionality."""
    from neural.config import get_config
    
    config = get_config()
    
    # Test property access
    assert hasattr(config, "core")
    assert hasattr(config, "api")
    assert hasattr(config, "storage")
    assert hasattr(config, "dashboard")
    assert hasattr(config, "no_code")
    assert hasattr(config, "hpo")
    assert hasattr(config, "automl")
    assert hasattr(config, "integrations")
    assert hasattr(config, "teams")
    assert hasattr(config, "monitoring")
    
    print("✓ ConfigManager has all subsystem properties")


def test_core_settings():
    """Test core settings."""
    from neural.config import get_config
    
    config = get_config()
    
    # Check defaults
    assert config.core.app_name == "Neural DSL"
    assert config.core.version == "0.3.0"
    assert config.core.default_backend in ["tensorflow", "pytorch", "onnx"]
    
    print("✓ Core settings accessible")


def test_api_settings():
    """Test API settings."""
    from neural.config import get_config
    
    config = get_config()
    
    # Check defaults
    assert isinstance(config.api.port, int)
    assert config.api.port > 0
    assert config.api.host
    
    # Check computed properties
    redis_url = config.api.redis_url
    assert "redis://" in redis_url
    
    broker_url = config.api.broker_url
    assert broker_url
    
    print("✓ API settings accessible")


def test_storage_settings():
    """Test storage settings."""
    from neural.config import get_config
    
    config = get_config()
    
    # Check defaults
    assert config.storage.base_path
    assert config.storage.auto_create_dirs is not None
    
    # Test get_full_path method
    models_path = config.storage.get_full_path("models")
    assert isinstance(models_path, Path)
    
    print("✓ Storage settings accessible")


def test_environment_variables():
    """Test environment variable override."""
    from neural.config import get_config, reset_config
    
    # Set environment variable
    os.environ["NEURAL_DEBUG"] = "true"
    
    # Reset and reload
    reset_config()
    config = get_config()
    
    # Check if environment variable was applied
    assert config.core.debug is True
    
    # Clean up
    del os.environ["NEURAL_DEBUG"]
    reset_config()
    
    print("✓ Environment variable override works")


def test_validation():
    """Test configuration validation."""
    from neural.config import get_config
    
    config = get_config()
    
    # This should not raise an exception
    config.validate_all()
    
    print("✓ Configuration validation successful")


def test_dump_config():
    """Test configuration export."""
    from neural.config import get_config
    
    config = get_config()
    
    # Test safe dump
    safe_config = config.dump_config(safe=True)
    assert isinstance(safe_config, dict)
    assert "core" in safe_config
    assert "api" in safe_config
    
    # Test unsafe dump
    unsafe_config = config.dump_config(safe=False)
    assert isinstance(unsafe_config, dict)
    
    print("✓ Configuration dump works")


def test_get_all_settings():
    """Test getting all settings."""
    from neural.config import get_config
    
    config = get_config()
    
    all_settings = config.get_all_settings()
    assert isinstance(all_settings, dict)
    assert len(all_settings) > 0
    
    # Check subsystems are present
    expected_subsystems = [
        "core", "api", "storage", "dashboard", "no_code",
        "hpo", "automl", "integrations", "teams", "monitoring"
    ]
    for subsystem in expected_subsystems:
        assert subsystem in all_settings
    
    print("✓ Get all settings works")


def test_environment_checks():
    """Test environment check methods."""
    from neural.config import get_config
    
    config = get_config()
    
    # These should not raise errors
    env = config.get_environment()
    assert isinstance(env, str)
    
    is_prod = config.is_production()
    assert isinstance(is_prod, bool)
    
    is_dev = config.is_development()
    assert isinstance(is_dev, bool)
    
    is_debug = config.is_debug()
    assert isinstance(is_debug, bool)
    
    print("✓ Environment checks work")


def test_teams_methods():
    """Test teams settings methods."""
    from neural.config import get_config
    
    config = get_config()
    
    # Test plan quotas
    quotas = config.teams.get_plan_quotas("professional")
    assert isinstance(quotas, dict)
    assert "max_models" in quotas
    
    # Test plan pricing
    pricing = config.teams.get_plan_pricing("starter")
    assert isinstance(pricing, dict)
    assert "monthly" in pricing
    assert "annual" in pricing
    
    print("✓ Teams settings methods work")


def test_utils():
    """Test configuration utilities."""
    from neural.config.utils import (
        get_config_summary,
        validate_environment,
    )
    
    # Test summary
    summary = get_config_summary()
    assert isinstance(summary, dict)
    assert "environment" in summary
    assert "subsystems" in summary
    
    # Test validation
    results = validate_environment()
    assert isinstance(results, dict)
    assert "valid" in results
    assert "errors" in results
    assert "warnings" in results
    
    print("✓ Configuration utilities work")


def run_all_tests():
    """Run all tests."""
    tests = [
        ("Import test", test_import),
        ("ConfigManager test", test_config_manager),
        ("Core settings test", test_core_settings),
        ("API settings test", test_api_settings),
        ("Storage settings test", test_storage_settings),
        ("Environment variables test", test_environment_variables),
        ("Validation test", test_validation),
        ("Dump config test", test_dump_config),
        ("Get all settings test", test_get_all_settings),
        ("Environment checks test", test_environment_checks),
        ("Teams methods test", test_teams_methods),
        ("Utils test", test_utils),
    ]
    
    print("Running configuration system tests...\n")
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
