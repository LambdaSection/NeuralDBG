"""
Tests for Neural DSL platform integrations base classes.

These tests verify the structure and basic functionality of the integrations module.
"""

import pytest

from neural.integrations import BaseConnector, JobResult, JobStatus, ResourceConfig


class TestResourceConfig:
    """Test ResourceConfig dataclass."""
    
    def test_default_config(self):
        """Test default resource configuration."""
        config = ResourceConfig(instance_type='test-instance')
        assert config.instance_type == 'test-instance'
        assert config.gpu_enabled is False
        assert config.gpu_count == 0
        assert config.auto_shutdown is True
    
    def test_gpu_config(self):
        """Test GPU resource configuration."""
        config = ResourceConfig(
            instance_type='gpu-instance',
            gpu_enabled=True,
            gpu_count=4,
            memory_gb=64
        )
        assert config.gpu_enabled is True
        assert config.gpu_count == 4
        assert config.memory_gb == 64
    
    def test_custom_params(self):
        """Test custom parameters."""
        config = ResourceConfig(
            instance_type='custom',
            custom_params={'priority': 'high', 'zone': 'us-east-1a'}
        )
        assert config.custom_params['priority'] == 'high'
        assert config.custom_params['zone'] == 'us-east-1a'


class TestJobStatus:
    """Test JobStatus enumeration."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert JobStatus.PENDING.value == 'pending'
        assert JobStatus.RUNNING.value == 'running'
        assert JobStatus.SUCCEEDED.value == 'succeeded'
        assert JobStatus.FAILED.value == 'failed'
        assert JobStatus.CANCELLED.value == 'cancelled'
        assert JobStatus.UNKNOWN.value == 'unknown'


class TestJobResult:
    """Test JobResult dataclass."""
    
    def test_basic_result(self):
        """Test basic job result."""
        result = JobResult(
            job_id='test-123',
            status=JobStatus.SUCCEEDED
        )
        assert result.job_id == 'test-123'
        assert result.status == JobStatus.SUCCEEDED
        assert result.output is None
        assert result.error is None
    
    def test_full_result(self):
        """Test full job result with all fields."""
        result = JobResult(
            job_id='test-456',
            status=JobStatus.SUCCEEDED,
            output='Model trained successfully',
            metrics={'accuracy': 0.95, 'loss': 0.12},
            artifacts=['model.h5', 'metrics.json'],
            logs_url='https://logs.example.com/test-456',
            duration_seconds=120.5
        )
        assert result.job_id == 'test-456'
        assert result.metrics['accuracy'] == 0.95
        assert len(result.artifacts) == 2
        assert result.duration_seconds == 120.5


class TestBaseConnector:
    """Test BaseConnector abstract class."""
    
    def test_base_connector_abstract(self):
        """Test that BaseConnector cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseConnector()
    
    def test_concrete_implementation(self):
        """Test a concrete implementation of BaseConnector."""
        
        class TestConnector(BaseConnector):
            def authenticate(self):
                return True
            
            def submit_job(self, code, resource_config=None, environment=None,
                          dependencies=None, job_name=None):
                return 'job-123'
            
            def get_job_status(self, job_id):
                return JobStatus.RUNNING
            
            def get_job_result(self, job_id):
                return JobResult(job_id=job_id, status=JobStatus.SUCCEEDED)
            
            def cancel_job(self, job_id):
                return True
            
            def list_jobs(self, limit=10, status_filter=None):
                return []
            
            def get_logs(self, job_id):
                return 'logs...'
        
        connector = TestConnector(credentials={'key': 'value'})
        assert connector.credentials == {'key': 'value'}
        assert connector.authenticated is False
        
        assert connector.authenticate() is True
        assert connector.submit_job('code') == 'job-123'
        assert connector.get_job_status('job-123') == JobStatus.RUNNING
        assert connector.cancel_job('job-123') is True
        assert connector.list_jobs() == []
        assert connector.get_logs('job-123') == 'logs...'
    
    def test_validate_credentials(self):
        """Test credential validation."""
        
        class TestConnector(BaseConnector):
            def authenticate(self):
                return True
            def submit_job(self, code, resource_config=None, environment=None,
                          dependencies=None, job_name=None):
                return 'job-123'
            def get_job_status(self, job_id):
                return JobStatus.RUNNING
            def get_job_result(self, job_id):
                return JobResult(job_id=job_id, status=JobStatus.SUCCEEDED)
            def cancel_job(self, job_id):
                return True
            def list_jobs(self, limit=10, status_filter=None):
                return []
            def get_logs(self, job_id):
                return 'logs...'
        
        connector_with_creds = TestConnector(credentials={'key': 'value'})
        assert connector_with_creds.validate_credentials() is True
        
        connector_without_creds = TestConnector(credentials={})
        assert connector_without_creds.validate_credentials() is False
    
    def test_optional_methods_not_implemented(self):
        """Test that optional methods raise NotImplementedError."""
        
        class TestConnector(BaseConnector):
            def authenticate(self):
                return True
            def submit_job(self, code, resource_config=None, environment=None,
                          dependencies=None, job_name=None):
                return 'job-123'
            def get_job_status(self, job_id):
                return JobStatus.RUNNING
            def get_job_result(self, job_id):
                return JobResult(job_id=job_id, status=JobStatus.SUCCEEDED)
            def cancel_job(self, job_id):
                return True
            def list_jobs(self, limit=10, status_filter=None):
                return []
            def get_logs(self, job_id):
                return 'logs...'
        
        connector = TestConnector()
        
        with pytest.raises(NotImplementedError):
            connector.upload_file('local', 'remote')
        
        with pytest.raises(NotImplementedError):
            connector.download_file('remote', 'local')
        
        with pytest.raises(NotImplementedError):
            connector.deploy_model('model', 'endpoint')
        
        with pytest.raises(NotImplementedError):
            connector.delete_endpoint('endpoint')
        
        with pytest.raises(NotImplementedError):
            connector.get_resource_usage()


class TestIntegrationImports:
    """Test that all integration components can be imported."""
    
    def test_import_base(self):
        """Test importing base components."""
        from neural.integrations.base import BaseConnector, JobResult, JobStatus, ResourceConfig
        assert BaseConnector is not None
        assert ResourceConfig is not None
        assert JobStatus is not None
        assert JobResult is not None
    
    def test_import_from_package(self):
        """Test importing from package."""
        from neural.integrations import BaseConnector, JobResult, JobStatus, ResourceConfig
        assert BaseConnector is not None
        assert ResourceConfig is not None
        assert JobStatus is not None
        assert JobResult is not None
    
    def test_import_utils(self):
        """Test importing utilities."""
        from neural.integrations.utils import (
            format_job_output,
            load_credentials_from_file,
            save_credentials_to_file,
        )
        assert format_job_output is not None
        assert load_credentials_from_file is not None
        assert save_credentials_to_file is not None


class TestUtilities:
    """Test utility functions."""
    
    def test_format_job_output(self):
        """Test job output formatting."""
        from neural.integrations.utils import format_job_output
        
        result = JobResult(
            job_id='test-123',
            status=JobStatus.SUCCEEDED,
            output='Test output',
            duration_seconds=10.5
        )
        
        formatted = format_job_output(result)
        assert 'test-123' in formatted
        assert 'succeeded' in formatted
        assert '10.50s' in formatted
    
    def test_format_job_output_with_metrics(self):
        """Test job output formatting with metrics."""
        from neural.integrations.utils import format_job_output
        
        result = JobResult(
            job_id='test-456',
            status=JobStatus.SUCCEEDED,
            metrics={'accuracy': 0.95, 'loss': 0.12}
        )
        
        formatted = format_job_output(result)
        assert 'accuracy' in formatted
        assert '0.95' in formatted
    
    def test_format_job_output_non_result(self):
        """Test formatting non-JobResult objects."""
        from neural.integrations.utils import format_job_output
        
        formatted = format_job_output("simple string")
        assert formatted == "simple string"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
