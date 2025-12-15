# Neural DSL Platform Integrations

Guide to building custom ML platform integrations for Neural DSL.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Creating Custom Connectors](#creating-custom-connectors)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

The Neural DSL integrations module provides base classes for building custom connectors to ML platforms. This allows you to execute Neural DSL models on your preferred infrastructure.

### Key Features

- **Abstract Base Classes**: Well-defined interface for platform connectors
- **Resource Configuration**: Flexible resource specification
- **Job Management**: Standard job lifecycle management
- **Extensible**: Implement only what you need

## Installation

```bash
pip install neural-dsl
```

No additional dependencies required for the base classes.

## Creating Custom Connectors

### Basic Connector

Create a custom connector by extending `BaseConnector` and implementing the required abstract methods:

```python
from neural.integrations import BaseConnector, JobStatus, JobResult, ResourceConfig

class MyPlatformConnector(BaseConnector):
    """Custom connector for MyPlatform."""
    
    def __init__(self, credentials=None):
        super().__init__(credentials)
        self.api_key = self.credentials.get('api_key', '')
        self.endpoint = self.credentials.get('endpoint', '')
    
    def authenticate(self) -> bool:
        """Authenticate with the platform."""
        if not self.api_key or not self.endpoint:
            return False
        
        # Add your authentication logic here
        # Example: call API to validate credentials
        try:
            # response = requests.post(f"{self.endpoint}/auth", 
            #                         headers={"X-API-Key": self.api_key})
            # if response.status_code == 200:
            self.authenticated = True
            return True
        except Exception as e:
            return False
    
    def submit_job(self, code, resource_config=None, environment=None,
                   dependencies=None, job_name=None) -> str:
        """Submit a job to the platform."""
        if not self.authenticated:
            raise Exception("Not authenticated")
        
        # Add your job submission logic here
        # Example: POST to platform API
        job_id = "unique-job-id"
        return job_id
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        # Add your status check logic here
        # Example: GET from platform API
        return JobStatus.RUNNING
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a job."""
        # Add your result retrieval logic here
        return JobResult(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            output="Job completed successfully",
            duration_seconds=10.5
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        # Add your cancellation logic here
        return True
    
    def list_jobs(self, limit=10, status_filter=None):
        """List recent jobs."""
        # Add your job listing logic here
        return [
            {'job_id': 'job-123', 'status': 'succeeded', 'name': 'example-job'}
        ]
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a job."""
        # Add your log retrieval logic here
        return "Job log output..."
```

### Using Your Connector

```python
# Initialize with credentials
connector = MyPlatformConnector(credentials={
    'api_key': 'your-api-key',
    'endpoint': 'https://api.myplatform.com'
})

# Authenticate
if connector.authenticate():
    print("Authentication successful")
    
    # Submit a job
    job_id = connector.submit_job(
        code="print('Hello from Neural DSL')",
        resource_config=ResourceConfig(
            instance_type='standard',
            gpu_enabled=False
        ),
        job_name='test-job'
    )
    
    # Check status
    status = connector.get_job_status(job_id)
    print(f"Job status: {status.value}")
    
    # Get result
    if status == JobStatus.SUCCEEDED:
        result = connector.get_job_result(job_id)
        print(f"Output: {result.output}")
```

## API Reference

### BaseConnector

Abstract base class for all platform connectors.

#### Required Methods

- **authenticate() -> bool**: Authenticate with the platform
- **submit_job(code, resource_config, environment, dependencies, job_name) -> str**: Submit a job
- **get_job_status(job_id) -> JobStatus**: Get job status
- **get_job_result(job_id) -> JobResult**: Get job result
- **cancel_job(job_id) -> bool**: Cancel a job
- **list_jobs(limit, status_filter) -> List[Dict]**: List jobs
- **get_logs(job_id) -> str**: Get job logs

#### Optional Methods

- **deploy_model(model_path, endpoint_name, resource_config) -> str**: Deploy a model
- **delete_endpoint(endpoint_name) -> bool**: Delete an endpoint
- **upload_file(local_path, remote_path) -> bool**: Upload a file
- **download_file(remote_path, local_path) -> bool**: Download a file
- **get_resource_usage() -> Dict**: Get resource usage statistics

### ResourceConfig

Dataclass for resource configuration.

```python
ResourceConfig(
    instance_type: str,
    gpu_enabled: bool = False,
    gpu_count: int = 0,
    memory_gb: Optional[int] = None,
    cpu_count: Optional[int] = None,
    disk_size_gb: Optional[int] = None,
    max_runtime_hours: Optional[int] = None,
    auto_shutdown: bool = True,
    custom_params: Dict[str, Any] = {}
)
```

### JobStatus

Enumeration for job statuses.

- `PENDING`: Job is queued
- `RUNNING`: Job is executing
- `SUCCEEDED`: Job completed successfully
- `FAILED`: Job failed
- `CANCELLED`: Job was cancelled
- `UNKNOWN`: Status cannot be determined

### JobResult

Dataclass for job results.

```python
JobResult(
    job_id: str,
    status: JobStatus,
    output: Optional[str] = None,
    error: Optional[str] = None,
    metrics: Dict[str, Any] = {},
    artifacts: List[str] = [],
    logs_url: Optional[str] = None,
    duration_seconds: Optional[float] = None
)
```

## Examples

### Example 1: Simple Connector

```python
from neural.integrations import BaseConnector, JobStatus, JobResult

class SimpleConnector(BaseConnector):
    def authenticate(self):
        return True
    
    def submit_job(self, code, **kwargs):
        return "job-123"
    
    def get_job_status(self, job_id):
        return JobStatus.SUCCEEDED
    
    def get_job_result(self, job_id):
        return JobResult(job_id=job_id, status=JobStatus.SUCCEEDED)
    
    def cancel_job(self, job_id):
        return True
    
    def list_jobs(self, limit=10, status_filter=None):
        return []
    
    def get_logs(self, job_id):
        return ""
```

### Example 2: Connector with Model Deployment

```python
class AdvancedConnector(BaseConnector):
    # ... implement required methods ...
    
    def deploy_model(self, model_path, endpoint_name, resource_config=None):
        """Deploy a model as an endpoint."""
        # Your deployment logic
        endpoint_url = f"https://api.platform.com/endpoints/{endpoint_name}"
        return endpoint_url
    
    def delete_endpoint(self, endpoint_name):
        """Delete a deployed endpoint."""
        # Your deletion logic
        return True
```

### Example 3: Polling for Job Completion

```python
import time

def wait_for_job(connector, job_id, poll_interval=30, timeout=3600):
    """Wait for a job to complete."""
    start_time = time.time()
    
    while True:
        status = connector.get_job_status(job_id)
        
        if status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return connector.get_job_result(job_id)
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job {job_id} timed out")
        
        time.sleep(poll_interval)

# Usage
result = wait_for_job(connector, job_id)
print(f"Job completed with status: {result.status}")
```

## Best Practices

### 1. Error Handling

Use the provided exception classes:

```python
from neural.exceptions import CloudConnectionError, CloudExecutionError, CloudException

def authenticate(self):
    try:
        # Authentication logic
        self.authenticated = True
        return True
    except Exception as e:
        raise CloudConnectionError(f"Authentication failed: {e}")
```

### 2. Credential Management

Use the utility functions for secure credential storage:

```python
from neural.integrations.utils import save_credentials_to_file, load_credentials_from_file

# Save credentials securely (with 600 permissions)
save_credentials_to_file({'api_key': 'secret'})

# Load credentials
credentials = load_credentials_from_file()
```

### 3. Resource Configuration

Provide sensible defaults and validate configurations:

```python
def submit_job(self, code, resource_config=None, **kwargs):
    if resource_config is None:
        resource_config = ResourceConfig(
            instance_type='standard',
            gpu_enabled=False
        )
    
    # Validate
    if resource_config.gpu_count > 8:
        raise ValueError("Maximum 8 GPUs supported")
    
    # Submit job
    ...
```

### 4. Logging

Use Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

def submit_job(self, code, **kwargs):
    logger.info(f"Submitting job to {self.endpoint}")
    # Job submission logic
    logger.info(f"Job submitted with ID: {job_id}")
```

### 5. Retry Logic

Implement retry logic for transient failures:

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
        return wrapper
    return decorator

class RobustConnector(BaseConnector):
    @retry(max_attempts=3)
    def get_job_status(self, job_id):
        # Status check logic that might fail transiently
        ...
```

### 6. Testing

Write tests for your connector:

```python
import pytest
from neural.integrations import JobStatus

def test_connector_authentication():
    connector = MyPlatformConnector(credentials={
        'api_key': 'test-key',
        'endpoint': 'https://test.api.com'
    })
    assert connector.authenticate() is True

def test_job_submission():
    connector = MyPlatformConnector(credentials={...})
    connector.authenticate()
    
    job_id = connector.submit_job("print('test')")
    assert job_id is not None
    assert isinstance(job_id, str)

def test_job_status():
    connector = MyPlatformConnector(credentials={...})
    connector.authenticate()
    
    status = connector.get_job_status('job-123')
    assert isinstance(status, JobStatus)
```

## Further Reading

- See `neural/integrations/examples.py` for complete examples
- See `neural/integrations/README.md` for API documentation
- See `neural/integrations/base.py` for implementation details
