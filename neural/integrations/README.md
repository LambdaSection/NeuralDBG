# Neural DSL Platform Integrations

Base classes for building ML platform integration connectors with unified API for authentication, remote execution, and resource management.

## Overview

The integrations module provides abstract base classes that you can extend to create custom connectors for any ML platform. This allows Neural DSL to work with your preferred cloud or on-premises infrastructure.

## Quick Start

### Creating a Custom Connector

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
        self.authenticated = True
        return True
    
    def submit_job(self, code, resource_config=None, environment=None,
                   dependencies=None, job_name=None) -> str:
        """Submit a job to the platform."""
        # Add your job submission logic here
        return "job-123"
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        # Add your status check logic here
        return JobStatus.RUNNING
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a job."""
        # Add your result retrieval logic here
        return JobResult(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            output="Job completed",
            duration_seconds=10.5
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        # Add your cancellation logic here
        return True
    
    def list_jobs(self, limit=10, status_filter=None):
        """List recent jobs."""
        # Add your job listing logic here
        return []
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a job."""
        # Add your log retrieval logic here
        return "Job logs..."

# Use your connector
connector = MyPlatformConnector(credentials={
    'api_key': 'your-api-key',
    'endpoint': 'https://api.myplatform.com'
})

connector.authenticate()

job_id = connector.submit_job(
    code="print('Hello from Neural DSL!')",
    resource_config=ResourceConfig(
        instance_type='standard',
        gpu_enabled=False
    )
)
```

## Resource Configuration

```python
from neural.integrations import ResourceConfig

# CPU-only configuration
cpu_config = ResourceConfig(
    instance_type='standard',
    gpu_enabled=False,
    memory_gb=16,
    cpu_count=4,
    disk_size_gb=100
)

# GPU configuration
gpu_config = ResourceConfig(
    instance_type='gpu-standard',
    gpu_enabled=True,
    gpu_count=2,
    memory_gb=64,
    max_runtime_hours=24,
    auto_shutdown=True
)

# Custom parameters
custom_config = ResourceConfig(
    instance_type='custom',
    custom_params={
        'priority': 'high',
        'preemptible': False,
        'zone': 'us-east-1a'
    }
)
```

## Job Management

### Submitting Jobs

```python
job_id = connector.submit_job(
    code=dsl_code,
    resource_config=gpu_config,
    environment={'PYTHONPATH': '/app'},
    dependencies=['tensorflow>=2.12', 'numpy>=1.23'],
    job_name='my-training-job'
)
```

### Monitoring Jobs

```python
# Check status
status = connector.get_job_status(job_id)
print(f"Status: {status.value}")

# Get full result
result = connector.get_job_result(job_id)
print(f"Status: {result.status}")
print(f"Output: {result.output}")
print(f"Metrics: {result.metrics}")
print(f"Duration: {result.duration_seconds}s")

# Get logs
logs = connector.get_logs(job_id)
print(logs)
```

### Listing Jobs

```python
from neural.integrations import JobStatus

# List all jobs
jobs = connector.list_jobs(limit=20)

# Filter by status
running_jobs = connector.list_jobs(limit=10, status_filter=JobStatus.RUNNING)
```

### Canceling Jobs

```python
success = connector.cancel_job(job_id)
if success:
    print("Job cancelled successfully")
```

## Optional Methods

The BaseConnector provides optional methods you can implement for additional functionality:

### Model Deployment

```python
def deploy_model(self, model_path, endpoint_name, resource_config=None) -> str:
    """Deploy a model as an endpoint."""
    # Your deployment logic here
    return "https://endpoint-url"

def delete_endpoint(self, endpoint_name: str) -> bool:
    """Delete a deployed endpoint."""
    # Your deletion logic here
    return True
```

### File Operations

```python
def upload_file(self, local_path: str, remote_path: str) -> bool:
    """Upload a file to the platform."""
    # Your upload logic here
    return True

def download_file(self, remote_path: str, local_path: str) -> bool:
    """Download a file from the platform."""
    # Your download logic here
    return True
```

### Resource Monitoring

```python
def get_resource_usage(self):
    """Get resource usage statistics."""
    # Your monitoring logic here
    return {
        'total_gpus': 8,
        'used_gpus': 3,
        'available_gpus': 5
    }
```

## Error Handling

```python
from neural.exceptions import (
    CloudConnectionError,
    CloudExecutionError,
    CloudException
)

try:
    connector.authenticate()
except CloudConnectionError as e:
    print(f"Authentication failed: {e}")

try:
    job_id = connector.submit_job(code=dsl_code)
except CloudExecutionError as e:
    print(f"Job submission failed: {e}")
```

## API Reference

### BaseConnector

Abstract base class for all platform connectors.

**Required Methods:**
- `authenticate()`: Authenticate with the platform
- `submit_job()`: Submit a job for execution
- `get_job_status()`: Get job status
- `get_job_result()`: Get job result
- `cancel_job()`: Cancel a job
- `list_jobs()`: List recent jobs
- `get_logs()`: Get job logs

**Optional Methods:**
- `deploy_model()`: Deploy a model
- `delete_endpoint()`: Delete an endpoint
- `upload_file()`: Upload a file
- `download_file()`: Download a file
- `get_resource_usage()`: Get resource usage

### ResourceConfig

Configuration for compute resources.

**Attributes:**
- `instance_type`: Instance type (platform-specific)
- `gpu_enabled`: Whether GPU is enabled
- `gpu_count`: Number of GPUs
- `memory_gb`: Memory in GB
- `cpu_count`: Number of CPUs
- `disk_size_gb`: Disk size in GB
- `max_runtime_hours`: Maximum runtime in hours
- `auto_shutdown`: Enable auto-shutdown
- `custom_params`: Dictionary of custom parameters

### JobStatus

Job status enumeration.

**Values:**
- `PENDING`: Job is pending
- `RUNNING`: Job is running
- `SUCCEEDED`: Job completed successfully
- `FAILED`: Job failed
- `CANCELLED`: Job was cancelled
- `UNKNOWN`: Status unknown

### JobResult

Result from a job execution.

**Attributes:**
- `job_id`: Job identifier
- `status`: Job status
- `output`: Job output
- `error`: Error message (if failed)
- `metrics`: Dictionary of performance metrics
- `artifacts`: List of output artifacts
- `logs_url`: URL to view logs
- `duration_seconds`: Execution duration in seconds

## Utilities

### Credential Management

```python
from neural.integrations.utils import (
    load_credentials_from_file,
    save_credentials_to_file
)

# Save credentials
save_credentials_to_file(
    credentials={'api_key': 'secret'},
    filepath='~/.neural/credentials.json'
)

# Load credentials
credentials = load_credentials_from_file(filepath='~/.neural/credentials.json')
```

### Output Formatting

```python
from neural.integrations.utils import format_job_output

result = connector.get_job_result(job_id)
formatted = format_job_output(result)
print(formatted)
```

## Examples

See `examples.py` for complete examples of implementing custom connectors.

## Integration Patterns

### Polling Pattern

```python
import time

def wait_for_completion(connector, job_id, poll_interval=30, timeout=3600):
    """Wait for a job to complete."""
    start_time = time.time()
    
    while True:
        status = connector.get_job_status(job_id)
        
        if status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return connector.get_job_result(job_id)
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
        
        time.sleep(poll_interval)
```

### Batch Submission Pattern

```python
def batch_submit_jobs(connector, jobs):
    """Submit multiple jobs."""
    job_ids = []
    
    for job_config in jobs:
        try:
            job_id = connector.submit_job(**job_config)
            job_ids.append(job_id)
        except Exception as e:
            print(f"Failed to submit job: {e}")
            job_ids.append(None)
    
    return job_ids
```

## Contributing

To add support for a new platform:

1. Create a new connector class that inherits from `BaseConnector`
2. Implement all required abstract methods
3. Add platform-specific authentication logic
4. Add tests for your connector
5. Update documentation with usage examples

## Best Practices

- Always validate credentials before attempting operations
- Implement proper error handling for network failures
- Use platform-specific resource configurations
- Cache authentication tokens when possible
- Implement retry logic for transient failures
- Log important operations for debugging
