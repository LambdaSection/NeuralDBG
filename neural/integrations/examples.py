"""
Example usage of Neural DSL platform integrations.

Demonstrates how to implement custom platform connectors using the base classes.
"""

from neural.integrations import BaseConnector, JobResult, JobStatus, ResourceConfig


class CustomConnector(BaseConnector):
    """
    Example custom connector implementation.
    
    This demonstrates how to create a connector for your own ML platform
    by subclassing BaseConnector and implementing the required methods.
    """
    
    def __init__(self, credentials=None):
        super().__init__(credentials)
        self.api_key = self.credentials.get('api_key', '')
        self.endpoint = self.credentials.get('endpoint', '')
    
    def authenticate(self) -> bool:
        """Authenticate with the platform."""
        if not self.api_key or not self.endpoint:
            return False
        self.authenticated = True
        return True
    
    def submit_job(self, code, resource_config=None, environment=None,
                   dependencies=None, job_name=None) -> str:
        """Submit a job to the platform."""
        if not self.authenticated:
            raise Exception("Not authenticated")
        return "job-123"
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job."""
        return JobStatus.SUCCEEDED
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a job."""
        return JobResult(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            output="Job completed successfully",
            duration_seconds=10.5
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        return True
    
    def list_jobs(self, limit=10, status_filter=None):
        """List recent jobs."""
        return [
            {'job_id': 'job-123', 'status': 'succeeded', 'name': 'example-job'}
        ]
    
    def get_logs(self, job_id: str) -> str:
        """Get logs for a job."""
        return "Job log output..."


def example_custom_connector():
    """Example using a custom connector."""
    print("=== Custom Connector Example ===\n")
    
    connector = CustomConnector(credentials={
        'api_key': 'your-api-key',
        'endpoint': 'https://api.example.com'
    })
    
    if connector.authenticate():
        print("Authentication successful")
        
        job_id = connector.submit_job(
            code="print('Hello from Neural DSL')",
            resource_config=ResourceConfig(
                instance_type='standard',
                gpu_enabled=False
            ),
            job_name='example-job'
        )
        
        print(f"Job submitted: {job_id}")
        
        status = connector.get_job_status(job_id)
        print(f"Job status: {status.value}")
        
        if status == JobStatus.SUCCEEDED:
            result = connector.get_job_result(job_id)
            print(f"Duration: {result.duration_seconds}s")
            print(f"Output: {result.output}")


def example_resource_config():
    """Example of creating resource configurations."""
    print("\n=== Resource Configuration Examples ===\n")
    
    basic_config = ResourceConfig(instance_type='t2.medium')
    print(f"Basic: {basic_config.instance_type}, GPU: {basic_config.gpu_enabled}")
    
    gpu_config = ResourceConfig(
        instance_type='p3.2xlarge',
        gpu_enabled=True,
        gpu_count=1,
        memory_gb=61,
        cpu_count=8,
        max_runtime_hours=4
    )
    print(f"GPU: {gpu_config.instance_type}, GPUs: {gpu_config.gpu_count}")
    
    custom_config = ResourceConfig(
        instance_type='custom',
        custom_params={'priority': 'high', 'preemptible': False}
    )
    print(f"Custom: {custom_config.custom_params}")


def example_job_status():
    """Example of job status handling."""
    print("\n=== Job Status Examples ===\n")
    
    statuses = [
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED
    ]
    
    for status in statuses:
        print(f"Status: {status.value}")


if __name__ == '__main__':
    print("Neural DSL Platform Integrations Examples")
    print("=" * 50)
    print("\nThese examples show how to implement custom connectors")
    print("using the Neural DSL integration base classes.\n")
    
    try:
        example_custom_connector()
        example_resource_config()
        example_job_status()
    except Exception as e:
        print(f"Example error: {e}")
