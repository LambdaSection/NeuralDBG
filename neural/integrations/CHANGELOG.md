# Neural DSL Integrations - Changelog

## [0.4.0] - Simplified Architecture

### Changed

#### Simplified Module Structure
- **Removed cloud platform connectors**: Databricks, SageMaker, Vertex AI, Azure ML, Paperspace, Run:AI
- **Removed PlatformManager**: Unified manager no longer needed
- **Focus on base classes**: Module now provides only essential base integration classes

#### Rationale
- Cloud platform connectors were not actively maintained with tests
- External dependencies (boto3, google-cloud, azure) removed from core
- Users can implement custom connectors by extending BaseConnector
- Reduces maintenance burden and dependency complexity

### Retained Components

#### Core Infrastructure
- **BaseConnector**: Abstract base class for building custom platform connectors
  - Authentication interface
  - Job submission and management
  - Optional model deployment
  - Optional resource management
  - Optional file operations

- **ResourceConfig**: Dataclass for resource configuration
  - Instance type selection
  - GPU configuration
  - Memory and CPU settings
  - Auto-shutdown and runtime limits
  - Custom parameters

- **JobStatus**: Enumeration for job states
  - PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED, UNKNOWN

- **JobResult**: Dataclass for job results
  - Status, output, and error information
  - Performance metrics
  - Artifacts and logs
  - Duration tracking

#### Utilities (Simplified)
- **load_credentials_from_file**: Load credentials from JSON
- **save_credentials_to_file**: Save credentials securely
- **format_job_output**: Format job results for display

#### Documentation
- **README.md**: Updated guide showing how to implement custom connectors
- **examples.py**: Updated examples demonstrating custom connector implementation

#### Testing
- **test_integrations.py**: Updated tests for base classes only
  - Connector implementation patterns
  - Resource configuration
  - Job status enumeration
  - Utility function tests

### Migration Guide

If you were using the removed platform connectors:

1. **Custom Implementation**: Implement your own connector by extending `BaseConnector`
2. **See Examples**: Check `examples.py` for implementation patterns
3. **Install Dependencies**: Install platform-specific dependencies directly in your project

Example migration:

```python
# Old code (removed):
from neural.integrations import DatabricksConnector, PlatformManager

# New approach:
from neural.integrations import BaseConnector, JobStatus, JobResult

class MyDatabricksConnector(BaseConnector):
    def authenticate(self):
        # Your auth logic
        pass
    # Implement other required methods...
```

### Benefits
- **Cleaner codebase**: Removed unmaintained code
- **Fewer dependencies**: No cloud provider SDKs in core
- **More flexible**: Users implement exactly what they need
- **Better maintained**: Focus on stable base classes
- **Easier to test**: Simple, focused API

## [0.3.0] - Initial Release

### Added
- Initial implementation with 6 cloud platform connectors
- PlatformManager for unified multi-platform management
- Comprehensive utilities and error handling
- Full documentation and examples
