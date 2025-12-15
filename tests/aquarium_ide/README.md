# Aquarium IDE Integration Tests

Comprehensive integration test suite for Aquarium IDE covering all major components and workflows.

## Test Structure

### `test_integration.py`
Core integration tests covering:
- **Welcome Screen Functionality**: Initialization, tab navigation, close actions, tutorial launch
- **Quick Start Templates**: Template loading, validation, difficulty levels, categories
- **Example Gallery API**: List endpoints, filtering, searching, loading examples
- **DSL Compilation**: Simple networks, CNNs, LSTMs, multi-backend compilation
- **Health Check Endpoints**: Root endpoint, health checks, response time validation
- **Backend API Endpoints**: Parse, shape propagation, code generation, compilation pipeline
- **Template Integration**: Compilation of all templates to verify correctness
- **Production Stability**: Error handling, graceful degradation, fallback mechanisms

### `test_backend_api.py`
Backend API endpoint tests using FastAPI TestClient:
- **Root Endpoints**: Service info, health checks
- **Parse Endpoint**: Valid/invalid DSL parsing, error handling
- **Shape Propagation**: Successful propagation, missing input handling
- **Code Generation**: TensorFlow and PyTorch code generation
- **Compilation Pipeline**: Full compilation workflow
- **Examples API**: List examples, load examples, handle missing files
- **Job Management**: List jobs, get status, stop jobs
- **Error Handling**: Invalid JSON, missing fields, proper error responses
- **CORS Configuration**: Verify CORS headers are present

### `test_welcome_components.py`
Frontend component integration tests:
- **Welcome Screen Component**: Props validation, tabs, callbacks, overlay styling
- **Quick Start Templates**: Structure, categories, difficulty colors, load callbacks
- **Example Gallery**: Structure, search, filtering, tags, loading states, error handling
- **Documentation Browser**: Structure, categories, search functionality
- **Video Tutorials**: Structure, categories, duration formatting
- **Component Integration**: Template/example to editor flow, state management
- **API Integration**: Mock API calls, error handling
- **Validation**: Template and example DSL syntax validation

### `test_e2e_workflows.py`
End-to-end workflow tests:
- **Welcome to Compilation**: Template selection, example loading, custom code workflows
- **Multi-Backend Workflow**: Compile same DSL to multiple backends
- **Shape Propagation**: Compilation with shape validation
- **Error Handling**: Invalid DSL, missing files, compilation error recovery
- **API Integration**: Full parse -> shape -> generate workflow
- **Example Gallery**: Browse, filter, load, compile workflow
- **Template Customization**: Load, modify, compile workflow
- **User Journeys**: New user and experienced user scenarios
- **Production Readiness**: High volume, concurrent users, error recovery
- **Performance**: Large model compilation

## Running Tests

### Run all Aquarium IDE tests:
```bash
pytest tests/aquarium_ide/ -v
```

### Run specific test file:
```bash
pytest tests/aquarium_ide/test_integration.py -v
pytest tests/aquarium_ide/test_backend_api.py -v
pytest tests/aquarium_ide/test_welcome_components.py -v
pytest tests/aquarium_ide/test_e2e_workflows.py -v
```

### Run specific test class:
```bash
pytest tests/aquarium_ide/test_integration.py::TestWelcomeScreen -v
pytest tests/aquarium_ide/test_backend_api.py::TestParseEndpoint -v
```

### Run specific test:
```bash
pytest tests/aquarium_ide/test_integration.py::TestWelcomeScreen::test_welcome_screen_initialization -v
```

### Run with coverage:
```bash
pytest tests/aquarium_ide/ --cov=neural.aquarium --cov-report=html
```

### Run with markers:
```bash
pytest tests/aquarium_ide/ -m integration -v
pytest tests/aquarium_ide/ -m "not slow" -v
```

## Test Coverage

The test suite covers:

### ✅ Welcome Screen
- Initialization and configuration
- Tab navigation (quickstart, examples, docs, videos)
- Close and skip actions
- Tutorial launch functionality

### ✅ Quick Start Templates
- All template structures and metadata
- Template loading and callbacks
- Difficulty level validation
- Category classification
- DSL syntax validation
- Compilation verification

### ✅ Example Gallery
- API endpoint structure (`/api/examples/list`)
- Example loading (`/api/examples/load`)
- Filtering by category and tags
- Search functionality
- Error handling and fallback to built-in examples
- Integration with compilation pipeline

### ✅ DSL Compilation
- Simple neural networks
- CNNs (Convolutional Neural Networks)
- LSTMs (Long Short-Term Memory)
- Invalid syntax error handling
- Multi-backend support (TensorFlow, PyTorch)
- Shape propagation validation

### ✅ Backend API Endpoints
- Root endpoint (`/`)
- Health check (`/health`)
- Parse DSL (`/api/parse`)
- Shape propagation (`/api/shape-propagation`)
- Code generation (`/api/generate-code`)
- Full compilation (`/api/compile`)
- Example management (`/api/examples/*`)
- Job management (`/api/jobs/*`)
- Documentation serving (`/api/docs/*`)

### ✅ Production Stability
- Error handling for invalid input
- Graceful degradation on API failures
- Fallback mechanisms (built-in examples)
- Missing file handling
- Invalid extension validation
- Response time validation
- Concurrent request handling
- Error recovery workflows

### ✅ End-to-End Workflows
- Welcome screen to compilation
- Template selection and customization
- Example loading and modification
- Multi-backend compilation
- Shape validation integration
- New user onboarding
- Experienced user shortcuts
- High-volume compilation
- Performance testing

## Requirements

The tests require the following dependencies:
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`
- `fastapi` (for TestClient)
- Core Neural DSL dependencies (parser, code generator, shape propagator)

Install with:
```bash
pip install -r requirements-dev.txt
```

## Test Fixtures

Common fixtures available in all tests:

- `sample_templates`: Sample quick start templates with metadata
- `example_files`: Temporary directory with example .neural files
- `test_client`: FastAPI TestClient for API testing
- `sample_dsl_code`: Common DSL code snippets
- `backend_url`: Default backend server URL

Additional fixtures from `tests/conftest.py`:
- `parser`, `transformer`: DSL parser and transformer
- `tmp_dir`: Temporary directory for file operations
- `sample_dsl_*`: Various DSL code samples

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Aquarium IDE tests
  run: |
    pytest tests/aquarium_ide/ -v --tb=short
```

## Test Categories

Tests are organized by category:

1. **Unit Tests**: Individual component behavior
2. **Integration Tests**: Component interaction
3. **API Tests**: HTTP endpoint testing
4. **E2E Tests**: Complete user workflows
5. **Performance Tests**: Large model handling

## Error Scenarios Covered

- Invalid DSL syntax
- Missing example files
- API failures and timeouts
- Invalid file extensions
- Missing required fields
- Compilation errors
- Network errors
- Concurrent access issues

## Production Readiness Checks

- ✅ Health check endpoints respond correctly
- ✅ Examples load with proper error handling
- ✅ Templates compile successfully
- ✅ Multi-backend support works
- ✅ Shape propagation validates correctly
- ✅ API returns proper error codes
- ✅ Graceful degradation on failures
- ✅ Response times are acceptable
- ✅ Concurrent requests handled
- ✅ Error recovery mechanisms work

## Contributing

When adding new features to Aquarium IDE, please add corresponding tests:

1. Add unit tests for new components
2. Add integration tests for component interactions
3. Add API tests for new endpoints
4. Add E2E tests for new workflows
5. Update this README with new test coverage

## Notes

- Tests use mocking for external dependencies where appropriate
- Temporary directories are cleaned up automatically
- FastAPI TestClient provides isolated test environment
- All tests are independent and can run in any order
- Tests verify both success and failure paths
