"""
Integration tests for Aquarium IDE backend API endpoints.

Tests actual FastAPI endpoints using test client.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from neural.aquarium.backend.server import create_app
    from fastapi.testclient import TestClient
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_dsl_code():
    """Sample DSL code for testing."""
    return """
    network TestModel {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(64, activation="relu")
            Dense(10, activation="softmax")
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """


class TestRootEndpoints:
    """Test root and health check endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test GET / returns service information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check_endpoint(self, test_client):
        """Test GET /health returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestParseEndpoint:
    """Test DSL parsing endpoint."""
    
    def test_parse_valid_dsl(self, test_client, sample_dsl_code):
        """Test parsing valid DSL code."""
        response = test_client.post(
            "/api/parse",
            json={
                "dsl_code": sample_dsl_code,
                "parser_type": "network"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_data"] is not None
        assert "layers" in data["model_data"]
    
    def test_parse_invalid_dsl(self, test_client):
        """Test parsing invalid DSL code."""
        response = test_client.post(
            "/api/parse",
            json={
                "dsl_code": "invalid dsl code",
                "parser_type": "network"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None
    
    def test_parse_missing_dsl_code(self, test_client):
        """Test parsing with missing DSL code."""
        response = test_client.post(
            "/api/parse",
            json={"parser_type": "network"}
        )
        
        assert response.status_code == 422


class TestShapePropagationEndpoint:
    """Test shape propagation endpoint."""
    
    def test_shape_propagation_success(self, test_client):
        """Test successful shape propagation."""
        model_data = {
            "input": {"shape": [28, 28, 1]},
            "layers": [
                {"type": "Flatten", "params": None},
                {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
                {"type": "Dense", "params": {"units": 10, "activation": "softmax"}}
            ]
        }
        
        response = test_client.post(
            "/api/shape-propagation",
            json={
                "model_data": model_data,
                "framework": "tensorflow"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "shape_history" in data
    
    def test_shape_propagation_missing_input(self, test_client):
        """Test shape propagation with missing input shape."""
        model_data = {
            "layers": [
                {"type": "Dense", "params": {"units": 64}}
            ]
        }
        
        response = test_client.post(
            "/api/shape-propagation",
            json={
                "model_data": model_data,
                "framework": "tensorflow"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestCodeGenerationEndpoint:
    """Test code generation endpoint."""
    
    def test_generate_tensorflow_code(self, test_client):
        """Test TensorFlow code generation."""
        model_data = {
            "input": {"shape": [28, 28, 1]},
            "layers": [
                {"type": "Flatten", "params": None},
                {"type": "Dense", "params": {"units": 10}}
            ]
        }
        
        response = test_client.post(
            "/api/generate-code",
            json={
                "model_data": model_data,
                "backend": "tensorflow"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["code"] is not None
        assert "tensorflow" in data["code"].lower() or "keras" in data["code"].lower()
    
    def test_generate_pytorch_code(self, test_client):
        """Test PyTorch code generation."""
        model_data = {
            "input": {"shape": [28, 28, 1]},
            "layers": [
                {"type": "Flatten", "params": None},
                {"type": "Dense", "params": {"units": 10}}
            ]
        }
        
        response = test_client.post(
            "/api/generate-code",
            json={
                "model_data": model_data,
                "backend": "pytorch"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["code"] is not None
        assert "torch" in data["code"].lower()


class TestCompileEndpoint:
    """Test complete compilation pipeline endpoint."""
    
    def test_compile_full_pipeline(self, test_client, sample_dsl_code):
        """Test complete compilation pipeline."""
        response = test_client.post(
            "/api/compile",
            json={
                "dsl_code": sample_dsl_code,
                "backend": "tensorflow",
                "parser_type": "network"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["code"] is not None
        assert data["model_data"] is not None
        assert "shape_history" in data
    
    def test_compile_with_different_backends(self, test_client, sample_dsl_code):
        """Test compilation with different backends."""
        backends = ["tensorflow", "pytorch"]
        
        for backend in backends:
            response = test_client.post(
                "/api/compile",
                json={
                    "dsl_code": sample_dsl_code,
                    "backend": backend,
                    "parser_type": "network"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["code"] is not None


class TestExamplesEndpoints:
    """Test example loading endpoints."""
    
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_list_examples_success(self, mock_exists, mock_glob, test_client):
        """Test listing examples."""
        mock_exists.return_value = True
        
        mock_file = MagicMock()
        mock_file.stem = "mnist_cnn"
        mock_file.suffix = ".neural"
        mock_file.relative_to.return_value = Path("examples/mnist_cnn.neural")
        
        mock_glob.return_value = [mock_file]
        
        with patch('builtins.open', MagicMock()):
            with patch('pathlib.Path.read_text', return_value="network Test {}"):
                response = test_client.get("/api/examples/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert "count" in data
    
    @patch('pathlib.Path.exists')
    def test_list_examples_empty_directory(self, mock_exists, test_client):
        """Test listing examples with empty directory."""
        mock_exists.return_value = False
        
        response = test_client.get("/api/examples/list")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
    
    def test_load_example_missing_path(self, test_client):
        """Test loading example without path parameter."""
        response = test_client.get("/api/examples/load")
        
        assert response.status_code == 422
    
    @patch('pathlib.Path.exists')
    def test_load_example_not_found(self, mock_exists, test_client):
        """Test loading non-existent example."""
        mock_exists.return_value = False
        
        response = test_client.get("/api/examples/load?path=nonexistent.neural")
        
        assert response.status_code == 404


class TestJobManagementEndpoints:
    """Test training job management endpoints."""
    
    def test_list_jobs_empty(self, test_client):
        """Test listing jobs when none exist."""
        response = test_client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
    
    def test_get_job_status_not_found(self, test_client):
        """Test getting status of non-existent job."""
        response = test_client.get("/api/jobs/nonexistent-job-id/status")
        
        assert response.status_code == 404
    
    def test_stop_job_not_found(self, test_client):
        """Test stopping non-existent job."""
        response = test_client.post("/api/jobs/nonexistent-job-id/stop")
        
        assert response.status_code == 404


class TestDocumentationEndpoint:
    """Test documentation serving endpoint."""
    
    @patch('pathlib.Path.exists')
    def test_get_documentation_not_found(self, mock_exists, test_client):
        """Test getting non-existent documentation."""
        mock_exists.return_value = False
        
        response = test_client.get("/api/docs/nonexistent.md")
        
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_invalid_json_body(self, test_client):
        """Test handling of invalid JSON in request body."""
        response = test_client.post(
            "/api/parse",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        response = test_client.post(
            "/api/parse",
            json={}
        )
        
        assert response.status_code == 422


class TestCORSHeaders:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present."""
        response = test_client.options("/api/parse")
        
        assert response.status_code in [200, 405]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
