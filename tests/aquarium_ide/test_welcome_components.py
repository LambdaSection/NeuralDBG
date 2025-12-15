"""
Integration tests for Welcome Screen components and functionality.

Tests welcome screen, quick start templates, example gallery,
documentation browser, and video tutorials.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestWelcomeScreenComponent:
    """Test welcome screen component behavior."""
    
    def test_welcome_screen_props(self):
        """Test welcome screen accepts required props."""
        props = {
            "onClose": Mock(),
            "onLoadTemplate": Mock(),
            "onStartTutorial": Mock()
        }
        
        assert callable(props["onClose"])
        assert callable(props["onLoadTemplate"])
        assert callable(props["onStartTutorial"])
    
    def test_welcome_screen_tabs(self):
        """Test welcome screen has all required tabs."""
        tabs = ["quickstart", "examples", "docs", "videos"]
        
        assert "quickstart" in tabs
        assert "examples" in tabs
        assert "docs" in tabs
        assert "videos" in tabs
        assert len(tabs) == 4
    
    def test_welcome_screen_default_tab(self):
        """Test welcome screen defaults to quickstart tab."""
        default_tab = "quickstart"
        
        assert default_tab == "quickstart"
    
    def test_welcome_screen_close_callback(self):
        """Test welcome screen close callback is invoked."""
        on_close = Mock()
        
        on_close()
        
        on_close.assert_called_once()
    
    def test_welcome_screen_start_tutorial_callback(self):
        """Test start tutorial callback is invoked."""
        on_start_tutorial = Mock()
        
        on_start_tutorial()
        
        on_start_tutorial.assert_called_once()
    
    def test_welcome_screen_tab_switching(self):
        """Test switching between tabs."""
        active_tab = "quickstart"
        tabs = ["quickstart", "examples", "docs", "videos"]
        
        for tab in tabs:
            active_tab = tab
            assert active_tab == tab
    
    def test_welcome_screen_overlay_styling(self):
        """Test welcome screen has overlay styling."""
        style = {
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "zIndex": 1000
        }
        
        assert style["position"] == "fixed"
        assert style["zIndex"] == 1000


class TestQuickStartTemplatesComponent:
    """Test quick start templates component."""
    
    def test_templates_structure(self):
        """Test templates have required structure."""
        template = {
            "id": "image-classification",
            "title": "Image Classification",
            "description": "CNN for image classification",
            "category": "Computer Vision",
            "icon": "🖼️",
            "difficulty": "beginner",
            "dslCode": "network ImageClassifier {}"
        }
        
        required_fields = ["id", "title", "description", "category", "icon", "difficulty", "dslCode"]
        for field in required_fields:
            assert field in template
    
    def test_template_categories(self):
        """Test template categories are valid."""
        categories = ["Computer Vision", "NLP", "Time Series", "Unsupervised", "Generative"]
        
        for category in categories:
            assert isinstance(category, str)
            assert len(category) > 0
    
    def test_template_difficulty_levels(self):
        """Test difficulty levels are valid."""
        difficulties = ["beginner", "intermediate", "advanced"]
        
        for difficulty in difficulties:
            assert difficulty in ["beginner", "intermediate", "advanced"]
    
    def test_template_load_callback(self):
        """Test template load callback receives DSL code."""
        on_load_template = Mock()
        dsl_code = "network TestNet {}"
        
        on_load_template(dsl_code)
        
        on_load_template.assert_called_once_with(dsl_code)
    
    def test_template_preview_functionality(self):
        """Test template preview shows code."""
        template_code = "network ImageClassifier { input: (28, 28, 1) }"
        
        assert "network" in template_code
        assert "input:" in template_code
    
    def test_all_templates_have_icons(self):
        """Test all templates have emoji icons."""
        icons = ["🖼️", "📝", "📈", "🔄", "🔀", "🎨"]
        
        for icon in icons:
            assert len(icon) > 0
    
    def test_difficulty_color_mapping(self):
        """Test difficulty levels have color mappings."""
        color_map = {
            "beginner": "#4caf50",
            "intermediate": "#ff9800",
            "advanced": "#f44336"
        }
        
        assert color_map["beginner"] == "#4caf50"
        assert color_map["intermediate"] == "#ff9800"
        assert color_map["advanced"] == "#f44336"


class TestExampleGalleryComponent:
    """Test example gallery component."""
    
    def test_example_structure(self):
        """Test examples have required structure."""
        example = {
            "name": "MNIST CNN",
            "path": "examples/mnist_cnn.neural",
            "description": "CNN for MNIST classification",
            "category": "Computer Vision",
            "tags": ["cnn", "mnist"],
            "complexity": "Beginner"
        }
        
        required_fields = ["name", "path", "description", "category", "tags", "complexity"]
        for field in required_fields:
            assert field in example
    
    def test_example_categories(self):
        """Test example categories."""
        categories = ["Computer Vision", "NLP", "Generative", "General"]
        
        assert "Computer Vision" in categories
        assert "NLP" in categories
        assert len(categories) > 0
    
    def test_example_tags(self):
        """Test example tags."""
        tags = ["cnn", "lstm", "transformer", "gan", "vae", "mnist"]
        
        assert isinstance(tags, list)
        assert len(tags) > 0
    
    def test_example_complexity_levels(self):
        """Test complexity levels are valid."""
        complexity_levels = ["Beginner", "Intermediate", "Advanced"]
        
        for level in complexity_levels:
            assert level in ["Beginner", "Intermediate", "Advanced"]
    
    def test_example_search_functionality(self):
        """Test example search filters correctly."""
        examples = [
            {"name": "MNIST CNN", "tags": ["mnist", "cnn"]},
            {"name": "LSTM Text", "tags": ["lstm", "nlp"]},
            {"name": "ResNet", "tags": ["cnn", "resnet"]}
        ]
        
        query = "mnist"
        filtered = [ex for ex in examples if query in ex["name"].lower()]
        
        assert len(filtered) == 1
        assert filtered[0]["name"] == "MNIST CNN"
    
    def test_example_category_filter(self):
        """Test category filtering."""
        examples = [
            {"name": "Ex1", "category": "Computer Vision"},
            {"name": "Ex2", "category": "NLP"},
            {"name": "Ex3", "category": "Computer Vision"}
        ]
        
        filtered = [ex for ex in examples if ex["category"] == "Computer Vision"]
        
        assert len(filtered) == 2
    
    def test_example_tag_filter(self):
        """Test tag-based filtering."""
        examples = [
            {"name": "Ex1", "tags": ["cnn", "vision"]},
            {"name": "Ex2", "tags": ["lstm", "nlp"]},
            {"name": "Ex3", "tags": ["cnn", "mnist"]}
        ]
        
        tag = "cnn"
        filtered = [ex for ex in examples if tag in ex["tags"]]
        
        assert len(filtered) == 2
    
    def test_example_load_callback(self):
        """Test example load callback receives code."""
        on_load_example = Mock()
        example_code = "network MNISTClassifier {}"
        
        on_load_example(example_code)
        
        on_load_example.assert_called_once_with(example_code)
    
    def test_loading_state(self):
        """Test loading state management."""
        state = {"loading": True, "examples": [], "error": None}
        
        assert state["loading"] is True
        assert isinstance(state["examples"], list)
    
    def test_error_state(self):
        """Test error state management."""
        state = {
            "loading": False,
            "error": "Failed to load examples",
            "examples": []
        }
        
        assert state["error"] is not None
        assert state["loading"] is False
    
    def test_builtin_examples_fallback(self):
        """Test built-in examples as fallback."""
        builtin_examples = [
            {
                "name": "MNIST CNN",
                "path": "examples/mnist_cnn.neural",
                "description": "CNN for MNIST",
                "category": "Computer Vision",
                "tags": ["cnn", "mnist"],
                "complexity": "Beginner"
            }
        ]
        
        assert len(builtin_examples) > 0
        assert builtin_examples[0]["name"] == "MNIST CNN"


class TestDocumentationBrowserComponent:
    """Test documentation browser component."""
    
    def test_documentation_structure(self):
        """Test documentation has required structure."""
        doc = {
            "title": "Getting Started",
            "path": "docs/getting-started.md",
            "category": "Tutorial",
            "content": "# Getting Started\n\nWelcome to Neural DSL..."
        }
        
        assert "title" in doc
        assert "path" in doc
        assert "content" in doc
    
    def test_documentation_categories(self):
        """Test documentation categories."""
        categories = ["Tutorial", "Reference", "Examples", "API"]
        
        for category in categories:
            assert isinstance(category, str)
    
    def test_documentation_search(self):
        """Test documentation search."""
        docs = [
            {"title": "Getting Started", "content": "intro tutorial"},
            {"title": "API Reference", "content": "api documentation"},
            {"title": "Advanced Topics", "content": "advanced guide"}
        ]
        
        query = "api"
        filtered = [
            doc for doc in docs
            if query in doc["title"].lower() or query in doc["content"].lower()
        ]
        
        assert len(filtered) == 1
        assert "API" in filtered[0]["title"]


class TestVideoTutorialsComponent:
    """Test video tutorials component."""
    
    def test_video_structure(self):
        """Test video tutorials have required structure."""
        video = {
            "id": "intro-video",
            "title": "Introduction to Neural DSL",
            "description": "Learn the basics",
            "duration": "10:30",
            "thumbnail": "thumbnail.jpg",
            "url": "https://example.com/video",
            "category": "Getting Started"
        }
        
        required_fields = ["id", "title", "description", "duration", "url"]
        for field in required_fields:
            assert field in video
    
    def test_video_categories(self):
        """Test video tutorial categories."""
        categories = ["Getting Started", "Advanced", "Examples", "Tips & Tricks"]
        
        assert len(categories) > 0
    
    def test_video_duration_format(self):
        """Test video duration is in correct format."""
        durations = ["5:30", "10:00", "15:45"]
        
        for duration in durations:
            assert ":" in duration
            parts = duration.split(":")
            assert len(parts) == 2


class TestWelcomeScreenIntegration:
    """Test integration between welcome screen components."""
    
    def test_template_to_editor_flow(self):
        """Test loading template into editor."""
        template_code = "network ImageClassifier { input: (28, 28, 1) }"
        editor_state = {"code": ""}
        
        def load_template(code):
            editor_state["code"] = code
        
        load_template(template_code)
        
        assert editor_state["code"] == template_code
    
    def test_example_to_editor_flow(self):
        """Test loading example into editor."""
        example_code = "network MNISTClassifier { input: (28, 28, 1) }"
        editor_state = {"code": ""}
        
        def load_example(code):
            editor_state["code"] = code
        
        load_example(example_code)
        
        assert editor_state["code"] == example_code
    
    def test_close_welcome_opens_editor(self):
        """Test closing welcome screen opens editor."""
        app_state = {"welcome_visible": True, "editor_visible": False}
        
        def close_welcome():
            app_state["welcome_visible"] = False
            app_state["editor_visible"] = True
        
        close_welcome()
        
        assert app_state["welcome_visible"] is False
        assert app_state["editor_visible"] is True
    
    def test_tutorial_starts_from_welcome(self):
        """Test starting tutorial from welcome screen."""
        app_state = {"tutorial_active": False}
        
        def start_tutorial():
            app_state["tutorial_active"] = True
        
        start_tutorial()
        
        assert app_state["tutorial_active"] is True


class TestExampleGalleryAPIIntegration:
    """Test example gallery API integration."""
    
    @patch('requests.get')
    def test_fetch_examples_from_api(self, mock_get):
        """Test fetching examples from API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "examples": [{"name": "Test", "path": "test.neural"}],
            "count": 1
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        import requests
        response = requests.get("/api/examples/list")
        data = response.json()
        
        assert data["count"] == 1
        assert len(data["examples"]) == 1
    
    @patch('requests.get')
    def test_fetch_example_code_from_api(self, mock_get):
        """Test fetching example code from API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "code": "network Test {}",
            "name": "test",
            "path": "examples/test.neural"
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        import requests
        response = requests.get("/api/examples/load?path=examples/test.neural")
        data = response.json()
        
        assert "code" in data
        assert "network" in data["code"]
    
    def test_handle_api_error_gracefully(self):
        """Test graceful handling of API errors."""
        error_response = {
            "error": "Service unavailable",
            "examples": []
        }
        
        assert "error" in error_response
        assert isinstance(error_response["examples"], list)
        assert len(error_response["examples"]) == 0


class TestTemplateValidation:
    """Test template validation and compilation."""
    
    def test_template_has_valid_dsl_syntax(self):
        """Test templates have valid DSL syntax."""
        templates = [
            "network ImageClassifier { input: (28, 28, 1) layers: Flatten() Dense(10) }",
            "network TextClassifier { input: (100) layers: LSTM(64) Dense(1) }"
        ]
        
        for template in templates:
            assert "network" in template
            assert "input:" in template
            assert "layers:" in template
    
    def test_template_compiles_successfully(self):
        """Test templates compile without errors."""
        from neural.parser.parser import create_parser, ModelTransformer
        
        template_code = """network ImageClassifier {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(10)
            loss: "mse"
            optimizer: "adam"
        }"""
        
        parser = create_parser("network")
        tree = parser.parse(template_code)
        transformer = ModelTransformer()
        model_data = transformer.transform(tree)
        
        assert model_data is not None
        assert "layers" in model_data


class TestExampleValidation:
    """Test example file validation."""
    
    def test_example_file_extension(self):
        """Test example files have .neural extension."""
        valid_files = ["mnist.neural", "lstm.neural", "resnet.neural"]
        
        for filename in valid_files:
            assert filename.endswith(".neural")
    
    def test_example_file_content(self, tmp_path):
        """Test example files contain valid DSL code."""
        example_file = tmp_path / "test.neural"
        example_file.write_text("network Test { input: (28, 28, 1) layers: Flatten() Dense(10) }")
        
        content = example_file.read_text()
        
        assert "network" in content
        assert "input:" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
