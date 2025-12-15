"""
E2E tests for DSL Editor functionality.

Tests DSL editing, parsing, and validation workflows.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.aquarium_e2e.page_objects import DSLEditorPage


SAMPLE_DSL = """network MNISTModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128, activation="relu")
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


INVALID_DSL = """network Invalid {
    input: (28, 28, 1)
    layers:
        InvalidLayer(param=value)
}"""


class TestDSLEditor:
    """Test suite for DSL Editor."""
    
    def test_editor_loads(self, page: Page):
        """Test that DSL editor loads correctly."""
        editor_page = DSLEditorPage(page)
        editor = editor_page.get_editor()
        
        expect(editor).to_be_visible()
        expect(editor).to_be_editable()
    
    def test_load_example(self, page: Page, take_screenshot):
        """Test loading example DSL code."""
        editor_page = DSLEditorPage(page)
        
        initial_content = editor_page.get_dsl_content()
        
        editor_page.click_load_example_button()
        page.wait_for_timeout(500)
        
        new_content = editor_page.get_dsl_content()
        
        assert new_content != initial_content, "Example was not loaded"
        assert "network" in new_content.lower(), "Loaded content is not valid DSL"
        
        take_screenshot("example_loaded")
    
    def test_parse_valid_dsl(self, page: Page, take_screenshot):
        """Test parsing valid DSL code."""
        editor_page = DSLEditorPage(page)
        
        editor_page.set_dsl_content(SAMPLE_DSL)
        take_screenshot("before_parse")
        
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status()
        
        take_screenshot("after_parse")
        
        assert editor_page.is_parse_successful(), "Valid DSL failed to parse"
        
        model_info = editor_page.get_model_info()
        assert "Input Shape" in model_info
        assert "Number of Layers" in model_info
        assert "Loss Function" in model_info
    
    def test_parse_invalid_dsl(self, page: Page, take_screenshot):
        """Test parsing invalid DSL code shows error."""
        editor_page = DSLEditorPage(page)
        
        editor_page.set_dsl_content(INVALID_DSL)
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status()
        
        take_screenshot("parse_error")
        
        assert not editor_page.is_parse_successful(), "Invalid DSL should not parse"
        
        status = editor_page.get_parse_status()
        assert "error" in status.lower() or "fail" in status.lower()
    
    def test_edit_dsl_content(self, page: Page):
        """Test editing DSL content."""
        editor_page = DSLEditorPage(page)
        
        custom_dsl = """network CustomModel {
    input: (32, 32, 3)
    layers:
        Dense(64, activation="relu")
        Output(10)
    optimizer: SGD(learning_rate=0.01)
    loss: mse
}"""
        
        editor_page.set_dsl_content(custom_dsl)
        retrieved_content = editor_page.get_dsl_content()
        
        assert retrieved_content == custom_dsl, "DSL content was not set correctly"
    
    def test_visualize_button(self, page: Page):
        """Test visualize button interaction."""
        editor_page = DSLEditorPage(page)
        
        editor_page.set_dsl_content(SAMPLE_DSL)
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status()
        
        visualize_btn = page.locator("#visualize-btn")
        expect(visualize_btn).to_be_visible()
        expect(visualize_btn).to_be_enabled()
    
    def test_model_info_details(self, page: Page):
        """Test that model info displays correct details after parsing."""
        editor_page = DSLEditorPage(page)
        
        editor_page.set_dsl_content(SAMPLE_DSL)
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status()
        
        model_info = editor_page.get_model_info()
        
        assert "28, 28, 1" in model_info
        assert "Conv2D" in model_info or "5" in model_info
        assert "categorical_crossentropy" in model_info
        assert "Adam" in model_info
    
    def test_multiple_parse_cycles(self, page: Page):
        """Test parsing multiple times with different DSL code."""
        editor_page = DSLEditorPage(page)
        
        dsl1 = """network Model1 {
    input: (10,)
    layers:
        Dense(5)
        Output(1)
    optimizer: Adam(learning_rate=0.001)
    loss: mse
}"""
        
        dsl2 = """network Model2 {
    input: (20,)
    layers:
        Dense(10)
        Dense(5)
        Output(2)
    optimizer: SGD(learning_rate=0.01)
    loss: binary_crossentropy
}"""
        
        editor_page.set_dsl_content(dsl1)
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status()
        assert editor_page.is_parse_successful()
        
        page.wait_for_timeout(500)
        
        editor_page.set_dsl_content(dsl2)
        editor_page.click_parse_button()
        editor_page.wait_for_parse_status(timeout=15000)
        assert editor_page.is_parse_successful()
        
        model_info = editor_page.get_model_info()
        assert "20" in model_info or "Model2" in model_info
