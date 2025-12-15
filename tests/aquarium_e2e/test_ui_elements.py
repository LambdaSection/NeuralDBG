"""
E2E tests for UI elements and interactions.

Tests basic UI elements, buttons, forms, and interactions.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect


class TestUIElements:
    """Test suite for UI elements."""
    
    def test_page_title(self, page: Page):
        """Test that page has correct title."""
        title = page.title()
        assert "Neural Aquarium IDE" in title or "Aquarium" in title
    
    def test_header_present(self, page: Page):
        """Test that main header is present."""
        header = page.locator("text=Neural Aquarium IDE")
        expect(header).to_be_visible()
    
    def test_dsl_editor_visible(self, page: Page):
        """Test that DSL editor is visible."""
        editor = page.locator("#dsl-editor")
        expect(editor).to_be_visible()
        expect(editor).to_be_editable()
    
    def test_action_buttons_present(self, page: Page):
        """Test that main action buttons are present."""
        parse_btn = page.locator("#parse-dsl-btn")
        visualize_btn = page.locator("#visualize-btn")
        load_example_btn = page.locator("#load-example-btn")
        
        expect(parse_btn).to_be_visible()
        expect(visualize_btn).to_be_visible()
        expect(load_example_btn).to_be_visible()
    
    def test_model_info_section_present(self, page: Page):
        """Test that model info section is present."""
        model_info = page.locator("#model-info")
        expect(model_info).to_be_visible()
    
    def test_runner_panel_elements(self, page: Page):
        """Test that runner panel elements are present."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-backend-select", state="visible")
        
        backend_select = page.locator("#runner-backend-select")
        dataset_select = page.locator("#runner-dataset-select")
        compile_btn = page.locator("#runner-compile-btn")
        console_output = page.locator("#runner-console-output")
        
        expect(backend_select).to_be_visible()
        expect(dataset_select).to_be_visible()
        expect(compile_btn).to_be_visible()
        expect(console_output).to_be_visible()
    
    def test_backend_options(self, page: Page):
        """Test that all backend options are available."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-backend-select", state="visible")
        
        backend_select = page.locator("#runner-backend-select")
        
        options_text = backend_select.inner_text()
        assert "TensorFlow" in options_text or "tensorflow" in options_text.lower()
        assert "PyTorch" in options_text or "pytorch" in options_text.lower()
        assert "ONNX" in options_text or "onnx" in options_text.lower()
    
    def test_dataset_options(self, page: Page):
        """Test that dataset options are available."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-dataset-select", state="visible")
        
        dataset_select = page.locator("#runner-dataset-select")
        
        options_text = dataset_select.inner_text()
        assert "MNIST" in options_text
        assert "CIFAR10" in options_text
    
    def test_training_configuration_inputs(self, page: Page):
        """Test that training configuration inputs are present."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-epochs", state="visible")
        
        epochs_input = page.locator("#runner-epochs")
        batch_size_input = page.locator("#runner-batch-size")
        val_split_input = page.locator("#runner-val-split")
        
        expect(epochs_input).to_be_visible()
        expect(batch_size_input).to_be_visible()
        expect(val_split_input).to_be_visible()
    
    def test_status_badge_present(self, page: Page):
        """Test that status badge is present in runner panel."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-status-badge", state="visible")
        
        status_badge = page.locator("#runner-status-badge")
        expect(status_badge).to_be_visible()
        
        badge_text = status_badge.inner_text()
        assert len(badge_text) > 0
    
    def test_console_output_styling(self, page: Page):
        """Test that console output has proper styling."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-console-output", state="visible")
        
        console = page.locator("#runner-console-output")
        
        bg_color = console.evaluate("el => window.getComputedStyle(el).backgroundColor")
        assert bg_color is not None
        
        font_family = console.evaluate("el => window.getComputedStyle(el).fontFamily")
        assert "monospace" in font_family.lower() or "console" in font_family.lower()
    
    def test_button_states(self, page: Page):
        """Test initial button states."""
        page.click("text=Runner")
        page.wait_for_selector("#runner-compile-btn", state="visible")
        
        compile_btn = page.locator("#runner-compile-btn")
        run_btn = page.locator("#runner-run-btn")
        export_btn = page.locator("#runner-export-btn")
        
        expect(compile_btn).to_be_enabled()
        expect(run_btn).to_be_disabled()
        expect(export_btn).to_be_disabled()
    
    def test_placeholder_text(self, page: Page):
        """Test that editor has placeholder text."""
        editor = page.locator("#dsl-editor")
        placeholder = editor.get_attribute("placeholder")
        
        assert placeholder is not None
        assert "network" in placeholder.lower() or "model" in placeholder.lower()
    
    def test_responsive_layout(self, page: Page):
        """Test that layout is responsive to viewport changes."""
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.wait_for_timeout(500)
        
        editor = page.locator("#dsl-editor")
        expect(editor).to_be_visible()
        
        page.set_viewport_size({"width": 1280, "height": 720})
        page.wait_for_timeout(500)
        
        expect(editor).to_be_visible()
    
    def test_icon_presence(self, page: Page):
        """Test that icons are present in buttons."""
        page.wait_for_selector(".fas", state="visible", timeout=5000)
        
        icons = page.locator(".fas").count()
        assert icons > 0, "No Font Awesome icons found"
    
    def test_card_layout(self, page: Page):
        """Test that card-based layout is present."""
        cards = page.locator(".card")
        card_count = cards.count()
        
        assert card_count > 0, "No Bootstrap cards found in layout"
