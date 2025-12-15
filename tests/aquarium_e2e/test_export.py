"""
E2E tests for model export and IDE integration.

Tests exporting compiled models and opening in external IDE.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.aquarium_e2e.page_objects import (
    AquariumWorkflow,
    DSLEditorPage,
    ExportModalPage,
    RunnerPanelPage,
)


SAMPLE_DSL = """network ExportModel {
    input: (16, 16, 3)
    layers:
        Conv2D(filters=16, kernel_size=3, activation="relu")
        Flatten()
        Dense(32, activation="relu")
        Output(5, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: sparse_categorical_crossentropy
}"""


class TestExport:
    """Test suite for model export functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_compiled_model(self, page: Page):
        """Setup: Compile a model before each test."""
        workflow = AquariumWorkflow(page)
        workflow.complete_basic_workflow(SAMPLE_DSL, compile_only=True)
    
    def test_export_modal_opens(self, page: Page):
        """Test that export modal opens when clicking export button."""
        runner = RunnerPanelPage(page)
        export_modal = ExportModalPage(page)
        
        runner.click_export_button()
        page.wait_for_timeout(500)
        
        assert export_modal.is_modal_open()
    
    def test_export_modal_cancel(self, page: Page):
        """Test canceling export modal."""
        runner = RunnerPanelPage(page)
        export_modal = ExportModalPage(page)
        
        runner.click_export_button()
        assert export_modal.is_modal_open()
        
        export_modal.click_export_cancel()
        export_modal.wait_for_modal_close()
        
        assert not export_modal.is_modal_open()
    
    def test_export_with_custom_filename(self, page: Page, take_screenshot):
        """Test exporting with custom filename."""
        runner = RunnerPanelPage(page)
        export_modal = ExportModalPage(page)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "my_custom_model.py"
            
            runner.click_export_button()
            assert export_modal.is_modal_open()
            
            take_screenshot("export_modal_open")
            
            export_modal.set_filename(filename)
            export_modal.set_location(tmpdir)
            
            export_modal.click_export_confirm()
            export_modal.wait_for_modal_close()
            
            page.wait_for_timeout(1000)
            take_screenshot("after_export")
            
            exported_file = Path(tmpdir) / filename
            assert exported_file.exists(), f"Exported file not found: {exported_file}"
            
            content = exported_file.read_text()
            assert len(content) > 0
            assert "import" in content.lower() or "def" in content.lower()
    
    def test_export_to_custom_location(self, page: Page):
        """Test exporting to custom location."""
        runner = RunnerPanelPage(page)
        export_modal = ExportModalPage(page)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "my_exports"
            
            runner.click_export_button()
            export_modal.set_filename("exported_model.py")
            export_modal.set_location(str(export_dir))
            export_modal.click_export_confirm()
            export_modal.wait_for_modal_close()
            
            page.wait_for_timeout(1000)
            
            assert export_dir.exists(), "Export directory was not created"
            exported_files = list(export_dir.glob("*.py"))
            assert len(exported_files) > 0, "No Python files exported"
    
    def test_export_different_backends(self, page: Page):
        """Test exporting models compiled with different backends."""
        workflow = AquariumWorkflow(page)
        
        backends = ["tensorflow", "pytorch"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for backend in backends:
                editor = DSLEditorPage(page)
                editor.set_dsl_content(SAMPLE_DSL)
                editor.click_parse_button()
                editor.wait_for_parse_status()
                
                workflow.nav.switch_to_runner_tab()
                workflow.runner.select_backend(backend)
                workflow.runner.click_compile_button()
                workflow.runner.wait_for_compilation()
                
                filename = f"model_{backend}.py"
                workflow.runner.click_export_button()
                workflow.export.set_filename(filename)
                workflow.export.set_location(tmpdir)
                workflow.export.click_export_confirm()
                workflow.export.wait_for_modal_close()
                
                page.wait_for_timeout(1000)
                
                exported_file = Path(tmpdir) / filename
                assert exported_file.exists(), f"Export failed for {backend}"
    
    def test_export_button_disabled_before_compile(self, page: Page):
        """Test that export button is disabled before compilation."""
        editor = DSLEditorPage(page)
        runner = RunnerPanelPage(page)
        
        editor.set_dsl_content(SAMPLE_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        page.click("text=Runner")
        page.wait_for_selector("#runner-backend-select", state="visible")
        
        assert not runner.is_export_button_enabled()
    
    def test_export_success_notification(self, page: Page):
        """Test that export shows success notification."""
        runner = RunnerPanelPage(page)
        export_modal = ExportModalPage(page)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.click_export_button()
            export_modal.set_filename("test_export.py")
            export_modal.set_location(tmpdir)
            export_modal.click_export_confirm()
            export_modal.wait_for_modal_close()
            
            page.wait_for_selector("#runner-notifications .alert-success", timeout=5000)
            
            notification = page.locator("#runner-notifications").inner_text()
            assert "success" in notification.lower() or "exported" in notification.lower()
    
    def test_open_in_ide_button_enabled_after_compile(self, page: Page):
        """Test that Open in IDE button is enabled after compilation."""
        runner = RunnerPanelPage(page)
        
        open_ide_btn = page.locator("#runner-open-ide-btn")
        expect(open_ide_btn).to_be_enabled()
