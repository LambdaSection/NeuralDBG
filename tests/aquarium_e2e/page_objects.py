"""
Page Object Models for Aquarium IDE E2E tests.

Encapsulates page interactions for better test maintainability.
"""
from __future__ import annotations

from typing import Optional

from playwright.sync_api import Page, expect


class AquariumIDEPage:
    """Base page object for Aquarium IDE."""
    
    def __init__(self, page: Page):
        self.page = page
    
    def wait_for_page_load(self, timeout: int = 30000):
        """Wait for page to be fully loaded."""
        self.page.wait_for_load_state("networkidle", timeout=timeout)
    
    def take_screenshot(self, name: str) -> str:
        """Take a screenshot."""
        path = f"screenshot_{name}.png"
        self.page.screenshot(path=path)
        return path


class DSLEditorPage(AquariumIDEPage):
    """Page object for DSL Editor interactions."""
    
    def get_editor(self):
        """Get the DSL editor textarea."""
        return self.page.locator("#dsl-editor")
    
    def set_dsl_content(self, content: str):
        """Set DSL content in the editor."""
        editor = self.get_editor()
        editor.clear()
        editor.fill(content)
    
    def get_dsl_content(self) -> str:
        """Get current DSL content from editor."""
        return self.get_editor().input_value()
    
    def click_parse_button(self):
        """Click the Parse DSL button."""
        self.page.click("#parse-dsl-btn")
    
    def click_visualize_button(self):
        """Click the Visualize button."""
        self.page.click("#visualize-btn")
    
    def click_load_example_button(self):
        """Click the Load Example button."""
        self.page.click("#load-example-btn")
    
    def wait_for_parse_status(self, timeout: int = 10000):
        """Wait for parse status to appear."""
        self.page.wait_for_selector("#parse-status", state="visible", timeout=timeout)
    
    def get_parse_status(self) -> str:
        """Get the parse status message."""
        return self.page.locator("#parse-status").inner_text()
    
    def is_parse_successful(self) -> bool:
        """Check if parsing was successful."""
        status = self.get_parse_status()
        return "successfully" in status.lower() or "success" in status.lower()
    
    def get_model_info(self) -> str:
        """Get model information text."""
        return self.page.locator("#model-info").inner_text()


class RunnerPanelPage(AquariumIDEPage):
    """Page object for Runner Panel interactions."""
    
    def select_backend(self, backend: str):
        """Select a backend (tensorflow, pytorch, onnx)."""
        self.page.select_option("#runner-backend-select", backend)
    
    def select_dataset(self, dataset: str):
        """Select a dataset (MNIST, CIFAR10, etc.)."""
        self.page.select_option("#runner-dataset-select", dataset)
    
    def set_epochs(self, epochs: int):
        """Set number of training epochs."""
        self.page.fill("#runner-epochs", str(epochs))
    
    def set_batch_size(self, batch_size: int):
        """Set batch size."""
        self.page.fill("#runner-batch-size", str(batch_size))
    
    def set_validation_split(self, val_split: float):
        """Set validation split."""
        self.page.fill("#runner-val-split", str(val_split))
    
    def click_compile_button(self):
        """Click the Compile button."""
        self.page.click("#runner-compile-btn")
    
    def click_run_button(self):
        """Click the Run button."""
        self.page.click("#runner-run-btn")
    
    def click_stop_button(self):
        """Click the Stop button."""
        self.page.click("#runner-stop-btn")
    
    def click_export_button(self):
        """Click the Export Script button."""
        self.page.click("#runner-export-btn")
    
    def click_open_ide_button(self):
        """Click the Open in IDE button."""
        self.page.click("#runner-open-ide-btn")
    
    def click_clear_button(self):
        """Click the Clear button."""
        self.page.click("#runner-clear-btn")
    
    def wait_for_compilation(self, timeout: int = 30000):
        """Wait for compilation to complete."""
        self.page.wait_for_function(
            "document.querySelector('#runner-status-badge').innerText === 'Compiled'",
            timeout=timeout
        )
    
    def get_console_output(self) -> str:
        """Get console output text."""
        return self.page.locator("#runner-console-output").inner_text()
    
    def get_status_badge(self) -> str:
        """Get status badge text."""
        return self.page.locator("#runner-status-badge").inner_text()
    
    def is_run_button_enabled(self) -> bool:
        """Check if Run button is enabled."""
        return not self.page.locator("#runner-run-btn").is_disabled()
    
    def is_export_button_enabled(self) -> bool:
        """Check if Export button is enabled."""
        return not self.page.locator("#runner-export-btn").is_disabled()
    
    def wait_for_console_output_contains(self, text: str, timeout: int = 10000):
        """Wait for console output to contain specific text."""
        self.page.wait_for_function(
            f"document.querySelector('#runner-console-output').innerText.includes('{text}')",
            timeout=timeout
        )


class ExportModalPage(AquariumIDEPage):
    """Page object for Export Modal interactions."""
    
    def is_modal_open(self) -> bool:
        """Check if export modal is open."""
        modal = self.page.locator("#runner-export-modal")
        return modal.is_visible()
    
    def set_filename(self, filename: str):
        """Set export filename."""
        self.page.fill("#runner-export-filename", filename)
    
    def set_location(self, location: str):
        """Set export location."""
        self.page.fill("#runner-export-location", location)
    
    def click_export_confirm(self):
        """Click the Export confirm button."""
        self.page.click("#runner-export-confirm")
    
    def click_export_cancel(self):
        """Click the Export cancel button."""
        self.page.click("#runner-export-cancel")
    
    def wait_for_modal_close(self, timeout: int = 5000):
        """Wait for modal to close."""
        self.page.wait_for_selector("#runner-export-modal", state="hidden", timeout=timeout)


class NavigationPage(AquariumIDEPage):
    """Page object for navigation interactions."""
    
    def switch_to_runner_tab(self):
        """Switch to Runner tab."""
        self.page.click("text=Runner")
        self.page.wait_for_selector("#runner-backend-select", state="visible")
    
    def switch_to_debugger_tab(self):
        """Switch to Debugger tab."""
        self.page.click("text=Debugger")
    
    def switch_to_visualization_tab(self):
        """Switch to Visualization tab."""
        self.page.click("text=Visualization")
    
    def switch_to_documentation_tab(self):
        """Switch to Documentation tab."""
        self.page.click("text=Documentation")
    
    def get_active_tab(self) -> str:
        """Get the currently active tab name."""
        active_tab = self.page.locator(".nav-link.active")
        return active_tab.inner_text()


class AquariumWorkflow:
    """High-level workflow orchestration combining multiple page objects."""
    
    def __init__(self, page: Page):
        self.page = page
        self.editor = DSLEditorPage(page)
        self.runner = RunnerPanelPage(page)
        self.export = ExportModalPage(page)
        self.nav = NavigationPage(page)
    
    def complete_basic_workflow(
        self,
        dsl_content: str,
        backend: str = "tensorflow",
        compile_only: bool = False
    ):
        """
        Execute complete basic workflow:
        1. Load DSL content
        2. Parse DSL
        3. Switch to runner
        4. Compile model
        5. Optionally run model
        """
        self.editor.set_dsl_content(dsl_content)
        self.editor.click_parse_button()
        self.editor.wait_for_parse_status()
        
        assert self.editor.is_parse_successful(), "DSL parsing failed"
        
        self.nav.switch_to_runner_tab()
        self.runner.select_backend(backend)
        self.runner.click_compile_button()
        self.runner.wait_for_compilation()
        
        assert self.runner.get_status_badge() == "Compiled", "Compilation failed"
        
        if not compile_only:
            self.runner.click_run_button()
            self.runner.wait_for_console_output_contains("RUN")
    
    def export_model_script(self, filename: str, location: str = "./exported_scripts"):
        """Export compiled model script."""
        self.runner.click_export_button()
        
        assert self.export.is_modal_open(), "Export modal did not open"
        
        self.export.set_filename(filename)
        self.export.set_location(location)
        self.export.click_export_confirm()
        self.export.wait_for_modal_close()
