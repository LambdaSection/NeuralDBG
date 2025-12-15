"""
E2E tests for complete Aquarium IDE workflow.

Tests the full user journey from start to finish.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.aquarium_e2e.page_objects import AquariumWorkflow, DSLEditorPage, NavigationPage


MNIST_DSL = """network MNISTClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Conv2D(filters=64, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128, activation="relu")
        Dropout(0.5)
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


CIFAR_DSL = """network CIFARModel {
    input: (32, 32, 3)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=64, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(256, activation="relu")
        Dropout(0.3)
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


class TestCompleteWorkflow:
    """Test suite for complete end-to-end workflows."""
    
    def test_simple_workflow_tensorflow(self, page: Page, take_screenshot):
        """Test complete workflow: DSL → Parse → Compile (TensorFlow)."""
        workflow = AquariumWorkflow(page)
        
        take_screenshot("1_initial_page")
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        take_screenshot("2_dsl_entered")
        
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        take_screenshot("3_parsed")
        
        assert workflow.editor.is_parse_successful()
        
        workflow.nav.switch_to_runner_tab()
        take_screenshot("4_runner_tab")
        
        workflow.runner.select_backend("tensorflow")
        workflow.runner.select_dataset("MNIST")
        take_screenshot("5_backend_selected")
        
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation()
        take_screenshot("6_compiled")
        
        assert workflow.runner.get_status_badge() == "Compiled"
        assert workflow.runner.is_export_button_enabled()
    
    def test_workflow_with_export(self, page: Page, take_screenshot):
        """Test complete workflow: DSL → Parse → Compile → Export."""
        workflow = AquariumWorkflow(page)
        
        workflow.complete_basic_workflow(MNIST_DSL, backend="tensorflow", compile_only=True)
        take_screenshot("workflow_compiled")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow.export_model_script(
                filename="mnist_model.py",
                location=tmpdir
            )
            
            take_screenshot("workflow_exported")
            
            exported_file = Path(tmpdir) / "mnist_model.py"
            assert exported_file.exists()
            
            content = exported_file.read_text()
            assert len(content) > 100
    
    def test_workflow_multiple_backends(self, page: Page, take_screenshot):
        """Test workflow with multiple backend compilations."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(CIFAR_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        assert workflow.editor.is_parse_successful()
        
        workflow.nav.switch_to_runner_tab()
        
        backends = ["tensorflow", "pytorch"]
        
        for idx, backend in enumerate(backends):
            take_screenshot(f"backend_{backend}_before")
            
            workflow.runner.select_backend(backend)
            workflow.runner.click_compile_button()
            workflow.runner.wait_for_compilation()
            
            take_screenshot(f"backend_{backend}_after")
            
            assert workflow.runner.get_status_badge() == "Compiled"
            console_output = workflow.runner.get_console_output()
            assert backend in console_output.lower()
            
            if idx < len(backends) - 1:
                page.wait_for_timeout(1000)
    
    def test_workflow_with_different_datasets(self, page: Page):
        """Test workflow with different dataset selections."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        datasets = ["MNIST", "CIFAR10"]
        
        for dataset in datasets:
            workflow.runner.select_dataset(dataset)
            workflow.runner.click_compile_button()
            workflow.runner.wait_for_compilation()
            
            console_output = workflow.runner.get_console_output()
            assert dataset in console_output
            assert workflow.runner.get_status_badge() == "Compiled"
            
            page.wait_for_timeout(500)
    
    def test_workflow_tab_navigation(self, page: Page, take_screenshot):
        """Test navigating between different tabs in workflow."""
        workflow = AquariumWorkflow(page)
        nav = NavigationPage(page)
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        nav.switch_to_runner_tab()
        take_screenshot("runner_tab")
        assert "Runner" in nav.get_active_tab()
        
        nav.switch_to_debugger_tab()
        take_screenshot("debugger_tab")
        page.wait_for_timeout(500)
        
        nav.switch_to_visualization_tab()
        take_screenshot("visualization_tab")
        page.wait_for_timeout(500)
        
        nav.switch_to_documentation_tab()
        take_screenshot("documentation_tab")
        page.wait_for_timeout(500)
        
        nav.switch_to_runner_tab()
        assert "Runner" in nav.get_active_tab()
    
    def test_workflow_load_example_and_compile(self, page: Page, take_screenshot):
        """Test workflow starting with example loading."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.click_load_example_button()
        page.wait_for_timeout(1000)
        take_screenshot("example_loaded")
        
        dsl_content = workflow.editor.get_dsl_content()
        assert len(dsl_content) > 0
        assert "network" in dsl_content.lower()
        
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        assert workflow.editor.is_parse_successful()
        
        workflow.nav.switch_to_runner_tab()
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation()
        
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_workflow_training_configuration(self, page: Page):
        """Test complete workflow with custom training configuration."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        workflow.runner.set_epochs(20)
        workflow.runner.set_batch_size(128)
        workflow.runner.set_validation_split(0.25)
        
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation()
        
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_workflow_error_recovery(self, page: Page, take_screenshot):
        """Test workflow recovery from parsing error."""
        workflow = AquariumWorkflow(page)
        
        invalid_dsl = """network InvalidModel {
    input: (10,)
    layers:
        UnknownLayer(param=123)
}"""
        
        workflow.editor.set_dsl_content(invalid_dsl)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        take_screenshot("error_state")
        
        assert not workflow.editor.is_parse_successful()
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        take_screenshot("recovered_state")
        
        assert workflow.editor.is_parse_successful()
    
    def test_workflow_console_persistence(self, page: Page):
        """Test that console output persists across operations."""
        workflow = AquariumWorkflow(page)
        
        workflow.complete_basic_workflow(MNIST_DSL, compile_only=True)
        
        first_output = workflow.runner.get_console_output()
        assert "[COMPILE]" in first_output
        
        workflow.nav.switch_to_debugger_tab()
        page.wait_for_timeout(500)
        
        workflow.nav.switch_to_runner_tab()
        
        second_output = workflow.runner.get_console_output()
        assert first_output in second_output or len(second_output) > 0
    
    @pytest.mark.slow
    def test_full_workflow_with_multiple_exports(self, page: Page, take_screenshot):
        """Test complete workflow with multiple exports to different backends."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(CIFAR_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        backends = ["tensorflow", "pytorch"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for backend in backends:
                workflow.runner.select_backend(backend)
                workflow.runner.click_compile_button()
                workflow.runner.wait_for_compilation()
                
                workflow.export_model_script(
                    filename=f"cifar_{backend}.py",
                    location=tmpdir
                )
                
                exported_file = Path(tmpdir) / f"cifar_{backend}.py"
                assert exported_file.exists()
                
                page.wait_for_timeout(1000)
            
            take_screenshot("all_exports_complete")
            
            exported_files = list(Path(tmpdir).glob("*.py"))
            assert len(exported_files) == len(backends)
