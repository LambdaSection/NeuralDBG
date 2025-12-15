"""
E2E tests for model compilation workflow.

Tests compilation to different backends (TensorFlow, PyTorch, ONNX).
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.aquarium_e2e.page_objects import AquariumWorkflow, DSLEditorPage, RunnerPanelPage


SAMPLE_DSL = """network ConvModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(64, activation="relu")
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}"""


class TestCompilation:
    """Test suite for model compilation."""
    
    @pytest.fixture(autouse=True)
    def setup_dsl(self, page: Page):
        """Setup: Parse DSL before each test."""
        editor = DSLEditorPage(page)
        editor.set_dsl_content(SAMPLE_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        nav_page = page.locator("text=Runner")
        nav_page.click()
        page.wait_for_selector("#runner-backend-select", state="visible")
    
    def test_compile_tensorflow(self, page: Page, take_screenshot):
        """Test compilation to TensorFlow backend."""
        runner = RunnerPanelPage(page)
        
        runner.select_backend("tensorflow")
        take_screenshot("before_compile_tf")
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        take_screenshot("after_compile_tf")
        
        assert runner.get_status_badge() == "Compiled"
        
        console_output = runner.get_console_output()
        assert "COMPILE" in console_output
        assert "tensorflow" in console_output.lower()
        assert "successful" in console_output.lower() or "✓" in console_output
        
        assert runner.is_run_button_enabled()
        assert runner.is_export_button_enabled()
    
    def test_compile_pytorch(self, page: Page, take_screenshot):
        """Test compilation to PyTorch backend."""
        runner = RunnerPanelPage(page)
        
        runner.select_backend("pytorch")
        take_screenshot("before_compile_pytorch")
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        take_screenshot("after_compile_pytorch")
        
        assert runner.get_status_badge() == "Compiled"
        
        console_output = runner.get_console_output()
        assert "pytorch" in console_output.lower()
        assert runner.is_run_button_enabled()
    
    def test_compile_onnx(self, page: Page, take_screenshot):
        """Test compilation to ONNX backend."""
        runner = RunnerPanelPage(page)
        
        runner.select_backend("onnx")
        take_screenshot("before_compile_onnx")
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        take_screenshot("after_compile_onnx")
        
        assert runner.get_status_badge() == "Compiled"
        
        console_output = runner.get_console_output()
        assert "onnx" in console_output.lower()
    
    def test_compilation_console_output(self, page: Page):
        """Test that compilation produces detailed console output."""
        runner = RunnerPanelPage(page)
        
        runner.select_backend("tensorflow")
        initial_output = runner.get_console_output()
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        final_output = runner.get_console_output()
        
        assert len(final_output) > len(initial_output)
        assert "[COMPILE]" in final_output
        assert "Backend:" in final_output
        assert "Dataset:" in final_output
        assert "Generated" in final_output
        assert "bytes of code" in final_output or "characters" in final_output
    
    def test_dataset_selection(self, page: Page):
        """Test dataset selection before compilation."""
        runner = RunnerPanelPage(page)
        
        datasets = ["MNIST", "CIFAR10", "CIFAR100"]
        
        for dataset in datasets:
            runner.select_dataset(dataset)
            runner.click_compile_button()
            runner.wait_for_compilation()
            
            console_output = runner.get_console_output()
            assert dataset in console_output
            
            assert runner.get_status_badge() == "Compiled"
            
            page.wait_for_timeout(500)
    
    def test_training_configuration(self, page: Page):
        """Test setting training configuration parameters."""
        runner = RunnerPanelPage(page)
        
        runner.set_epochs(5)
        runner.set_batch_size(64)
        runner.set_validation_split(0.15)
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        assert runner.get_status_badge() == "Compiled"
    
    def test_recompile_after_changes(self, page: Page):
        """Test recompiling after changing backend."""
        runner = RunnerPanelPage(page)
        
        runner.select_backend("tensorflow")
        runner.click_compile_button()
        runner.wait_for_compilation()
        assert "tensorflow" in runner.get_console_output().lower()
        
        page.wait_for_timeout(1000)
        
        runner.select_backend("pytorch")
        runner.click_compile_button()
        runner.wait_for_compilation(timeout=40000)
        
        console_output = runner.get_console_output()
        assert "pytorch" in console_output.lower()
    
    def test_clear_console(self, page: Page):
        """Test clearing console output."""
        runner = RunnerPanelPage(page)
        
        runner.click_compile_button()
        runner.wait_for_compilation()
        
        output_before = runner.get_console_output()
        assert len(output_before) > 50
        
        runner.click_clear_button()
        page.wait_for_timeout(500)
        
        output_after = runner.get_console_output()
        assert len(output_after) < len(output_before)
        assert "cleared" in output_after.lower() or "ready" in output_after.lower()
