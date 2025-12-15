"""
E2E performance and stress tests for Aquarium IDE.

Tests performance characteristics and system behavior under load.
"""
from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page

from tests.aquarium_e2e.page_objects import AquariumWorkflow, DSLEditorPage, RunnerPanelPage
from tests.aquarium_e2e.utils import PerformanceTimer, create_sample_dsl


COMPLEX_DSL = """network ComplexModel {
    input: (224, 224, 3)
    layers:
        Conv2D(filters=64, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=128, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=256, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Conv2D(filters=512, kernel_size=3, activation="relu")
        BatchNormalization()
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(1024, activation="relu")
        Dropout(0.5)
        Dense(512, activation="relu")
        Dropout(0.3)
        Output(1000, activation="softmax")
    optimizer: Adam(learning_rate=0.0001)
    loss: categorical_crossentropy
}"""


@pytest.mark.slow
class TestPerformance:
    """Test suite for performance characteristics."""
    
    def test_parsing_performance(self, page: Page):
        """Test DSL parsing performance."""
        editor = DSLEditorPage(page)
        editor.set_dsl_content(COMPLEX_DSL)
        
        with PerformanceTimer("DSL Parsing") as timer:
            editor.click_parse_button()
            editor.wait_for_parse_status()
        
        assert timer.duration < 5.0, f"Parsing took too long: {timer.duration}s"
        assert editor.is_parse_successful()
    
    def test_compilation_performance_tensorflow(self, page: Page):
        """Test TensorFlow compilation performance."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(COMPLEX_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        workflow.runner.select_backend("tensorflow")
        
        with PerformanceTimer("TensorFlow Compilation") as timer:
            workflow.runner.click_compile_button()
            workflow.runner.wait_for_compilation(timeout=60000)
        
        assert timer.duration < 30.0, f"Compilation took too long: {timer.duration}s"
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_compilation_performance_pytorch(self, page: Page):
        """Test PyTorch compilation performance."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(COMPLEX_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        workflow.runner.select_backend("pytorch")
        
        with PerformanceTimer("PyTorch Compilation") as timer:
            workflow.runner.click_compile_button()
            workflow.runner.wait_for_compilation(timeout=60000)
        
        assert timer.duration < 30.0, f"Compilation took too long: {timer.duration}s"
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_multiple_sequential_compilations(self, page: Page):
        """Test performance of multiple sequential compilations."""
        workflow = AquariumWorkflow(page)
        
        sample_dsl = create_sample_dsl("TestModel", (28, 28, 1), 3)
        workflow.editor.set_dsl_content(sample_dsl)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        backends = ["tensorflow", "pytorch", "onnx"]
        
        with PerformanceTimer("Multiple Sequential Compilations") as timer:
            for backend in backends:
                workflow.runner.select_backend(backend)
                workflow.runner.click_compile_button()
                workflow.runner.wait_for_compilation(timeout=60000)
                page.wait_for_timeout(500)
        
        avg_time = timer.duration / len(backends)
        assert avg_time < 20.0, f"Average compilation time too slow: {avg_time}s"
    
    def test_large_dsl_content_handling(self, page: Page):
        """Test handling of large DSL content."""
        editor = DSLEditorPage(page)
        
        large_dsl_parts = [
            "network LargeModel {",
            "    input: (100, 100, 3)",
            "    layers:"
        ]
        
        for i in range(50):
            large_dsl_parts.append(
                f"        Dense({128 - i}, activation=\"relu\")"
            )
        
        large_dsl_parts.extend([
            "        Output(10, activation=\"softmax\")",
            "    optimizer: Adam(learning_rate=0.001)",
            "    loss: categorical_crossentropy",
            "}"
        ])
        
        large_dsl = "\n".join(large_dsl_parts)
        
        with PerformanceTimer("Large DSL Input") as timer:
            editor.set_dsl_content(large_dsl)
        
        assert timer.duration < 2.0, f"Setting large DSL took too long: {timer.duration}s"
        
        content = editor.get_dsl_content()
        assert len(content) > 1000
    
    def test_rapid_tab_switching(self, page: Page):
        """Test performance of rapid tab switching."""
        workflow = AquariumWorkflow(page)
        
        with PerformanceTimer("Rapid Tab Switching") as timer:
            for _ in range(10):
                workflow.nav.switch_to_runner_tab()
                page.wait_for_timeout(100)
                workflow.nav.switch_to_debugger_tab()
                page.wait_for_timeout(100)
                workflow.nav.switch_to_visualization_tab()
                page.wait_for_timeout(100)
                workflow.nav.switch_to_documentation_tab()
                page.wait_for_timeout(100)
        
        avg_switch_time = timer.duration / 40
        assert avg_switch_time < 0.5, f"Tab switching too slow: {avg_switch_time}s per switch"
    
    def test_console_output_performance(self, page: Page):
        """Test console output handling performance."""
        workflow = AquariumWorkflow(page)
        
        sample_dsl = create_sample_dsl("PerfTest", (10,), 3)
        workflow.editor.set_dsl_content(sample_dsl)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        for _ in range(5):
            workflow.runner.click_compile_button()
            workflow.runner.wait_for_compilation()
            page.wait_for_timeout(500)
        
        with PerformanceTimer("Console Output Retrieval") as timer:
            console_output = workflow.runner.get_console_output()
        
        assert timer.duration < 1.0, f"Console output retrieval too slow: {timer.duration}s"
        assert len(console_output) > 0
    
    def test_page_load_performance(self, page: Page):
        """Test initial page load performance."""
        start_time = time.time()
        
        page.goto(page.url, wait_until="networkidle")
        
        load_time = time.time() - start_time
        
        assert load_time < 10.0, f"Page load too slow: {load_time}s"
    
    def test_memory_stability_multiple_operations(self, page: Page):
        """Test memory stability with multiple operations."""
        workflow = AquariumWorkflow(page)
        
        for i in range(10):
            sample_dsl = create_sample_dsl(f"Model{i}", (28, 28, 1), 3)
            workflow.editor.set_dsl_content(sample_dsl)
            workflow.editor.click_parse_button()
            workflow.editor.wait_for_parse_status()
            
            if i % 3 == 0:
                workflow.nav.switch_to_runner_tab()
                workflow.runner.click_compile_button()
                workflow.runner.wait_for_compilation()
                page.wait_for_timeout(300)
        
        page.wait_for_timeout(1000)


@pytest.mark.slow
class TestStress:
    """Test suite for stress testing."""
    
    def test_rapid_parse_requests(self, page: Page):
        """Test handling of rapid parse requests."""
        editor = DSLEditorPage(page)
        
        sample_dsl = create_sample_dsl("StressTest", (10,), 2)
        editor.set_dsl_content(sample_dsl)
        
        for _ in range(10):
            editor.click_parse_button()
            page.wait_for_timeout(200)
        
        editor.wait_for_parse_status(timeout=15000)
        assert editor.is_parse_successful()
    
    def test_multiple_backend_switches(self, page: Page):
        """Test stress of multiple backend switches."""
        workflow = AquariumWorkflow(page)
        
        sample_dsl = create_sample_dsl("BackendStress", (10,), 2)
        workflow.editor.set_dsl_content(sample_dsl)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        
        backends = ["tensorflow", "pytorch", "onnx"]
        
        for _ in range(5):
            for backend in backends:
                workflow.runner.select_backend(backend)
                page.wait_for_timeout(100)
        
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation()
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_long_running_session(self, page: Page):
        """Test stability during long-running session."""
        workflow = AquariumWorkflow(page)
        
        operations = [
            ("parse", lambda: (
                workflow.editor.set_dsl_content(create_sample_dsl("Test", (10,), 2)),
                workflow.editor.click_parse_button(),
                workflow.editor.wait_for_parse_status()
            )),
            ("switch_tab", lambda: workflow.nav.switch_to_runner_tab()),
            ("compile", lambda: (
                workflow.runner.click_compile_button(),
                workflow.runner.wait_for_compilation()
            )),
            ("clear", lambda: workflow.runner.click_clear_button()),
        ]
        
        for i in range(20):
            op_name, op_func = operations[i % len(operations)]
            
            try:
                op_func()
                page.wait_for_timeout(200)
            except Exception as e:
                pytest.fail(f"Operation {op_name} failed at iteration {i}: {e}")
        
        assert True
