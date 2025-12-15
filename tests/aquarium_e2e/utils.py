"""
Utility functions for Aquarium E2E tests.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from playwright.sync_api import Page


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.5,
    error_message: str = "Condition not met within timeout"
) -> bool:
    """
    Wait for a condition to become true.
    
    Args:
        condition: Callable that returns bool
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
        error_message: Error message if timeout occurs
    
    Returns:
        True if condition met, raises TimeoutError otherwise
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(interval)
    
    raise TimeoutError(error_message)


def wait_for_text_in_element(
    page: Page,
    selector: str,
    text: str,
    timeout: int = 10000
):
    """
    Wait for specific text to appear in an element.
    
    Args:
        page: Playwright page
        selector: CSS selector
        text: Text to wait for
        timeout: Timeout in milliseconds
    """
    page.wait_for_function(
        f"document.querySelector('{selector}').innerText.includes('{text}')",
        timeout=timeout
    )


def get_element_text_when_visible(page: Page, selector: str, timeout: int = 10000) -> str:
    """
    Get element text after ensuring it's visible.
    
    Args:
        page: Playwright page
        selector: CSS selector
        timeout: Timeout in milliseconds
    
    Returns:
        Element text content
    """
    page.wait_for_selector(selector, state="visible", timeout=timeout)
    return page.locator(selector).inner_text()


def take_screenshot_on_failure(page: Page, test_name: str, output_dir: Path):
    """
    Take a screenshot when test fails.
    
    Args:
        page: Playwright page
        test_name: Name of the test
        output_dir: Directory to save screenshot
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    filepath = output_dir / f"failure_{test_name}_{int(time.time())}.png"
    page.screenshot(path=str(filepath))
    return filepath


def clear_and_type(page: Page, selector: str, text: str):
    """
    Clear input and type new text.
    
    Args:
        page: Playwright page
        selector: CSS selector for input
        text: Text to type
    """
    element = page.locator(selector)
    element.clear()
    element.fill(text)


def wait_for_server_ready(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """
    Wait for server to be ready by checking health endpoint.
    
    Args:
        url: Base URL of server
        timeout: Maximum time to wait
        interval: Check interval
    
    Returns:
        True if server is ready
    """
    import requests
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    
    return False


def get_console_logs(page: Page) -> list[str]:
    """
    Get browser console logs.
    
    Args:
        page: Playwright page
    
    Returns:
        List of console messages
    """
    logs = []
    
    def handle_console(msg):
        logs.append(f"[{msg.type}] {msg.text}")
    
    page.on("console", handle_console)
    
    return logs


def verify_no_console_errors(page: Page) -> bool:
    """
    Verify there are no console errors on the page.
    
    Args:
        page: Playwright page
    
    Returns:
        True if no errors, False otherwise
    """
    errors = []
    
    def handle_console(msg):
        if msg.type == "error":
            errors.append(msg.text)
    
    page.on("console", handle_console)
    
    return len(errors) == 0


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.2f} seconds")


def create_sample_dsl(
    model_name: str = "TestModel",
    input_shape: tuple = (28, 28, 1),
    num_layers: int = 3
) -> str:
    """
    Create a sample DSL for testing.
    
    Args:
        model_name: Name of the model
        input_shape: Input shape tuple
        num_layers: Number of layers to include
    
    Returns:
        DSL string
    """
    layers = []
    
    if len(input_shape) > 1:
        layers.append("Flatten()")
    
    for i in range(num_layers - 1):
        units = 128 // (2 ** i)
        layers.append(f'Dense({units}, activation="relu")')
    
    layers.append('Output(10, activation="softmax")')
    
    layers_str = "\n        ".join(layers)
    
    return f"""network {model_name} {{
    input: {input_shape}
    layers:
        {layers_str}
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}}"""


def verify_file_exported(export_dir: Path, filename: str) -> bool:
    """
    Verify that a file was exported correctly.
    
    Args:
        export_dir: Export directory
        filename: Expected filename
    
    Returns:
        True if file exists and is not empty
    """
    filepath = export_dir / filename
    
    if not filepath.exists():
        return False
    
    if filepath.stat().st_size == 0:
        return False
    
    return True
