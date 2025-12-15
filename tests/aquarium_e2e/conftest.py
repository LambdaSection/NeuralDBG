"""
Pytest configuration and fixtures for Aquarium IDE E2E tests.
"""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


AQUARIUM_PORT = 8052
AQUARIUM_URL = f"http://localhost:{AQUARIUM_PORT}"
SERVER_STARTUP_TIMEOUT = 30
SERVER_CHECK_INTERVAL = 0.5


@pytest.fixture(scope="session")
def aquarium_server():
    """Start Aquarium IDE server for testing."""
    process = None
    try:
        neural_path = Path(__file__).parent.parent.parent
        aquarium_module = "neural.aquarium.aquarium"
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(neural_path)
        
        process = subprocess.Popen(
            [sys.executable, "-m", aquarium_module, "--port", str(AQUARIUM_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(neural_path)
        )
        
        start_time = time.time()
        server_ready = False
        
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                import requests
                response = requests.get(f"{AQUARIUM_URL}/health", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                pass
            time.sleep(SERVER_CHECK_INTERVAL)
        
        if not server_ready:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            raise RuntimeError(
                f"Aquarium server failed to start within {SERVER_STARTUP_TIMEOUT}s"
            )
        
        yield AQUARIUM_URL
        
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Browser launch arguments."""
    return {
        "headless": os.environ.get("HEADLESS", "true").lower() == "true",
        "slow_mo": int(os.environ.get("SLOW_MO", "0")),
    }


@pytest.fixture(scope="session")
def playwright_instance():
    """Playwright instance."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright, browser_type_launch_args):
    """Browser instance."""
    browser = playwright_instance.chromium.launch(**browser_type_launch_args)
    yield browser
    browser.close()


@pytest.fixture
def context(browser: Browser):
    """Browser context for test isolation."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
    )
    yield context
    context.close()


@pytest.fixture
def page(context: BrowserContext, aquarium_server):
    """Page instance with Aquarium loaded."""
    page = context.new_page()
    page.goto(aquarium_server)
    page.wait_for_load_state("networkidle")
    yield page
    page.close()


@pytest.fixture
def screenshot_dir():
    """Directory for test screenshots."""
    screenshot_path = Path(__file__).parent / "screenshots"
    screenshot_path.mkdir(exist_ok=True)
    return screenshot_path


@pytest.fixture
def take_screenshot(page: Page, screenshot_dir: Path, request):
    """Helper to take screenshots during tests."""
    def _take_screenshot(name: str):
        test_name = request.node.name
        filepath = screenshot_dir / f"{test_name}_{name}.png"
        page.screenshot(path=str(filepath))
        return filepath
    
    return _take_screenshot
