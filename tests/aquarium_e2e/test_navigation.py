"""
E2E tests for UI navigation and tab switching.

Tests navigation between different sections of Aquarium IDE.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.aquarium_e2e.page_objects import NavigationPage


class TestNavigation:
    """Test suite for UI navigation."""
    
    def test_initial_tab_is_runner(self, page: Page):
        """Test that initial active tab is Runner."""
        nav = NavigationPage(page)
        
        page.wait_for_selector("#runner-backend-select", state="visible", timeout=10000)
        
        active_tab = nav.get_active_tab()
        assert "Runner" in active_tab
    
    def test_switch_to_debugger_tab(self, page: Page):
        """Test switching to Debugger tab."""
        nav = NavigationPage(page)
        
        nav.switch_to_debugger_tab()
        page.wait_for_timeout(500)
        
        debugger_content = page.locator("text=NeuralDbg Integration")
        expect(debugger_content).to_be_visible()
    
    def test_switch_to_visualization_tab(self, page: Page):
        """Test switching to Visualization tab."""
        nav = NavigationPage(page)
        
        nav.switch_to_visualization_tab()
        page.wait_for_timeout(500)
        
        viz_content = page.locator("text=Model Architecture")
        expect(viz_content).to_be_visible()
    
    def test_switch_to_documentation_tab(self, page: Page):
        """Test switching to Documentation tab."""
        nav = NavigationPage(page)
        
        nav.switch_to_documentation_tab()
        page.wait_for_timeout(500)
        
        doc_content = page.locator("text=Neural DSL Documentation")
        expect(doc_content).to_be_visible()
    
    def test_tab_switching_sequence(self, page: Page, take_screenshot):
        """Test switching through all tabs in sequence."""
        nav = NavigationPage(page)
        
        tabs = ["Runner", "Debugger", "Visualization", "Documentation"]
        
        for tab in tabs:
            if tab == "Runner":
                nav.switch_to_runner_tab()
            elif tab == "Debugger":
                nav.switch_to_debugger_tab()
            elif tab == "Visualization":
                nav.switch_to_visualization_tab()
            elif tab == "Documentation":
                nav.switch_to_documentation_tab()
            
            page.wait_for_timeout(300)
            take_screenshot(f"tab_{tab.lower()}")
    
    def test_return_to_runner_from_other_tabs(self, page: Page):
        """Test returning to Runner tab from other tabs."""
        nav = NavigationPage(page)
        
        other_tabs = [
            nav.switch_to_debugger_tab,
            nav.switch_to_visualization_tab,
            nav.switch_to_documentation_tab
        ]
        
        for switch_func in other_tabs:
            switch_func()
            page.wait_for_timeout(300)
            
            nav.switch_to_runner_tab()
            page.wait_for_timeout(300)
            
            page.wait_for_selector("#runner-backend-select", state="visible")
            active_tab = nav.get_active_tab()
            assert "Runner" in active_tab
    
    def test_tab_content_persists(self, page: Page):
        """Test that tab content persists when switching tabs."""
        nav = NavigationPage(page)
        
        nav.switch_to_runner_tab()
        backend_select = page.locator("#runner-backend-select")
        backend_select.select_option("pytorch")
        
        nav.switch_to_debugger_tab()
        page.wait_for_timeout(300)
        
        nav.switch_to_runner_tab()
        page.wait_for_timeout(300)
        
        selected_value = backend_select.input_value()
        assert selected_value == "pytorch"
    
    def test_navigation_with_keyboard(self, page: Page):
        """Test keyboard navigation if supported."""
        runner_tab = page.locator("text=Runner")
        expect(runner_tab).to_be_visible()
        
        runner_tab.focus()
        
        page.keyboard.press("Tab")
        page.wait_for_timeout(200)
    
    def test_all_tabs_visible(self, page: Page):
        """Test that all main tabs are visible."""
        tabs = ["Runner", "Debugger", "Visualization", "Documentation"]
        
        for tab_name in tabs:
            tab = page.locator(f"text={tab_name}")
            expect(tab).to_be_visible()
