"""
E2E tests for different model variations.

Tests various DSL model types and configurations.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page

from tests.aquarium_e2e.page_objects import AquariumWorkflow, DSLEditorPage
from tests.aquarium_e2e.test_data import (
    AUTOENCODER_DSL,
    CIFAR10_DSL,
    IMAGENET_DSL,
    MNIST_DSL,
    RNN_DSL,
    SIMPLE_DSL,
    TRANSFORMER_DSL,
    get_dsl_by_name,
)


class TestModelVariations:
    """Test suite for different model architectures."""
    
    @pytest.mark.parametrize("model_name,dsl_content", [
        ("simple", SIMPLE_DSL),
        ("mnist", MNIST_DSL),
        ("cifar10", CIFAR10_DSL),
    ])
    def test_parse_model_variations(self, page: Page, model_name: str, dsl_content: str):
        """Test parsing different model architectures."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(dsl_content)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        assert editor.is_parse_successful(), f"{model_name} failed to parse"
        
        model_info = editor.get_model_info()
        assert len(model_info) > 0
    
    def test_simple_model_workflow(self, page: Page):
        """Test complete workflow with simple model."""
        workflow = AquariumWorkflow(page)
        workflow.complete_basic_workflow(SIMPLE_DSL, backend="tensorflow", compile_only=True)
        
        assert workflow.runner.get_status_badge() == "Compiled"
    
    def test_mnist_cnn_model_workflow(self, page: Page):
        """Test complete workflow with MNIST CNN model."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        assert workflow.editor.is_parse_successful()
        
        model_info = workflow.editor.get_model_info()
        assert "28, 28, 1" in model_info
        assert "Conv2D" in model_info or "Convolutional" in model_info
    
    def test_cifar10_model_with_batch_norm(self, page: Page):
        """Test CIFAR10 model with batch normalization."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(CIFAR10_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        assert editor.is_parse_successful()
        
        model_info = editor.get_model_info()
        assert "32, 32, 3" in model_info
    
    @pytest.mark.slow
    def test_imagenet_large_model(self, page: Page):
        """Test large ImageNet-style model."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(IMAGENET_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status(timeout=15000)
        
        assert editor.is_parse_successful()
        
        model_info = editor.get_model_info()
        assert "224, 224, 3" in model_info
    
    def test_rnn_sequence_model(self, page: Page):
        """Test RNN/LSTM sequence model."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(RNN_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        assert editor.is_parse_successful()
        
        model_info = editor.get_model_info()
        assert "100, 128" in model_info
    
    def test_transformer_attention_model(self, page: Page):
        """Test Transformer model with attention."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(TRANSFORMER_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        assert editor.is_parse_successful()
    
    def test_autoencoder_model(self, page: Page):
        """Test autoencoder architecture."""
        editor = DSLEditorPage(page)
        
        editor.set_dsl_content(AUTOENCODER_DSL)
        editor.click_parse_button()
        editor.wait_for_parse_status()
        
        assert editor.is_parse_successful()
        
        model_info = editor.get_model_info()
        assert "784" in model_info
    
    @pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
    def test_compile_mnist_multiple_backends(self, page: Page, backend: str):
        """Test compiling MNIST model to different backends."""
        workflow = AquariumWorkflow(page)
        
        workflow.editor.set_dsl_content(MNIST_DSL)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        workflow.runner.select_backend(backend)
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation(timeout=40000)
        
        assert workflow.runner.get_status_badge() == "Compiled"
        console_output = workflow.runner.get_console_output()
        assert backend in console_output.lower()
    
    def test_switch_between_models(self, page: Page, take_screenshot):
        """Test switching between different model architectures."""
        editor = DSLEditorPage(page)
        
        models = [
            ("simple", SIMPLE_DSL),
            ("mnist", MNIST_DSL),
            ("cifar10", CIFAR10_DSL)
        ]
        
        for model_name, dsl_content in models:
            editor.set_dsl_content(dsl_content)
            editor.click_parse_button()
            editor.wait_for_parse_status(timeout=15000)
            
            take_screenshot(f"model_{model_name}")
            
            assert editor.is_parse_successful(), f"Failed to parse {model_name}"
            
            page.wait_for_timeout(500)
    
    @pytest.mark.parametrize("dataset", ["MNIST", "CIFAR10"])
    def test_dataset_compatibility(self, page: Page, dataset: str):
        """Test model compilation with different datasets."""
        workflow = AquariumWorkflow(page)
        
        dsl_map = {
            "MNIST": MNIST_DSL,
            "CIFAR10": CIFAR10_DSL
        }
        
        dsl_content = dsl_map.get(dataset, SIMPLE_DSL)
        
        workflow.editor.set_dsl_content(dsl_content)
        workflow.editor.click_parse_button()
        workflow.editor.wait_for_parse_status()
        
        workflow.nav.switch_to_runner_tab()
        workflow.runner.select_dataset(dataset)
        workflow.runner.click_compile_button()
        workflow.runner.wait_for_compilation(timeout=40000)
        
        console_output = workflow.runner.get_console_output()
        assert dataset in console_output
