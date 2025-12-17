"""
Unit tests for the Event class in NeuralDBG.

These tests focus on individual methods and components of the Event class,
ensuring each part functions correctly in isolation.
"""

import torch
import pytest
from neuraldbg import Event


class TestEventUnit:
    """Unit tests for individual Event methods."""

    def test_init_sets_attributes(self):
        """
        Test that __init__ correctly sets all attributes.

        Verifies that step, layer_name are set, tensors are cloned and detached,
        and gradient is initialized to None.
        """
        # Create sample tensors with gradients enabled
        input_tensor = torch.randn(3, 3, requires_grad=True)
        output_tensor = torch.randn(3, 3, requires_grad=True)

        # Initialize event
        event = Event(step=1, layer_name='conv1', input_tensor=input_tensor, output_tensor=output_tensor)

        # Verify basic attributes
        assert event.step == 1
        assert event.layer_name == 'conv1'

        # Verify tensors are copied correctly
        assert torch.equal(event.input, input_tensor)
        assert torch.equal(event.output, output_tensor)

        # Verify gradient initialization
        assert event.gradient is None

        # Verify tensors are detached and cloned (independent copies)
        assert not event.input.requires_grad, "Input tensor should be detached"
        assert not event.output.requires_grad, "Output tensor should be detached"
        assert event.input is not input_tensor, "Input should be a separate object"
        assert event.output is not output_tensor, "Output should be a separate object"

    def test_capture_gradient_with_grad(self):
        """
        Test capture_gradient when tensor has gradients.

        Ensures gradients are captured, detached, and cloned when available.
        """
        # Create tensor and compute gradients
        tensor = torch.randn(3, 3, requires_grad=True)
        loss = tensor.sum()
        loss.backward()

        # Create event and capture gradient
        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        # Verify gradient was captured
        assert event.gradient is not None, "Gradient should be captured when available"
        assert torch.equal(event.gradient, tensor.grad), "Captured gradient should match tensor.grad"

        # Verify gradient is detached and cloned
        assert not event.gradient.requires_grad, "Captured gradient should be detached"
        assert event.gradient is not tensor.grad, "Captured gradient should be a separate object"

    def test_capture_gradient_without_grad(self):
        """
        Test capture_gradient when tensor has no gradients.

        Ensures gradient remains None when no gradients are computed.
        """
        # Create tensor without computing gradients
        tensor = torch.randn(3, 3, requires_grad=False)

        # Create event and attempt to capture gradient
        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        # Verify gradient remains None
        assert event.gradient is None, "Gradient should remain None when tensor has no gradients"
