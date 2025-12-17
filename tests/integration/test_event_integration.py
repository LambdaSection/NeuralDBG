"""
Integration tests for the Event class in NeuralDBG.

These tests verify how Event objects work together with PyTorch's
autograd system in simulated neural network scenarios, testing
the interaction between forward passes, backward passes, and
gradient capture.
"""

import torch
import pytest
from neuraldbg import Event


class TestEventIntegration:
    """Integration tests for Event in neural network contexts."""

    def test_forward_backward_simulation(self):
        """
        Test Events capturing forward and backward passes in a simulated network.

        Simulates a linear layer computation with forward pass and gradient computation,
        then verifies that Event correctly captures all relevant information.
        """
        # Simulate input parameters for a linear layer: y = Wx + b
        batch_size, input_dim = 10, 5
        output_dim = 3

        # Create model parameters (weights and bias)
        input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
        weight = torch.randn(output_dim, input_dim, requires_grad=True)
        bias = torch.randn(output_dim, requires_grad=True)

        # Forward pass: compute output
        output_tensor = torch.matmul(input_tensor, weight.t()) + bias

        # Enable gradient capture on output tensor (non-leaf node)
        output_tensor.retain_grad()

        # Backward pass: compute gradients
        loss = output_tensor.sum()
        loss.backward()

        # Create event to capture this computation step
        forward_event = Event(
            step=1,
            layer_name='linear',
            input_tensor=input_tensor,
            output_tensor=output_tensor
        )

        # Capture gradients after backward pass
        forward_event.capture_gradient(output_tensor)

        # Verify event captured all information correctly
        assert forward_event.step == 1, "Step should be set correctly"
        assert forward_event.layer_name == 'linear', "Layer name should be set correctly"

        # Verify input/output tensors match original computation
        assert torch.equal(forward_event.input, input_tensor), "Input tensor should be captured"
        assert torch.equal(forward_event.output, output_tensor), "Output tensor should be captured"

        # Verify gradients were captured
        assert forward_event.gradient is not None, "Gradient should be captured after backward pass"
        assert torch.equal(forward_event.gradient, output_tensor.grad), "Captured gradient should match computed gradient"

        # Verify captured tensors are independent (detached and cloned)
        assert not forward_event.input.requires_grad, "Captured input should be detached"
        assert not forward_event.output.requires_grad, "Captured output should be detached"
        assert not forward_event.gradient.requires_grad, "Captured gradient should be detached"
