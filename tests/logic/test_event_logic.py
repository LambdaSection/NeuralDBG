"""
Logic tests for the Event class in NeuralDBG.

These tests focus on edge cases, boundary conditions, and specific
behaviors that ensure the Event class handles various scenarios
correctly, including tensor independence, multiple gradient captures,
and error conditions.
"""

import torch
import pytest
from neuraldbg import Event


class TestEventLogic:
    """Logic tests for edge cases and specific behaviors."""

    def test_tensor_independence(self):
        """
        Test that Event's tensors are independent of original tensors.

        Ensures that modifications to original tensors don't affect
        the Event's stored copies, maintaining data integrity.
        """
        # Create original tensors
        original_input = torch.randn(2, 2, requires_grad=True)
        original_output = torch.randn(2, 2, requires_grad=True)

        # Create event (this clones and detaches the tensors)
        event = Event(step=1, layer_name='test', input_tensor=original_input, output_tensor=original_output)

        # Create working copies of originals to modify
        original_input_copy = original_input.clone()
        original_output_copy = original_output.clone()

        # Modify the copies (simulate changes in the original computation)
        original_input_copy.add_(1.0)  # Add 1 to all elements
        original_output_copy.mul_(2.0)  # Multiply all elements by 2

        # Verify Event's tensors remain unchanged
        assert torch.equal(event.input, original_input), "Event input should match original before modification"
        assert torch.equal(event.output, original_output), "Event output should match original before modification"

        # Verify Event's tensors differ from modified copies
        assert not torch.equal(event.input, original_input_copy), "Event input should differ from modified copy"
        assert not torch.equal(event.output, original_output_copy), "Event output should differ from modified copy"

    def test_gradient_capture_multiple_times(self):
        """
        Test capturing gradients multiple times updates correctly.

        Ensures that calling capture_gradient multiple times properly
        updates the stored gradient with the latest computation.
        """
        # Create tensor for gradient computation
        tensor = torch.randn(3, requires_grad=True)

        # First backward pass with sum loss (gradients = ones)
        loss1 = tensor.sum()
        loss1.backward()

        # Create event and capture initial gradients
        event = Event(step=1, layer_name='test', input_tensor=tensor, output_tensor=tensor)
        event.capture_gradient(tensor)

        # Store initial gradient for comparison
        first_grad = event.gradient.clone()

        # Second backward pass with different loss (gradients = 2 * tensor)
        tensor.grad.zero_()  # Reset gradients
        loss2 = (tensor ** 2).sum()
        loss2.backward()

        # Capture updated gradients
        event.capture_gradient(tensor)

        # Verify gradient was updated
        assert not torch.equal(event.gradient, first_grad), "Gradient should be updated after second capture"
        assert torch.equal(event.gradient, tensor.grad), "Captured gradient should match current tensor.grad"

    def test_capture_gradient_on_different_tensor(self):
        """
        Test capturing gradient from a tensor that hasn't computed gradients.

        Ensures that capture_gradient only stores gradients when they
        actually exist, preventing incorrect gradient storage.
        """
        # Create two tensors - one with gradients, one without
        output_tensor = torch.randn(3, requires_grad=True)
        other_tensor = torch.randn(3, requires_grad=True)  # No backward pass on this one

        # Compute gradients only for output_tensor
        loss = output_tensor.sum()
        loss.backward()

        # Create event
        event = Event(step=1, layer_name='test', input_tensor=output_tensor, output_tensor=output_tensor)

        # Attempt to capture gradient from tensor without gradients
        event.capture_gradient(other_tensor)

        # Verify no gradient was captured
        assert event.gradient is None, "Gradient should remain None when capturing from tensor without gradients"
