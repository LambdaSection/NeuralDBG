"""
NeuralDbg Causal Inference Engine

This module defines the NeuralDbg class, which is a causal inference engine for deep learning training dynamics.
It extracts semantic events from training, compresses them into causal patterns, and provides
post-mortem reasoning about training failures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    """Types of semantic events that can occur during training."""
    GRADIENT_HEALTH_TRANSITION = "gradient_health_transition"
    ACTIVATION_REGIME_SHIFT = "activation_regime_shift"
    OPTIMIZER_INSTABILITY = "optimizer_instability"
    DATA_ANOMALY = "data_anomaly"

class GradientHealth(Enum):
    """Gradient health states."""
    HEALTHY = "healthy"
    VANISHING = "vanishing"
    EXPLODING = "exploding"
    SATURATED = "saturated"

@dataclass
class SemanticEvent:
    """
    Represents a meaningful transition in training dynamics.

    Unlike raw tensor snapshots, semantic events capture high-level changes
    that are relevant for causal inference.
    """
    event_type: EventType
    layer_name: str
    step: int
    from_state: Any
    to_state: Any
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class CausalHypothesis:
    """A ranked hypothesis about the cause of a training failure."""
    description: str
    confidence: float
    evidence: List[SemanticEvent]
    causal_chain: List[str]

class NeuralDbg:
    """
    Causal inference engine for deep learning training dynamics.

    Monitors training loops to extract semantic events, detect patterns,
    and provide post-mortem explanations for training failures.
    """

    def __init__(self, model: nn.Module, threshold_vanishing: float = 1e-6, threshold_exploding: float = 1e3):
        """
        Initialize the causal inference engine.

        Args:
            model: The PyTorch model to monitor
            threshold_vanishing: Gradient norm threshold for vanishing detection
            threshold_exploding: Gradient norm threshold for exploding detection
        """
        self.model = model
        self.threshold_vanishing = threshold_vanishing
        self.threshold_exploding = threshold_exploding

        # Semantic event storage (not tensors!)
        self.events: List[SemanticEvent] = []

        # Previous state tracking for transition detection
        self.previous_gradient_norms: Dict[str, float] = {}
        self.previous_activation_stats: Dict[str, Dict[str, float]] = {}

        # Hook storage for automatic monitoring
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Training state
        self.step = 0
        self.is_monitoring = False

    def __enter__(self):
        """Start monitoring the training loop."""
        self._install_hooks()
        self.is_monitoring = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and cleanup."""
        self._remove_hooks()
        self.is_monitoring = False

    def _install_hooks(self):
        """Install forward and backward hooks to extract semantic events."""
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            """Extract semantic events from forward pass."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # Extract activation regime information
            if isinstance(output, torch.Tensor):
                activation_stats = self._compute_activation_stats(output)

                # Detect activation regime shifts
                if layer_name in self.previous_activation_stats:
                    prev_stats = self.previous_activation_stats[layer_name]
                    shift = self._detect_activation_shift(prev_stats, activation_stats)
                    if shift:
                        event = SemanticEvent(
                            event_type=EventType.ACTIVATION_REGIME_SHIFT,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=prev_stats,
                            to_state=activation_stats,
                            confidence=shift['confidence'],
                            metadata=shift
                        )
                        self.events.append(event)

                self.previous_activation_stats[layer_name] = activation_stats

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            """Extract semantic events from backward pass."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # Extract gradient health information
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()

                # Detect gradient health transitions
                if layer_name in self.previous_gradient_norms:
                    prev_norm = self.previous_gradient_norms[layer_name]
                    transition = self._detect_gradient_transition(prev_norm, grad_norm)
                    if transition:
                        event = SemanticEvent(
                            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=self._classify_gradient_health(prev_norm),
                            to_state=self._classify_gradient_health(grad_norm),
                            confidence=transition['confidence'],
                            metadata={
                                'prev_norm': prev_norm,
                                'current_norm': grad_norm,
                                'transition_type': transition['type']
                            }
                        )
                        self.events.append(event)

                self.previous_gradient_norms[layer_name] = grad_norm

        # Install hooks on all modules (including root for full coverage)
        for name, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(forward_hook))
            self.hooks.append(module.register_backward_hook(backward_hook))

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _get_layer_name(self, module: nn.Module) -> str:
        """Get the name of a module from the model."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name or "root"
        return "unknown"

    def _compute_activation_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistical summary of activation tensor."""
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'sparsity': (tensor == 0).float().mean().item(),
            'norm': tensor.norm().item()
        }

    def _detect_activation_shift(self, prev_stats: Dict[str, float], current_stats: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect significant shifts in activation patterns."""
        # Simple heuristic: large change in sparsity or norm indicates regime shift
        sparsity_change = abs(current_stats['sparsity'] - prev_stats['sparsity'])
        norm_change = abs(current_stats['norm'] - prev_stats['norm']) / max(prev_stats['norm'], 1e-6)

        if sparsity_change > 0.1 or norm_change > 0.5:  # Thresholds for detection
            confidence = min(sparsity_change * 10, norm_change * 2, 1.0)
            return {
                'confidence': confidence,
                'sparsity_change': sparsity_change,
                'norm_change': norm_change
            }
        return None

    def _classify_gradient_health(self, norm: float) -> GradientHealth:
        """Classify gradient health based on norm."""
        if norm < self.threshold_vanishing:
            return GradientHealth.VANISHING
        elif norm > self.threshold_exploding:
            return GradientHealth.EXPLODING
        elif norm > 1.0:  # Lower threshold for saturation
            return GradientHealth.SATURATED
        else:
            return GradientHealth.HEALTHY

    def _detect_gradient_transition(self, prev_norm: float, current_norm: float) -> Optional[Dict[str, Any]]:
        """Detect transitions in gradient health."""
        prev_health = self._classify_gradient_health(prev_norm)
        current_health = self._classify_gradient_health(current_norm)

        if prev_health != current_health:
            # Calculate confidence based on magnitude of change
            if prev_norm > 0:
                ratio = abs(current_norm - prev_norm) / prev_norm
            else:
                ratio = abs(current_norm)  # Handle zero case

            confidence = min(ratio * 0.1, 1.0)  # Scale down the confidence

            return {
                'type': f"{prev_health.value}_to_{current_health.value}",
                'confidence': confidence
            }
        return None

    def explain_failure(self, failure_type: str = "vanishing_gradients") -> List[CausalHypothesis]:
        """
        Provide ranked causal hypotheses for a training failure.

        Args:
            failure_type: Type of failure to explain

        Returns:
            List of ranked hypotheses with confidence scores
        """
        hypotheses = []

        if failure_type == "vanishing_gradients":
            hypotheses = self._explain_vanishing_gradients()

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    def _explain_vanishing_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for vanishing gradient failures."""
        hypotheses = []

        # Find first vanishing gradient event
        vanishing_events = [e for e in self.events
                          if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                          and e.to_state == GradientHealth.VANISHING]

        if not vanishing_events:
            return hypotheses

        first_vanishing = min(vanishing_events, key=lambda e: e.step)

        # Hypothesis 1: Originated in this layer
        hypotheses.append(CausalHypothesis(
            description=f"Gradient vanishing originated in layer '{first_vanishing.layer_name}' at step {first_vanishing.step}",
            confidence=first_vanishing.confidence,
            evidence=[first_vanishing],
            causal_chain=[f"Vanishing detected in {first_vanishing.layer_name}"]
        ))

        # Hypothesis 2: Check for activation saturation coupling
        saturation_events = [e for e in self.events
                           if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
                           and e.step <= first_vanishing.step + 10]  # Nearby in time

        if saturation_events:
            nearest_sat = min(saturation_events, key=lambda e: abs(e.step - first_vanishing.step))
            hypotheses.append(CausalHypothesis(
                description=f"Gradient vanishing likely due to LR × activation mismatch - saturation in '{nearest_sat.layer_name}' preceded vanishing",
                confidence=min(first_vanishing.confidence, nearest_sat.confidence),
                evidence=[first_vanishing, nearest_sat],
                causal_chain=[
                    f"Saturation in {nearest_sat.layer_name} at step {nearest_sat.step}",
                    f"Led to vanishing gradients in {first_vanishing.layer_name} at step {first_vanishing.step}"
                ]
            ))

        return hypotheses

    def get_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Get all current causal hypotheses."""
        return self.explain_failure()

    def trace_causal_chain(self, event_type: str) -> List[str]:
        """Trace the causal chain for a specific type of event."""
        # Simple implementation - in practice this would be more sophisticated
        relevant_events = [e for e in self.events if e.event_type.value == event_type]
        return [f"{e.layer_name} at step {e.step}: {e.metadata}" for e in relevant_events]

    def detect_coupled_failures(self) -> List[Dict[str, Any]]:
        """Detect coupled failures (events that occur together)."""
        couplings = []

        # Simple coupling detection: events within 5 steps of each other
        for i, event1 in enumerate(self.events):
            for event2 in self.events[i+1:]:
                if abs(event1.step - event2.step) <= 5 and event1.layer_name != event2.layer_name:
                    couplings.append({
                        'event1': f"{event1.event_type.value} in {event1.layer_name}",
                        'event2': f"{event2.event_type.value} in {event2.layer_name}",
                        'step_difference': abs(event1.step - event2.step),
                        'confidence': min(event1.confidence, event2.confidence)
                    })

        return couplings
