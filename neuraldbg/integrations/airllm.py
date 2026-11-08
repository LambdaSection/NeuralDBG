"""
AirLLM Integration — NeuralDBG observer for layer-streamed inference.

AirLLM streams one decoder layer at a time (disk -> GPU -> meta) via
forward pre/post hooks on each big module. Wrapping that model directly
with ``NeuralDbg(model).__enter__()`` is wrong: the model lives on ``meta``
and auto leaf-hooks would see empty tensors.

This observer is non-intrusive:
- it never installs NeuralDbg auto-hooks on the streamed model,
- it uses NeuralDbg manual APIs (stats + record_*) on layer outputs,
- it times ``generate()`` to flag I/O stalls (issues #364/#374).

Usage:
    from neuraldbg.integrations.airllm import AirLLMObserver

    obs = AirLLMObserver(airllm_wrapper=model)  # or model=nn.Module
    with obs:
        out = model.generate(ids, max_new_tokens=20)
    print(obs.report()["summary_text"])
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

try:
    import airllm  # noqa: F401 — optional, only to flag availability

    HAS_AIRLLM = True
except ImportError:
    HAS_AIRLLM = False


class AirLLMObserver:
    """Non-intrusive NeuralDBG observer for AirLLM-streamed models."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        airllm_wrapper: Optional[Any] = None,
        family: str = "LLM",
        slow_token_threshold_s: float = 5.0,
        strict_mode: bool = False,
        **dbg_kwargs,
    ):
        """
        Args:
            model: plain nn.Module to observe (for tests / CPU).
            airllm_wrapper: AirLLMBaseModel instance (has .model, .layers,
                .generate). Takes precedence for layer discovery + timing.
            family: NeuralDBG family calibration (LLM by default, x2.0).
            slow_token_threshold_s: generate() slower than this is flagged
                as I/O stall (AirLLM issues #364/#374).
        """
        self.wrapper = airllm_wrapper
        self.family = family
        self.slow_token_threshold_s = slow_token_threshold_s

        inner = self._resolve_inner_model(model, airllm_wrapper)
        self.inner_model = inner

        self._dbg: Optional[NeuralDbg] = None
        if inner is not None:
            try:
                self._dbg = NeuralDbg(
                    inner, family=family, strict_mode=strict_mode, **dbg_kwargs
                )
            except Exception as e:  # noqa: BLE001 — never break inference
                warnings.warn(f"AirLLMObserver: NeuralDbg init failed: {e}")
                self._dbg = None

        self._handles: List[Any] = []
        self._layer_stats: Dict[str, Dict[str, float]] = {}
        self._timings: List[Dict[str, Any]] = []
        self._slow_calls = 0
        self._step = 0
        self._orig_generate = None
        self._attached = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AirLLMObserver":
        self.attach()
        return self

    def __exit__(self, *args) -> None:
        self.detach()

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self) -> "AirLLMObserver":
        """Attach output-observation hooks AFTER AirLLM streaming hooks."""
        if self._attached:
            return self
        layers = self._discover_layers()
        for name, mod in layers:
            try:
                h = mod.register_forward_hook(self._make_layer_hook(name))
                self._handles.append(h)
            except Exception:  # noqa: BLE001 — one bad layer must not block
                continue
        self._wrap_generate()
        self._attached = True
        return self

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        if self.wrapper is not None and self._orig_generate is not None:
            try:
                self.wrapper.generate = self._orig_generate
            except Exception:  # noqa: BLE001
                pass
            self._orig_generate = None
        self._attached = False

    # ------------------------------------------------------------------
    # Core observation (testable without hooks)
    # ------------------------------------------------------------------

    def observe_layer_output(
        self, layer_name: str, tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Inspect one layer output. Returns stats dict, records events."""
        self._step += 1
        stats: Dict[str, Any] = {"layer": layer_name, "step": self._step}

        if not isinstance(tensor, torch.Tensor):
            stats["skipped"] = "not_a_tensor"
            return stats
        if not torch.is_floating_point(tensor):
            stats["skipped"] = "non_float"
            return stats
        try:
            has_nan = bool(torch.isnan(tensor).any().item())
            has_inf = bool(torch.isinf(tensor).any().item())
        except Exception:  # noqa: BLE001 — meta tensors can raise
            stats["skipped"] = "meta_or_empty"
            return stats

        stats["has_nan"] = has_nan
        stats["has_inf"] = has_inf

        if has_nan and self._dbg is not None:
            try:
                self._dbg.record_gradient_anomaly(
                    layer_name,
                    kind="nan",
                    metadata={"source": "airllm", "step": self._step},
                )
            except Exception:  # noqa: BLE001
                pass
        if has_inf and self._dbg is not None:
            try:
                self._dbg.record_gradient_anomaly(
                    layer_name,
                    kind="inf",
                    metadata={"source": "airllm", "step": self._step},
                )
            except Exception:  # noqa: BLE001
                pass

        if self._dbg is not None:
            try:
                act = self._dbg._compute_activation_stats(tensor)
                stats.update(act)
                self._layer_stats[layer_name] = dict(act)
            except Exception:  # noqa: BLE001
                pass
        return stats

    def step_loss(self, loss_value: float) -> None:
        """Record a training loss (future AirLLMLoRA.train_step path)."""
        self._step += 1
        if self._dbg is None:
            return
        try:
            self._dbg.record_loss(float(loss_value))
            self._dbg.step_iteration()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_events(self) -> List:
        if self._dbg is None:
            return []
        try:
            return self._dbg.get_events()
        except Exception:  # noqa: BLE001
            return []

    def report(self) -> Dict[str, Any]:
        events = self.get_events()
        anomaly_kw = (
            "nan",
            "inf",
            "vanishing",
            "exploding",
            "dead",
            "saturated",
            "anomalous",
            "diverging",
            "spike",
        )
        anomalies = [
            e
            for e in events
            if any(
                k in str(getattr(e, "to_state", "")).lower()
                or k in str(getattr(e, "event_type", "")).lower()
                for k in anomaly_kw
            )
        ]
        slow = [
            t
            for t in self._timings
            if t.get("elapsed_s", 0) >= self.slow_token_threshold_s
        ]
        if not anomalies and not slow:
            text = (
                f"AirLLMObserver: healthy — {len(events)} events, "
                f"{len(self._timings)} generate() calls, 0 stalls."
            )
        else:
            text = (
                f"AirLLMObserver: {len(anomalies)} anomalies / "
                f"{len(events)} events, {len(slow)} slow generate() "
                f"(>={self.slow_token_threshold_s}s). "
                f"Check first-failure layer + I/O timings."
            )
        return {
            "summary_text": text,
            "total_events": len(events),
            "anomaly_events": len(anomalies),
            "generate_calls": len(self._timings),
            "slow_calls": len(slow),
            "timings": list(self._timings[-10:]),
            "layers_observed": sorted(self._layer_stats.keys()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_inner_model(model, wrapper) -> Optional[nn.Module]:
        if wrapper is not None:
            inner = getattr(wrapper, "model", None)
            if isinstance(inner, nn.Module):
                return inner
            if isinstance(wrapper, nn.Module):
                return wrapper
        return model if isinstance(model, nn.Module) else None

    def _discover_layers(self) -> List[Any]:
        """Prefer wrapper.layers (AirLLM big modules), else leaf modules."""
        if self.wrapper is not None:
            layers = getattr(self.wrapper, "layers", None)
            names = getattr(self.wrapper, "layer_names", None)
            if isinstance(layers, (list, tuple)) and layers:
                out = []
                for i, mod in enumerate(layers):
                    if isinstance(mod, nn.Module):
                        n = names[i] if names and i < len(names) else f"layer_{i}"
                        out.append((str(n), mod))
                if out:
                    return out
        if self.inner_model is not None:
            return [
                (n if n else "root", m)
                for n, m in self.inner_model.named_modules()
                if isinstance(m, nn.Module) and not list(m.children())
            ][:32]
        return []

    def _make_layer_hook(self, layer_name: str):
        def _hook(module, args, output):
            try:
                tensor = output[0] if isinstance(output, tuple) else output
                # Unwrap (hidden, extra) tuples; only inspect tensors
                if isinstance(tensor, torch.Tensor):
                    self.observe_layer_output(layer_name, tensor.detach())
            except Exception:  # noqa: BLE001 — observer never breaks forward
                pass
            return output

        return _hook

    def _wrap_generate(self) -> None:
        w = self.wrapper
        if w is None or not hasattr(w, "generate"):
            return
        if self._orig_generate is not None:
            return
        try:
            self._orig_generate = w.generate

            def _timed_generate(*args, **kwargs):
                t0 = time.time()
                try:
                    return self._orig_generate(*args, **kwargs)
                finally:
                    el = time.time() - t0
                    self._timings.append(
                        {
                            "elapsed_s": el,
                            "step": self._step,
                            "slow": el >= self.slow_token_threshold_s,
                        }
                    )
                    if el >= self.slow_token_threshold_s:
                        self._slow_calls += 1

            w.generate = _timed_generate
        except Exception:  # noqa: BLE001
            self._orig_generate = None
