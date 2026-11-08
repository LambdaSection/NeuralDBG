"""CPU tests for AirLLM observer — no GPU, no download, no airllm dep."""

import time

import torch
import torch.nn as nn

from neuraldbg.integrations.airllm import AirLLMObserver


class _FakeWrapper:
    """Minimal AirLLMBaseModel stand-in: .model/.layers/.generate."""

    def __init__(self):
        self.model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.layers = list(self.model)
        self.layer_names = [f"layer_{i}" for i in range(len(self.layers))]
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        with torch.no_grad():
            x = torch.randn(1, 4)
            return self.model(x)


def test_observe_healthy_records_no_nan():
    model = nn.Linear(4, 2)
    obs = AirLLMObserver(model=model)
    stats = obs.observe_layer_output("l0", torch.randn(2, 4))
    assert stats["has_nan"] is False
    assert stats["has_inf"] is False
    assert obs.report()["anomaly_events"] == 0


def test_observe_nan_and_inf_raise_events():
    model = nn.Linear(4, 2)
    obs = AirLLMObserver(model=model)
    nan_t = torch.tensor([[1.0, float("nan")]])
    inf_t = torch.tensor([[1.0, float("inf")]])
    obs.observe_layer_output("dec.layer_3", nan_t)
    obs.observe_layer_output("dec.layer_5", inf_t)
    rep = obs.report()
    assert rep["total_events"] >= 2
    assert rep["anomaly_events"] >= 2


def test_step_loss_does_not_crash():
    model = nn.Linear(4, 2)
    obs = AirLLMObserver(model=model)
    for v in (1.0, 0.8, 0.6):
        obs.step_loss(v)
    assert obs.report()["total_events"] >= 0


def test_attach_wrap_generate_and_detach():
    w = _FakeWrapper()
    obs = AirLLMObserver(airllm_wrapper=w, slow_token_threshold_s=100.0)
    with obs:
        assert len(obs._handles) > 0
        out = w.generate()
        assert out.shape == (1, 2)
        # forward hooks fired during generate -> layers observed
        assert obs.report()["generate_calls"] == 1
        assert obs.report()["slow_calls"] == 0
    # after exit, generate restored and hooks removed
    assert obs._handles == []
    out2 = w.generate()
    assert out2.shape == (1, 2)
    assert obs.report()["generate_calls"] == 1  # no more timing


def test_slow_generate_flagged():
    class _Slow:
        def __init__(self):
            self.model = nn.Linear(2, 2)
            self.layers = [self.model]
            self.layer_names = ["slow_layer"]

        def generate(self, *a, **k):
            time.sleep(0.05)
            return torch.zeros(1, 2)

    w = _Slow()
    obs = AirLLMObserver(airllm_wrapper=w, slow_token_threshold_s=0.01)
    with obs:
        w.generate()
    assert obs.report()["slow_calls"] == 1
