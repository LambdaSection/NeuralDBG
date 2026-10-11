#!/usr/bin/env python3
"""
Panos/TrainProof case 2a — NaN from gradient overflow (huge LR).

Same log symptom as case 2b (loss=NaN from step K) but the cause lives in
the optimizer/layer, not the data: LR=5.0 explodes gradients in Linear_4
(first, @step 2: 33.6 -> 1.95B) until overflow (inf @step 3, diverging
[8.3, 2262, 7.6e18, NaN]). NeuralDBG localizes exploding grads BEFORE
the NaN, with clean inputs. Caveat: spurious data distribution_shift
root @step 1 (seed 11 warmup); per-step grad series not in archive;
tensor_cache .pt files evicted by disk_cache.cleanup on context exit
(see bundle script — patched to relative paths or NOT-CHECKED).

Run:  python examples/repro_panos_case2a.py  (from repo root)
"""

import json

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

torch.manual_seed(11)

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16),
                      nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=5.0)  # FAULT: absurd LR
crit = nn.MSELoss()

TRUE_W = torch.randn(8, 1)
nan_step = None
with NeuralDbg(model) as dbg:
    for step in range(200):
        dbg.step = step
        opt.zero_grad()
        x = torch.randn(8, 8)  # data always clean
        loss = crit(model(x), x @ TRUE_W)
        if not torch.isfinite(loss):
            nan_step = step
            dbg.record_loss(float("nan"))
            break
        loss.backward()
        dbg.record_loss(loss.item())
        opt.step()
    from _panos_bundle_helper import preserve_cache

    _cache_map = preserve_cache(dbg)

print(f"first NaN loss at step: {nan_step}")

print("\nNeuralDBG hypotheses:")
hyps = dbg.explain_failure() or []
for h in hyps:
    print(f"  - [{h.confidence:.0%}] {h.description}")
dbg.export_aquarium_package("outputs/panos_case2a_events.json")
from _panos_bundle_helper import patch_package as _patch2a

try:
    import torch as _torch2a

    _env2a = {
        "torch": _torch2a.__version__,
        "seed": 11,
        "fault": "LR=5.0, clean data",
        "command": "python examples/repro_panos_case2a.py (from repo root)",
    }
except Exception:
    _env2a = {"seed": 11}
_n_p, _n_m = _patch2a(
    "outputs/panos_case2a_events.json",
    _cache_map,
    {"case": "panos_case2a", "env": _env2a, "nan_step": nan_step,
     "cause": "gradient overflow (LR=5.0)", "data": "clean"},
)
print(f"bundle: {_n_p} tensor refs bundled, {_n_m} NOT-CHECKED")
print("\nEvents -> outputs/panos_case2a_events.json")
print(json.dumps({"nan_step": nan_step, "cause": "gradient overflow (LR=5.0)",
                  "data": "clean", "n_hypotheses": len(hyps)}, indent=1))
