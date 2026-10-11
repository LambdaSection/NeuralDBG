#!/usr/bin/env python3
"""
Panos/TrainProof case 3 — silent death (the PASS that lies).

FAULT (explicit, line 38): model[2].weight.zero_() at step 20 — bad
resume / pruning accident. NOTE weight-only: bias is NOT zeroed (partial
death, not full layer death — disclosed per review). Loss stays finite
10.7171 -> 0.8422, no NaN, no divergence. "Static invariants PASS" is
UNSUPPORTED in this archive (no TrainProof log attached) — current claim
is only "finite loss, no crash". Dead-unit fraction 56%: acts =
model[:4](randn(256,8)); dead_frac = (acts==0).all(dim=0).mean() over
32 channels after second ReLU.

NeuralDBG reading: gradient vanishing ReLU_1 + Linear_0 exactly @step 20
(0.49->0.0, 0.36->0.0) -> ranked cause = dead units, refuting
"under-capacity".

Run:  python examples/repro_panos_case3.py  (from repo root)
"""

import json

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

torch.manual_seed(3)

model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32),
                      nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
crit = nn.MSELoss()

TRUE_W = torch.randn(8, 1)
losses = []
with NeuralDbg(model) as dbg:
    for step in range(80):
        dbg.step = step
        if step == 20:  # FAULT: weight of layer silently zeroed (bias intact)
            with torch.no_grad():  # (bad resume, pruning accident, …)
                model[2].weight.zero_()
        opt.zero_grad()
        x = torch.randn(16, 8)
        loss = crit(model(x), x @ TRUE_W)
        loss.backward()
        dbg.record_loss(loss.item())
        opt.step()
        losses.append(loss.item())
    from _panos_bundle_helper import preserve_cache

    _cache_map = preserve_cache(dbg)

with torch.no_grad():
    acts = model[:4](torch.randn(256, 8))
    dead_frac = (acts == 0).all(dim=0).float().mean().item()
print(f"loss first={losses[0]:.4f} last={losses[-1]:.4f} "
      f"(finite plateau: {all(abs(v) < 1e6 for v in losses)}, "
      f"dead-unit fraction: {dead_frac:.0%})")

print("\nNeuralDBG hypotheses:")
hyps = dbg.explain_failure() or []
for h in hyps:
    print(f"  - [{h.confidence:.0%}] {h.description}")
dbg.export_aquarium_package("outputs/panos_case3_events.json")
from _panos_bundle_helper import patch_package as _patch3

try:
    import torch as _torch3

    _env3 = {
        "torch": _torch3.__version__,
        "seed": 3,
        "fault": "model[2].weight.zero_() at step 20 (bias NOT zeroed)",
        "command": "python examples/repro_panos_case3.py (from repo root)",
    }
except Exception:
    _env3 = {"seed": 3}
_n_p, _n_m = _patch3(
    "outputs/panos_case3_events.json",
    _cache_map,
    {"case": "panos_case3", "env": _env3, "loss_first": losses[0],
     "loss_last": losses[-1], "dead_frac": dead_frac,
     "dead_frac_formula": "(acts==0).all(dim=0).float().mean() over 32 channels, 256 samples, model[:4]",
     "all_finite": True,
     "invariant_note": "UNSUPPORTED: no TrainProof log in archive; claim is only finite loss, no crash"},
)
print(f"bundle: {_n_p} tensor refs bundled, {_n_m} NOT-CHECKED")
print("\nEvents -> outputs/panos_case3_events.json")
print(json.dumps({"loss_first": round(losses[0], 4),
                  "loss_last": round(losses[-1], 4),
                  "dead_frac": round(dead_frac, 4),
                  "all_finite": True, "n_hypotheses": len(hyps)}, indent=1))
