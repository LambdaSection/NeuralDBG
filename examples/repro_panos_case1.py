#!/usr/bin/env python3
"""
Panos/TrainProof case 1 — "DIVERGE that isn't".

One corrupted batch (inputs x8.0) at step 1204 of an otherwise healthy run.
Train loss spikes x446 at step 1204 (0.0209 -> 9.338); loss[1205]=6.516
is still 311x baseline (NOT recovered next step). First back within 3x
is step 1242 (within 5x: step 1221). Held-out on clean data keeps
improving (7.4041 -> 0.0162, probe every 50 steps, stdout + JSON).

Deterministic reading (loss series only): DIVERGE / strong FAIL.
NeuralDBG reading: optimizer_instability loss_spike @1204 +
loss_spike->stable @1205 (threshold artefact at 6.5, not true recovery);
NO data_anomaly @1204. Top hypotheses rank step 1 / step 946, NOT the
injected 1204 fault — ranking limitation, disclosed.

Run:  python examples/repro_panos_case1.py  (from repo root)
"""

import json

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

torch.manual_seed(7)

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16),
                      nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
crit = nn.MSELoss()

TRUE_W = torch.randn(8, 1)
losses, heldout = [], []
heldout_steps = []
with NeuralDbg(model) as dbg:
    for step in range(1250):
        dbg.step = step
        opt.zero_grad()
        x = torch.randn(8, 8)
        if step == 1204:  # FAULT: one corrupted batch (large but survivable)
            x = x * 8.0
        y = x @ TRUE_W + 0.01 * torch.randn(8, 1)
        loss = crit(model(x), y)
        loss.backward()
        dbg.record_loss(loss.item())
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:  # held-out probe: clean data, no grad
            with torch.no_grad():
                xh = torch.randn(64, 8)
                heldout.append(crit(model(xh), xh @ TRUE_W).item())
                heldout_steps.append(step)
    # Preserve tensor_cache BEFORE __exit__ cleanup deletes it (see helper).
    from _panos_bundle_helper import preserve_cache

    _cache_map = preserve_cache(dbg)

spike = losses[1204] / max(losses[1203], 1e-12)
print(f"loss[1203]={losses[1203]:.4f} loss[1204]={losses[1204]:.4f} "
      f"(x{spike:.0f}) loss[1205]={losses[1205]:.4f}")
print(f"held-out improving: {heldout[-1] < heldout[0]} "
      f"({heldout[0]:.4f} -> {heldout[-1]:.4f})")

print("\nNeuralDBG hypotheses:")
hyps = dbg.explain_failure() or []
for h in hyps:
    print(f"  - [{h.confidence:.0%}] {h.description}")
dbg.export_aquarium_package("outputs/panos_case1_events.json")
# Patch to reproducible bundle: relative tensor paths + held-out series.
from _panos_bundle_helper import patch_package

try:
    import torch as _torch

    _env = {
        "torch": _torch.__version__,
        "seed": 7,
        "fault": "inputs x8.0 at step 1204 only",
        "command": "python examples/repro_panos_case1.py (from repo root)",
    }
except Exception:
    _env = {"seed": 7}
_n_p, _n_m = patch_package(
    "outputs/panos_case1_events.json",
    _cache_map,
    {
        "case": "panos_case1",
        "env": _env,
        "loss_1203": losses[1203],
        "loss_1204": losses[1204],
        "loss_1205": losses[1205],
        "spike_factor": spike,
        "heldout_steps": heldout_steps,
        "heldout_history": heldout,
        "heldout_first": heldout[0],
        "heldout_last": heldout[-1],
        "recovery_note": "loss[1205]=6.516 still 311x baseline; first back within 3x is step 1242",
    },
)
print(f"bundle: {_n_p} tensor refs bundled, {_n_m} NOT-CHECKED")
print("\nEvents -> outputs/panos_case1_events.json")
print(json.dumps({"spike_factor": round(spike, 1),
                  "heldout_first": round(heldout[0], 4),
                  "heldout_last": round(heldout[-1], 4),
                  "n_hypotheses": len(hyps)}, indent=1))
