#!/usr/bin/env python3
"""
Panos/TrainProof case 2b — NaN from corrupted inputs (healthy optimizer).

Same log symptom as case 2a (loss=NaN from step K) but the cause lives in
the data: one batch carries NaN inputs while gradients were healthy until
contamination. The fix is opposite (filter the source, keep LR).
Evidence: nan_detected Linear_0 @step 25 (nan_count 1) fanning same-step
to ReLU_1 / Linear_2 / ReLU_3 / Linear_4 / root, optimizer diverging same
step [...5.16, NaN] as consequence — zero gradient prodrome. Caveat:
spurious distribution_shift root @step 1 (seed 11 warmup); tensor_cache
.pt files evicted on context exit (patched to relative or NOT-CHECKED).

Run:  python examples/repro_panos_case2b.py  (from repo root)
"""

import json

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

torch.manual_seed(11)  # same seed as 2a: only the fault differs

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16),
                      nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.01)  # sane optimizer
crit = nn.MSELoss()

TRUE_W = torch.randn(8, 1)
nan_step = None
with NeuralDbg(model) as dbg:
    for step in range(60):
        dbg.step = step
        opt.zero_grad()
        x = torch.randn(8, 8)
        if step == 25:  # FAULT: NaN injected in the inputs
            x[0, 0] = float("nan")
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
dbg.export_aquarium_package("outputs/panos_case2b_events.json")
from _panos_bundle_helper import patch_package as _patch2b

try:
    import torch as _torch2b

    _env2b = {
        "torch": _torch2b.__version__,
        "seed": 11,
        "fault": "NaN in input batch at step 25",
        "command": "python examples/repro_panos_case2b.py (from repo root)",
    }
except Exception:
    _env2b = {"seed": 11}
_n_p, _n_m = _patch2b(
    "outputs/panos_case2b_events.json",
    _cache_map,
    {"case": "panos_case2b", "env": _env2b, "nan_step": nan_step,
     "cause": "NaN in input batch (step 25)", "optimizer": "sane (LR=0.01)"},
)
print(f"bundle: {_n_p} tensor refs bundled, {_n_m} NOT-CHECKED")
print("\nEvents -> outputs/panos_case2b_events.json")
print(json.dumps({"nan_step": nan_step, "cause": "NaN in input batch (step 25)",
                  "optimizer": "sane (LR=0.01)", "n_hypotheses": len(hyps)},
                 indent=1))
