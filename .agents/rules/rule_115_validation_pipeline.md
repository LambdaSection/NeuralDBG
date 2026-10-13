---
name: rule-115-validation-pipeline
description: 'RULE 115: Validation Pipeline — Progressive Gates (MANDATORY)'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_115_validation_pipeline.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 115: Validation Pipeline — Progressive Gates (MANDATORY)

## Purpose
NeuralDBG must prove it is NOT overfitted to its training data. The only way to do this is through a **progressive, gated validation pipeline** where each stage must pass before the next begins. No stage can be skipped. No result can be hand-waved.

---

## The 5-Stage Pipeline

```
STAGE 1          STAGE 2           STAGE 3              STAGE 4           STAGE 5
Fuzzer      →    Stress       →    Combinatorial   →    OOS          →    Competitive
(discover)       (resilience)       (coverage)           (generalize)       (superiority)
```

### Stage 1: Architecture Fuzzer (DISCOVER)
**Question**: Can NeuralDBG handle architectures it has NEVER seen?
**Method**: Randomly generate 50+ valid PyTorch architectures, inject 7 bug types, train, detect.
**Gate**: ≥ 80% detection rate on fuzzer-generated architectures.
**If fail**: New architecture patterns found → add to training data → retrain → re-fuzz.
**Output**: `fuzzer_results.json` + new black-swan family candidates.

### Stage 2: Stress Test Suite (RESILIENCE)
**Question**: Does NeuralDBG crash, false-positive, or miss under extreme conditions?
**Method**: 15+ stress scenarios (extreme gradients, 100-layer nets, mixed precision, NaN cascade, etc.).
**Gate**: 15/15 (100%) — NO crashes, NO false positives on healthy baseline.
**If fail**: Fix the crash/FP before ANY other work.
**Output**: `stress_results.json` + new stress scenarios discovered by Fuzzer.

### Stage 3: Detection Combinatorial (COVERAGE)
**Question**: Across ALL known architecture families × ALL bug types, what's the detection rate?
**Method**: Cross product of {MLP, CNN, RNN, Transformer, Hybrid, BlackSwan} × {exploding, vanishing, nan, zero_init, dead_bias, divergence, mixed_precision} × 10 configs each.
**Gate**: ≥ 85% overall, ≥ 70% per family.
**If fail**: Weakest family gets priority tuning.
**Output**: `combinatorial_results.json`.

### Stage 4: Out-of-Sample Validation (GENERALIZE) 🔴 ANTI-OVERFITTING GATE
**Question**: Does NeuralDBG work on a REAL architecture it was NEVER trained on?
**Method**: Pick a torchvision/timm model (ResNet, ViT, EfficientNet) NOT in any training data. Run 6 bug scenarios + healthy baseline.
**Gate**: 6/6 (100%) detection, ≤ 10 FP events on healthy baseline.
**If fail**: NeuralDBG IS overfitted. STOP everything. Fix overfitting before proceeding.
**Output**: `oos_validation_report.json`.

### Stage 5: Competitive Benchmark (SUPERIORITY)
**Question**: Is NeuralDBG OBJECTIVELY better than ALL alternatives?
**Method**: Run identical failure scenarios through NeuralDBG AND real instances of W&B, TensorBoard, Captum, torch.autograd.anomaly_mode.
**Gate**: ≥ 50% better than the BEST competitor on detection rate AND root cause identification AND time-to-diagnosis.
**If fail**: Identify specific gap, close it, re-benchmark.
**Output**: `benchmark_comparison.json` + `benchmark_comparison.html`.

---

## Pipeline Automation

The ENTIRE pipeline MUST be runnable with ONE command:
```bash
python run_validation_pipeline.py --all
```

Individual stages:
```bash
python run_validation_pipeline.py --stage fuzzer
python run_validation_pipeline.py --stage stress
python run_validation_pipeline.py --stage combinatorial
python run_validation_pipeline.py --stage oos
python run_validation_pipeline.py --stage benchmark
```

The pipeline script MUST:
1. Run stages in order (1→5)
2. HALT if any stage fails its gate
3. Produce a single `pipeline_report.json` with all results
4. Update `PLAN.md` dashboard automatically

---

## New Families Discovery Rule

When the Fuzzer or Stress Test discovers a NEW failure mode:
1. It becomes a **candidate family** in the Black-Swan catalog
2. It gets added to the Combinatorial matrix
3. It gets its own OOS test
4. It gets added to the Competitive Benchmark

**Example**: If Stress Test discovers that `torch.compile` + `fp16` causes silent gradient corruption → new family `CompiledMixedPrecision` → added to all 5 stages.

---

## Anti-Overfitting Enforcement

At ANY point, if OOS (Stage 4) drops below 100%:
- 🛑 **ALL feature work stops**
- 🛑 No new bug types, no new architectures, no new signals
- ✅ Only work allowed: fix overfitting
- ✅ Re-run pipeline after fix
- ✅ OOS must return to 100% before any other work resumes

This is the SINGLE most important gate. A tool that only works on its training data is worthless.

---

## Competitive Benchmark Targets

| Metric | Target vs Best Competitor |
|--------|--------------------------|
| Detection rate | ≥ +50% (NeuralDBG detects 50% more failures) |
| Root cause accuracy | ≥ +70% (NeuralDBG identifies root cause 70% more often) |
| Time-to-diagnosis | ≥ 5× faster |
| False positive rate | ≤ competitor's FP rate |
| Causal chains | Competitors have 0 — we must have ≥ 10 per scenario |

---

## Relation to Other Rules
- **R82**: Pipeline results update SESSION_SUMMARY.md Decision Log
- **R102**: Each stage must have ≥ 60% test coverage of its own code
- **R14**: Each stage has its own failure mode table
- **R19**: Pipeline runs trigger version tags (vX.Y.Z-kuro)
- **R104**: Pipeline failures auto-create GitHub Issues
