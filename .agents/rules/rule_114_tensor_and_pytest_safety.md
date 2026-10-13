---
name: rule-114-tensor-and-pytest-safety
description: 'RULE 114: Tensor Operations and Test Suite Warning Governance'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_114_tensor_and_pytest_safety.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 114: Tensor Operations and Test Suite Warning Governance

## Rule

ML/DL training dynamic causal inference hooks, anomaly tracking, and test suites MUST be built defensively to prevent data-type crashes and log spam.

## Implementation Standards

### 1. PyTorch Tensor Dtype Guarding
When implementing hooks or dynamic tracking that monitor training activations and gradients:
- **Floating-Point Checks**: Always verify that a tensor is a floating-point type (`torch.is_floating_point(t)`) before computing floating-point operations (e.g., `.mean()`, `.std()`, `.var()`) or applying epsilon thresholds.
- **Underflow Protection**: Use dtype-aware epsilon scaling to prevent division-by-zero or precision underflow. For instance, scale eps to `1e-4` for FP16/BF16, and `1e-9` for FP32/FP64.
- **Bypass Non-Floats**: Skip statistical computations on non-floating-point tensors (like token indices, label/target masks, class indices) to avoid runtime errors.

### 2. PyTest Warning Filters (Zero-Warnings Policy)
To keep CI and test logs clean and maintain the Zero-Warnings policy:
- **Filter Third-Party Noise**: Silence warnings originating from third-party libraries (e.g., deprecation warnings from packages like MLflow, numpy, or internal PyTorch hooks) via configuration in `pyproject.toml` or `pytest.ini`.
- **Expose Internal Warnings**: Only leave active internal warnings that are actionable or denote critical failure conditions in our code.

## Verification

```
WHEN modifying tensor activation monitoring:
  1. Does the function check torch.is_floating_point(t) before applying mathematical statistics?
  2. Does it use dtype-aware epsilon scaling to prevent underflow in FP16/BF16?
  
WHEN running tests:
  1. Does the test suite run with minimal deprecation warnings?
  2. Are all third-party warnings filtered in pyproject.toml?
```

## Enforcement

```
IF a test suite fails due to third-party warnings noise:
  ACTION: Add specific ignore filters in pyproject.toml [tool.pytest.ini_options].

IF a runtime error occurs during training tracking on non-float tensors:
  ACTION: Add a torch.is_floating_point guard to bypass statistical checks.
```

---
**Created**: 2026-05-20
**Trigger**: Activation stats crashed on token indices; test logs polluted by MLflow and PyTorch deprecations.
**Enforcement**: MANDATORY
