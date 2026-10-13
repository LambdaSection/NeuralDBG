---
name: rule-102-test-coverage
description: ML test coverage 80%, hermetic tests
trigger: auto
scope: company
priority: 3
status: active
paths:
- tests/**
- src/**
- '**/*.py'
---

<!-- Generated from rules/rule_102_test_coverage.md — DO NOT EDIT. scope=company priority=3 status=active -->

# RULE 102: ML Project Test Coverage — Mandatory Standards

## Rule

All Python machine learning projects MUST maintain a minimum test coverage of **80% total** with branch coverage enabled. This rule extends R5 (60% minimum) specifically for ML/DL inference engines and monitoring tools.

## Coverage Targets

| Module Type | Line Coverage | Branch Coverage |
|---|---|---|
| Core engine (`__init__.py`) | ≥ 88% | ≥ 75% |
| Standalone fallback paths | 100% | 100% |
| Enhanced reasoning modules | ≥ 90% | ≥ 80% |
| Error/exception paths | ≥ 85% | ≥ 75% |
| Export utilities | ≥ 90% | ≥ 80% |
| **TOTAL** | **≥ 80%** | **≥ 70%** |

## Test Pyramid (MANDATORY for OpenQuant)

| Layer | % Tests | Source | Exemple |
|---|---|---|---|
| Unit (hermetic, no network) | 70% | `tests/unit/test_*.py` | `episode/memory.py`, `robot/risk_manager.py` |
| Integration (mocked I/O) | 20% | `tests/integration/test_*.py` | `robot/daemon_v2.py` avec `Memory` fake |
| E2E (paper -> trade) | 10% | `tests/e2e/test_*.py` | `auto_code` walkforward complet |

Règles : `load_dotenv` mocké, `yfinance` mocké, `Alpaca` fake, `0% module = bloquant`.

## Mandatory Test Categories

### 1. Standalone Fallback Tests (CRITICAL)
Every method that delegates to a proprietary/optional engine MUST have standalone tests that:
- Force `_causal_engine = None` explicitly
- Assert the fallback logic returns correct types (list, dict, str, etc.)
- Verify no `AttributeError` or `NotImplementedError` is raised

### 2. Enhanced Module Tests (CRITICAL)
Any module with 0% coverage MUST be treated as a blocking issue:
- **No PR/commit is allowed while any module has 0% coverage**
- New modules must have tests written BEFORE or AT THE SAME TIME as the module

### 3. Error Path Tests
- All `try/except` blocks must have tests that trigger the exception
- All `__del__` and `cleanup()` methods must be tested with corrupted state
- RuntimeError handlers in hooks must be tested with patched/mocked failures

### 4. Export Method Tests
- All export methods (`export_*`) must produce valid parseable outputs (JSON, Mermaid, etc.)
- Empty-state exports (no events) must not crash
- Metadata with non-serializable values must be filtered, not crash

## Configuration Standards

### pyproject.toml
```toml
[tool.pytest.ini_options]
addopts = "-ra -q --cov=neuraldbg --cov-report=term-missing"

[tool.coverage.run]
branch = true
source = ["neuraldbg"]
omit = ["tests/*", "examples/*"]

[tool.coverage.report]
fail_under = 75
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "def __repr__",
]
```

### Warning Policy
```toml
filterwarnings = [
    # Only silence 3rd-party noise. Keep internal warnings visible.
    "ignore::DeprecationWarning:mlflow.*",
    "ignore::FutureWarning:torch.*",
    # Document and silence EXPECTED internal warnings (design decisions)
    "ignore:NeuralDbg\\: Model is wrapped in DataParallel.*:UserWarning",
]
```

## Verification

```
BEFORE every commit:
  1. Run: pytest --cov=neuraldbg --cov-report=term-missing
  2. Verify total coverage ≥ 80%
  3. Verify no module has 0% coverage
  4. Verify 0 test failures
  5. Check warning output — only expected warnings should appear

WHEN adding a new module:
  1. Create tests/unit/test_<module_name>.py IMMEDIATELY
  2. Achieve ≥ 90% coverage before marking the feature complete
  3. Run full suite to verify no regression
```

## Enforcement

```
IF total coverage < 80%:
  ACTION: STOP new features. Run audit: pytest --cov --cov-report=term-missing
  IDENTIFY: Which module has the largest miss count
  WRITE: Tests for that module first

IF any module has 0% coverage:
  ACTION: BLOCKING. No PR allowed. Write tests immediately.

IF test suite passes but warnings are noisy:
  ACTION: Add specific filterwarnings for expected internal warnings
  NEVER: Silence all warnings globally (ignore::* is forbidden)
```

## Test File Naming Convention

```
tests/
  unit/
    test_<module_name>.py          # Core module tests
    test_<feature>_fallbacks.py    # Standalone fallback tests
    test_<feature>_error_paths.py  # Error and exception tests
  integration/
    test_<scenario>_demo.py        # End-to-end scenario tests
```

## Special Rule: numpy Dependency in Tests

When `enhanced_causality.py` or any module using `numpy` is tested:
- `numpy` must be listed in `[dev]` optional dependencies
- Import errors for `numpy` in test environment = BLOCKING issue

---
**Created**: 2026-05-29
**Trigger**: NeuralDBG pre-launch audit revealed `enhanced_causality.py` at 0% coverage and total coverage at 65% (below R5 minimum).
**Project Scope**: NeuralDBG, Neural-Agent, NeuralDBG-Engine, all future ML projects
**Enforcement**: MANDATORY — extends R5
