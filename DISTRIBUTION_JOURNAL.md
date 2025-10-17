# Distribution Journal

## 2025-10-17 (Latest Update)
- **Comprehensive Debug Session Completed**
- **Current Status Assessment:**
  - CLI system: WORKING (imports and help command functional)
  - DSL Parser: WORKING (basic parsing and transformation functional)
  - Core framework dependencies: MIXED (torch module missing, causing import failures)
  - Shape propagation: REQUIRES DEPENDENCY RESOLUTION
- **Issues Identified:**
  - Missing torch dependency - critical for core imports
  - Optional dependencies (pytest_json, triton) missing - causing test runner issues
  - Framework needs dependency resolution before full operation
- **Next Steps:**
  - Resolve dependency issues (install torch and optional packages)
  - Run full test suite after dependencies resolved
  - Test end-to-end CLI workflows (compile, run, debug commands)
  - Validate shape propagation and neural network generation

## 2025-10-17
- Bugs fixed:
  - Removed strict 4D input validation for Output layer (now accepts higher dimensions)
  - Made pysnooper optional dependency in parser.py and cli.py (to avoid import errors when missing)
  - Removed @pysnooper.snoop() decorators from critical methods in parser.py (hpo_range, dense)
  - Fixed unicode emojis in test_runner.py to ensure ASCII compatibility
  - Fixed missing dependencies by handling triton import gracefully
  - Fixed CUSTOM_LAYER regex to exclude built-in layer types, preventing Dense(128) from being parsed as custom layers
  - Added NeuralParser class to provide test interface expected by fuzzing tests
  - Fixed parameter parsing for Dense layers - now correctly extracts units from Dense(128) format
- Code quality:
  - Parser validation relaxed for Output layer to allow flexible input shapes
  - Enhanced error handling for optional dependencies across the codebase

## 2025-10-16
- Features added:
  - `neural docs` command (Markdown; PDF via Pandoc if available)
  - Auto-flatten policy flag for Dense/Output; strict by default
- Bugs fixed:
  - TensorFlow codegen now applies layers to `x` (e.g., `Dense(...)(x)`)
  - Output-on-4D inputs now enforced (error by default; optional flatten)
- Tests added:
  - TF Dense is-applied test
  - Output-on-4D strict vs. auto-flatten tests
  - Docgen smoke test
- Packaging groundwork:
  - Added `pyproject.toml` (build isolation)
  - Updated setup.py version to `0.3.0.dev0`
  - Version detection now tries `neural-dsl` then `neural`, fallback `0.3.0-dev`


- Environment setup:
  - Installed optional dev dependencies: jax[cpu], optuna (for version and CI checks)
- Repo hygiene:
  - Added ROADMAP.md to .gitignore (keep roadmap local, reduce distribution noise)

- CI skeleton:
  - Added .github/workflows/ci.yml (Ubuntu+Windows, Python 3.11): ruff, mypy, pytest, pip-audit (non-blocking)
- Static analysis config:
  - Added ruff config in pyproject.toml
  - Added mypy.ini with permissive baseline (to tighten incrementally)

- README updates:
  - Added CI badge and Reproducibility section (seed utility usage)
- CI enhancement:
  - Added nightly schedule (02:00 UTC) to CI workflow
- Tests:
  - Added tests/code_generator/test_policy_and_parity.py for flatten policy and TF/PT parity
  - Adjusted tests to expect ValueError when auto_flatten_output=False (strict policy)
- Security and CI:
  - Installed ruff, mypy, pip-audit locally and ran scans
  - Resolved pip-audit findings by upgrading setuptools; environment now clean
  - Enhanced Dependabot to also track github-actions updates
- Docs:
  - README: added Optional Dependencies section with guidance
- Codegen:
  - Extracted Dense/Output 2D policy into helper functions for TF/PT
  - Added tests/tests/code_generator/test_policy_helpers.py



- Docs:
  - README: added a Contributing section with local dev workflow (venv, install, lint/type/test, pip-audit)

- CLI:
  - Hardened `neural clean`: safe patterns only; dry-run by default; `--yes` to apply; `--all` removes caches
- DocGen:
  - Upgraded to v1.1: added Summary section and warnings capture; preserved math + shapes
- Docs:
  - README top: added Contributions badge and end-to-end quick commands
- Tests:
  - Added tests/cli/test_clean_command.py for clean behavior
  - Added tests/docgen/test_docgen_v11.py for DocGen v1.1


- Parser hardening:
  - Fixed architectural mismatch between `basic_layer` and layer methods by introducing a token-shift helper that removes leading layer tokens when present.
  - Updated Dense, Conv2D, Dropout, Flatten, MaxPooling2D, LSTM, and BatchNormalization methods to accept both direct grammar calls and basic_layer-dispatched calls.
  - Preserved `params: None` for layers without parameters (e.g., Flatten(), BatchNormalization()) and avoided clobbering by `basic_layer`.
  - Improved device spec extraction for `@ "cuda:0"` syntax.
  - Targeted tests will be (re)run next to validate fixes across parser layer suite.
