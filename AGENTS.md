1→# Neural DSL - Agent Guide
2→
3→## Setup
4→```bash
5→python -m venv .venv                    # Create venv (convention: .venv or venv)
6→.\.venv\Scripts\Activate                # Windows activation
7→pip install -e .                        # Install in editable mode
8→pip install -e ".[full]"                # Install with all optional dependencies
9→```
10→
11→## Commands
12→- **Build**: N/A (pure Python, no build step)
13→- **Lint**: `python -m ruff check .` or `python -m pylint neural/`
14→- **Test**: `python -m pytest tests/ -v` or `pytest --cov=neural --cov-report=term`
15→- **Dev Server**: `python neural/dashboard/dashboard.py` (NeuralDbg on :8050) or `python neural/no_code/no_code.py` (No-code GUI on :8051)
16→
17→## Tech Stack
18→- **Language**: Python 3.8+ with type hints
19→- **Core**: Lark (DSL parser), Click (CLI), Flask/Dash (dashboards)
20→- **ML Backends**: TensorFlow, PyTorch, ONNX (all optional)
21→- **Tools**: pytest, ruff/pylint, pre-commit, mypy
22→
23→## Architecture
24→- `neural/cli/` - CLI commands (compile, run, visualize, debug)
25→- `neural/parser/` - DSL parser and AST transformer
26→- `neural/code_generation/` - Multi-backend code generators (TF/PyTorch/ONNX)
27→- `neural/shape_propagation/` - Shape validation and propagation
28→- `neural/dashboard/` - NeuralDbg real-time debugger
29→- `neural/no_code/` - No-code web interface
30→
31→## Code Style
32→- Follow PEP 8, 100-char line length (Ruff configured)
33→- Use type hints (`from __future__ import annotations` for forward refs)
34→- Docstrings with numpy-style parameters
35→- No comments unless complex logic requires context
36→- Functional over classes where reasonable
37→