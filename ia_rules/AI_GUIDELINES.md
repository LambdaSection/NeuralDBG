# NeuralDBG AI Guidelines

This file captures AI assistant rules and contributor guidance adapted for `NeuralDBG`.

## For Developers
If you are using an AI coding assistant (Cursor, Windsurf, Copilot, etc.), ensure it follows the repo rules:
- **Cursor/Windsurf**: `.cursorrules` (in root)
- **GitHub Copilot**: `.github/copilot-instructions.md`

## Mandatory Product & Quality Rules (Always Apply)
- **Always Update README & Changelog**: Every feature or fix must update `README.md` and `CHANGELOG.md` (create if missing).
- **Zero Friction for Users**: Tools must work out of the box. Minimal config, clear defaults, copy-paste examples that run.
- **Solve Real Pain Points**: Before building, ask: "Does this fix a real user pain?" No speculative features; validate need first.
- **Security & Quality Tooling**: CI must include **CodeQL**, **SonarQube**, and **Codacy** (or equivalent). No shortcuts on static analysis.

## Tools for the AI Era
When AI agents can write and debug code, specialized tools still matter:
- **Structured Observability**: Tools produce machine-readable data (`SemanticEvent`, causal chains). AI consumes it; humans get explanations.
- **Bidirectional Tooling**: Build tools that feed AI assistants *and* present to humans. The value is the structured bridge.
- **Reduced Cognitive Load**: Semantic events give users the vocabulary to ask the right questions; AI can then act on it.

---

## Core Principles

### 1. Framework Neutrality ("Switzerland")
`NeuralDBG` is focused on semantic event extraction and causal reasoning for training dynamics. Keep core logic framework-agnostic and prefer lightweight adapters for framework-specific traces.

### 2. Explanation & Artifact Quality
- The generated explanations, reports, and visual artifacts are the product. Make them clear, reviewer-friendly, and self-contained.

### 3. Architecture
- **Input**: Instrumentation/trace (PyTorch training traces, JSON exports)
- **Core**: `SemanticEvent` model and compact graph representation
- **Output**: Human-readable reports, HTML artifacts, and API-returnable `CausalHypothesis` objects

### 4. Design Principles
- **SRP**: Single Responsibility for modules (monitoring, analysis, export).
- **DRY / KISS / YAGNI / SOLID**: Follow the project's coding philosophy.
- **Duck Typing**: Use pragmatic, Pythonic interfaces where appropriate.

### 5. Tooling
- **Pre-commit**: `ruff`, `mypy`, `pylint`.
- **Diagrams**: Update `logic_graph.md` or README when architecture changes.

## Adding Features (Guidance)

1.  **Get Representative Traces**: Use real or synthetic PyTorch traces or JSON exports from integrations.
2.  **Map to `SemanticEvent`s**: Identify which events matter (gradient transitions, activation shifts, optimizer anomalies).
3.  **Implement Adapter/Parser**: Add a small adapter that converts raw traces into `SemanticEvent`s or the internal graph structures.
4.  **Test Thoroughly**: Unit, integration, logic, and fuzzy tests. Aim for >=60% coverage for new modules.

## Critical Thinking — "Devil's Advocate" Mode
Treat AI assistants as co-engineers. Before and during coding, ask whether a change genuinely helps users, whether a simpler path exists, and what could break.

**During implementation:** flag code smells, question scope creep, and propose tests for risky logic.

**After implementation:** review diffs, suggest improvements, and document any technical debt.

### Traceability — "Always Leave a Trail"
Every session should produce a structured session summary and follow conventional commits.

**Commit discipline:**
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Keep commits small and focused.

**Session summary (MANDATORY at end of every session):**
```
## Session Summary — [DATE]
**What was done:** (bullet list of changes)
**Files changed:** (list)
**Tests:** X passing, Y% coverage
**Next steps:** (what remains)
**Blockers:** (if any)
```

### Protocol
- **Step-by-Step**: Follow the plan and verify phase completion before proceeding.
- **Context Persistence**: Maintain artifacts under `./.antigravity/artifacts/` when applicable.

### Updating Visuals & Demos
1.  Edit visualization/export code under `neuraldbg` or `demo_vanishing_gradients.py` as needed.
2.  Regenerate demo outputs and verify examples in `demo_vanishing_gradients.py` or `tests/integration`.
    ```bash
    python3 demo_vanishing_gradients.py
    ```
3.  Add integration tests to `tests/integration/` to verify generated artifacts when possible.

Keep this document focused on practical, repo-specific rules for AI assistants and human contributors.

