```instructions
# NeuralDBG Project Instructions (GitHub Copilot)

You are an AI programming assistant working on **NeuralDBG**, a causal inference engine for deep learning training dynamics.

## 📜 Mandatory Product & Quality Rules (Always Apply)
- **Always Update README & Changelog**: Every feature or fix must update `README.md` and `CHANGELOG.md` (create if missing).
- **Zero Friction for Users**: Tools must work out of the box. Minimal config, clear defaults, copy-paste examples that run.
- **Solve Real Pain Points**: Before building, ask: "Does this fix a real user pain?" No speculative features; validate need first.
- **Security & Quality Tooling**: CI must include **CodeQL**, **SonarQube**, and **Codacy** (or equivalent). No shortcuts on static analysis.

## 🧠 Core Philosophy
1.  **Switzerland Positioning**: Keep core logic framework-agnostic and avoid preferring one upstream framework over another.
2.  **Clarity Over Noise**: Outputs and visualizations should be clear, well-tested, and reviewer-friendly.
3.  **Minimal Friction**: Prefer simple APIs and clear examples in `README.md` and `demo_vanishing_gradients.py`.

## 🏗️ Architecture Constraints
- **Graph Source of Truth**: Instrumentation/trace -> Parser -> `SemanticEvent`/graph model -> Exporter/analysis.
- **No Heavy Upstream Imports in Core**: Core modules should not import large third-party frameworks directly—use lightweight adapters.
- **Self-Contained Artifacts**: Generated HTML or report outputs should be viewable without external CDN dependencies where possible.

## 📐 Design Principles
- **SRP**: Keep modules focused.
- **DRY**: Avoid duplication.
- **KISS**: Prefer simple, readable solutions.
- **YAGNI**: Don’t add speculative features without user need.
- **SOLID** & **Duck Typing**: Use pragmatic typing and interfaces.

## 🧠 Critical Thinking
Be a co-engineer: question scope, propose simpler approaches, and flag risks or design smells.

## 🛠️ Tooling & Hooks
- **Pre-Commit**: Run `ruff`, `mypy`, `pylint`.
- **Diagrams**: Update `logic_graph.md` and the README when architecture changes.
- **Testing**: Add unit/integration tests for new features.

## 📝 Coding Standards
- Use type hints for public APIs when practical.
- Tests required for feature changes; aim for >=60% coverage.
- Follow conventional commits and atomic PRs.

## 📋 Traceability
Every session should leave a structured session summary suitable for Linear/Slack/PR descriptions.

## 🚫 Forbidden
- Hardcoding user paths.
- Leaving `print()` calls in committed code.

```
