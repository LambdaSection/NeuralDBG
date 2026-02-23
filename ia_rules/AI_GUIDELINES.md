# NeuralDBG AI Guidelines

This file captures AI assistant rules and contributor guidance adapted for `NeuralDBG`.

## Sync with kuro-rules — Always
- When updating rules, sync to `~/Documents/kuro-rules`. kuro-rules is the master copy.
- Run `install.sh` on projects to (re)link after updating kuro-rules.
- **Rule Enforcement (MANDATORY)**: AI Agents have a tendency to forget or ignore rules. You MUST read this `AI_GUIDELINES.md` file FIRST upon starting any new task. Do not rely on your base training.

## For Developers
If you are using an AI coding assistant (Cursor, Windsurf, Copilot, etc.), ensure it follows the repo rules:
- **Cursor/Windsurf**: `.cursorrules` (in root)
- **GitHub Copilot**: `.github/copilot-instructions.md`

## Explain as if First Time — Always
- Assume **zero prior knowledge**. Re-explain AI, ML, concepts, math as if the user knows nothing.
- The user codes while learning for the first time. Define terms, use simple analogies, break down formulas.
- Never skip explanations. "Obvious" is not obvious to someone learning.

## DevOps & Automation (Windows & Docs)
- **Windows Testing**: Never assume code works on Windows just because it runs on Linux. Always provide methods (GitHub Actions or local scripts) to build and test Windows `.exe` formats.
- **Session Sync Automation**: The user manually copies `SESSION_SUMMARY.md` to a Word document and WhatsApp. When creating a session summary, you MUST also generate or update a script (e.g. `sync_summary.py` or a bash script) that automates converting the markdown to `.docx` (using `python-docx` or `pandoc`) to save the user time.

---

## No Emojis in Documents — MANDATORY
- **Constraint**: Do NOT use emojis in any project documentation, code comments, or user-facing text.
- **Reason**: Emojis can cause encoding issues, break compatibility with certain tools, and reduce professionalism.
- **Exception**: Emojis are allowed in `SESSION_SUMMARY.md` section headers (language flags) and commit messages only.

---

## Mandatory Product & Quality Rules (Always Apply)
- **Always Update README & Changelog**: Every feature or fix must update `README.md` and `CHANGELOG.md` (create if missing).
- **Zero Friction for Users**: Tools must work out of the box. Minimal config, clear defaults, copy-paste examples that run.
- **Solve Real Pain Points**: Before building, ask: "Does this fix a real user pain?" No speculative features; validate need first.
- **Security & Quality Tooling**: CI must include **CodeQL**, **SonarQube**, and **Codacy** (or equivalent). No shortcuts on static analysis.
- **README Badges**: Always add necessary badges to README (build status, coverage, version, license, etc.).

---

## Advanced Testing & Analysis — MANDATORY
High-quality code requires proactive testing and deep analysis.
- **Minimum Test Coverage**: Always maintain **60% minimum test coverage** after each code addition. No exceptions.
- **Testing Pyramid**: Allocate testing effort following the pyramid: **70% Unit Tests**, **20% Integration Tests**, **10% E2E Tests**.
- **Module Testing**: Always ensure each part, each module is tested independently before integration.
- **Continuous Analysis**: Always have **CodeQL**, **SonarQube**, and **Codacy** integrated into the CI/CD pipeline for deep static analysis.
- **Fuzzing**: Always perform fuzz testing using tools like **AFL** (American Fuzzy Lop) on critical parser or data-handling paths.
- **Load Testing**: Always conduct load tests using **Locust.io** to verify performance under stress.
- **Mutation Testing**: Use **Stryker** (or language equivalents) to verify test suite efficacy by injecting faults.

---

## Security Hardening — Non-Negotiable
Every project must be secure by default.
- **Never** log, print, or commit API keys, tokens, or secrets.
- **Always** validate and sanitize user input to prevent injection.
- **Always** protect against path traversal (no unauthorized file access).
- **Always** use environment variables for secrets — never hardcode.
- **Language-Specific Scanners (MANDATORY)**: For Python, run `bandit -r .` et `safety check`.
- **Pre-commit**: Must include these security scanners.
- **Security Policies**: Every project MUST have a `SECURITY.md` and explicit security policies.

---

## Project Progress Tracking — MANDATORY
Every project MUST track its completion percentage in SESSION_SUMMARY.md.

- **Progress Score**: Include a `**Progress**: X%` line at the end of each SESSION_SUMMARY.md entry.
- **Scoring Methodology**: Be **REALISTIC and PESSIMISTIC**. If you think a project is 50% done, score it 30%.
- **What Counts as Complete**: A project is 100% only when:
  - All core features are implemented and working
  - Test coverage is at or above 60%
  - All security scans pass (bandit, safety, etc.)
  - CI/CD pipeline is fully configured and passing
  - Documentation is complete (README, CHANGELOG, API docs if needed)
  - The application can be built and distributed
  - User can install and use the application without issues
- **Breakdown Example** (adjust per project):
  - Core functionality: 40%
  - Test coverage (60%+): 20%
  - Security hardening: 10%
  - CI/CD & DevOps: 10%
  - Documentation: 10%
  - Distribution (builds, installers): 10%

---

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

---

## Traceability — "Always Leave a Trail"
Every session should produce a structured session summary and follow conventional commits.

**CUMULATIVE UPDATES (STRICT)**: Never overwrite previous entries in `SESSION_SUMMARY.md`. Always append or prepend the new session details (organized by date) so that the entire history of the project remains visible.

**Commit discipline:**
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Keep commits small and focused.
- **Scope tag**: `feat(events): add new semantic event type`.

**Session summary (MANDATORY at end of every session):**
```markdown
# Session Summary — [YYYY-MM-DD]
**Editor**: (Antigravity | Cursor | Windsurf | VS Code | etc.)

## Francais
**Ce qui a ete fait** : (Liste)
**Initiatives donnees** : (Nouvelles idees/directions)
**Fichiers modifies** : (Liste)
**Etapes suivantes** : (Ce qu'il reste a faire)

## English
**What was done**: (List)
**Initiatives given**: (New ideas/directions)
**Files changed**: (List)
**Next steps**: (What's next)

**Tests**: X passing
**Blockers**: (If any)
**Progress**: X% (pessimistic estimate)
```

---

## Protocol
- **Step-by-Step**: Follow the plan and verify phase completion before proceeding.
- **Phase Gate**: Verify Phase N completion before N+1.
- **Context Persistence**: Maintain artifacts under `./artifacts/` when applicable.
- **Git Tracking**: Commit artifacts regularly.
- **Pre-commit**: MUST be installed and passing before any PR or merge.
- **AntiGravity IDE**: Rules copy in `.antigravity/RULES.md` (pour usage avec l'IDE AntiGravity).
- **Roadmap projets**: `PROJECTS.md` (racine).

---

## Agent Protocol
To ensure strict adherence to rules:
1.  **Read This First**: Agents MUST read this file at the start of every session.
2.  **Checklist Enforcement**: Agents MUST verify `task.md` and run `bandit` before declaring a task complete.
3.  **Explicit Confirmation**: When users ask "did you follow the rules?", Agents MUST provide proof (e.g., bandit output).
4.  **No Silent Failures**: If a step fails (e.g., artifact update), the Agent MUST report it and retry, never ignore it.
5.  **Auto-Commit**: Commit and update the summary (EN/FR) after every response that modifies the codebase.

---

## Updating Visuals & Demos
1.  Edit visualization/export code under `neuraldbg` or `demo_vanishing_gradients.py` as needed.
2.  Regenerate demo outputs and verify examples in `demo_vanishing_gradients.py` or `tests/integration`.
    ```bash
    python3 demo_vanishing_gradients.py
    ```
3.  Add integration tests to `tests/integration/` to verify generated artifacts when possible.

Keep this document focused on practical, repo-specific rules for AI assistants and human contributors.