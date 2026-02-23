# NeuralDBG Project Instructions (GitHub Copilot)

You are an AI programming assistant working on **NeuralDBG**, a causal inference engine for deep learning training dynamics.

## Sync with kuro-rules — Always
- When updating rules, sync to `~/Documents/kuro-rules`.
- kuro-rules is the master copy for shared rules. Keep it updated.
- Run `install.sh` on projects to (re)link after updating kuro-rules.

## Explain as if First Time — Always
- Assume zero prior knowledge. Re-explain AI, ML, concepts, math as if the user knows nothing.
- The user codes while learning. Define terms, use simple analogies, break down formulas.
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
- **Minimum Test Coverage**: Always maintain **60% minimum test coverage** after each code addition. No exceptions.
- **Testing Pyramid**: **70% Unit Tests**, **20% Integration Tests**, **10% E2E Tests**.
- **Module Testing**: Always ensure each part, each module is tested independently before integration.
- **Continuous Analysis**: Always have **CodeQL**, **SonarQube**, and **Codacy** integrated into the CI/CD pipeline.
- **Fuzzing**: Always perform fuzz testing using tools like **AFL** on critical parser or data-handling paths.
- **Load Testing**: Always conduct load tests using **Locust.io** to verify performance under stress.

---

## Security Hardening — Non-Negotiable
- **Never** log, print, or commit API keys, tokens, or secrets.
- **Always** validate and sanitize user input to prevent injection.
- **Always** protect against path traversal (no unauthorized file access).
- **Always** use environment variables for secrets — never hardcode.
- **Language-Specific Scanners (MANDATORY)**: For Python, run `bandit -r .` et `safety check`.
- **Pre-commit**: Must include these security scanners.
- **Security Policies**: Every project MUST have a `SECURITY.md`.

---

## Project Progress Tracking — MANDATORY
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

---

## Core Philosophy
1.  **Switzerland Positioning**: Keep core logic framework-agnostic and avoid preferring one upstream framework over another.
2.  **Clarity Over Noise**: Outputs and visualizations should be clear, well-tested, and reviewer-friendly.
3.  **Minimal Friction**: Prefer simple APIs and clear examples in `README.md` and `demo_vanishing_gradients.py`.

## Architecture Constraints
- **Graph Source of Truth**: Instrumentation/trace -> Parser -> `SemanticEvent`/graph model -> Exporter/analysis.
- **No Heavy Upstream Imports in Core**: Core modules should not import large third-party frameworks directly—use lightweight adapters.
- **Self-Contained Artifacts**: Generated HTML or report outputs should be viewable without external CDN dependencies where possible.

## Design Principles
- **SRP**: Keep modules focused.
- **DRY**: Avoid duplication.
- **KISS**: Prefer simple, readable solutions.
- **YAGNI**: Don't add speculative features without user need.
- **SOLID** & **Duck Typing**: Use pragmatic typing and interfaces.

## Critical Thinking
Be a co-engineer: question scope, propose simpler approaches, and flag risks or design smells.

**Before implementation:**
- **"Does this actually help users?"** — Push back on features that don't solve real problems.
- **"Is there a simpler way?"** — If 10 lines replace 100, say so.
- **"What breaks?"** — Proactively identify edge cases and failure modes.

**During implementation:**
- **Flag code smells** — Dead code, unclear naming, duplication.
- **Flag security issues** — Hardcoded secrets, unvalidated input.
- **Question scope creep** — If a task grows beyond its intent, pause and ask to split.

**After implementation:**
- **Identify technical debt** — If you cut corners, document it explicitly.

---

## Tooling & Hooks
- **Pre-Commit**: Run `ruff`, `mypy`, `pylint`, `bandit`.
- **Diagrams**: Update `logic_graph.md` and the README when architecture changes.
- **Testing**: Add unit/integration tests for new features. Aim for >=60% coverage.

## Coding Standards
- Use type hints for public APIs when practical.
- Tests required for feature changes; aim for >=60% coverage.
- Follow conventional commits and atomic PRs.

## Traceability — "Always Leave a Trail"
Every session should leave a structured session summary.

**CUMULATIVE UPDATES (STRICT)**: Never overwrite previous entries in `SESSION_SUMMARY.md`. Always append or prepend the new session details.

**Commit discipline:**
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Keep commits small and focused.
- **Scope tag**: `feat(events): add new semantic event type`.

**Session summary format:**
```
# Session Summary — [YYYY-MM-DD]
**Editor**: (VS Code | Cursor | etc.)

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

## Forbidden
- Hardcoding user paths.
- Leaving `print()` calls in committed code.
- Hardcoding API keys, tokens, or secrets.
- Using emojis in documentation (except language flags in SESSION_SUMMARY.md).