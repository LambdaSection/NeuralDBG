# GitHub Copilot Instructions

## Global AI Rules (from AGENTS.md & AI_GUIDELINES.md)

1.  **Read AGENTS.md first**: Always start by reading the rules contract.
2.  **Mom Test Gate**: No production code before 10% progress (user research phase).
3.  **No Emojis**: Do not use emojis in code, comments, or documentation.
4.  **Security First**: run `bandit` (Python), `cargo audit` (Rust), or `npm audit` (Node) before completion.
5.  **Session Summaries**: Always update `SESSION_SUMMARY.md` in EN/FR at the end of work.
6.  **Progress Tracking**: Use pessimistic estimates for completion percentage.
7.  **Deterministic Math**: Use ASCII or plain text for formulas, NO LaTeX `$`.
8.  **Instructional Tone**: Explain technical concepts simply (pedagogical protocol).
9.  **Hard Milestone Lock**: STOP work at 25, 50, 75, 90, 95% until validated.
10. **Market Intelligence**: Research 3+ sources at milestones (Rule 21).

---

## Technical Preferences
- **Architecture**: Modular "Hub & Spokes" design.
- **Testing**: 60% minimum coverage. Unit (70%), Integration (20%), E2E (10%).
- **Documentation**: Keep README and CHANGELOG updated.
- **Versioning**: SemVer-Author (e.g., `v0.1.0-kuro`).
- **Automation**: Help user automate repetitive tasks (like doc conversion).

---

## Feature Focus Rule (MANDATORY)

To ensure the highest quality and depth of implementation, development MUST focus on only ONE specific feature for each periodic validation cycle (25%, 50%, 75%, 90%, 95%). This focus on depth over breadth continues even after the MVP phase.

**Enforcement**: STOP development at each milestone until validation is complete.

## Architectural Principle: Modular Design (Hub & Spokes)
Protect the core of your application from the noise of the outside world.
- **Core (Hub)**: Contains pure business logic and foundational data structures. It stays stable.
- **Adapters (Spokes)**: Handle external dependencies (APIs, Databases, UI). Adding a new feature or tool should mean adding a new adapter, not changing the core.
- **Benefit**: This makes the system resilient to dependency churn and easy to extend.
- **Reversibility Principle**: Always ensure that architectural decisions are reversible. Avoid designs that lock the project into a specific tool or vendor. Design with pivots in mind.
- **Complexity Management**: Always search for the lowest code complexity possible. Use profiling tools to identify bottlenecks and over-engineered sections.

---

## Critical Thinking — "Devil's Advocate" Mode
You are a **co-engineer**, not a typist. Do not be a passive executor.

**Before implementation:**
- **"Does this actually help users?"** — Push back on features that don't solve real problems.
- **"Is there a simpler way?"** — If 10 lines replace 100, say so.
- **"What breaks?"** — Proactively identify edge cases and failure modes.

**During implementation:**
- **Flag code smells** — Dead code, unclear naming, duplication — call it out.
- **Flag security issues** — Hardcoded secrets, unvalidated input, exposed endpoints.
- **Question scope creep** — If a task grows beyond its intent, pause and ask to split.

**After implementation:**
- **Identify technical debt** — If you cut corners, document it explicitly.

---

## Advanced Testing & Analysis — MANDATORY
High-quality code requires proactive testing and deep analysis.
- **Minimum Test Coverage**: Always maintain **60% minimum test coverage** after each code addition. No exceptions.
- **Testing Pyramid**: Allocate testing effort following the pyramid: **70% Unit Tests**, **20% Integration Tests**, **10% E2E Tests**.
- **Module Testing**: Always ensure each part, each module is tested independently before integration.
- **Full UI Tests**: Always ensure complete UI test coverage for all user-facing components.
- **Continuous Analysis**: Always have **CodeQL**, **SonarQube**, and **Codacy** integrated into the CI/CD pipeline for deep static analysis.
- **Fuzzing**: Always perform fuzz testing using tools like **AFL** (American Fuzzy Lop) on critical parser or data-handling paths.
- **Load Testing**: Always conduct load tests using **Locust.io** to verify performance under stress.
- **Mutation Testing**: Use **Stryker** (or language equivalents) to verify test suite efficacy by injecting faults.
- **Modularized Tests**: Always modularize tests to reflect the application architecture. Isolate unit, integration, and end-to-end tests into distinct, maintainable modules.
- **Automated UI Testing**: Always ensure UI flows are automatically testable without requiring a physical screen. Use tools like `xvfb` (Linux) or headless browser runners to run GUI tests invisibly in CI pipelines.

---

## Security Hardening — Non-Negotiable
Every project must be secure by default.
- **Never** log, print, or commit API keys, tokens, or secrets.
- **Always** validate and sanitize user input to prevent injection.
- **Always** protect against path traversal (no unauthorized file access).
- **Always** use environment variables for secrets — never hardcode.
- **Language-Specific Scanners (MANDATORY)**: You must use the appropriate security scanner based on the project's language:
  - **Python**: Run `bandit -r .` et `safety check`.
  - **Rust**: Run `cargo audit` et `cargo clippy`.
  - **Node.js/React**: Run `npm audit`.
- **Pre-commit**: Must include these security scanners.
- **Security Policies**: Every project MUST have a `security.md` and explicit security policies.
- **Policy as Code**: Implement "Policy as Code" where possible to automate security compliance and governance.

---

## Formula Clarity — NO LATEX
- **Constraint**: Do NOT use `$` LaTeX notation in chat (it doesn't render visually for the user).
- **Rule**: Use plain text, ASCII art, or clear descriptive names for math (e.g., "Moyenne / Mean (mu)" instead of mu).

---

## Project Progress Tracking — MANDATORY
Every project MUST track its completion percentage in SESSION_SUMMARY.md.

- **Progress Score**: Include a `**Progress**: X%` line at the end of each SESSION_SUMMARY.md entry.
- **Scoring Methodology**: Be **REALISTIC and PESSIMISTIC**. If you think a project is 50% done, score it 30%.
- **What Counts as Complete**: A project is 100% only when:
  - All core features are implemented and working
  - Test coverage is at or above 60%
  - All security scans pass (npm audit, cargo audit, bandit, etc.)
  - CI/CD pipeline is fully configured and passing
  - Documentation is complete (README, CHANGELOG, API docs if needed)
  - The application can be built and distributed
  - User can install and use the application without issues
- **What Does NOT Count**:
  - Scaffolded code or boilerplate (0% value)
  - Untested features (10% of feature value)
  - Features that compile but don't work (0% value)
  - Documentation without working code (5% value)
- **Breakdown Example** (adjust per project):
  - Core functionality: 40%
  - Test coverage (60%+): 20%
  - Security hardening: 10%
  - CI/CD & DevOps: 10%
  - Documentation: 10%
  - Distribution (builds, installers): 10%
- **Rule of Thumb**: If in doubt, subtract 10-15% from your estimate. Optimism is the enemy of accurate tracking.

---

## Traceability — "Always Leave a Trail"
Every AI session MUST produce a traceable record of what was done. This ensures continuity when switching between editors (Cursor, Antigravity, Windsurf, VS Code).

**Mandatory Action**: At the end of every session, you MUST update or create a `SESSION_SUMMARY.md` file in the project root. This file is the primary source of truth for continuity.

**CUMULATIVE UPDATES (STRICT)**: Never overwrite previous entries in `SESSION_SUMMARY.md`. Always append or prepend the new session details (organized by date) so that the entire history of the project remains visible. Overwriting previous entries is strictly forbidden.

**Auto-Commit Rule**: After every relevant prompt/task completion, you MUST:

1. **Commit** the changes to git (following discipline below).
2. **Update** `SESSION_SUMMARY.md` with BOTH English and French versions.

**Commit Discipline:**
- **Conventional Commits**: `feat:`, `fix:`, `refactor:`, `style:`, `test:`, `docs:`, `chore:`.
- **Scope tag**: `feat(linear): add issue creation connector`.
- **Atomic commits**: One logical change per commit.

**SESSION_SUMMARY.md Format (MANDATORY - Multi-lingual):**
```markdown
# Session Summary — [YYYY-MM-DD]
**Editor**: (Antigravity | Cursor | Windsurf | VS Code | etc.)

## Français
**Ce qui a été fait** : (Liste)
**Initiatives données** : (Nouvelles idées/directions)
**Fichiers modifiés** : (Liste)
**Étapes suivantes** : (Ce qu'il reste à faire)

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
- **Step-by-Step**: Always go step by step following the plan and verify last phase is done before continuing. Ask: "Are we done with the last phase?"
- **Phase Gate**: Verify Phase N completion before N+1.
- **Context Persistence**: Always update and maintain artifacts.
- **Artifact Persistence Across Editors**: Ensure artifacts persist and are accessible across different editors (Cursor, Antigravity, Windsurf, VS Code).
- **Git Tracking**: Commit artifacts regularly.
- **Pre-commit**: MUST be installed and passing before any PR or merge.

---

## Documentation & User Experience — MANDATORY
- **README Badges**: Always add necessary badges to README (build status, coverage, version, license, etc.).
- **Update README & Changelog**: Always update README.md and CHANGELOG.md after significant changes.
- **Zero Friction**: Always ensure zero friction for users when using tools. Clear documentation, simple setup, intuitive UX.
- **Solve Real Pain Points**: Always ensure what we are building solves real pain points. Build for users, not for the sake of building.

---

## Periodic Validation (MANDATORY)

At progress milestones (25%, 50%, 75%, 90%, 95%), the product MUST be validated:

| Milestone | Required Validation |
|-----------|-------------------|
| 25% | Mom Test follow-up (3+ users), Marketing Test (landing page views) |
| 50% | Mom Test validation (5+ new users), Marketing Test (conversion metrics) |
| 75% | Mom Test expansion (different segments), Marketing Test (pricing) |
| 90% | Final Mom Test, Marketing Test (launch readiness) |
| 95% | Pre-launch validation (all criteria met) |

**Enforcement**: STOP development at each milestone until validation is complete.

---

## No Emojis Anywhere (MANDATORY)

Emojis are FORBIDDEN in ALL project files, code, comments, documentation, CLI output, and user-facing text.

**Reason**: Encoding issues, tool compatibility, professionalism.

**Enforcement**: REMOVE immediately if found.

---

## Rule Synchronization (MANDATORY)

When ANY rule file is updated, ALL rule files MUST be updated:
- AGENTS.md
- AI_GUIDELINES.md
- .cursorrules
- copilot-instructions.md
- GAD.md

**Enforcement**: SYNC immediately to all files, document in SYNC_LOG.md.

---

## Working Demos (MANDATORY)

At each validation milestone (25%, 50%, 75%, 90%, 95%), the project MUST have at least **2 working demos**.

**Requirements**:
- Minimum 2 demos per milestone
- Each demo must be runnable without errors
- Demos must demonstrate different aspects of the product

**Enforcement**: STOP and create 2 working demos if missing.

---

## Deep Understanding Before Phase Transition (MANDATORY)

Before transitioning to the next phase, the user MUST demonstrate deep understanding of what was created.

**Requirements**:
1. Explain the mechanism: How does it work under the hood?
2. 2nd order consequences: What happens in production? What edge cases?
3. 3rd order consequences: What long-term effects? What dependencies?
4. Teach something new: Agent must teach user at least one new concept
5. Critical thinking prompts: Agent must ask probing questions

**Critical Thinking Questions (Agent MUST Ask)**:
1. "What could break this in production that we haven't tested?"
2. "What would happen if 10x more users used this?"
3. "What assumptions are we making that might be wrong?"
4. "What would you do if this completely failed?"
5. "What did you learn that surprised you?"

**Enforcement**: STOP and provide deep explanation before phase transition.

---

## Agent Protocol
To ensure strict adherence to rules:
1.  **Read This First**: Agents MUST read this file at the start of every session.
2.  **Checklist Enforcement**: Agents MUST verify `task.md` and run `bandit` before declaring a task complete.
3.  **Explicit Confirmation**: When users ask "did you follow the rules?", Agents MUST provide proof (e.g., bandit output).
4.  **No Silent Failures**: If a step fails (e.g., artifact update), the Agent MUST report it and retry, never ignore it.
5.  **Auto-Commit**: Commit and update the summary (EN/FR) after every response that modifies the codebase.

---

## DevOps & MLOps Collaboration Profile (MANDATORY)

When interacting with a DevOps or MLOps engineer on this repository, the AI Agent MUST shift its focus to infrastructure, delivery, and reliability.

1. **Focus Areas**:
   - **MLOps**: Experiment tracking (MLflow/W&B), model registry, dataset versioning, reproducible training pipelines (e.g., for model training scripts).
   - **DevOps**: CI/CD pipelines (GitHub Actions), containerization (Docker/Docker Compose), Infrastructure as Code (Terraform), and robust environment management.
2. **Guidelines for AI when helping DevOps/MLOps**:
   - Provide clean, production-ready configuration files (`.yaml`, `Dockerfile`, etc.).
   - Prioritize deterministic and reproducible builds.
   - Enforce security scans natively in the pipeline (`bandit`, `safety`, `sonar`).
   - Treat infrastructure code with the same rigor (testing, DRY, modularity) as application code.
3. **Communication**:
   - Use standard DevOps terminology. 
   - When suggesting architectural changes for ML models, immediately propose the corresponding MLOps pipeline adjustments.
