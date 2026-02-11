# NeuralDBG AI Guidelines

This project uses specific rules for AI assistants to ensure quality and consistency.

## For Developers
If you are using an AI coding assistant (Cursor, Windsurf, Copilot, etc.), please ensure it is aware of the context rules:
- **Cursor/Windsurf**: `.cursorrules` (in root)
- **GitHub Copilot**: `.github/copilot-instructions.md`

## Core Principles

### 1. Causal Reasoning First
NeuralDBG is a **Causal Inference Engine** for deep learning training.
- We answer "why did training fail?" not just "what happened?"
- We focus on semantic events, not raw metrics.
- We provide structured explanations with confidence scores.

### 2. PyTorch Native
- The tool is built for PyTorch users.
- It must work seamlessly with `torch.compile`.
- Non-invasive: wraps existing training loops via context manager.

### 3. Architecture
- **Input**: Training loop events (gradients, activations, optimizer state)
- **Core**: Semantic event extraction → Causal compression → Hypothesis generation
- **Output**: Ranked failure explanations with confidence scores

### 4. Design Principles
- **SRP**: Single Responsibility. No "God Classes".
- **DRY**: Detailed logic belongs in utilities, not duplicated.
- **KISS**: Simple is better than complex.
- **YAGNI**: Don't overengineer for hypothetical futures.
- **SOLID**: Follow the 5 commandments of OOD.
- **Duck Typing**: Embrace Python's dynamic nature.
- **Clean Code**: Readable names, small functions, no side effects.
- **Agile**: Ship > Argue. Code wins arguments.

### 5. Tooling
- **Pre-commit**: Ruff, Mypy, Pylint.
- **Testing**: Pytest with comprehensive coverage.
- **Security**: CodeQL scanning.

## Adding Features

### Adding a New Detector
1. **Identify the Pattern**: What semantic event are you detecting?
2. **Implement Detector**: Add class in appropriate module.
3. **Test**: Write tests covering all layers:
   - **Coverage**: 60% Minimum.
   - **Module Coverage**: Ensure EACH part, EACH module is tested.
   - **Unit**: Individual methods.
   - **Integration**: Full detection pipeline.
   - **Logic**: Event extraction verification.
   - **Edge Cases**: Boundary conditions and failure modes.

### 6. Critical Thinking — "Devil's Advocate" Mode
AI assistants working on this project MUST NOT be passive executors. You are a **co-engineer**, not a typist.

**Before writing any code, always ask yourself:**
- **"Does this actually help users?"** — If a feature doesn't solve a real problem, push back. Ask the human to justify it.
- **"Is there a simpler way?"** — Challenge over-engineering. If 10 lines replace 100, say so.
- **"What breaks?"** — Proactively identify edge cases, failure modes, and security risks before they become bugs.
- **"Who is this for?"** — If the target user wouldn't understand or benefit from a change, flag it.
- **"Does this already exist?"** — Before building, check if a library, pattern, or existing code already solves the problem.

**During implementation:**
- **Flag code smells** — If you see dead code, unclear naming, duplicated logic, or tight coupling, call it out even if you weren't asked to.
- **Question scope creep** — If a task is growing beyond its original intent, pause and ask: "Should we split this?"
- **Suggest tests for the scary parts** — If a piece of logic is complex or critical, proactively suggest testing it.
- **Challenge assumptions** — If the human says "we need X", it's OK to ask "why not Y?" if Y is demonstrably better.

**After implementation:**
- **Review your own work** — Before declaring done, re-read the diff. Would you approve this PR?
- **Suggest improvements** — "This works, but here's how we could make it better in the future: ..."
- **Identify technical debt** — If you had to cut corners, document it explicitly.

> **The goal: every interaction should leave the codebase better than we found it, and every feature should genuinely serve the people who use NeuralDBG.**

### 7. Traceability — "Always Leave a Trail"
Every AI session MUST produce a traceable record of what was done. This is non-negotiable.

**Commit discipline:**
- **Conventional Commits**: Use prefixes: `feat:`, `fix:`, `refactor:`, `style:`, `test:`, `docs:`, `chore:`.
- **Scope tag**: Include the module in parentheses: `feat(detectors): add gradient vanishing detector`.
- **Linear issue IDs**: If a Linear issue exists, reference it: `feat(detectors): add gradient vanishing detector [NDB-42]`.
- **Atomic commits**: One logical change per commit. Don't bundle unrelated changes.

**Session summary (MANDATORY at end of every session):**
Before finishing, provide a structured summary the human can paste into Linear/Slack/anywhere:
```
## Session Summary — [DATE]
**What was done:** (bullet list of changes)
**Files changed:** (list)
**Tests:** X passing, Y% coverage
**Next steps:** (what remains)
**Blockers:** (if any)
```

**Why:** Multiple editors (Cursor, Augment, Copilot, Antigravity) work on this project. Git history + structured summaries are the universal source of truth that lets the team follow progress regardless of which tool was used.

### 8. Protocol
- **Step-by-Step**: Stick to the plan.
- **Phase Gate**: Verify Phase N completion before N+1.
- **Context Persistence**: Always update and maintain artifacts in `./.antigravity/artifacts/` (tasks, plans, walkthroughs). These are the single source of truth for project evolution across AI assistants and sessions.
- **Git Tracking**: Ensure these artifacts are committed regularly to maintain project context across different environments.

### 9. Testing Philosophy
- **Always Have Full Tests**: Every feature must be accompanied by comprehensive tests.
- **Test Real Scenarios**: Use actual PyTorch models and training loops in integration tests.
- **Mock Carefully**: Only mock external dependencies, not core logic.
- **Verify Explanations**: Test that causal hypotheses make sense and have reasonable confidence scores.

