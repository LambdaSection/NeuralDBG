# Rules for AI (AntiGravity / Cursor / Copilot)

> Copie des règles du projet — à appliquer par tout assistant IA (AntiGravity, Cursor, etc.)

---

## 1. Always Update README & Changelog
- Every feature or fix must update `README.md` (usage, examples, API if changed).
- Update `CHANGELOG.md` (create if missing) with conventional commit-style entries.

## 2. Zero Friction for Users
- Tools must work out of the box. Minimal config, clear defaults.
- Provide copy-paste examples that run without extra setup.
- No hidden requirements; document any prerequisite explicitly.

## 3. Solve Real Pain Points
- Before building: *"Does this fix a real user pain?"*
- No speculative features; validate need first.
- Prefer solving existing problems over adding new capabilities.

## 4. Security & Quality Tooling
- CI must include **CodeQL**, **SonarQube**, and **Codacy** (or equivalent).
- No shortcuts on static analysis. Fail builds on critical issues.

## 5. One Project at a Time
- Work on **one project at a time**. No parallel development.
- Finish project A or B before switching to the other.

---

> Roadmap des projets A & B : `.antigravity/PROJECTS.md`
