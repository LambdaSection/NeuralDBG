---
name: rule-106-plan-roadmap-split
description: 'RULE 106: Private Plan + Public Roadmap Split — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_106_plan_roadmap_split.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 106: Private Plan + Public Roadmap Split — MANDATORY

## Problem

Projects need detailed internal planning (honest metrics, failure assessments, sensitive strategy) but public repos should only show a clean, trustworthy roadmap. Without a dedicated rule:

- Agents commit PLAN.md with internal metrics (download counts, failure assessments, strategy notes) to public repos
- Sensitive information leaks into GitHub (launch failures, competitor weaknesses, contacts)
- OR agents keep everything private and the public has no visibility into the project direction

## Solution

Every project MUST maintain two separate documents:

### 1. `PLAN.md` (PRIVATE — gitignored)

Internal tactical document. Contains:

- Honest status assessments (what's failing, what's blocked)
- Internal metrics (download counts, star counts, failure rates)
- Sensitive strategy (competitor weaknesses, contacts, business model details)
- Raw post-mortems and lessons learned
- Detailed task breakdowns with personal notes

**MUST be in `.gitignore`.** Never committed to a public repository.

### 2. `ROADMAP.md` (PUBLIC — tracked)

Clean public roadmap. Contains:

- Vision and architecture (no sensitive details)
- Version roadmap with checkboxes (what's done, what's next)
- Install instructions and quick start
- High-level milestones (no internal metrics)
- License

**MUST be committed** to the public repository.

## What Goes Where

| Content | PLAN.md (private) | ROADMAP.md (public) |
|---------|-------------------|---------------------|
| Vision | Yes | Yes (trimmed) |
| Honest status ("launch failed") | Yes | No |
| Metrics (stars, downloads) | Yes | No |
| Failure assessments | Yes | No |
| Competitor weaknesses | Yes | No |
| Version roadmap | Yes | Yes (clean) |
| Install/quickstart | Yes | Yes |
| Internal contacts | Yes | No |
| Architecture diagram | Yes | Yes (public-friendly) |
| Bug catalog status | Yes | High-level only |

## Process

### At project creation:

1. Create `PLAN.md` with full internal detail
2. Add `PLAN.md` to `.gitignore`
3. Create `ROADMAP.md` with public-friendly content
4. Commit `ROADMAP.md` to the repo

### During development:

1. Update `PLAN.md` after every session (honest status)
2. Update `ROADMAP.md` when milestones are reached (clean summary)
3. NEVER copy sensitive content from PLAN.md to ROADMAP.md

### At session start (R100 integration):

1. Read `PLAN.md` for internal context
2. Verify `PLAN.md` is in `.gitignore`
3. Read `ROADMAP.md` to understand public-facing state

## Verification

```
At project start:
  IF PLAN.md exists AND is NOT in .gitignore:
    VIOLATION: Private plan is tracked
    ACTION: git rm --cached PLAN.md && add to .gitignore

  IF ROADMAP.md does not exist:
    VIOLATION: No public roadmap
    ACTION: Create ROADMAP.md from PLAN.md (remove sensitive content)

  IF PLAN.md contains ROADMAP.md content (verbatim):
    WARNING: Content may be duplicated — keep PLAN.md internal

At session end:
  IF PLAN.md was modified:
    VERIFY: Changes are NOT in ROADMAP.md verbatim
    VERIFY: No sensitive metrics in git diff
```

## Violation Examples

**VIOLATION**: PLAN.md committed with download counts, failure metrics, and competitor analysis.
**CORRECT**: PLAN.md in .gitignore. ROADMAP.md has clean version roadmap only.

**VIOLATION**: No ROADMAP.md — public has no visibility into project direction.
**CORRECT**: ROADMAP.md exists with vision, milestones, install instructions.

**VIOLATION**: ROADMAP.md contains "launch failed, only 21 stars, HN blocked."
**CORRECT**: ROADMAP.md says "Phase 10 ongoing, community posts planned."

## Rationale

Open-source projects need both:
- **Internal honesty** (what's working, what's failing, what to fix) — for the team
- **Public trust** (clear roadmap, clean communication) — for users and contributors

Mixing these two damages both: internal honesty gets leaked (embarrassing), or public communication gets polluted with internal noise (confusing).

---

**Created**: 2026-06-07
**Trigger**: PLAN.md rewrite revealed it was gitignored — needed a formal rule for the private/public split
**Applies to**: All projects with both internal planning and public presence
**Enforcement**: MANDATORY
**Pairs with**: R76 (.gitignore), R97 (launch planning), R12 (plan existence), R104 (issue tracking)
