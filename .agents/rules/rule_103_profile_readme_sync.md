---
name: rule-103-profile-readme-sync
description: 'RULE 103: Profile README Sync — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_103_profile_readme_sync.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 103: Profile README Sync — MANDATORY

## Rule

AI agents MUST verify that BOTH GitHub profile READMEs are up-to-date whenever a project is added, renamed, or its status changes.

## Profile READMEs

| Account | Repo | README path | Portfolio |
|---------|------|-------------|-----------|
| **LambdaSection** | `LambdaSection/.github` | `profile/README.md` | — |
| **Lemniscate-world** | `Lemniscate-world/Lemniscate-world` | `README.md` (root) | `index.html` (GitHub Pages) |

## Portfolio Dashboard (GitHub Pages)

- **URL**: https://lemniscate-world.github.io/Lemniscate-world/
- **Source**: Auto-generated from `Epingle_Projets.md`
- **Generation**: `python ~/Documents/kuro-rules/scripts/generate_portfolio.py`
- **The README.md MUST feature the dashboard link as its FIRST link**

## When This Rule Triggers

- Adding a new project to any LambdaSection repo
- Changing a project's status (e.g., Private Beta → Stable)
- Updating project descriptions or links
- Any portfolio update (R85, R80)
- User asks "tout est à jour ?" or "profile à jour ?"

## Verification Steps

### MANDATORY: Profile Sync Check
```
ACTION: Fetch current content of both profile READMEs
  - LambdaSection: gh api repos/LambdaSection/.github/contents/profile/README.md
  - Lemniscate-world: gh api repos/Lemniscate-world/Lemniscate-world/contents/README.md

ACTION: Compare projects listed in profiles vs actual repos
  - gh repo list LambdaSection --limit 50
  - gh repo list Lemniscate-world --limit 50

VERIFY: Every public/active repo appears in at least one profile README
```

### What Must Be Listed

Each profile MUST include:
1. **Project name** with correct GitHub link
2. **One-line description** matching the repo description
3. **Current status** (Stable, Beta, Active, etc.)
4. **Version or stars** when relevant

### LambdaSection Profile
- Focus: LambdaSection organization projects
- Style: Clean table format with links
- Must include: NeuralDBG, Astral, Aquarium, Metatron, TokenWise, Datalint, Sugar, Logos, Odin, Neural-Agent

### Lemniscate-world Profile
- Focus: Personal projects + LambdaSection highlights
- Style: Personal/bio style with project cards
- Must include: All LambdaSection projects + personal repos (OpenQuant, Charmed, Dissect, etc.)

## Enforcement

### Violation Detection
```
VIOLATION: New project added to LambdaSection but not in either profile README
VIOLATION: Project status changed (e.g., Stable → Deprecated) but profiles not updated
VIOLATION: Profile README links to wrong repo (e.g., Lemniscate-world/NeuralDBG instead of LambdaSection/NeuralDBG)
```

### Auto-Fix Procedure
1. Clone the profile repo
2. Update the README with accurate project info
3. Commit and push
4. Report: "Updated profile README for [account] — added/updated [N] projects"

## Integration with Other Rules

- **R85 (Portfolio Completeness)**: Profile sync is a SUB-CHECK of portfolio completeness
- **R80 (Epingle Projets)**: Profile READMEs must match Epingle_Projets.md project list
- **R87 (Ownership Verification)**: Verify correct repo links (LambdaSection vs Lemniscate-world)

## Quick Reference

```bash
# Check LambdaSection profile
gh api repos/LambdaSection/.github/contents/profile/README.md --jq '.size'

# Check Lemniscate-world profile
gh api repos/Lemniscate-world/Lemniscate-world/contents/README.md --jq '.size'

# List all repos to compare
gh repo list LambdaSection --limit 50 --json name,description,visibility
gh repo list Lemniscate-world --limit 50 --json name,description,visibility
```

---

**Created**: 2026-05-30
**Applies to**: All project additions, status changes, portfolio updates
**Enforcement**: MANDATORY — Profile desync MUST be fixed before session ends
