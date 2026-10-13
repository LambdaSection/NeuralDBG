---
name: rule-105-multirepo-governance
description: 'RULE 105: Multi-Repo & Monorepo Governance — MANDATORY'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_105_multirepo_governance.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 105: Multi-Repo & Monorepo Governance — MANDATORY

## Problem

Kuro Rules currently treat every repository as an isolated unit. The portfolio contains
explicit multi-repo and monorepo structures that are not covered by any existing rule:

1. **VS Code multi-root workspaces** (e.g. `DEMETER.code-workspace` linking
   `Constant_Yield`, `DevDemeterDAO`, `XP_Farming_System`, `GALT-Flatcoin-Concept`).
2. **Git submodules** inside an existing project (e.g. `lib/openzeppelin-contracts`,
   `lib/osx` in DevDemeterDAO).
3. **Cross-repo dependencies** declared via Solidity import paths
   (`../../DevDemeterDAO/contracts/interfaces/IVeDummy.sol`) and shared interfaces
   (`IProductReporter`, `IStabilityEngine`).
4. **Coordinated phases** spanning multiple repos (Phase 0.5 in `DEV_ROADMAP.md`).

Without a dedicated rule, AI agents:
- Do not see the dependency graph when planning a change.
- Modify one repo without auditing the impact on consumers.
- Misclassify the workspace structure (treats a multi-root as a single project).
- Risk breaking consumers of shared interfaces when upgrading semantics.

## Solution

Treat any structure meeting one of the detection criteria below as a **Multi-Repo
Ecosystem (MRE)** or **Monorepo**. The agent MUST follow the **Ecosystem Lifecycle**
defined in this rule.

### 1. Detection (MANDATORY at session start)

The agent MUST run the following checks when a session starts in a project that
may be part of a multi-repo or monorepo:

| Signal | Path / pattern | Interpretation |
|---|---|---|
| Multi-root workspace | `*.code-workspace` at repo root | MRE (VS Code multi-root) |
| Git submodules | `.gitmodules` | MRE with submodules |
| pnpm workspace | `pnpm-workspace.yaml` | Monorepo |
| npm workspaces | `package.json` `"workspaces": [...]` | Monorepo |
| Lerna / Nx / Turborepo | `lerna.json`, `nx.json`, `turbo.json` | Monorepo |
| Cross-repo imports | any path that resolves `../../<OtherRepo>/...` | MRE (Sol/TS/Python) |

If any signal matches, activate **Mode Multi-Repo** for the session.

### 2. Ecosystem Artifacts (MUST exist)

For every MRE, the following files MUST be present at the **hub repo** (the repo
that owns the shared interfaces, by default the one referenced first in the
workspace file or the one containing `ECOSYSTEM_MAP.md`):

#### `ECOSYSTEM_MAP.md` (REQUIRED)
- List of every repo in the MRE with: path, role (hub/consumer/library), remote URL, ownership class (R87).
- Directional dependency graph (which repo depends on which).
- Interface ownership (which repo defines `IXxx`, which consume it).
- Deployment order.

#### `COMPATIBILITY_MATRIX.md` (REQUIRED)
- SemVer matrix: rows = repos, columns = versions.
- Status column: `compatible`, `breaking-pending`, `incompatible`.
- Last integration test date and result.
- Required upgrade path when one component moves a major version.

#### `WORKSPACE_PROTOCOLS.md` (REQUIRED for MRE, OPTIONAL for monorepos)
- Branch strategy per repo (R30).
- Tag coordination protocol.
- CI cross-repo rules.
- Lock-step release policy (when applicable).

### 3. AI Agent Behaviour in Mode Multi-Repo

Before any code change in a member repo:

1. **Read** `ECOSYSTEM_MAP.md` and `COMPATIBILITY_MATRIX.md` of the hub repo.
2. **Identify** consumers: every repo that imports from the file being modified.
3. **Impact Matrix**: for each consumer, list: breaking risk, migration cost, test impact.
4. **Plan synchronisation**: if the change is breaking, propose a migration path in
   `COMPATIBILITY_MATRIX.md` BEFORE merging.
5. **Lock the cross-repo contract**: changes to shared interfaces require the
   `ECOSYSTEM_OWNER` to approve (in practice: a comment in the issue and a bump in
   the matrix).

### 4. Plan Coverage (R12 extension)

The `PLAN.md` / `DEV_ROADMAP.md` / `ROADMAP.md` MUST contain an **Écosystème**
section when the repo is part of a MRE:

```markdown
## Écosystème

- **Workspace**: `~/Documents/DEMETER.code-workspace`
- **Repos liés** (owner = role):
  - `DevDemeterDAO` (EXTERNAL per R87 — `Demeter-Financial-Labs/`) — hub interfaces
  - `Constant_Yield` (EXTERNAL) — consumer of IVeDummy
  - `XP_Farming_System` (EXTERNAL) — consumer of IProductReporter
  - `GALT-Flatcoin-Concept` (EXTERNAL) — future consumer of IStabilityEngine
- **Coordination**: voir `ECOSYSTEM_MAP.md` et `COMPATIBILITY_MATRIX.md`
```

### 5. Issue Tracking (R104 extension)

Cross-repo issues MUST be prefixed with `[CROSS]` and link every repo affected:

```markdown
## [CROSS] Phase 0.5 — Centraliser IVeDummy.sol

**Projets :** DevDemeterDAO (hub) / Constant_Yield / XP_Farming_System
**Coordination requise :** bump COMPATIBILITY_MATRIX.md + tag sync
```

### 6. Session Summary (R86 extension)

When ending a session in a member repo, the `SESSION_SUMMARY.md` MUST include
a section:

```markdown
### Impact écosystème
- Repos impactés : [list]
- Breaking change : [oui/non]
- COMPATIBILITY_MATRIX.md mis à jour : [oui/non]
- Issues cross-repo créées : [liens]
```

### 7. Sync Rules (R87 + R11 integration)

`sync-rules.ps1` MUST classify every project before sync:

- **OWNED** (`Lemniscate-world/`, `Lemniscate-SHA-256/`, `pbakaus/`):
  sync `AGENTS.md`, `.cursorrules`, `copilot-instructions.md`, `.windsurfrules`,
  `AI_GUIDELINES.md` in full.
- **EXTERNAL** (e.g. `Demeter-Financial-Labs/`, unknown org):
  sync ONLY the `AGENTS.md` redirector. **NEVER** sync the full rule files
  (`.cursorrules`, `copilot-instructions.md`, `.windsurfrules`, `AI_GUIDELINES.md`)
  — those contain governance rules the external org did not approve.

If a project has no remote or the org is unknown, ASK the user before classifying
it as OWNED.

### 8. Submodules

For git submodules inside a repo:

- The submodule SHA MUST be pinned to a tag, not a branch, before any production
  deployment.
- The `ECOSYSTEM_MAP.md` MUST list every submodule under a dedicated section
  `Sous-modules` with the pinned tag.
- AI agents MUST NOT modify files inside a submodule — open a PR in the
  submodule's source repo instead.

## Verification

```
At session start in a repo:
  IF .gitmodules exists OR *.code-workspace exists OR pnpm-workspace.yaml exists OR
     cross-repo imports detected:
    ACTIVATE: Mode Multi-Repo
    VERIFY: ECOSYSTEM_MAP.md + COMPATIBILITY_MATRIX.md present at hub
    IF missing: STOP -> create them before any code change
    VERIFY: PLAN.md / ROADMAP.md / DEV_ROADMAP.md contains an "Ecosysteme" section
            that references ECOSYSTEM_MAP.md and COMPATIBILITY_MATRIX.md
    IF missing: STOP -> add the section before any code change

Before any code change in a shared interface:
  READ: consumers list in ECOSYSTEM_MAP.md
  WRITE: impact matrix in the issue
  BUMP: COMPATIBILITY_MATRIX.md

Before running sync-rules.ps1:
  CLASSIFY: every project as OWNED, EXTERNAL, or UNKNOWN (KuroUtils.psm1)
  FILTER: EXTERNAL and UNKNOWN projects receive only AGENTS.md (R87)
  NEVER sync .cursorrules / copilot-instructions.md / .windsurfrules /
         AI_GUIDELINES.md to EXTERNAL or UNKNOWN projects
```

## Violation Examples

**VIOLATION**: Modifying `IVeDummy.sol` in DevDemeterDAO without updating
`COMPATIBILITY_MATRIX.md` and without checking Constant_Yield consumers.
**CORRECT**: Open `[CROSS] IVeDummy signature change` issue, list consumers,
bump matrix, then merge with a SemVer major tag.

**VIOLATION**: Running `sync-rules.ps1` without classification, which writes
full rule files into an EXTERNAL repo (Demeter-Financial-Labs).
**CORRECT**: Detect `Demeter-Financial-Labs` org, skip `.cursorrules` /
`AI_GUIDELINES.md` / `copilot-instructions.md` / `.windsurfrules` for that
project.

**VIOLATION**: Treating a `*.code-workspace` as a single project.
**CORRECT**: Open the workspace file, enumerate folders, treat each as a
distinct member of the MRE.

**VIOLATION**: Editing a file inside `lib/openzeppelin-contracts` (submodule).
**CORRECT**: Open a PR upstream, bump the pinned tag in the parent repo only
after the upstream PR is merged.

## Rationale

The Demeter ecosystem, like most modern DeFi/AI portfolios, is structurally
multi-repo. Treating each repo as an island guarantees:
- silent breaking changes,
- inconsistent governance,
- duplicated rules drifting between repos.

R105 forces a hub-and-spoke discipline that the existing rules (R11 sync,
R12 plan, R87 ownership, R91 versioning, R104 issues) only enforce on isolated
repos.

---

**Created**: 2026-06-07
**Trigger**: Audit of `DevDemeterDAO/DEV_ROADMAP.md` Phase 0.5 + absence of
multi-repo rule in kuro-rules
**Applies to**: All projects listed in a `*.code-workspace`, all repos with
`.gitmodules`, all monorepos (pnpm/npm/Lerna/Nx/Turbo)
**Enforcement**: MANDATORY
**Pairs with**: R11, R12, R15, R30, R87, R88, R91, R104
