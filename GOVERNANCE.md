# Governance

## Overview

NeuralDBG is maintained by **LambdaSection** as an open-source MIT project. We follow a lightweight maintainer model inspired by PyTorch Ecosystem requirements.

## Maintainers

| Name | GitHub | Role | Since |
|------|--------|------|-------|
| Jacques-Charles SENOUVO (Kuro) | `@Lemniscate-world` | Lead Maintainer, BDFL | 2025 |

Currently single active maintainer. Second maintainer position is open — see `Roles` below.

Core maintainer has merge rights and is listed in `.github/CODEOWNERS`.

Historical contributors: see `https://github.com/LambdaSection/NeuralDBG/graphs/contributors` and `CHANGELOG.md`.

## Emeritus

| Name | GitHub | Role | Period |
|------|--------|------|--------|
| P3niel | `@P3niel` | Emeritus — docs, integrations, community (past contributions) | 2025 |

## Decision Making

- **Trivial** (docs, typos, CI): single maintainer approval.
- **Standard** (features, bug fixes): PR + maintainer review, CI green.
- **Major** (API breaking, license, governance): issue/RFC + lead maintainer approval. Second maintainer approval required once the position is filled.

Lazy consensus: if no objection within 72h after review, the PR may be merged.

## Roles

- **Maintainer**: review/merge PRs, cut releases, triage issues, enforce Code of Conduct.
- **Contributor**: anyone submitting PRs/issues. Recognized after 5 merged PRs or significant feature.
- **Emeritus**: former maintainers retaining advisory role with no merge obligations.

Becoming a maintainer: sustained contributions (≥3 months, ≥10 merged PRs or equivalent) + nomination by existing maintainer + lead approval. Open call: second maintainer actively sought to meet PyTorch Ecosystem ≥2 requirement.

## Release Methodology

- **Versioning**: SemVer (`MAJOR.MINOR.PATCH`), documented in `CHANGELOG.md`.
- **Cadence**: Monthly minor releases; patches as needed for security/bug fixes.
- **Process**: tag `vX.Y.Z` → `publish.yml` builds and publishes to PyPI → GitHub Release notes.
- **Support**: latest minor (`1.5.x`) actively maintained; see `SECURITY.md` for supported versions.
- **Changelog**: Keep a Changelog format.

## Communication

- Issues/PRs: GitHub
- Security: `SECURITY.md`
- Email: `lemniscate_zero@proton.me` (primary), `neuraldbg@lemniscate.ai`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Contributing: `CONTRIBUTING.md`

## Ecosystem

Related repos:

- `LambdaSection/NeuralDBG` (this repo) — MIT, public
- `LambdaSection/Aquarium` — causal chain visualizer. Currently private during active UX iteration; public viewer is `docs/aquarium.html` in this repo (zero-dependency). Will be made public when stable; see `docs/ecosystem.md`.

## Amending Governance

Changes to this document require PR + lead maintainer approval (unanimous core approval once a second maintainer is onboarded).
