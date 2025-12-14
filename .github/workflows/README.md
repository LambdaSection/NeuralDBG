# GitHub Actions Workflows

This directory contains the consolidated CI/CD workflows for Neural DSL. The workflows have been streamlined from 20+ to 7 essential workflows.

## Active Workflows

### 1. CI (`ci.yml`)
**Triggers:** Push, Pull Request, Daily Schedule (2 AM UTC)

Comprehensive continuous integration workflow that includes:
- **Lint & Type Check**: Ruff, mypy, flake8
- **Unit Tests**: Python 3.8-3.12 on Ubuntu & Windows with coverage
- **Integration Tests**: Multi-version, multi-platform with coverage
- **E2E Tests**: End-to-end testing across platforms
- **UI Tests**: Dashboard interface testing (Python 3.11 only)
- **Supply Chain Audit**: Bandit, safety, pip-audit security scans
- **Coverage Report**: Aggregated coverage upload to Codecov

This is the primary workflow for all code quality checks and testing.

### 2. Security Scanning (`security.yml`)
**Triggers:** Push to main/develop, Pull Requests, Manual dispatch

Multi-layered security analysis:
- **Bandit**: Python code security scanner
- **Safety**: Dependency vulnerability checker
- **Git Secrets**: Scans for committed secrets/credentials
- **TruffleHog**: Secret detection in git history

### 3. CodeQL Advanced (`codeql.yml`)
**Triggers:** Push to main, Pull Requests, Weekly schedule (Monday 5:43 PM UTC)

GitHub's semantic code analysis for:
- JavaScript/TypeScript code
- Python code

Identifies security vulnerabilities and code quality issues.

### 4. Benchmarks (`benchmarks.yml`)
**Triggers:** Manual dispatch, Weekly schedule (Sunday), Push to benchmarks code

Performance testing workflow:
- Quick benchmarks on Python 3.9-3.11
- Full benchmarks (manual/scheduled only)
- Results uploaded as artifacts
- Optional GitHub Pages publishing

### 5. Validate Examples (`validate_examples.yml`)
**Triggers:** Push/PR to examples or neural code, Daily schedule (3 AM UTC)

Ensures example code works correctly:
- DSL syntax validation
- Compilation tests (TensorFlow & PyTorch)
- Visualization generation
- Notebook format validation
- End-to-end workflow testing

### 6. Automated Release (`automated_release.yml`)
**Triggers:** Manual dispatch (with version bump options), Push to tags (v*)

Comprehensive release automation:
- Pre-release validation (lint, type check, tests)
- Automatic version bumping (major/minor/patch)
- Changelog parsing
- GitHub Release creation
- PyPI publishing (with trusted publishing)
- TestPyPI support for testing
- Draft release option

### 7. Aquarium Release (`aquarium-release.yml`)
**Triggers:** Push to tags (aquarium-v*.*.*), Manual dispatch

Builds and releases the Neural Aquarium desktop application:
- Multi-platform builds (Windows, macOS, Linux)
- Code signing support (when secrets configured)
- Installer creation (.exe, .msi, .dmg, .AppImage, .deb, .rpm)
- Checksum generation
- GitHub Release creation
- Auto-update support

## Removed Workflows

The following workflows were removed during consolidation:

### Deprecated/Low-Value
- `complexity.yml` - Code complexity metrics (low value)
- `metrics.yml` - GitHub metrics with lowlighter/metrics (low value)
- `post_release.yml` - Twitter posting (manual if needed)

### Duplicates
- `pytest-to-issues.yml` - Functionality covered by ci.yml
- `pylint.yml` - Covered by ci.yml lint job
- `pre-commit.yml` - Redundant with ci.yml
- `security-audit.yml` - Merged into security.yml and ci.yml supply-chain-audit
- `snyk-security.yml` - Incomplete, requires token, covered by other security tools
- `pypi.yml` - Duplicate of python-publish.yml
- `python-publish.yml` - Superseded by automated_release.yml
- `release.yml` - Superseded by automated_release.yml
- `close-fixed-issues.yml` - Dependent on removed pytest-to-issues workflow
- `periodic_tasks.yml` - Covered by ci.yml and validate_examples.yml schedules

## Workflow Matrix

| Workflow | Lint | Test | Security | Build | Publish | Schedule |
|----------|------|------|----------|-------|---------|----------|
| CI | ✅ | ✅ | ✅ | - | - | Daily 2 AM |
| Security | - | - | ✅ | - | - | - |
| CodeQL | - | - | ✅ | - | - | Weekly Mon |
| Benchmarks | - | ✅ | - | - | - | Weekly Sun |
| Validate Examples | - | ✅ | - | - | - | Daily 3 AM |
| Automated Release | ✅ | ✅ | - | ✅ | ✅ | - |
| Aquarium Release | - | - | - | ✅ | ✅ | - |

## Usage

### Running Tests
Tests run automatically on push/PR via `ci.yml`. No manual intervention needed.

### Creating a Release
1. **Automated**: Trigger `automated_release.yml` manually from GitHub Actions tab
   - Select version bump type (major/minor/patch)
   - Optionally skip tests/lint
   - Optionally create draft or publish to TestPyPI
2. **Tag-based**: Push a tag like `v1.2.3` to trigger release automatically

### Releasing Aquarium
1. **Automated**: Trigger `aquarium-release.yml` manually with version number
2. **Tag-based**: Push a tag like `aquarium-v0.3.0` to trigger build

### Security Scans
- Run automatically on push/PR
- Can be triggered manually via workflow_dispatch
- Weekly CodeQL scans run automatically

## Maintenance

When adding new workflows:
1. Ensure they don't duplicate existing functionality
2. Document them in this README
3. Consider adding to the workflow matrix above
4. Use appropriate triggers (avoid over-scheduling)
