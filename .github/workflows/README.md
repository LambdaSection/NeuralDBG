# GitHub Actions Workflows

This directory contains the essential CI/CD workflows for the Neural DSL project. The workflows have been consolidated from 20+ files to 4 essential ones for better maintainability and faster execution.

## Active Workflows

### 1. essential-ci.yml - Main CI/CD Pipeline

**Triggers:**
- Push to main/develop branches
- Pull requests to main/develop branches
- Nightly scheduled run at 02:00 UTC

**Jobs:**

#### Lint Job
- Runs on: Ubuntu latest, Python 3.11
- Tools: Ruff (fast Python linter)
- Quick code quality check

#### Test Job
- Matrix: Python 3.8, 3.11, 3.12 × Ubuntu & Windows
- Tests full test suite with pytest
- Generates code coverage reports
- Uploads coverage to Codecov (Ubuntu + Python 3.11 only)

#### Security Job
- Runs on: Ubuntu latest, Python 3.11
- Tools: Bandit (SAST), Safety (dependency vulnerabilities), pip-audit (supply chain)
- Continues on error to avoid blocking CI

**Purpose:** Ensures code quality, compatibility, and security for every change.

### 2. release.yml - Release Automation

**Triggers:**
- Push of version tags (v*)

**Jobs:**

#### Build Job
- Builds source distribution and wheel
- Validates distributions with twine
- Uploads artifacts for publishing jobs

#### PyPI Publishing Job
- Uses trusted publishing (OIDC)
- Publishes to PyPI automatically
- Requires: `pypi` environment configured

#### GitHub Release Job
- Creates GitHub release with tag
- Attaches distribution artifacts
- Uses CHANGELOG.md for release notes

**Purpose:** Automates the complete release process from git tag to published package.

### 3. codeql.yml - Security Analysis

**Triggers:**
- Push to main branch
- Pull requests to main branch
- Weekly scheduled scan (Monday 03:00 UTC)

**Jobs:**

#### Analyze Job
- Matrix: Python, JavaScript/TypeScript
- Uses GitHub CodeQL for deep security analysis
- Uploads results to GitHub Security tab

**Purpose:** Continuous security scanning for vulnerabilities and code quality issues.

### 4. validate-examples.yml - Example Validation

**Triggers:**
- Push/PR changes to examples/ or neural/ directories
- Daily scheduled validation at 03:00 UTC

**Jobs:**

#### Validate Job
- Runs example validation script
- Tests DSL compilation (dry-run)
- Generates visualizations
- Continues on error to avoid blocking

**Purpose:** Ensures example files remain valid and compilation works.

## Workflow Features

### Caching Strategy
All workflows use pip caching for faster dependency installation:
```yaml
cache: 'pip'
cache-dependency-path: |
  setup.py
  requirements*.txt
```

### Matrix Testing
The main CI workflow tests across:
- Python versions: 3.8, 3.11, 3.12
- Operating systems: Ubuntu, Windows
- Total combinations: 6 (3 versions × 2 OS)

### Security Tools
- **Ruff**: Fast linting (replaces flake8, pylint for speed)
- **Mypy**: Type checking
- **Bandit**: Security issue scanner (SAST)
- **Safety**: Dependency vulnerability scanner
- **pip-audit**: Supply chain security audit
- **CodeQL**: Advanced security analysis by GitHub

### Secrets Required

For full functionality, configure these secrets:

| Secret | Purpose | Required For |
|--------|---------|--------------|
| CODECOV_TOKEN | Code coverage reporting | essential-ci.yml |
| PYPI_API_TOKEN | PyPI publishing | release.yml (or use trusted publishing) |
| GITHUB_TOKEN | Automatic (GitHub provides) | All workflows |

## Removed Workflows (Consolidated)

The following workflows were removed during cleanup:

### Replaced Workflows
- `ci.yml` → Replaced by `essential-ci.yml`
- `pylint.yml` → Consolidated into `essential-ci.yml` (lint job)
- `pre-commit.yml` → Consolidated into `essential-ci.yml` (lint job)
- `pypi.yml` → Replaced by `release.yml`
- `python-publish.yml` → Replaced by `release.yml`
- `validate_examples.yml` → Replaced by `validate-examples.yml` (renamed)

### Consolidated Security Workflows
- `security.yml` → Consolidated into `essential-ci.yml` (security job)
- `security-audit.yml` → Consolidated into `essential-ci.yml` (security job)
- `snyk-security.yml` → Removed (redundant with CodeQL)

### Removed Specialized Workflows
- `aquarium-release.yml` - Feature-specific release
- `automated_release.yml` - Redundant automation
- `benchmarks.yml` - Run manually when needed
- `close-fixed-issues.yml` - Issue automation (not essential)
- `complexity.yml` - Metrics (not essential for CI)
- `metrics.yml` - Metrics (not essential for CI)
- `periodic_tasks.yml` - Consolidated into other workflows
- `post_release.yml` - Post-release automation (manual)
- `pytest-to-issues.yml` - Issue automation (not essential)

## Local Development

Before pushing, run the same checks locally:

```bash
# Lint
python -m ruff check .

# Type check
python -m mypy neural/ --ignore-missing-imports

# Tests
python -m pytest tests/ -v

# Security scan
python -m bandit -r neural/ -ll
python -m pip_audit -l
```

## Workflow Maintenance

### Adding a New Workflow
1. Create `.yml` file in this directory
2. Follow existing patterns for caching, matrix, etc.
3. Test with `act` (GitHub Actions local runner) if possible
4. Update this README

### Modifying Existing Workflows
1. Test changes on a feature branch first
2. Monitor workflow runs in Actions tab
3. Check that all jobs complete successfully
4. Update this README if behavior changes

### Troubleshooting

**Workflow fails on Windows but passes on Ubuntu:**
- Check for path separator issues (`/` vs `\`)
- Verify line endings (CRLF vs LF)
- Test locally on Windows if possible

**Security job always fails:**
- Check if new dependency has known vulnerabilities
- Update dependencies: `pip install -U package-name`
- Add exceptions if false positive (document why)

**Coverage upload fails:**
- Verify CODECOV_TOKEN is set
- Check Codecov service status
- Ensure coverage.xml is generated

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI/CD Best Practices](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Security Workflows Guide](https://docs.github.com/en/code-security/code-scanning)
- [Trusted Publishing for PyPI](https://docs.pypi.org/trusted-publishers/)

## Contact

For questions about workflows:
- Open an issue with the `ci/cd` label
- Ask in Discord: https://discord.gg/KFku4KvS
- Ping maintainers in PR comments
