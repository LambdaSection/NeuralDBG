# Workflow Migration Guide

This guide helps you understand the changes to GitHub Actions workflows and how to adapt if you were relying on specific workflows.

## Overview

We consolidated 20+ workflows into 4 essential ones for better maintainability and performance. This guide maps old workflows to their new equivalents.

## Workflow Mapping

### CI/CD Workflows

| Old Workflow | New Workflow | Notes |
|--------------|--------------|-------|
| `ci.yml` | `essential-ci.yml` | Comprehensive CI with lint, test, security |
| `pylint.yml` | `essential-ci.yml` (lint job) | Now uses Ruff for speed |
| `pre-commit.yml` | `essential-ci.yml` (lint job) | Pre-commit hooks still work locally |

**Changes:**
- Linting moved from Pylint to Ruff (faster, same coverage)
- Pre-commit checks now part of main CI
- All CI jobs consolidated into single workflow

**Migration:**
- Update CI badge in README: `ci.yml` → `essential-ci.yml`
- Pre-commit hooks unchanged (still run locally)
- Ruff config in `pyproject.toml` (compatible with existing rules)

### Release Workflows

| Old Workflow | New Workflow | Notes |
|--------------|--------------|-------|
| `pypi.yml` | `release.yml` | Unified PyPI + GitHub releases |
| `python-publish.yml` | `release.yml` | Same functionality, streamlined |

**Changes:**
- Single workflow handles both PyPI and GitHub releases
- Uses trusted publishing (OIDC) for PyPI (more secure)
- Build job separate from publish jobs

**Migration:**
- Configure PyPI trusted publisher (recommended):
  1. Go to PyPI project settings
  2. Add GitHub Actions trusted publisher
  3. Remove PYPI_API_TOKEN secret (optional)
- Or keep using API token in secret

### Security Workflows

| Old Workflow | New Workflow | Notes |
|--------------|--------------|-------|
| `security.yml` | `essential-ci.yml` (security job) | Bandit, Safety, pip-audit |
| `security-audit.yml` | `essential-ci.yml` (security job) | Scheduled runs preserved |
| `snyk-security.yml` | `codeql.yml` | CodeQL covers same + more |

**Changes:**
- Security scans run on every push/PR (was weekly)
- Consolidated into main CI for faster feedback
- CodeQL replaces Snyk (GitHub-native, more comprehensive)

**Migration:**
- No action needed (automatic in CI)
- Security reports in Actions tab (same location)
- SNYK_TOKEN secret no longer needed (can remove)

### Example Validation

| Old Workflow | New Workflow | Notes |
|--------------|--------------|-------|
| `validate_examples.yml` | `validate-examples.yml` | Renamed for consistency |

**Changes:**
- Renamed to use hyphen instead of underscore
- Same functionality, same schedule

**Migration:**
- Update any external references to use new name
- Workflow behavior unchanged

### Removed Workflows

#### Benchmarks
| Old Workflow | Replacement | Notes |
|--------------|-------------|-------|
| `benchmarks.yml` | Manual execution | Run locally when needed |

**Migration:**
```bash
# Run benchmarks locally
python -m pytest tests/benchmarks/ -v
python neural/benchmarks/quick_start.py
```

#### Metrics and Complexity
| Old Workflow | Replacement | Notes |
|--------------|-------------|-------|
| `complexity.yml` | Local tools | Run locally with radon |
| `metrics.yml` | Local tools | Run locally with metrics tools |

**Migration:**
```bash
# Generate complexity metrics locally
pip install radon
radon cc neural/ -a

# Generate code metrics
pip install pylint
pylint neural/ --output-format=text
```

#### Issue Automation
| Old Workflow | Replacement | Notes |
|--------------|-------------|-------|
| `pytest-to-issues.yml` | Manual issue creation | Create issues manually |
| `close-fixed-issues.yml` | Manual issue closing | Close issues in PRs |

**Migration:**
- Create issues manually when tests fail
- Link PRs to issues: "Fixes #123" in PR description
- Issues auto-close when PR merges

#### Release Automation
| Old Workflow | Replacement | Notes |
|--------------|-------------|-------|
| `aquarium-release.yml` | Manual release | Release Aquarium separately |
| `automated_release.yml` | `release.yml` | Unified release process |
| `post_release.yml` | Manual tasks | Post-release tasks manual |
| `periodic_tasks.yml` | Manual execution | Run periodic tasks manually |

**Migration:**
- Aquarium releases: Consider separate repository
- Post-release tasks: Run manually after release
- Periodic tasks: Schedule locally or run as needed

## Breaking Changes

### None for Most Users

The workflow consolidation has no breaking changes for:
- Contributors (CI runs automatically)
- Users (releases work the same)
- Maintainers (same triggers, better performance)

### Potential Issues

#### 1. Direct Workflow References

If you reference workflows by name in documentation:

```markdown
# Before
[![CI](https://github.com/org/repo/actions/workflows/ci.yml/badge.svg)](...)

# After
[![CI](https://github.com/org/repo/actions/workflows/essential-ci.yml/badge.svg)](...)
```

#### 2. Workflow Dependencies

If external tools depend on specific workflows:

- Update tool configuration to use new workflow names
- Check webhook/notification configurations
- Update any automation that triggers workflows

#### 3. Secret Usage

Secrets no longer needed:
- `SNYK_TOKEN` (if using Snyk)
- `DISCORD_WEBHOOK_URL` (if using Discord notifications)

Secrets still needed:
- `CODECOV_TOKEN` (for coverage)
- `PYPI_API_TOKEN` (if not using trusted publishing)
- `GITHUB_TOKEN` (automatic, no action needed)

## New Features

### Consolidated CI

**Benefits:**
- Single workflow status to check
- Faster execution (parallel jobs)
- Better resource usage
- Easier to maintain

**Jobs:**
1. **Lint** - Code quality (Ruff)
2. **Test** - Multi-version, multi-OS testing
3. **Security** - Security scanning

### Matrix Testing

Now testing across:
- Python 3.8, 3.11, 3.12
- Ubuntu and Windows
- 6 combinations total

### Security Every PR

Security scans run on every push/PR instead of weekly:
- Faster vulnerability detection
- Immediate feedback on new dependencies
- No waiting for weekly scan

## Local Development

Recommended workflow before pushing:

```bash
# 1. Run lint
python -m ruff check .

# 2. Run type check
python -m mypy neural/ --ignore-missing-imports

# 3. Run tests
python -m pytest tests/ -v

# 4. Check security (optional)
python -m bandit -r neural/ -ll
python -m pip_audit -l
```

This matches what CI will run, catching issues early.

## Troubleshooting

### CI Fails After Migration

**Symptom:** Workflow fails with "workflow not found"

**Solution:**
- Check workflow filename: `essential-ci.yml` (not `ci.yml`)
- Clear GitHub Actions cache
- Re-run workflow

### Badge Shows "Unknown"

**Symptom:** CI badge in README shows "unknown" status

**Solution:**
- Update badge URL to reference `essential-ci.yml`
- Wait for workflow to run once
- Check workflow name matches exactly

### Security Scan Fails

**Symptom:** Security job fails on dependency vulnerabilities

**Solution:**
- Update vulnerable dependencies: `pip install -U package`
- Check if vulnerability is real or false positive
- If false positive, document and continue-on-error

### Release Fails

**Symptom:** Release workflow can't publish to PyPI

**Solution:**
- Configure trusted publishing on PyPI (recommended)
- Or set PYPI_API_TOKEN secret
- Verify tag format: `v1.2.3` (must start with 'v')

## Rollback Instructions

If you need to restore old workflows:

```bash
# 1. Check out from git history
git checkout HEAD~1 .github/workflows/

# 2. Or restore specific workflow
git checkout HEAD~1 .github/workflows/ci.yml

# 3. Commit and push
git add .github/workflows/
git commit -m "Restore old workflows"
git push
```

## Questions?

- **Documentation**: See `.github/workflows/README.md`
- **Issues**: Open issue with `ci/cd` label
- **Discord**: https://discord.gg/KFku4KvS
- **Examples**: Check workflow files for inline comments

## Timeline

- **Before**: 20+ workflows, complex maintenance
- **After**: 4 essential workflows, streamlined
- **Migration**: Automatic for most users
- **Support**: Ongoing for questions/issues

## Feedback

We welcome feedback on the new workflow structure:
- Open an issue if something doesn't work
- Suggest improvements in discussions
- Contribute workflow enhancements via PR

Thank you for adapting to the improved CI/CD setup!
