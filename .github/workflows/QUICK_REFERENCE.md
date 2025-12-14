# GitHub Actions Quick Reference

## Daily Development

### What Runs Automatically?
- **On every push/PR:**
  - CI workflow: linting, type checking, tests (all platforms/versions)
  - Security scanning: bandit, safety, git-secrets, trufflehog
  - CodeQL analysis (on main branch)
  
- **Nightly (scheduled):**
  - Full CI suite (2 AM UTC)
  - Example validation (3 AM UTC)
  
- **Weekly:**
  - Benchmarks (Sunday midnight)
  - CodeQL deep scan (Monday 5:43 PM)

### Common Tasks

#### Running Tests Locally
```bash
# Lint
python -m ruff check .

# Type check
python -m mypy .

# Tests
python -m pytest tests/ -v
```

#### Creating a Release
1. Go to **Actions** → **Automated Release**
2. Click **Run workflow**
3. Select options:
   - Version bump: `patch` (1.2.3 → 1.2.4), `minor` (1.2.3 → 1.3.0), or `major` (1.2.3 → 2.0.0)
   - Skip tests: Only for hotfixes
   - Draft: Review before publishing
   - TestPyPI: Test the release first

#### Releasing Aquarium Desktop App
1. Go to **Actions** → **Aquarium Release**
2. Click **Run workflow**
3. Enter version (e.g., `0.3.0`)
4. Builds will create installers for Windows, macOS, and Linux

#### Manual Security Scan
1. Go to **Actions** → **Security Scanning**
2. Click **Run workflow**

## Workflow Reference

### CI (ci.yml)
**When:** Every push, PR, daily 2 AM UTC  
**Purpose:** Main quality gate - linting, testing, coverage  
**Duration:** ~15-30 minutes (parallel jobs)

### Security (security.yml)
**When:** Push/PR to main/develop  
**Purpose:** Security scanning with multiple tools  
**Duration:** ~5-10 minutes

### CodeQL (codeql.yml)
**When:** Push/PR to main, weekly Mon 5:43 PM  
**Purpose:** GitHub semantic security analysis  
**Duration:** ~10-15 minutes

### Benchmarks (benchmarks.yml)
**When:** Weekly Sunday, manual, push to benchmark code  
**Purpose:** Performance regression detection  
**Duration:** ~10-20 minutes

### Validate Examples (validate_examples.yml)
**When:** Push/PR to examples, daily 3 AM UTC  
**Purpose:** Ensure examples compile and run  
**Duration:** ~8-12 minutes

### Automated Release (automated_release.yml)
**When:** Manual trigger, tag push  
**Purpose:** Version bump, build, publish to PyPI  
**Duration:** ~10-15 minutes

### Aquarium Release (aquarium-release.yml)
**When:** Manual trigger, tag push (aquarium-v*)  
**Purpose:** Build desktop app installers  
**Duration:** ~20-40 minutes (parallel builds)

## Troubleshooting

### CI Failing?
1. Check which job failed (lint, unit-tests, integration-tests, e2e-tests)
2. Run same command locally (see test commands above)
3. Fix issues and push again

### Security Alerts?
1. Review bandit/safety reports in workflow artifacts
2. Fix actual vulnerabilities
3. For false positives, add exclusions in `pyproject.toml`

### Release Failed?
1. Check pre-release-validation job logs
2. Ensure tests pass: `pytest tests/`
3. Ensure linting passes: `ruff check .`
4. Try with `skip_tests: true` only for urgent hotfixes

### Benchmarks Failing?
1. Check if performance degraded significantly
2. Review code changes affecting performance
3. Benchmarks can be re-run manually

## Notifications

### When to Check Workflows
- **Before merging PRs:** Ensure all CI checks pass
- **After release:** Verify PyPI publication
- **Weekly:** Review security scan results
- **Monthly:** Check benchmark trends

### Getting Alerts
Configure GitHub notifications:
- Settings → Notifications → Actions
- Enable for: Failures, workflow runs on your PRs

## Migration from Old Workflows

### If you used:
- **pytest.yml** → Now in `ci.yml` unit-tests job
- **pylint.yml** → Use `ruff check .` (faster, better)
- **pre-commit.yml** → Still use pre-commit locally
- **release.yml** → Use `automated_release.yml` workflow
- **pypi.yml** → Use `automated_release.yml` workflow
- **post_release.yml** → Announce releases manually
- **complexity.yml** → Use `ruff check .` for complexity
- **metrics.yml** → View on GitHub Insights
- **periodic_tasks.yml** → Covered by ci.yml schedule

## Tips

### Speed Up CI
- Run tests locally before pushing
- Use `ruff` instead of `pylint` (10x faster)
- Leverage pip caching (automatic in workflows)

### Best Practices
- Keep commits atomic and well-described
- Run linting before committing: `ruff check . && ruff format .`
- Write tests for new features
- Update examples when changing DSL syntax

### Debugging Failed Workflows
1. Click on failed job
2. Expand failed step
3. Copy error message
4. Reproduce locally
5. Fix and push

## Need Help?

- **Workflow syntax:** [GitHub Actions Docs](https://docs.github.com/en/actions)
- **CI issues:** Check `.github/workflows/README.md`
- **Consolidation details:** Check `.github/workflows/CONSOLIDATION_SUMMARY.md`
