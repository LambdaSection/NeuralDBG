# Neural DSL Automation Scripts

This directory contains automation scripts for:
- Blog post generation
- GitHub releases
- PyPI publishing
- Example validation
- Test automation
- Social media posts

## Scripts

### `blog_generator.py`
Generates blog posts from CHANGELOG.md for multiple platforms.

**Usage:**
```bash
python scripts/automation/blog_generator.py [version]
```

**Output:**
- `docs/blog/medium_v{version}_release.md`
- `docs/blog/devto_v{version}_release.md`
- `docs/blog/github_v{version}_release.md`

### `release_automation.py`
Automates the full release process: version bumping, testing, GitHub releases, PyPI publishing.

**Usage:**
```bash
# Patch release (default)
python scripts/automation/release_automation.py

# Minor release
python scripts/automation/release_automation.py --version-type minor

# Major release
python scripts/automation/release_automation.py --version-type major

# Draft release (for testing)
python scripts/automation/release_automation.py --draft

# Skip tests
python scripts/automation/release_automation.py --skip-tests

# Publish to TestPyPI first
python scripts/automation/release_automation.py --test-pypi
```

### `example_validator.py`
Validates all examples in the `examples/` directory.

**Usage:**
```bash
python scripts/automation/example_validator.py
```

**Output:**
- `examples_validation_report.md`

### `test_automation.py`
Runs tests and generates reports.

**Usage:**
```bash
# Run tests with coverage
python scripts/automation/test_automation.py

# Or import and use
from scripts.automation.test_automation import TestAutomation
automation = TestAutomation()
automation.run_and_report(coverage=True)
```

**Output:**
- `test_report.md`
- `test_results.json`
- `htmlcov/` (if coverage enabled)

### `social_media_generator.py`
Generates social media posts from release information.

**Usage:**
```bash
python scripts/automation/social_media_generator.py
```

**Output:**
- `docs/social/twitter_v{version}.txt`
- `docs/social/linkedin_v{version}.txt`

## GitHub Actions

### Automated Release Workflow
Located at `.github/workflows/automated_release.yml`

**Triggers:**
- Manual dispatch (with options)
- Tag push (v*)

**Actions:**
1. Run tests
2. Generate blog posts
3. Validate examples
4. Bump version (if manual)
5. Create GitHub release
6. Upload artifacts

### Periodic Tasks Workflow
Located at `.github/workflows/periodic_tasks.yml`

**Schedule:** Daily at 2 AM UTC

**Actions:**
1. Run tests
2. Validate examples
3. Generate reports
4. Upload artifacts

## Setup

### Required Tools

1. **GitHub CLI** (for releases):
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   
   # Or download from https://cli.github.com/
   ```

2. **Python packages**:
   ```bash
   pip install build twine pytest pytest-json-report
   ```

3. **GitHub Actions Secrets** (if publishing to PyPI):
   - `PYPI_API_TOKEN` - PyPI API token
   - `TEST_PYPI_API_TOKEN` - TestPyPI API token (optional)

## Workflow

### Typical Release Process

1. **Update CHANGELOG.md** with new features/fixes
2. **Run automation**:
   ```bash
   python scripts/automation/release_automation.py --version-type patch
   ```
3. **Review generated files**:
   - Blog posts in `docs/blog/`
   - Social media posts in `docs/social/`
   - Release notes
4. **Manual steps** (if needed):
   - Review and edit blog posts
   - Post to social media
   - Update documentation

### Automated Daily Tasks

GitHub Actions runs daily to:
- Validate all examples
- Run test suite
- Generate reports
- Upload artifacts

## Customization

### Blog Post Templates
Edit `blog_generator.py` to customize:
- Post format
- Platform-specific formatting
- Additional sections

### Social Media Posts
Edit `social_media_generator.py` to customize:
- Post length
- Hashtags
- Formatting

### Release Process
Edit `release_automation.py` to customize:
- Version bumping logic
- Release steps
- Notification methods

## Troubleshooting

### GitHub CLI not found
Install from https://cli.github.com/ or use manual release creation.

### PyPI upload fails
Check API tokens in GitHub Secrets or use `--test-pypi` first.

### Tests fail
Review test output and fix issues before releasing.

## Future Enhancements

- [ ] Automated documentation generation
- [ ] Automated example generation
- [ ] Automated dependency updates
- [ ] Automated security scanning
- [ ] Automated performance benchmarking
- [ ] Automated changelog generation from commits

