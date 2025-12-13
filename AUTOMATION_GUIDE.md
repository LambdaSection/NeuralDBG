# Neural DSL Automation Guide

## Overview

Neural DSL now has comprehensive automation for:
- ✅ **Blog Post Generation** - Auto-generate posts from CHANGELOG
- ✅ **Blog Publishing** - Auto-publish to Dev.to and Medium via API
- ✅ **Social Media Posting** - Auto-post to Twitter/X and LinkedIn via API
- ✅ **GitHub Discussions** - Auto-create release announcements
- ✅ **GitHub Releases** - Automated version bumping and releases
- ✅ **PyPI Publishing** - Automated package publishing
- ✅ **Post-Release Automation** - Version updates, discussions, deployments, notifications
- ✅ **Example Validation** - Validate all examples automatically
- ✅ **Test Automation** - Run tests and generate reports
- ✅ **Periodic Tasks** - Daily automated maintenance

## Quick Start

### Generate Blog Posts

```bash
python scripts/automation/master_automation.py --blog
```

This generates:
- `docs/blog/medium_v{version}_release.md`
- `docs/blog/devto_v{version}_release.md`
- `docs/blog/github_v{version}_release.md`

### Run Tests and Validation

```bash
python scripts/automation/master_automation.py --test --validate
```

### Full Release

```bash
# Patch release (0.3.0 -> 0.3.1)
python scripts/automation/master_automation.py --release --version-type patch

# Minor release (0.3.0 -> 0.4.0)
python scripts/automation/master_automation.py --release --version-type minor

# Major release (0.3.0 -> 1.0.0)
python scripts/automation/master_automation.py --release --version-type major
```

### Daily Maintenance

```bash
python scripts/automation/master_automation.py --daily
```

Or let GitHub Actions handle it automatically (runs daily at 2 AM UTC).

## Automation Scripts

### 1. Blog Generator (`blog_generator.py`)

**What it does:**
- Reads CHANGELOG.md
- Extracts release notes for current version
- Generates formatted blog posts for multiple platforms

**Usage:**
```bash
python scripts/automation/blog_generator.py [version]
```

**Output:**
- Medium post
- Dev.to post
- GitHub release notes

### 2. Release Automation (`release_automation.py`)

**What it does:**
- Bumps version in setup.py and __init__.py
- Runs test suite
- Generates release notes
- Creates GitHub release
- Publishes to PyPI
- Generates blog posts

**Usage:**
```bash
python scripts/automation/release_automation.py \
  --version-type patch \
  --skip-tests false \
  --draft false \
  --test-pypi false
```

### 3. Example Validator (`example_validator.py`)

**What it does:**
- Finds all .neural files in examples/
- Validates each example
- Generates validation report

**Usage:**
```bash
python scripts/automation/example_validator.py
```

**Output:**
- `examples_validation_report.md`

### 4. Test Automation (`test_automation.py`)

**What it does:**
- Runs pytest test suite
- Generates coverage reports
- Creates test reports

**Usage:**
```bash
python scripts/automation/test_automation.py
```

**Output:**
- `test_report.md`
- `test_results.json`
- `htmlcov/` (if coverage enabled)

### 5. Social Media Generator (`social_media_generator.py`)

**What it does:**
- Generates Twitter/X posts
- Generates LinkedIn posts
- Formats for each platform

**Usage:**
```bash
python scripts/automation/social_media_generator.py
```

**Output:**
- `docs/social/twitter_v{version}.txt`
- `docs/social/linkedin_v{version}.txt`

## GitHub Actions Workflows

### Marketing Automation (`.github/workflows/marketing_automation.yml`) ⭐ NEW

**The complete marketing automation solution!**

**Triggers:**
- Release publication (automatic)
- Manual dispatch

**Actions:**
1. Validate API secrets
2. Generate blog posts and social media content
3. Publish to Dev.to via API (immediate)
4. Publish to Medium via API (draft)
5. Post to Twitter/X via API
6. Post to LinkedIn via API
7. Create GitHub Discussion with announcement
8. Commit generated files to repository

**Features:**
- ✅ Comprehensive error handling
- ✅ Secret validation before publishing
- ✅ Continue-on-error for individual platforms
- ✅ Detailed workflow summary
- ✅ Artifact upload for all generated content

**Documentation:**
- [Full Guide](docs/MARKETING_AUTOMATION_GUIDE.md)
- [Quick Reference](docs/MARKETING_AUTOMATION_QUICK_REF.md)
- [Workflows README](.github/workflows/README.md)

### Automated Release (`.github/workflows/automated_release.yml`)

**Triggers:**
- Manual dispatch (with options)
- Tag push (v*)

**Actions:**
1. Checkout code
2. Set up Python
3. Install dependencies
4. Run tests
5. Generate blog posts
6. Validate examples
7. Bump version (if manual)
8. Create GitHub release
9. Upload artifacts

**Usage:**
1. Go to GitHub Actions tab
2. Select "Automated Release"
3. Click "Run workflow"
4. Choose options:
   - Version type (patch/minor/major)
   - Skip tests (true/false)
   - Draft release (true/false)

### Periodic Tasks (`.github/workflows/periodic_tasks.yml`)

**Schedule:** Daily at 2 AM UTC

**Actions:**
1. Run tests
2. Validate examples
3. Generate reports
4. Upload artifacts (kept for 30 days)

**No manual action needed** - runs automatically!

### Post-Release Automation (`.github/workflows/post_release_automation.yml`)

**Triggers:**
- Automatic: After release is published
- Manual: Via workflow dispatch

**Actions:**
1. Update version to next dev version (e.g., 0.3.0 → 0.3.1.dev0)
2. Create GitHub Discussion announcement
3. Update documentation links
4. Trigger Netlify/Vercel deployments
5. Send Discord notification
6. Create planning issue for next release

**See:** [POST_RELEASE_AUTOMATION_QUICK_REF.md](POST_RELEASE_AUTOMATION_QUICK_REF.md) for details

## Typical Workflow

### For a New Release

1. **Update CHANGELOG.md** with new features/fixes
2. **Run automation:**
   ```bash
   python scripts/automation/master_automation.py --release --version-type patch
   ```
3. **Review generated files:**
   - Blog posts in `docs/blog/`
   - Social media posts in `docs/social/`
   - Release notes
4. **Manual steps** (optional):
   - Edit blog posts if needed
   - Post to social media
   - Update documentation

### For Daily Maintenance

**Automatic** - GitHub Actions handles it!

Or run manually:
```bash
python scripts/automation/master_automation.py --daily
```

## Setup Requirements

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install build twine pytest pytest-json-report
   ```

2. **Install GitHub CLI** (for releases):
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   ```

3. **Authenticate GitHub CLI:**
   ```bash
   gh auth login
   ```

### GitHub Actions Setup

1. **Secrets for PyPI Publishing:**
   - `PYPI_API_TOKEN` - PyPI API token
   - `TEST_PYPI_API_TOKEN` - TestPyPI API token (optional)

2. **Secrets for Marketing Automation:**
   - `DEVTO_API_KEY` - Dev.to API key
   - `MEDIUM_API_KEY` - Medium integration token
   - `MEDIUM_USER_ID` - Medium user ID (optional)
   - `TWITTER_API_KEY` - Twitter API key
   - `TWITTER_API_SECRET` - Twitter API secret
   - `TWITTER_ACCESS_TOKEN` - Twitter access token
   - `TWITTER_ACCESS_TOKEN_SECRET` - Twitter access token secret
   - `LINKEDIN_ACCESS_TOKEN` - LinkedIn access token
   - `LINKEDIN_PERSON_URN` - LinkedIn person URN (optional)

3. **Permissions:**
   - Contents: write (for releases and commits)
   - Actions: read (for workflows)
   - Discussions: write (for announcements)

## Customization

### Blog Post Templates

Edit `scripts/automation/blog_generator.py`:
- Customize post format
- Add platform-specific sections
- Modify styling

### Social Media Posts

Edit `scripts/automation/social_media_generator.py`:
- Adjust post length
- Customize hashtags
- Change formatting

### Release Process

Edit `scripts/automation/release_automation.py`:
- Modify version bumping logic
- Add custom release steps
- Change notification methods

## Troubleshooting

### GitHub CLI not found
```bash
# Install from https://cli.github.com/
# Or use manual release creation in GitHub UI
```

### PyPI upload fails
- Check API tokens in GitHub Secrets
- Use `--test-pypi` first to test
- Verify package name and version

### Tests fail
- Review test output
- Fix failing tests
- Use `--skip-tests` only for emergency releases

### Blog posts not generating
- Check CHANGELOG.md format
- Verify version exists in changelog
- Check file permissions

## Future Enhancements

- [x] ~~Automated blog post publishing to Dev.to and Medium~~ ✅ DONE
- [x] ~~Automated social media posting to Twitter/X and LinkedIn~~ ✅ DONE
- [x] ~~Automated GitHub Discussions creation~~ ✅ DONE
- [ ] Automated documentation generation
- [ ] Automated example generation from templates
- [ ] Automated dependency updates (Dependabot)
- [ ] Automated security scanning
- [ ] Automated performance benchmarking
- [ ] Automated changelog generation from commits
- [ ] Automated translation of blog posts
- [ ] Automated newsletter generation
- [ ] Hashnode and other blog platform support

## Support

For issues or questions:
- Open an issue on GitHub
- Check `scripts/automation/README.md` for detailed docs
- Review GitHub Actions logs for errors

---

**Last Updated:** October 18, 2025
**Version:** 1.0.0

