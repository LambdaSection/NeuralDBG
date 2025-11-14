# 🤖 Neural DSL Automation - Quick Reference

## What's Automated?

✅ **Blog Posts** - Auto-generated from CHANGELOG.md  
✅ **GitHub Releases** - Automated version bumping and releases  
✅ **PyPI Publishing** - Automated package publishing  
✅ **Example Validation** - All examples validated automatically  
✅ **Test Reports** - Automated test running and reporting  
✅ **Social Media** - Auto-generated posts for Twitter/LinkedIn  
✅ **Daily Maintenance** - Runs automatically via GitHub Actions  

---

## Quick Commands

```bash
# Generate blog posts
python scripts/automation/master_automation.py --blog

# Run tests and validate
python scripts/automation/master_automation.py --test --validate

# Generate social media posts
python scripts/automation/master_automation.py --social

# Full release (patch/minor/major)
python scripts/automation/master_automation.py --release --version-type patch

# Daily maintenance
python scripts/automation/master_automation.py --daily
```

---

## GitHub Actions

### Automated Release
- **Location:** Actions → "Automated Release"
- **Trigger:** Manual dispatch or tag push
- **Actions:** Tests, blog generation, release creation

### Periodic Tasks
- **Schedule:** Daily at 2 AM UTC
- **Actions:** Tests, validation, reports
- **No action needed** - runs automatically!

---

## Files Generated

### Blog Posts
- `docs/blog/medium_v{version}_release.md`
- `docs/blog/devto_v{version}_release.md`
- `docs/blog/github_v{version}_release.md`

### Social Media
- `docs/social/twitter_v{version}.txt`
- `docs/social/linkedin_v{version}.txt`

### Reports
- `test_report.md`
- `examples_validation_report.md`

---

## Setup (One-Time)

1. **Install dependencies:**
   ```bash
   pip install build twine pytest pytest-json-report
   ```

2. **Install GitHub CLI** (optional, for releases):
   ```bash
   brew install gh  # macOS
   # or https://cli.github.com/
   ```

3. **Set GitHub Secrets** (optional, for PyPI):
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`

---

## Documentation

- **Full Guide:** `AUTOMATION_GUIDE.md`
- **Quick Start:** `QUICK_START_AUTOMATION.md`
- **Script Docs:** `scripts/automation/README.md`

---

**That's it!** Everything is automated. 🎉

