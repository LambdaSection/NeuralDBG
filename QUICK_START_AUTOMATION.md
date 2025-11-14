# Quick Start: Neural DSL Automation

## 🚀 Get Started in 5 Minutes

### 1. Generate Blog Posts

```bash
python scripts/automation/master_automation.py --blog
```

**Output:**
- `docs/blog/medium_v0.3.0_release.md`
- `docs/blog/devto_v0.3.0_release.md`
- `docs/blog/github_v0.3.0_release.md`

### 2. Run Tests and Validate Examples

```bash
python scripts/automation/master_automation.py --test --validate
```

**Output:**
- `test_report.md`
- `examples_validation_report.md`

### 3. Generate Social Media Posts

```bash
python scripts/automation/master_automation.py --social
```

**Output:**
- `docs/social/twitter_v0.3.0.txt`
- `docs/social/linkedin_v0.3.0.txt`

### 4. Daily Maintenance

```bash
python scripts/automation/master_automation.py --daily
```

Or let GitHub Actions handle it automatically (runs daily at 2 AM UTC).

### 5. Full Release (When Ready)

```bash
# Test release (draft, won't publish)
python scripts/automation/master_automation.py --release --version-type patch --draft

# Real release
python scripts/automation/master_automation.py --release --version-type patch
```

---

## 📋 Prerequisites

### Required
- Python 3.8+
- Neural DSL installed (`pip install -e .`)

### Optional (for full automation)
- GitHub CLI (`gh`) - for automated releases
- PyPI API tokens - for publishing
- LLM API keys - for AI features

---

## 🎯 Common Tasks

### Before a Release
1. Update `CHANGELOG.md` with new features
2. Run: `python scripts/automation/master_automation.py --blog --social`
3. Review generated files
4. Run: `python scripts/automation/master_automation.py --release --version-type patch`

### Daily Check
- GitHub Actions runs automatically
- Or run: `python scripts/automation/master_automation.py --daily`

### After a Release
- Blog posts are ready in `docs/blog/`
- Social media posts are ready in `docs/social/`
- GitHub release is created automatically
- PyPI package is published automatically

---

## 🔧 Troubleshooting

### "Module not found"
```bash
pip install -e .
```

### "GitHub CLI not found"
Install from https://cli.github.com/ or use manual releases.

### "Tests fail"
Review test output and fix issues before releasing.

---

## 📚 More Information

- **Full Guide:** `AUTOMATION_GUIDE.md`
- **Script Docs:** `scripts/automation/README.md`
- **AI Guide:** `docs/ai_integration_guide.md`

---

**That's it!** You're ready to automate everything. 🎉

