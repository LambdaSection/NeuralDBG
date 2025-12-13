# Automation Reference Guide

Complete reference for Neural DSL's automation systems including releases, content generation, and maintenance.

## Quick Start

### One-Command Release

```bash
# Complete automated release (recommended)
python scripts/automation/master_automation.py --release --version-type patch
```

This single command replaces 20+ manual steps and reduces release time from 2-3 hours to ~15 minutes.

### Common Tasks

```bash
# Generate blog posts
python scripts/automation/master_automation.py --blog

# Generate social media content
python scripts/automation/master_automation.py --social

# Run daily maintenance
python scripts/automation/master_automation.py --daily

# Run tests and validate examples
python scripts/automation/master_automation.py --test --validate
```

## Versioning Strategy

### Semantic Versioning Rules
- **Patch (0.0.X)**: Every 15 bugs fixed
- **Minor (0.X.0)**: Each new feature added
- **Major (X.0.0)**: Stable, bug-free release

### Version Commands

```bash
# Patch release (0.3.0 → 0.3.1)
python scripts/automation/master_automation.py --release --version-type patch

# Minor release (0.3.0 → 0.4.0)
python scripts/automation/master_automation.py --release --version-type minor

# Major release (0.3.0 → 1.0.0)
python scripts/automation/master_automation.py --release --version-type major

# Test release (draft, won't publish)
python scripts/automation/master_automation.py --release --version-type patch --draft
```

## Release Workflow

### Automated Steps

When you run a release, these steps are automated:

1. **Version Update**
   - Bump version in `setup.py` and `neural/__init__.py`
   - Update README badges
   - Commit changes
   - Create and push git tag

2. **Validation** (5-10 minutes)
   - Linting with Ruff
   - Type checking with mypy
   - Full test suite
   - Example validation

3. **Build & Publish**
   - Build source distribution and wheel
   - Verify package integrity
   - Parse CHANGELOG for release notes
   - Create GitHub release
   - Publish to PyPI via trusted publishing

4. **Content Generation**
   - Generate blog posts (Dev.to, Medium, GitHub)
   - Generate social media posts (X/Twitter, LinkedIn)

### What's Still Manual

- GIF/Video creation
- Community posting (Reddit, Discord, ProductHunt, Hacker News)
- Influencer outreach (DMs)
- YouTube/Vimeo content creation

## Content Generation

### Blog Posts

```bash
# Generate all blog posts
python scripts/automation/master_automation.py --blog
```

**Output:**
- `docs/blog/medium_v0.3.0_release.md`
- `docs/blog/devto_v0.3.0_release.md`
- `docs/blog/github_v0.3.0_release.md`

**Content includes:**
- Main feature highlights
- Bug fixes from closed issues
- Code examples from README and docs
- Benchmark comparisons (when available)

### Social Media

```bash
# Generate social media posts
python scripts/automation/master_automation.py --social
```

**Output:**
- `docs/social/twitter_v0.3.0.txt` (5 posts for the week)
- `docs/social/linkedin_v0.3.0.txt`

**Hashtags used:**
- #MachineLearning #buildinpublic #AI #DeepLearning
- #MLTools #opensource #Automation

## Post-Release Automation

After publishing a release (e.g., v0.3.0), the post-release workflow automatically:

1. ✅ Updates version to next dev cycle (0.3.1.dev0)
2. ✅ Creates GitHub Discussion announcement
3. ✅ Updates documentation links
4. ✅ Triggers website deployments (Netlify/Vercel)
5. ✅ Sends Discord notification
6. ✅ Creates planning issue for next release

### Setup (One-Time)

**Enable Discussions:**
```
Repository → Settings → Features → ☑ Discussions
```

**Set Permissions:**
```
Repository → Settings → Actions → General → Workflow permissions
☑ Read and write permissions
```

**Add Secrets (Optional):**
```
Repository → Settings → Secrets → Actions → New repository secret
```

| Secret Name | Get From | Required |
|-------------|----------|----------|
| `NETLIFY_BUILD_HOOK` | Netlify → Build Hooks | Optional |
| `VERCEL_DEPLOY_HOOK` | Vercel → Deploy Hooks | Optional |
| `DISCORD_WEBHOOK_URL` | Discord → Channel → Webhooks | Optional |

### Trigger Post-Release Automation

**Automatic (Recommended):**
```bash
# Create and push tag
git tag v0.3.0
git push origin v0.3.0
# Workflow runs automatically after release is published
```

**Manual Trigger:**
```bash
gh workflow run post_release_automation.yml -f version="0.3.0"
```

### Verify Post-Release

After workflow completes (2-5 minutes), check:

- [ ] `setup.py` version is `0.3.1.dev0`
- [ ] `neural/__init__.py` `__version__` is `0.3.1.dev0`
- [ ] Discussions tab has new announcement
- [ ] Issues tab has planning issue (labels: `release`, `planning`)
- [ ] Netlify/Vercel deployment triggered
- [ ] Discord notification sent

## Environment Setup

### Local Development

Create `.env` file in project root:

```bash
# .env (add to .gitignore!)
PYPI_TOKEN=your_token_here
TWITTER_API_KEY=your_key_here
TWITTER_API_SECRET=your_secret_here
TWITTER_ACCESS_TOKEN=your_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_secret_here
```

### GitHub Actions

Add secrets in repository settings:

```
Settings → Secrets and variables → Actions → New repository secret
```

Required secrets:
- `PYPI_API_TOKEN`
- `TWITTER_API_KEY`
- `TWITTER_API_SECRET`
- `TWITTER_ACCESS_TOKEN`
- `TWITTER_ACCESS_TOKEN_SECRET`

## Weekly Workflow

### Monday-Tuesday
- Code development
- Fix bugs
- Implement features

### Wednesday
- Generate release content if ready
- Create demonstration content
- **REVIEW** blog post for Thursday

### Thursday
- **POST** blog to Dev.to
- Share on X/Twitter and LinkedIn
- Post to Reddit and Discord

### Friday
- Monitor community feedback
- Plan next week's features
- DM influencers (3 people)

### Weekend
- YouTube/Vimeo content creation
- Benchmark experiments
- Research new features

## Troubleshooting

### Tests Fail

```bash
# Run locally to debug
pytest tests/ -v -x

# Fix issues, then re-run workflow
```

### PyPI Publishing Fails

1. Check environment name is `pypi` (exact match)
2. Verify trusted publisher configuration on PyPI
3. Ensure workflow has `id-token: write` permission
4. Check workflow logs for specific error

### Version Already Exists

PyPI doesn't allow re-uploading. Increment version:

```bash
python scripts/automation/master_automation.py --release --version-type patch
```

### Module Not Found

```bash
# Ensure Neural is installed
pip install -e .
```

### GitHub CLI Not Found

Install from https://cli.github.com/ or use manual releases.

### Version Not Updated (Post-Release)

```bash
# Check if commit was made
git log --oneline -5

# Manually update if missing
```

### Discussion Not Created

```bash
# Ensure Discussions are enabled in Settings
# Try manual creation:
python scripts/automation/post_release_helper.py --action discussion --version "0.3.0"
```

### Deployment Not Triggered

```bash
# Test webhooks manually:
curl -X POST "$NETLIFY_BUILD_HOOK"
curl -X POST "$VERCEL_DEPLOY_HOOK"
```

### Discord Not Notified

```bash
# Test webhook:
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"Test"}' "$DISCORD_WEBHOOK_URL"
```

## Command Reference

### Master Automation Script

```bash
# Full release
python scripts/automation/master_automation.py --release --version-type patch

# Blog only
python scripts/automation/master_automation.py --blog

# Social media only
python scripts/automation/master_automation.py --social

# Daily tasks
python scripts/automation/master_automation.py --daily

# Tests and validation
python scripts/automation/master_automation.py --test --validate
```

### Release Automation

```bash
# Basic release
python scripts/automation/release_automation.py --version-type patch

# Test on TestPyPI first
python scripts/automation/release_automation.py --test-pypi

# Create draft release
python scripts/automation/release_automation.py --draft

# Skip tests (not recommended)
python scripts/automation/release_automation.py --skip-tests

# Use trusted publishing (for GitHub Actions)
python scripts/automation/release_automation.py --use-trusted-publishing
```

### GitHub CLI Commands

```bash
# List recent runs
gh run list --workflow=post_release_automation.yml

# Watch current run
gh run watch

# View run logs
gh run view --log

# Manual trigger with all options
gh workflow run post_release_automation.yml \
  -f version="0.3.0" \
  -f skip_version_bump=false \
  -f skip_discussion=false \
  -f skip_deployment=false \
  -f skip_notifications=false
```

## Comparison: Manual vs Automated

| Step | Before (Manual) | After (Automated) |
|------|-----------------|-------------------|
| Version bump | Manual edit | `--release` flag |
| CHANGELOG | Manual write | Auto from issues |
| Blog posts | Manual write | Auto-generated |
| Social media | Manual write | Auto-generated |
| PyPI upload | Manual twine | GitHub Actions |
| Git tags | Manual commands | Auto in script |
| **Total Time** | **2-3 hours** | **15 minutes** |

## Pre-Release Checklist

Before running a release:

- [ ] Update `CHANGELOG.md` with new version section
- [ ] All tests passing locally: `pytest tests/ -v`
- [ ] Linting clean: `ruff check .`
- [ ] Type checking clean: `mypy neural/code_generation neural/utils`
- [ ] Version number correct in changelog
- [ ] All features documented
- [ ] Breaking changes clearly marked

## Integration Flow

Complete release and post-release flow:

```
1. git tag v0.3.0 && git push origin v0.3.0
   ↓
2. release.yml (creates GitHub Release)
   ↓
3. pypi.yml (publishes to PyPI)
   ↓
4. post_release.yml (Twitter announcement)
   ↓
5. post_release_automation.yml
   - Bump to 0.3.1.dev0
   - Create discussion
   - Update docs
   - Trigger deployments
   - Send notifications
   - Create planning issue
   ↓
6. Continue development on v0.3.1.dev0
```

## Additional Resources

- **Full Automation Guide**: [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)
- **Dependency Guide**: [DEPENDENCY_GUIDE.md](DEPENDENCY_GUIDE.md)
- **Migration Guide**: [MIGRATION_GUIDE_DEPENDENCIES.md](MIGRATION_GUIDE_DEPENDENCIES.md)
- **Release Workflow**: [RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md)
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
- **Discord**: [Join our community](https://discord.gg/KFku4KvS)
- **Email**: Lemniscate_zero@proton.me
