# Optimized Distribution Plan for Neural DSL

## Versioning Strategy

### Semantic Versioning Rules
- **Patch (0.0.X)**: Every 15 bugs fixed
- **Minor (0.X.0)**: New feature added
- **Major (X.0.0)**: Stable version with no known bugs (see GitHub Issues)

## One-Command Release Process

### Full Automated Release
```bash
# Complete release workflow (recommended)
python scripts/automation/master_automation.py --release --version-type patch

# This automatically:
# 1. Bumps version in setup.py
# 2. Updates CHANGELOG.md from closed issues
# 3. Generates blog posts (Dev.to, Medium, GitHub)
# 4. Creates social media posts (X, LinkedIn)
# 5. Creates GitHub release
# 6. Triggers PyPI upload via GitHub Actions
```

### Step-by-Step (For Manual Control)

#### Phase 1: Preparation
```bash
# 1. Generate blog posts and social content
python scripts/automation/master_automation.py --blog --social

# 2. Review generated content in:
# - docs/blog/devto_v*.md
# - docs/blog/medium_v*.md
# - docs/social/twitter_v*.txt
# - docs/social/linkedin_v*.txt
```

#### Phase 2: Release
```bash
# 3. Execute release
python scripts/automation/release_automation.py --version-type patch

# 4. Push tags (automated in release script)
# git tag v*.*.*
# git push origin v*.*.*
```

#### Phase 3: Distribution
```bash
# 5. Generate architecture diagrams
pyreverse neural -o png

# 6. PyPI upload (automatically handled by GitHub Actions)
# Manual alternative:
# python setup.py sdist bdist_wheel
# twine upload dist/*
```

## Manual Content Creation

### Required Manual Steps
1. **GIF/Video Creation**: Create demonstration GIF or video
   - Tool: OBS Studio, Peek, or ScreenToGif
   - Focus: Highlight the main feature of the release

2. **Community Posting**:
   - Reddit (r/MachineLearning, r/Python, r/learnmachinelearning)
   - Discord (AI/ML communities)
   - ProductHunt (for major releases)
   - Hacker News (for significant features)

3. **Influencer Outreach** (DM List):
   - Arjun
   - Petar Veličković
   - Sander Dieleman
   - Farhad Salehi, PhD
   - Cassie Kozyrkov
   - Allie K. Miller
   - Demis Hassabis
   - Bernard Marr
   - Andrew Ng

4. **YouTube/Vimeo Content** (Weekly/Bi-weekly):
   - Script with Manim animations
   - Feature demonstrations
   - Comparisons/Benchmarks

## Automated Tools

### Blog Generation
```bash
# Generate blog with insights from closed issues
python scripts/automation/blog_generator.py

# Emphasizes:
# - Main feature of the release
# - Bug fixes from closed issues
# - Code examples from README.md and dsl.md
# - Benchmark comparisons (when available)
```

### Social Media Posts
```bash
# Generate 5 X posts for the week
python scripts/automation/social_media_generator.py

# Hashtags: #MachineLearning #buildinpublic #AI #DeepLearning #MLTools #opensource #Automation
# Communities: buildinpublic, AI/ML/DataScience, quant/acc, Startup Community
```

## GitHub Actions Automation

### Automated Release Workflow
**Trigger**: Push tag `v*.*.*` or manual dispatch

**Actions**:
1. Run tests
2. Generate blog posts
3. Create GitHub release
4. Build Python package
5. Upload to PyPI
6. Post to social media (if configured)

### Periodic Tasks
**Schedule**: Daily at 2 AM UTC

**Actions**:
1. Run test suite
2. Validate examples
3. Generate reports
4. Check for dependency updates

## Environment Setup

### Required Secrets (GitHub)
```bash
# Set these in GitHub repository secrets
PYPI_API_TOKEN=<your_token>
TWITTER_API_KEY=<your_key>
TWITTER_API_SECRET=<your_secret>
TWITTER_ACCESS_TOKEN=<your_token>
TWITTER_ACCESS_TOKEN_SECRET=<your_secret>
```

### Local Environment (.env file)
```bash
# Create .env in project root (add to .gitignore!)
PYPI_TOKEN=your_token_here
TWITTER_API_KEY=your_key_here
TWITTER_API_SECRET=your_secret_here
TWITTER_ACCESS_TOKEN=your_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_secret_here
```

## Weekly Schedule Template

### Monday
- Code development
- Fix bugs
- Implement features

### Tuesday
- Run tests
- Review PRs
- Update documentation

### Wednesday
- Generate release content (if releasing)
- Create demonstration content
- **REVIEW** blog post for Thursday

### Thursday
- **POST** blog to Dev.to
- Share on X, LinkedIn
- Post to Reddit, Discord

### Friday
- Monitor community feedback
- Plan next week's features
- DM influencers (3 people)

### Weekend
- YouTube/Vimeo content creation
- Benchmark experiments
- Research new features

## Quick Reference

### Single Command Releases
```bash
# Patch release (bug fixes)
python scripts/automation/master_automation.py --release --version-type patch

# Minor release (new feature)
python scripts/automation/master_automation.py --release --version-type minor

# Major release (stable, bug-free)
python scripts/automation/master_automation.py --release --version-type major
```

### Daily Automation
```bash
# Run all daily tasks
python scripts/automation/master_automation.py --daily
```

### Content Only (No Release)
```bash
# Just generate blog posts
python scripts/automation/master_automation.py --blog

# Just generate social media
python scripts/automation/master_automation.py --social
```

## Comparison: Before vs After

| Step | Before (Manual) | After (Automated) |
|------|-----------------|-------------------|
| Version bump | Manual edit | `--release` flag |
| CHANGELOG | Manual write | Auto from issues |
| Blog posts | Manual write | Auto-generated |
| Social media | Manual write | Auto-generated |
| PyPI upload | Manual twine | GitHub Actions |
| Git tags | Manual commands | Auto in script |
| **Total Time** | **2-3 hours** | **15 minutes** |

## Notes

- All automation scripts are in `scripts/automation/`
- See `AUTOMATION_GUIDE.md` for detailed documentation
- Credentials should **never** be committed to git
- Use `.env` for local development, GitHub Secrets for CI/CD
