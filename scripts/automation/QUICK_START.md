# Quick Start Guide: Blog Publishing Automation

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Get API Keys

**Dev.to**:
- Visit: https://dev.to/settings/extensions
- Click "Generate API Key"
- Copy the key

**Medium**:
- Visit: https://medium.com/me/settings/security
- Click "Integration tokens"
- Enter description, click "Get token"
- Copy the token

### 3. Configure Credentials

**Option A: Environment Variables**
```bash
export DEVTO_API_KEY="your_devto_api_key_here"
export MEDIUM_API_TOKEN="your_medium_token_here"
```

**Option B: .env File**
```bash
cd scripts/automation
cp .env.example .env
# Edit .env with your credentials
```

### 4. Test the Setup

```bash
# Test blog generation
python scripts/automation/blog_generator.py

# Test Dev.to (dry run - generates files only)
python scripts/automation/master_automation.py --blog

# Check generated files
ls docs/blog/
```

## Common Commands

### Generate Content Only

```bash
# Generate blog posts
python scripts/automation/master_automation.py --blog

# Generate social media posts
python scripts/automation/master_automation.py --social

# Generate everything
python scripts/automation/master_automation.py --marketing
```

### Publish to Dev.to

```bash
# Publish as draft
python scripts/automation/devto_publisher.py --file docs/blog/devto_v0.3.0_release.md

# Publish immediately
python scripts/automation/devto_publisher.py --file article.md --publish

# Batch publish all devto_*.md files
python scripts/automation/devto_publisher.py --directory docs/blog
```

### Publish to Medium

```bash
# Publish as draft
python scripts/automation/medium_publisher.py --file docs/blog/medium_v0.3.0_release.md

# Publish as public
python scripts/automation/medium_publisher.py --file article.md --status public

# List your publications
python scripts/automation/medium_publisher.py --list-publications

# Publish to a publication
python scripts/automation/medium_publisher.py --file article.md --publication-id YOUR_PUB_ID
```

### One-Command Marketing Automation

```bash
# Generate and publish everywhere (as drafts)
python scripts/automation/master_automation.py --marketing --publish-devto --publish-medium

# Generate and publish immediately
python scripts/automation/master_automation.py --marketing \
  --publish-devto --devto-public \
  --publish-medium --medium-status public
```

## Article Format

Create markdown files with frontmatter:

**Dev.to Format** (`devto_article.md`):
```markdown
---
title: My Awesome Article
published: false
description: A brief description
tags: python, machinelearning, tutorial
canonical_url: https://example.com/original
series: My Series Name
---

# My Awesome Article

Article content goes here...
```

**Medium Format** (`medium_article.md`):
```markdown
---
title: My Awesome Article
status: draft
tags: python, machine-learning, tutorial
canonical_url: https://example.com/original
license: all-rights-reserved
---

# My Awesome Article

Article content goes here...
```

## Troubleshooting

### "ImportError: requests library required"
```bash
pip install requests
```

### "ValueError: Dev.to API key required"
```bash
export DEVTO_API_KEY="your_key"
```

### "ValueError: Medium API token required"
```bash
export MEDIUM_API_TOKEN="your_token"
```

### "401 Unauthorized"
- Check your API key/token is correct
- Regenerate if necessary
- Make sure environment variable is set

### Article not updating on Dev.to
- Use `--update` flag
- Ensure article title matches exactly (case-sensitive)

### Medium creates duplicate posts
- This is expected - Medium API doesn't support updates
- Each publish creates a new post

## Workflow Examples

### Release Workflow

```bash
# 1. Update CHANGELOG.md with new version

# 2. Generate and review content
python scripts/automation/master_automation.py --marketing

# 3. Review generated files
cat docs/blog/devto_v0.3.0_release.md
cat docs/blog/medium_v0.3.0_release.md
cat docs/social/twitter_v0.3.0.txt

# 4. Edit if needed (optional)
nano docs/blog/devto_v0.3.0_release.md

# 5. Publish to Dev.to as draft
python scripts/automation/devto_publisher.py \
  --file docs/blog/devto_v0.3.0_release.md

# 6. Publish to Medium as draft
python scripts/automation/medium_publisher.py \
  --file docs/blog/medium_v0.3.0_release.md

# 7. Review on platforms, then make public manually
```

### Custom Article Workflow

```bash
# 1. Create article with frontmatter
cat > my_article.md << 'EOF'
---
title: Custom Article
published: false
tags: python, tutorial
---

# Custom Article

Content here...
EOF

# 2. Publish to Dev.to
python scripts/automation/devto_publisher.py --file my_article.md

# 3. Publish to Medium
python scripts/automation/medium_publisher.py --file my_article.md
```

### Batch Publishing Workflow

```bash
# 1. Create multiple articles in a directory
mkdir -p my_articles
echo "..." > my_articles/devto_article1.md
echo "..." > my_articles/devto_article2.md

# 2. Batch publish to Dev.to
python scripts/automation/devto_publisher.py \
  --directory my_articles \
  --pattern "devto_*.md"

# 3. Or use Medium
python scripts/automation/medium_publisher.py \
  --directory my_articles \
  --pattern "medium_*.md"
```

## Tips & Best Practices

1. **Always review generated content** before publishing
2. **Start with drafts** - use `--publish` or `--status public` only when ready
3. **Use descriptive filenames** - e.g., `devto_v0.3.0_release.md`
4. **Keep API keys secure** - never commit to git
5. **Test with one article** before batch publishing
6. **Use canonical URLs** when cross-posting
7. **Optimize tags** - use relevant, popular tags
8. **Check rate limits** - don't spam the APIs

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- See [example_usage.py](example_usage.py) for code examples
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for features

## Need Help?

1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify API credentials are correct
4. Check API key permissions
5. Look at example files in `docs/blog/`
6. Test with a simple article first

## Quick Reference

```bash
# Generate only
master_automation.py --marketing

# Generate + Publish (draft)
master_automation.py --marketing --publish-devto --publish-medium

# Generate + Publish (public)
master_automation.py --marketing --publish-devto --devto-public --publish-medium --medium-status public

# Dev.to standalone
devto_publisher.py --file article.md [--publish] [--update]

# Medium standalone
medium_publisher.py --file article.md [--status draft|public|unlisted]

# List Medium publications
medium_publisher.py --list-publications
```

Happy Publishing! 🚀
