# Blog Publishing Automation - Setup Checklist

## Initial Setup

- [ ] Install `requests` library: `pip install requests`
- [ ] Get Dev.to API key from https://dev.to/settings/extensions
- [ ] Get Medium API token from https://medium.com/me/settings/security
- [ ] Set environment variables or create `.env` file
- [ ] Test connection: `python scripts/automation/devto_publisher.py --help`

## Pre-Publishing Checklist

- [ ] Update CHANGELOG.md with release notes
- [ ] Generate blog posts: `python scripts/automation/master_automation.py --marketing`
- [ ] Review generated content in `docs/blog/`
- [ ] Review social media posts in `docs/social/`
- [ ] Edit content if needed (optional)
- [ ] Test with a draft publish first

## Publishing Checklist

### Dev.to

- [ ] Verify frontmatter is correct (title, tags, description)
- [ ] Check tags (max 4)
- [ ] Set canonical URL if cross-posting
- [ ] Publish as draft first
- [ ] Review on Dev.to platform
- [ ] Make public when ready (via platform or `--publish` flag)

### Medium

- [ ] Verify frontmatter is correct (title, tags, status)
- [ ] Check tags (max 5)
- [ ] Set canonical URL if cross-posting
- [ ] Choose correct license
- [ ] Publish as draft first
- [ ] Review on Medium platform
- [ ] Make public when ready (via platform or `--status public`)

## Post-Publishing Checklist

- [ ] Verify article is published and accessible
- [ ] Check formatting looks correct
- [ ] Verify all links work
- [ ] Post to social media using generated content
- [ ] Monitor comments and engagement
- [ ] Update canonical URLs if needed

## Security Checklist

- [ ] API keys are NOT committed to git
- [ ] `.env` file is in `.gitignore`
- [ ] Generated blog files are in `.gitignore`
- [ ] API keys have minimal required permissions
- [ ] API keys are rotated periodically

## Troubleshooting Checklist

If publish fails:

- [ ] Check API key is set correctly
- [ ] Verify article format (frontmatter syntax)
- [ ] Check required fields are present
- [ ] Look for error messages in output
- [ ] Test with a simple article first
- [ ] Check API rate limits
- [ ] Verify network connection

## Automation Workflow Checklist

For full release:

- [ ] Run tests: `python scripts/automation/master_automation.py --test`
- [ ] Validate examples: `python scripts/automation/master_automation.py --validate`
- [ ] Generate marketing content: `python scripts/automation/master_automation.py --marketing`
- [ ] Review and edit content
- [ ] Publish to Dev.to (draft)
- [ ] Publish to Medium (draft)
- [ ] Review on platforms
- [ ] Make articles public
- [ ] Post to social media
- [ ] Monitor engagement

## Maintenance Checklist

Weekly:

- [ ] Check for API updates
- [ ] Review published articles
- [ ] Respond to comments
- [ ] Update templates if needed

Monthly:

- [ ] Rotate API keys
- [ ] Review publishing analytics
- [ ] Update documentation
- [ ] Check for new platform features

## Quality Checklist

Before publishing:

- [ ] Title is clear and engaging
- [ ] Description is informative
- [ ] Tags are relevant and popular
- [ ] Content is well-formatted
- [ ] Code examples work correctly
- [ ] Links are correct and working
- [ ] Images are included (if applicable)
- [ ] Grammar and spelling are correct

## Documentation Checklist

Have you read:

- [ ] [QUICK_START.md](QUICK_START.md) - Getting started guide
- [ ] [README.md](README.md) - Full documentation
- [ ] [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [ ] [example_usage.py](example_usage.py) - Code examples

## Success Criteria

- [ ] Articles publish without errors
- [ ] Content appears correctly on platforms
- [ ] All formatting is preserved
- [ ] Tags and metadata are correct
- [ ] Canonical URLs work
- [ ] Social media posts are generated
- [ ] Workflow is smooth and efficient

---

**Note**: This checklist is a guide. Adapt it to your specific workflow and needs.
