# Marketing Automation Guide

This guide explains how to configure and use the automated marketing workflow that publishes release announcements across multiple platforms.

## Overview

The marketing automation workflow (`.github/workflows/marketing_automation.yml`) automatically:

1. **Generates marketing content** from CHANGELOG.md
2. **Publishes blog posts** to Dev.to and Medium
3. **Posts to social media** (Twitter/X and LinkedIn)
4. **Creates GitHub Discussions** with release announcements
5. **Commits generated files** to the repository

## Triggers

The workflow runs automatically when:
- A new release is published on GitHub
- Manually triggered via workflow dispatch

## Prerequisites

### Required Secrets

Configure the following secrets in your GitHub repository settings (`Settings` → `Secrets and variables` → `Actions`):

#### Dev.to API Key

1. Go to [Dev.to Settings](https://dev.to/settings/extensions)
2. Generate an API key
3. Add as `DEVTO_API_KEY` secret

#### Medium Integration Token

1. Go to [Medium Settings](https://medium.com/me/settings/security)
2. Generate an integration token
3. Add as `MEDIUM_API_KEY` secret
4. (Optional) Add your user ID as `MEDIUM_USER_ID` (auto-detected if not provided)

#### Twitter/X API Credentials

1. Apply for Twitter Developer access at [developer.twitter.com](https://developer.twitter.com/)
2. Create an app and generate API keys
3. Add the following secrets:
   - `TWITTER_API_KEY`
   - `TWITTER_API_SECRET`
   - `TWITTER_ACCESS_TOKEN`
   - `TWITTER_ACCESS_TOKEN_SECRET`

#### LinkedIn API Token

1. Create a LinkedIn app at [LinkedIn Developers](https://www.linkedin.com/developers/)
2. Get OAuth 2.0 access token with `w_member_social` permission
3. Add as `LINKEDIN_ACCESS_TOKEN` secret
4. (Optional) Add your person URN as `LINKEDIN_PERSON_URN` (auto-detected if not provided)

## Usage

### Automatic Trigger (Recommended)

When you publish a release on GitHub:

1. Create and publish a release via GitHub UI or API
2. The workflow automatically triggers
3. Marketing content is generated and published
4. Check the Actions tab for workflow status

### Manual Trigger

To manually run the workflow:

1. Go to `Actions` → `Marketing Automation`
2. Click `Run workflow`
3. Enter the version number (e.g., `0.3.0`)
4. Optionally skip blog publishing or social media posting
5. Click `Run workflow`

## Workflow Steps

### 1. Validate Secrets

Checks which API credentials are configured and outputs the status.

### 2. Generate Content

- Reads CHANGELOG.md for release notes
- Generates formatted blog posts for Dev.to, Medium, and GitHub
- Creates social media posts for Twitter/X and LinkedIn
- Saves all generated content as artifacts

### 3. Publish to Dev.to

**Requirements:** `DEVTO_API_KEY`

- Publishes article to Dev.to with proper formatting
- Tags: `neuralnetworks`, `python`, `machinelearning`, `deeplearning`
- Series: "Neural DSL Releases"
- Status: Published immediately

### 4. Publish to Medium

**Requirements:** `MEDIUM_API_KEY`, optional `MEDIUM_USER_ID`

- Creates a **draft** article on Medium for review
- Tags: `neural-networks`, `python`, `machine-learning`, `deep-learning`, `ai`
- Status: Draft (requires manual publishing)

### 5. Post to Twitter/X

**Requirements:** Twitter API credentials

- Posts release announcement tweet
- Includes top features and installation instructions
- Automatically truncates to 280 characters if needed
- Includes relevant hashtags

### 6. Post to LinkedIn

**Requirements:** `LINKEDIN_ACCESS_TOKEN`, optional `LINKEDIN_PERSON_URN`

- Posts professional release announcement
- Lists key features
- Includes links and hashtags
- Published to public feed

### 7. Update GitHub Discussions

**Requirements:** `GITHUB_TOKEN` (automatically provided)

- Creates a new discussion in the "Announcements" category
- Includes full release notes
- Links to all published content (Dev.to, Medium, Twitter/X, LinkedIn)
- Requires "Announcements" discussion category to exist

### 8. Commit Blog Files

**Requirements:** `GITHUB_TOKEN` (automatically provided)

- Commits generated blog files to `docs/blog/`
- Commits social media posts to `docs/social/`
- Creates commit: `docs: add generated marketing content for vX.X.X`

## Generated Files

The workflow generates and commits the following files:

```
docs/
├── blog/
│   ├── devto_vX.X.X_release.md      # Dev.to formatted post
│   ├── medium_vX.X.X_release.md     # Medium formatted post
│   └── github_vX.X.X_release.md     # GitHub release notes
└── social/
    ├── twitter_vX.X.X.txt           # Twitter/X post text
    └── linkedin_vX.X.X.txt          # LinkedIn post text
```

## Error Handling

The workflow includes comprehensive error handling:

- **Secret Validation**: Checks for required secrets before attempting to publish
- **Continue on Error**: Individual publishing steps won't fail the entire workflow
- **Detailed Logs**: Each step outputs detailed success/failure messages
- **Summary Report**: Final summary shows status of all steps

## Troubleshooting

### Dev.to Publishing Fails

**Common Issues:**
- Invalid API key
- Rate limiting (Dev.to has rate limits)
- Duplicate article (article with same title already exists)

**Solutions:**
- Verify API key is correct and active
- Wait and retry if rate limited
- Check Dev.to dashboard for existing drafts

### Medium Publishing Fails

**Common Issues:**
- Invalid integration token
- Token expired or revoked
- User ID not found

**Solutions:**
- Regenerate integration token
- Ensure token has not been revoked
- Verify Medium account is active

### Twitter/X Posting Fails

**Common Issues:**
- Invalid credentials
- Duplicate tweet (Twitter prevents identical tweets)
- API access level insufficient

**Solutions:**
- Verify all four credential secrets are correct
- Ensure Twitter Developer account has write permissions
- Check Twitter API tier limits

### LinkedIn Posting Fails

**Common Issues:**
- Access token expired (LinkedIn tokens expire after 60 days)
- Insufficient permissions
- Person URN not found

**Solutions:**
- Regenerate access token
- Ensure token has `w_member_social` permission
- Verify LinkedIn app is approved

### GitHub Discussions Fails

**Common Issues:**
- "Announcements" category doesn't exist
- Insufficient permissions

**Solutions:**
- Create "Announcements" category in repository settings
- Ensure workflow has `discussions: write` permission (automatic with `GITHUB_TOKEN`)

## Customization

### Modify Content Templates

Edit the following files to customize generated content:

- `scripts/automation/blog_generator.py` - Blog post templates
- `scripts/automation/social_media_generator.py` - Social media post templates

### Modify Workflow Behavior

Edit `.github/workflows/marketing_automation.yml` to:

- Change when workflow triggers
- Add/remove publishing platforms
- Modify error handling behavior
- Customize commit messages

### Add New Platforms

To add support for new platforms:

1. Add new job to workflow (e.g., `post-hashnode`)
2. Add required secrets
3. Implement publishing logic using platform's API
4. Update secret validation step
5. Update summary report

## Best Practices

1. **Test with Manual Trigger**: Test the workflow manually before relying on automatic triggers
2. **Review Medium Drafts**: Medium posts are created as drafts - review before publishing
3. **Monitor Rate Limits**: Be aware of API rate limits for each platform
4. **Keep Secrets Updated**: Regularly rotate and update API credentials
5. **Check Logs**: Review workflow logs after each run to catch issues early
6. **Update CHANGELOG.md**: Ensure CHANGELOG.md is well-formatted for best results

## Security Considerations

- **Never commit secrets** to the repository
- **Use repository secrets** for all API credentials
- **Rotate credentials regularly** (especially LinkedIn tokens which expire)
- **Monitor API usage** to detect unauthorized access
- **Limit workflow permissions** to only what's necessary

## Example Workflow Run

```
1. Release v0.3.0 published on GitHub
2. Workflow triggered automatically
3. Content generated from CHANGELOG.md
   ✓ Blog posts created
   ✓ Social media posts created
4. Publishing begins:
   ✓ Dev.to article published
   ✓ Medium draft created
   ✓ Twitter/X post published
   ✓ LinkedIn post published
5. GitHub Discussion created
6. Files committed to repository
7. Summary report generated
```

## Support

If you encounter issues with the marketing automation workflow:

1. Check the workflow logs in the Actions tab
2. Verify all required secrets are configured
3. Review this guide for troubleshooting steps
4. Open an issue on GitHub with relevant logs

## Related Documentation

- [Automation Guide](../AUTOMATION_GUIDE.md) - Overall automation strategy
- [Blog README](blog/README.md) - Blog content management
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute

## API Documentation Links

- [Dev.to API](https://developers.forem.com/api/)
- [Medium API](https://github.com/Medium/medium-api-docs)
- [Twitter API](https://developer.twitter.com/en/docs/twitter-api)
- [LinkedIn API](https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/share-api)
- [GitHub GraphQL API](https://docs.github.com/en/graphql)
