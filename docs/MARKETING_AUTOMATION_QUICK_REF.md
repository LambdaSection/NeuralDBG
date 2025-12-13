# Marketing Automation - Quick Reference

Quick reference for using the automated marketing workflow.

## 🚀 Quick Start

### Automatic (Recommended)
1. Publish a release on GitHub
2. Marketing automation runs automatically
3. Check Actions tab for status

### Manual
```bash
# Go to: Actions → Marketing Automation → Run workflow
# Enter version: 0.3.0
# Run workflow
```

## 🔐 Required Secrets

Configure in: `Settings` → `Secrets and variables` → `Actions`

| Platform | Secret | Required |
|----------|--------|----------|
| **Dev.to** | `DEVTO_API_KEY` | Yes |
| **Medium** | `MEDIUM_API_KEY` | Yes |
| **Medium** | `MEDIUM_USER_ID` | Optional |
| **Twitter/X** | `TWITTER_API_KEY` | Yes |
| **Twitter/X** | `TWITTER_API_SECRET` | Yes |
| **Twitter/X** | `TWITTER_ACCESS_TOKEN` | Yes |
| **Twitter/X** | `TWITTER_ACCESS_TOKEN_SECRET` | Yes |
| **LinkedIn** | `LINKEDIN_ACCESS_TOKEN` | Yes |
| **LinkedIn** | `LINKEDIN_PERSON_URN` | Optional |

## 📝 What Gets Published

### Dev.to
- ✅ Published immediately
- 📌 Tags: neuralnetworks, python, machinelearning, deeplearning
- 📚 Series: "Neural DSL Releases"

### Medium
- ⚠️ Created as **DRAFT** (review before publishing)
- 📌 Tags: neural-networks, python, machine-learning, deep-learning, ai

### Twitter/X
- ✅ Posted immediately
- 🎯 Max 280 characters
- #️⃣ Hashtags included

### LinkedIn
- ✅ Posted immediately
- 🌐 Public visibility
- 💼 Professional format

### GitHub
- 💬 Creates discussion in "Announcements" category
- 🔗 Links to all published content

## 📁 Generated Files

```
docs/
├── blog/
│   ├── devto_vX.X.X_release.md
│   ├── medium_vX.X.X_release.md
│   └── github_vX.X.X_release.md
└── social/
    ├── twitter_vX.X.X.txt
    └── linkedin_vX.X.X.txt
```

## 🔧 Common Tasks

### Get API Keys

#### Dev.to
1. Go to https://dev.to/settings/extensions
2. Generate API key
3. Add as `DEVTO_API_KEY`

#### Medium
1. Go to https://medium.com/me/settings/security
2. Generate integration token
3. Add as `MEDIUM_API_KEY`

#### Twitter/X
1. Apply at https://developer.twitter.com/
2. Create app
3. Generate API keys and tokens
4. Add all four secrets

#### LinkedIn
1. Create app at https://www.linkedin.com/developers/
2. Get OAuth 2.0 token with `w_member_social` permission
3. Add as `LINKEDIN_ACCESS_TOKEN`

### Test Without Publishing

```bash
# Manual run with:
# - skip_blog_publish: true
# - skip_social: true
```

This generates content without publishing.

### Review Generated Content

1. Go to Actions → Workflow run
2. Download "marketing-content" artifact
3. Review files before next release

## ❗ Troubleshooting

### Secret Issues
```
⚠️ Error: "401 Unauthorized"
→ Check secret name (case-sensitive)
→ Verify secret value is correct
→ Regenerate API key if needed
```

### Dev.to Issues
```
⚠️ Error: "Duplicate article"
→ Article with same title exists
→ Check Dev.to dashboard
→ Delete or rename existing draft
```

### Medium Issues
```
⚠️ Error: "403 Forbidden"
→ Token expired or revoked
→ Regenerate integration token
→ Ensure token has write permissions
```

### Twitter/X Issues
```
⚠️ Error: "403 Forbidden" or "187 Duplicate"
→ Check all 4 credentials are set
→ Verify app has write permissions
→ Twitter blocks duplicate tweets
```

### LinkedIn Issues
```
⚠️ Error: "401 Unauthorized"
→ Token expires after 60 days
→ Regenerate access token
→ Ensure w_member_social permission
```

## 📊 Workflow Status

Check workflow status:
```
Actions → Marketing Automation → Latest run
```

View summary:
- ✅ Green checkmark = Success
- ❌ Red X = Failed
- ⚠️ Orange = Warning
- ⏭️ Skipped

## 🎯 Best Practices

1. ✅ **Test manually first** before relying on automation
2. ✅ **Review Medium drafts** before publishing
3. ✅ **Keep CHANGELOG.md updated** for best results
4. ✅ **Rotate secrets regularly** (especially LinkedIn)
5. ✅ **Monitor rate limits** for each platform
6. ✅ **Check logs** after each run

## 📚 Full Documentation

For detailed information, see:
- [Marketing Automation Guide](MARKETING_AUTOMATION_GUIDE.md)
- [Workflows README](.github/workflows/README.md)
- [Automation Guide](../AUTOMATION_GUIDE.md)

## 🆘 Need Help?

1. Check workflow logs in Actions tab
2. Review troubleshooting section above
3. See full guide: [MARKETING_AUTOMATION_GUIDE.md](MARKETING_AUTOMATION_GUIDE.md)
4. Open issue with logs attached

## 🔗 API Documentation

- [Dev.to API](https://developers.forem.com/api/)
- [Medium API](https://github.com/Medium/medium-api-docs)
- [Twitter API](https://developer.twitter.com/en/docs/twitter-api)
- [LinkedIn API](https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/share-api)
