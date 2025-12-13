# Marketing Automation Setup Checklist

Use this checklist to ensure your marketing automation is properly configured.

## Pre-Setup

- [ ] Read [Marketing Automation Guide](../docs/MARKETING_AUTOMATION_GUIDE.md)
- [ ] Review [Setup Guide](../docs/MARKETING_AUTOMATION_SETUP.md)
- [ ] Have admin access to GitHub repository
- [ ] Have accounts on all target platforms

## Platform Setup

### Dev.to

- [ ] Create/verify Dev.to account
- [ ] Navigate to Settings → Extensions
- [ ] Generate API key
- [ ] Test API key with curl command
- [ ] Add `DEVTO_API_KEY` secret to GitHub
- [ ] Verify secret is saved correctly

### Medium

- [ ] Create/verify Medium account
- [ ] Navigate to Settings → Security
- [ ] Generate integration token
- [ ] Copy token immediately (shown once!)
- [ ] Test token with curl command
- [ ] Add `MEDIUM_API_KEY` secret to GitHub
- [ ] (Optional) Add `MEDIUM_USER_ID` secret
- [ ] Verify secrets are saved correctly

### Twitter/X

- [ ] Create/verify Twitter account
- [ ] Apply for Twitter Developer access
- [ ] Wait for approval (if needed)
- [ ] Create app in Developer Portal
- [ ] Generate API Key and Secret
- [ ] Generate Access Token and Secret
- [ ] Set app permissions to "Read and write"
- [ ] Test credentials with tweepy
- [ ] Add `TWITTER_API_KEY` secret to GitHub
- [ ] Add `TWITTER_API_SECRET` secret to GitHub
- [ ] Add `TWITTER_ACCESS_TOKEN` secret to GitHub
- [ ] Add `TWITTER_ACCESS_TOKEN_SECRET` secret to GitHub
- [ ] Verify all 4 secrets are saved correctly

### LinkedIn

- [ ] Create/verify LinkedIn account
- [ ] Create app in LinkedIn Developers
- [ ] Request "Share on LinkedIn" product access
- [ ] Wait for approval (if needed)
- [ ] Set up OAuth 2.0 flow
- [ ] Generate access token with `w_member_social` scope
- [ ] **Set calendar reminder** for 60 days (token expires!)
- [ ] Test token with curl command
- [ ] Add `LINKEDIN_ACCESS_TOKEN` secret to GitHub
- [ ] (Optional) Add `LINKEDIN_PERSON_URN` secret
- [ ] Verify secrets are saved correctly

### GitHub Discussions

- [ ] Enable Discussions in repository settings
- [ ] Create "Announcements" category
- [ ] Set format to "Announcement" type
- [ ] Configure permissions (maintainers only can post)
- [ ] Verify category exists and is accessible

## Secret Verification

- [ ] Go to Settings → Secrets and variables → Actions
- [ ] Verify `DEVTO_API_KEY` is listed
- [ ] Verify `MEDIUM_API_KEY` is listed
- [ ] Verify `TWITTER_API_KEY` is listed
- [ ] Verify `TWITTER_API_SECRET` is listed
- [ ] Verify `TWITTER_ACCESS_TOKEN` is listed
- [ ] Verify `TWITTER_ACCESS_TOKEN_SECRET` is listed
- [ ] Verify `LINKEDIN_ACCESS_TOKEN` is listed
- [ ] Check for typos in secret names (case-sensitive!)

## Testing

### Phase 1: Content Generation Only

- [ ] Go to Actions → Marketing Automation
- [ ] Click "Run workflow"
- [ ] Enter version: `0.0.0-test`
- [ ] Set `skip_blog_publish`: `true`
- [ ] Set `skip_social`: `true`
- [ ] Click "Run workflow"
- [ ] Wait for completion
- [ ] Download "marketing-content" artifact
- [ ] Review generated blog posts
- [ ] Review generated social media posts
- [ ] Verify content quality and formatting

### Phase 2: Dev.to Publishing Test

- [ ] Run workflow with only Dev.to configured
- [ ] Set `skip_blog_publish`: `false`
- [ ] Set `skip_social`: `true`
- [ ] Check workflow logs for errors
- [ ] Visit Dev.to to verify article published
- [ ] Delete test article from Dev.to

### Phase 3: Medium Publishing Test

- [ ] Run workflow with only Medium configured
- [ ] Set `skip_blog_publish`: `false`
- [ ] Set `skip_social`: `true`
- [ ] Check workflow logs for errors
- [ ] Visit Medium to verify draft created
- [ ] Review and delete test draft

### Phase 4: Twitter/X Posting Test

- [ ] Run workflow with only Twitter configured
- [ ] Set `skip_blog_publish`: `true`
- [ ] Set `skip_social`: `false`
- [ ] Check workflow logs for errors
- [ ] Visit Twitter to verify post published
- [ ] Delete test tweet

### Phase 5: LinkedIn Posting Test

- [ ] Run workflow with only LinkedIn configured
- [ ] Set `skip_blog_publish`: `true`
- [ ] Set `skip_social`: `false`
- [ ] Check workflow logs for errors
- [ ] Visit LinkedIn to verify post published
- [ ] Delete test post

### Phase 6: Full Integration Test

- [ ] Configure all platform secrets
- [ ] Create a test release (can be draft)
- [ ] Let workflow run automatically
- [ ] Monitor workflow progress
- [ ] Verify all steps complete successfully
- [ ] Check Dev.to for article
- [ ] Check Medium for draft
- [ ] Check Twitter for post
- [ ] Check LinkedIn for post
- [ ] Check GitHub Discussions for announcement
- [ ] Check repository for committed files
- [ ] Clean up test content from all platforms
- [ ] Delete test release

## Post-Setup

- [ ] Document any custom configurations
- [ ] Share setup with team members
- [ ] Set up monitoring/alerts
- [ ] Add calendar reminder for LinkedIn token renewal (60 days)
- [ ] Create backup of all API keys (securely!)
- [ ] Update team documentation
- [ ] Train team on manual workflow trigger
- [ ] Plan first real release announcement

## Ongoing Maintenance

### Monthly Tasks

- [ ] Review workflow run history
- [ ] Check for failed runs
- [ ] Monitor API rate limits
- [ ] Verify all secrets still valid
- [ ] Review generated content quality

### Every 60 Days

- [ ] **Regenerate LinkedIn access token** (CRITICAL!)
- [ ] Update `LINKEDIN_ACCESS_TOKEN` secret
- [ ] Test LinkedIn posting
- [ ] Set reminder for next renewal

### Quarterly Tasks

- [ ] Rotate all API keys/tokens (security best practice)
- [ ] Review and update content templates
- [ ] Check for new API features
- [ ] Update documentation
- [ ] Test complete workflow end-to-end

### As Needed

- [ ] Add new platforms
- [ ] Customize content templates
- [ ] Adjust error handling
- [ ] Update troubleshooting guides

## Troubleshooting Checklist

If workflow fails:

- [ ] Check workflow logs in Actions tab
- [ ] Verify secret names match exactly (case-sensitive)
- [ ] Test API credentials manually (curl commands)
- [ ] Check for API rate limits
- [ ] Verify token hasn't expired (LinkedIn!)
- [ ] Check platform-specific status pages
- [ ] Review error messages in logs
- [ ] Consult [Troubleshooting Guide](../docs/MARKETING_AUTOMATION_GUIDE.md#troubleshooting)
- [ ] Open issue with logs if problem persists

## Success Criteria

- [ ] Workflow runs without errors
- [ ] Content generated correctly
- [ ] Dev.to article published (if configured)
- [ ] Medium draft created (if configured)
- [ ] Twitter post published (if configured)
- [ ] LinkedIn post published (if configured)
- [ ] GitHub Discussion created
- [ ] Blog files committed to repository
- [ ] Summary report shows all green checkmarks
- [ ] Team can manually trigger workflow
- [ ] Documentation complete and accessible

## Security Verification

- [ ] No secrets in code or logs
- [ ] All secrets use GitHub Secrets
- [ ] `.env` files in `.gitignore`
- [ ] Team knows not to commit secrets
- [ ] Have process for rotating credentials
- [ ] Have process for revoking if compromised
- [ ] Monitoring set up for unusual API activity
- [ ] Backup of credentials stored securely
- [ ] Know how to regenerate each credential

## Documentation Verification

- [ ] Team has access to all guides
- [ ] Quick reference is accessible
- [ ] Setup guide is complete
- [ ] Troubleshooting section is helpful
- [ ] API documentation links work
- [ ] Examples are clear
- [ ] Screenshots/diagrams included where helpful

## Final Sign-Off

- [ ] All checklist items completed
- [ ] Testing successful
- [ ] Documentation reviewed
- [ ] Team trained
- [ ] Ready for first production release
- [ ] Monitoring in place
- [ ] Support process defined

---

## Resources

- [Marketing Automation Guide](../docs/MARKETING_AUTOMATION_GUIDE.md)
- [Quick Reference](../docs/MARKETING_AUTOMATION_QUICK_REF.md)
- [Setup Guide](../docs/MARKETING_AUTOMATION_SETUP.md)
- [Workflows README](workflows/README.md)
- [Implementation Summary](MARKETING_AUTOMATION_SUMMARY.md)

---

**Completion Date:** _______________

**Completed By:** _______________

**Review Date:** _______________

**Status:** [ ] Not Started  [ ] In Progress  [ ] Complete  [ ] Verified
