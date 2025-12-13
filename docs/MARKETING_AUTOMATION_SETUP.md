# Marketing Automation Setup Guide

Step-by-step instructions for setting up API credentials for the marketing automation workflow.

## Prerequisites

- Repository admin access (to configure secrets)
- Accounts on target platforms (Dev.to, Medium, Twitter/X, LinkedIn)
- Developer access on each platform

## Setup Checklist

- [ ] Dev.to API key configured
- [ ] Medium integration token configured
- [ ] Twitter/X API credentials configured (all 4 secrets)
- [ ] LinkedIn access token configured
- [ ] GitHub "Announcements" discussion category created
- [ ] Test workflow run completed

## Step-by-Step Setup

### 1. Dev.to API Key

#### Get API Key

1. Log in to [Dev.to](https://dev.to/)
2. Navigate to [Settings → Extensions](https://dev.to/settings/extensions)
3. Scroll to "DEV Community API Keys"
4. Click "Generate API Key"
5. Give it a name (e.g., "Neural DSL Marketing Automation")
6. Copy the generated API key

#### Add to GitHub

1. Go to your repository on GitHub
2. Navigate to `Settings` → `Secrets and variables` → `Actions`
3. Click `New repository secret`
4. Name: `DEVTO_API_KEY`
5. Value: Paste your Dev.to API key
6. Click `Add secret`

#### Test API Key

```bash
curl -H "api-key: YOUR_API_KEY" https://dev.to/api/articles/me
```

Should return your articles if the key is valid.

---

### 2. Medium Integration Token

#### Get Integration Token

1. Log in to [Medium](https://medium.com/)
2. Navigate to [Settings → Security and apps](https://medium.com/me/settings/security)
3. Scroll to "Integration tokens"
4. Enter description (e.g., "Neural DSL Marketing")
5. Click "Get integration token"
6. Copy the token immediately (shown only once)

#### Add to GitHub

1. Go to repository `Settings` → `Secrets and variables` → `Actions`
2. Click `New repository secret`
3. Name: `MEDIUM_API_KEY`
4. Value: Paste your Medium integration token
5. Click `Add secret`

#### Optional: Get User ID

The workflow can auto-detect your user ID, but you can provide it manually:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.medium.com/v1/me
```

Extract the `id` field from the response and add as `MEDIUM_USER_ID` secret (optional).

#### Test Integration Token

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     https://api.medium.com/v1/me
```

Should return your user information if the token is valid.

---

### 3. Twitter/X API Credentials

Twitter API setup is the most complex as it requires 4 separate credentials.

#### Apply for Developer Access

1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Sign in with your Twitter account
3. Click "Apply for a developer account"
4. Fill out the application form:
   - Choose "Making a bot" or "Doing something else"
   - Explain: "Automated release announcements for open source project"
   - Provide project details
5. Submit and wait for approval (usually instant to 24 hours)

#### Create App

1. Once approved, go to [Developer Dashboard](https://developer.twitter.com/en/portal/dashboard)
2. Click "Create Project"
3. Name your project (e.g., "Neural DSL")
4. Select use case: "Making a bot"
5. Provide description
6. Click "Next" to create an app
7. Name your app (e.g., "Neural DSL Release Bot")

#### Generate API Keys

1. In your app settings, navigate to "Keys and tokens" tab
2. Under "Consumer Keys", click "Regenerate" to get:
   - **API Key** (Consumer Key)
   - **API Secret** (Consumer Secret)
   - Copy both and save securely

#### Generate Access Tokens

1. Scroll to "Authentication Tokens"
2. Click "Generate" under "Access Token and Secret"
3. Copy both:
   - **Access Token**
   - **Access Token Secret**

#### Set App Permissions

1. In app settings, go to "User authentication settings"
2. Click "Set up"
3. Enable "OAuth 1.0a"
4. App permissions: Select "Read and write"
5. Type of App: "Automated App or bot"
6. Save settings

#### Add to GitHub

Add all 4 credentials as secrets:

1. `TWITTER_API_KEY` → API Key (Consumer Key)
2. `TWITTER_API_SECRET` → API Secret (Consumer Secret)
3. `TWITTER_ACCESS_TOKEN` → Access Token
4. `TWITTER_ACCESS_TOKEN_SECRET` → Access Token Secret

#### Test Credentials

```python
import tweepy

auth = tweepy.OAuthHandler("API_KEY", "API_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
api = tweepy.API(auth)

# Test by verifying credentials
try:
    api.verify_credentials()
    print("Authentication successful!")
except Exception as e:
    print(f"Authentication failed: {e}")
```

---

### 4. LinkedIn Access Token

LinkedIn API requires OAuth 2.0 flow, which is more complex.

#### Create LinkedIn App

1. Go to [LinkedIn Developers](https://www.linkedin.com/developers/)
2. Click "Create app"
3. Fill in app details:
   - App name: "Neural DSL Marketing"
   - LinkedIn Page: Select your company page (or create one)
   - App logo: Upload Neural DSL logo
   - Legal agreement: Check the box
4. Click "Create app"

#### Configure App

1. In your app, go to "Auth" tab
2. Note your Client ID and Client Secret
3. Add OAuth 2.0 redirect URLs:
   - For local testing: `http://localhost:3000/callback`
4. Go to "Products" tab
5. Request access to "Share on LinkedIn" product
6. Wait for approval (usually instant for personal pages)

#### Get Access Token

Since OAuth 2.0 requires a web flow, use one of these methods:

**Option A: Use LinkedIn OAuth Playground**
1. Go to a LinkedIn OAuth tool (search for "LinkedIn OAuth token generator")
2. Follow the OAuth flow
3. Get access token with `w_member_social` scope

**Option B: Manual OAuth Flow**

```python
# Step 1: Generate authorization URL
auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=w_member_social"
print(f"Visit: {auth_url}")

# Step 2: After authorization, exchange code for token
import requests

token_url = "https://www.linkedin.com/oauth/v2/accessToken"
data = {
    'grant_type': 'authorization_code',
    'code': 'AUTHORIZATION_CODE',
    'redirect_uri': REDIRECT_URI,
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET
}
response = requests.post(token_url, data=data)
access_token = response.json()['access_token']
```

**Option C: Use Simple Python Script**

Create `get_linkedin_token.py`:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from urllib.parse import urlparse, parse_qs
import webbrowser

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
REDIRECT_URI = "http://localhost:3000/callback"

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urlparse(self.path).query
        params = parse_qs(query)
        
        if 'code' in params:
            code = params['code'][0]
            
            # Exchange code for token
            token_url = "https://www.linkedin.com/oauth/v2/accessToken"
            data = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': REDIRECT_URI,
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET
            }
            
            response = requests.post(token_url, data=data)
            token_data = response.json()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            if 'access_token' in token_data:
                access_token = token_data['access_token']
                self.wfile.write(f"<h1>Success!</h1><p>Access Token: {access_token}</p>".encode())
                print(f"\n\nAccess Token: {access_token}\n\n")
            else:
                self.wfile.write(f"<h1>Error</h1><p>{token_data}</p>".encode())

# Start server
auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=w_member_social"
print(f"Opening browser for authorization...")
webbrowser.open(auth_url)

server = HTTPServer(('localhost', 3000), OAuthHandler)
print("Waiting for authorization...")
server.handle_request()
```

#### Add to GitHub

1. Add `LINKEDIN_ACCESS_TOKEN` secret with your access token
2. (Optional) Add `LINKEDIN_PERSON_URN` if you want to skip auto-detection

#### Important Notes

- LinkedIn access tokens expire after 60 days
- You'll need to regenerate tokens periodically
- Consider setting up refresh token flow for long-term use

#### Test Access Token

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.linkedin.com/v2/me
```

Should return your profile information if the token is valid.

---

### 5. GitHub Discussion Category

The workflow creates discussions in the "Announcements" category.

#### Create Category

1. Go to your repository on GitHub
2. Click on "Discussions" tab
3. If discussions aren't enabled:
   - Go to `Settings` → `Features`
   - Check "Discussions"
4. Click "Categories" (pencil icon)
5. Create a new category:
   - Name: "Announcements"
   - Description: "Release announcements and updates"
   - Format: "Announcement" (lock discussions, only maintainers can post)
6. Save

The workflow will now be able to post to this category.

---

## Testing the Setup

### Test Individual Components

1. **Generate content only** (no publishing):
   ```bash
   # Manual workflow run with both skip flags enabled
   ```

2. **Check generated content**:
   - Download artifacts from workflow run
   - Review blog posts and social media content

3. **Test single platform**:
   - Configure only one platform's secrets
   - Run workflow
   - Verify that platform publishes correctly

### Test Complete Workflow

1. Create a test release (can be draft)
2. Workflow triggers automatically
3. Check Actions tab for progress
4. Verify each step:
   - ✓ Content generated
   - ✓ Dev.to published
   - ✓ Medium draft created
   - ✓ Twitter/X posted
   - ✓ LinkedIn posted
   - ✓ Discussion created
   - ✓ Files committed

### Troubleshooting Tests

If a step fails:
1. Check the workflow logs
2. Verify the API key/token is correct
3. Test the API manually (using curl commands above)
4. Check rate limits
5. Verify permissions

---

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use repository secrets**, not environment variables
3. **Rotate credentials regularly**:
   - Dev.to: No expiration, but rotate periodically
   - Medium: No expiration, but rotate periodically
   - Twitter: No expiration, but rotate periodically
   - LinkedIn: **Expires after 60 days** - set calendar reminder
4. **Limit scope** to minimum required permissions
5. **Monitor API usage** for unexpected activity
6. **Revoke immediately** if compromised
7. **Document who has access** to credentials

---

## Maintenance

### Regular Tasks

**Monthly:**
- Review API usage and rate limits
- Check for new API features or changes
- Verify all secrets are still valid

**Every 60 Days:**
- Regenerate LinkedIn access token
- Test all publishing endpoints

**Quarterly:**
- Rotate all API keys and tokens
- Review and update documentation
- Test complete workflow end-to-end

### Monitoring

Set up monitoring for:
- Workflow failures (GitHub notifications)
- API rate limit warnings
- Token expiration alerts (especially LinkedIn)
- Unusual API activity

---

## Cost Considerations

All platforms used in this workflow offer free tiers:

- **Dev.to**: Free, unlimited
- **Medium**: Free, no API costs
- **Twitter/X**: 
  - Free tier: 1,500 posts/month (more than enough)
  - Basic tier: $100/month for higher limits (not needed)
- **LinkedIn**: Free for personal use
- **GitHub**: Free for public repositories

Expected monthly costs: **$0** 💰

---

## Support and Resources

### Official API Documentation
- [Dev.to API Docs](https://developers.forem.com/api/)
- [Medium API Docs](https://github.com/Medium/medium-api-docs)
- [Twitter API Docs](https://developer.twitter.com/en/docs/twitter-api)
- [LinkedIn API Docs](https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/share-api)

### Getting Help
- Check workflow logs in Actions tab
- Review [Marketing Automation Guide](MARKETING_AUTOMATION_GUIDE.md)
- Open issue on GitHub
- Check platform-specific status pages

### Common Issues
- [Troubleshooting Guide](MARKETING_AUTOMATION_GUIDE.md#troubleshooting)
- [Quick Reference](MARKETING_AUTOMATION_QUICK_REF.md)

---

## Next Steps

After completing setup:

1. ✅ Test with a manual workflow run
2. ✅ Review generated content
3. ✅ Publish a test release
4. ✅ Monitor the first few automated runs
5. ✅ Document any custom configurations
6. ✅ Share with team members
7. ✅ Set up monitoring and alerts

---

**Setup Complete!** 🎉

Your marketing automation is now ready to handle release announcements automatically across all platforms.
