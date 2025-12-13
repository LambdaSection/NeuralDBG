# Marketing Automation Workflow Diagram

Visual representation of the marketing automation workflow.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRIGGER EVENT                                │
│  ┌────────────────────────┐      ┌────────────────────────┐        │
│  │  Release Published     │  OR  │  Manual Workflow       │        │
│  │  (Automatic)          │      │  Dispatch              │        │
│  └────────────────────────┘      └────────────────────────┘        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VALIDATE SECRETS                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Dev.to   │ │ Medium   │ │ Twitter  │ │ LinkedIn │              │
│  │ API Key? │ │ Token?   │ │ Creds?   │ │ Token?   │              │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘              │
│        │            │            │            │                     │
│        └────────────┴────────────┴────────────┘                     │
│                     │                                                │
│                     ▼                                                │
│         ┌──────────────────────┐                                    │
│         │  Report Available    │                                    │
│         │  Platforms          │                                    │
│         └──────────────────────┘                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATE CONTENT                                  │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Read CHANGELOG.md                                   │          │
│  │  Extract release notes for current version           │          │
│  │  Parse sections (Added, Fixed, Changed, etc.)        │          │
│  └────────────────────────┬─────────────────────────────┘          │
│                           │                                          │
│       ┌───────────────────┴──────────────────┐                     │
│       │                                       │                     │
│       ▼                                       ▼                     │
│  ┌─────────────────┐                   ┌──────────────────┐       │
│  │  Blog Posts     │                   │  Social Media    │       │
│  │  • Dev.to       │                   │  Posts           │       │
│  │  • Medium       │                   │  • Twitter/X     │       │
│  │  • GitHub       │                   │  • LinkedIn      │       │
│  └─────────────────┘                   └──────────────────┘       │
│                                                                     │
│  Upload as artifacts for debugging/review                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PUBLISH CONTENT (PARALLEL)                           │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │   Dev.to API    │  │   Medium API    │  │   Twitter API    │  │
│  │                 │  │                 │  │                  │  │
│  │ ✓ Immediate     │  │ ✓ Create draft  │  │ ✓ Post tweet     │  │
│  │   publish       │  │   for review    │  │   (max 280 char) │  │
│  │                 │  │                 │  │                  │  │
│  │ • Title         │  │ • Title         │  │ • Top features   │  │
│  │ • Body markdown │  │ • Body markdown │  │ • Install cmd    │  │
│  │ • Tags          │  │ • Tags          │  │ • Link           │  │
│  │ • Series        │  │                 │  │ • Hashtags       │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬────────┘  │
│           │                    │                      │            │
│           └────────────────────┴──────────────────────┘            │
│                                │                                    │
│                                ▼                                    │
│                    ┌──────────────────┐                            │
│                    │   LinkedIn API   │                            │
│                    │                  │                            │
│                    │ ✓ Post to feed   │                            │
│                    │   (public)       │                            │
│                    │                  │                            │
│                    │ • Full text      │                            │
│                    │ • Features list  │                            │
│                    │ • Professional   │                            │
│                    │   format         │                            │
│                    └────────┬─────────┘                            │
│                             │                                       │
│         Save URLs/IDs for later linking                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              UPDATE GITHUB DISCUSSIONS                               │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Create new discussion in "Announcements" category   │          │
│  │                                                       │          │
│  │  Content:                                             │          │
│  │  • Full release notes                                 │          │
│  │  • Link to Dev.to article (if published)             │          │
│  │  • Link to Medium draft (if created)                 │          │
│  │  • Link to Twitter/X post (if posted)                │          │
│  │  • LinkedIn post notice                               │          │
│  │  • Installation instructions                          │          │
│  │  • Documentation links                                │          │
│  └──────────────────────────────────────────────────────┘          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                COMMIT GENERATED FILES                                │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  git add docs/blog/*.md docs/social/*.txt            │          │
│  │  git commit -m "docs: add marketing content vX.X.X"  │          │
│  │  git push                                             │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                      │
│  Files committed:                                                   │
│  • docs/blog/devto_vX.X.X_release.md                               │
│  • docs/blog/medium_vX.X.X_release.md                              │
│  • docs/blog/github_vX.X.X_release.md                              │
│  • docs/social/twitter_vX.X.X.txt                                  │
│  • docs/social/linkedin_vX.X.X.txt                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   GENERATE SUMMARY                                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Workflow Summary in GitHub Actions UI:              │          │
│  │                                                       │          │
│  │  ✅ Content Generation: Success                      │          │
│  │  ✅ Dev.to Publishing: Success                       │          │
│  │  ✅ Medium Publishing: Success (Draft)               │          │
│  │  ✅ Twitter/X Posting: Success                       │          │
│  │  ✅ LinkedIn Posting: Success                        │          │
│  │  ✅ GitHub Discussions: Updated                      │          │
│  │  ✅ Blog Files: Committed                            │          │
│  │                                                       │          │
│  │  📝 Required Secrets Documentation                   │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
CHANGELOG.md
     │
     ├─→ Parse & Extract
     │
     ├─→ Generate Blog Posts
     │   ├─→ Dev.to Format (markdown + frontmatter)
     │   ├─→ Medium Format (markdown)
     │   └─→ GitHub Format (markdown)
     │
     └─→ Generate Social Posts
         ├─→ Twitter/X Format (280 char limit)
         └─→ LinkedIn Format (professional)
```

## Error Handling Flow

```
Step Execution
     │
     ├─→ Success
     │   ├─→ Continue to next step
     │   └─→ Upload artifacts
     │
     └─→ Failure
         ├─→ Log detailed error
         ├─→ Continue anyway (continue-on-error: true)
         ├─→ Mark step as failed in summary
         └─→ Workflow continues to completion
```

## Secret Validation Logic

```
Check Secret Exists
     │
     ├─→ YES
     │   ├─→ Mark platform available
     │   ├─→ Run publishing step
     │   └─→ Report success/failure
     │
     └─→ NO
         ├─→ Mark platform unavailable
         ├─→ Skip publishing step
         └─→ Report in summary
```

## Platform-Specific Workflows

### Dev.to Publishing

```
1. Read generated blog post
2. Parse frontmatter
3. Extract body markdown
4. Call Dev.to API:
   POST /api/articles
   Headers: api-key
   Body: {
     article: {
       title: "...",
       published: true,
       body_markdown: "...",
       tags: [...],
       series: "..."
     }
   }
5. Get article URL
6. Save for discussion linking
```

### Medium Publishing

```
1. Read generated blog post
2. Get user ID (if not provided)
   GET /v1/me
3. Create draft post:
   POST /v1/users/{userId}/posts
   Headers: Authorization Bearer
   Body: {
     title: "...",
     contentFormat: "markdown",
     content: "...",
     tags: [...],
     publishStatus: "draft"
   }
4. Get draft URL
5. Save for discussion linking
```

### Twitter/X Posting

```
1. Read generated post
2. Authenticate with OAuth 1.0a
3. Ensure <= 280 characters
4. Post status update:
   POST /statuses/update
   Body: { status: "..." }
5. Get tweet URL
6. Save for discussion linking
```

### LinkedIn Posting

```
1. Read generated post
2. Get person URN (if not provided)
   GET /v2/me
3. Create UGC post:
   POST /v2/ugcPosts
   Headers: Authorization Bearer
   Body: {
     author: "urn:li:person:...",
     lifecycleState: "PUBLISHED",
     specificContent: {...},
     visibility: "PUBLIC"
   }
4. Get post ID
5. Save for discussion linking
```

## Timing & Performance

```
Typical Workflow Duration: 3-5 minutes

Breakdown:
├─ Secret Validation:     10-15 seconds
├─ Content Generation:    30-45 seconds
├─ Dev.to Publishing:     5-10 seconds
├─ Medium Publishing:     5-10 seconds
├─ Twitter Posting:       5-10 seconds
├─ LinkedIn Posting:      5-10 seconds
├─ GitHub Discussion:     10-15 seconds
├─ File Commit:          10-15 seconds
└─ Summary Generation:    5-10 seconds
```

## Artifact Structure

```
marketing-content.zip
├── docs/
│   ├── blog/
│   │   ├── devto_vX.X.X_release.md
│   │   ├── medium_vX.X.X_release.md
│   │   └── github_vX.X.X_release.md
│   └── social/
│       ├── twitter_vX.X.X.txt
│       └── linkedin_vX.X.X.txt
```

## State Management

```
Workflow State (Outputs)
├─ version: "X.X.X"
├─ has_devto: true/false
├─ has_medium: true/false
├─ has_twitter: true/false
└─ has_linkedin: true/false

Step Dependencies
├─ validate-secrets (no dependencies)
├─ generate-content (depends: validate-secrets)
├─ publish-devto (depends: generate-content)
├─ publish-medium (depends: generate-content)
├─ post-twitter (depends: generate-content)
├─ post-linkedin (depends: generate-content)
├─ update-discussions (depends: all publish/post jobs)
├─ commit-files (depends: all publish/post jobs)
└─ summary (depends: all jobs)
```

## Conditional Execution

```
Job Execution Conditions:

publish-devto:
  if: has_devto == true && !skip_blog_publish

publish-medium:
  if: has_medium == true && !skip_blog_publish

post-twitter:
  if: has_twitter == true && !skip_social

post-linkedin:
  if: has_linkedin == true && !skip_social

update-discussions:
  if: always()  # Run even if some publishes failed

commit-files:
  if: always()  # Always commit generated files

summary:
  if: always()  # Always generate summary
```

## Success Criteria

```
Workflow Success = ALL of:
  ✅ Content generated successfully
  ✅ At least one platform published (or all skipped intentionally)
  ✅ Files committed to repository
  ✅ Summary generated

Individual Step Failure is OK due to continue-on-error!
```

## Resources

- [Workflow File](../.github/workflows/marketing_automation.yml)
- [Full Guide](MARKETING_AUTOMATION_GUIDE.md)
- [Quick Reference](MARKETING_AUTOMATION_QUICK_REF.md)
- [Setup Guide](MARKETING_AUTOMATION_SETUP.md)
