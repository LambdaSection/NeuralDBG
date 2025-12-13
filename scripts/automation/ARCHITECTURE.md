# Automation Architecture

## Overview

The Neural DSL automation system provides a comprehensive suite of tools for managing releases, testing, blog publishing, and marketing automation.

## Component Architecture

```
scripts/automation/
├── Core Generators
│   ├── blog_generator.py          - Generate blog posts from CHANGELOG
│   └── social_media_generator.py  - Generate social media content
│
├── Publishers
│   ├── devto_publisher.py         - Dev.to API integration
│   └── medium_publisher.py        - Medium API integration
│
├── Automation
│   ├── master_automation.py       - Orchestrates all tasks
│   ├── release_automation.py      - Handles releases
│   ├── test_automation.py         - Test execution
│   └── example_validator.py       - Example validation
│
├── Documentation
│   ├── README.md                  - Usage guide
│   ├── ARCHITECTURE.md           - This file
│   └── .env.example              - Configuration template
│
└── Examples
    └── example_usage.py          - Code examples
```

## Data Flow

### Marketing Automation Flow

```
CHANGELOG.md
    │
    ├─> BlogGenerator
    │       │
    │       ├─> Medium Post (docs/blog/medium_*.md)
    │       ├─> Dev.to Post (docs/blog/devto_*.md)
    │       └─> GitHub Notes (docs/blog/github_*.md)
    │
    ├─> SocialMediaGenerator
    │       │
    │       ├─> Twitter Post (docs/social/twitter_*.txt)
    │       └─> LinkedIn Post (docs/social/linkedin_*.txt)
    │
    └─> Publishers (if API keys configured)
            │
            ├─> DevToPublisher ──> Dev.to Platform
            └─> MediumPublisher ──> Medium Platform
```

### Release Automation Flow

```
User Trigger (--release)
    │
    ├─> Version Bump (setup.py, __init__.py)
    │
    ├─> Run Tests (TestAutomation)
    │
    ├─> Validate Examples (ExampleValidator)
    │
    ├─> Build Package (python -m build)
    │
    ├─> Create GitHub Release (gh CLI)
    │
    ├─> Publish to PyPI (twine)
    │
    └─> Marketing Automation
            │
            ├─> Generate Content
            └─> Publish (optional)
```

## Component Details

### 1. BlogGenerator

**Purpose**: Extracts release notes from CHANGELOG.md and generates platform-specific blog posts.

**Input**: 
- CHANGELOG.md
- Version number (optional, auto-detected)

**Output**:
- Medium-formatted post
- Dev.to-formatted post (with frontmatter)
- GitHub release notes

**Key Methods**:
- `_detect_version()` - Auto-detect current version
- `_extract_release_notes()` - Parse CHANGELOG
- `generate_medium_post()` - Format for Medium
- `generate_devto_post()` - Format for Dev.to with frontmatter

### 2. DevToPublisher

**Purpose**: Publishes articles to Dev.to via REST API.

**Features**:
- YAML frontmatter parsing
- Article creation and updates
- Draft/published status control
- Tag management (max 4)
- Canonical URL support
- Find article by title
- Batch publishing

**API Endpoints**:
- `POST /api/articles` - Create article
- `PUT /api/articles/{id}` - Update article
- `GET /api/articles/me` - List user articles

**Configuration**:
- `DEVTO_API_KEY` environment variable

### 3. MediumPublisher

**Purpose**: Publishes articles to Medium via REST API.

**Features**:
- Markdown content support
- Draft/public/unlisted status
- Tag management (max 5)
- Publication support
- License configuration
- User info retrieval

**API Endpoints**:
- `GET /v1/me` - Get user info
- `POST /v1/users/{userId}/posts` - Create post
- `GET /v1/users/{userId}/publications` - List publications
- `POST /v1/publications/{pubId}/posts` - Create publication post

**Configuration**:
- `MEDIUM_API_TOKEN` environment variable

**Note**: Medium API doesn't support updating existing posts.

### 4. MasterAutomation

**Purpose**: Orchestrates all automation workflows.

**Workflows**:

#### Marketing Workflow (`--marketing`)
1. Generate blog posts
2. Generate social media posts
3. Publish to Dev.to (if `--publish-devto`)
4. Publish to Medium (if `--publish-medium`)

#### Release Workflow (`--release`)
1. Run tests
2. Validate examples
3. Bump version
4. Build package
5. Create GitHub release
6. Publish to PyPI
7. Run marketing automation

#### Daily Tasks (`--daily`)
1. Run test suite
2. Validate examples
3. Generate reports

**Command Examples**:
```bash
# Marketing only
python master_automation.py --marketing

# Marketing + publishing
python master_automation.py --marketing --publish-devto --publish-medium

# Full release
python master_automation.py --release --version-type patch

# Daily maintenance
python master_automation.py --daily
```

### 5. SocialMediaGenerator

**Purpose**: Generates social media posts from release notes.

**Output**:
- Twitter/X post (280 char limit)
- LinkedIn post (extended format)

**Features**:
- Automatic truncation for Twitter
- Hashtag management
- Feature highlighting (top features)
- Link inclusion

## Configuration

### Environment Variables

```bash
# Required for publishing
DEVTO_API_KEY=your_devto_api_key
MEDIUM_API_TOKEN=your_medium_api_token

# Optional for releases
GITHUB_TOKEN=your_github_token
PYPI_API_TOKEN=your_pypi_token
TEST_PYPI_API_TOKEN=your_test_pypi_token
```

### File Structure

```
project_root/
├── CHANGELOG.md                    # Source for release notes
├── setup.py                        # Version info
├── neural/__init__.py             # Version info
├── scripts/automation/
│   ├── *.py                       # Automation scripts
│   └── .env                       # API credentials (not committed)
└── docs/
    ├── blog/                      # Generated blog posts
    │   ├── medium_v*.md
    │   ├── devto_v*.md
    │   └── github_v*.md
    └── social/                    # Generated social posts
        ├── twitter_v*.txt
        └── linkedin_v*.txt
```

## Error Handling

### DevToPublisher
- API authentication errors → Check DEVTO_API_KEY
- 422 errors → Validate article format
- Rate limiting → Exponential backoff
- Duplicate titles → Use update flag

### MediumPublisher
- API authentication errors → Check MEDIUM_API_TOKEN
- Publication errors → Verify publication ID and permissions
- Rate limiting → Exponential backoff
- No update support → Each publish creates new post

### General
- Missing dependencies → Install requests library
- File not found → Check paths and file generation
- Network errors → Retry with exponential backoff

## Extension Points

### Adding New Platforms

1. **Create Publisher Class**:
   ```python
   class NewPlatformPublisher:
       def __init__(self, api_key):
           self.api_key = api_key
       
       def publish_from_file(self, file_path):
           # Implementation
           pass
   ```

2. **Add to BlogGenerator**:
   ```python
   def generate_newplatform_post(self):
       # Format post for new platform
       pass
   ```

3. **Integrate in MasterAutomation**:
   ```python
   # Add command-line flag
   parser.add_argument("--publish-newplatform")
   
   # Add to marketing workflow
   if args.publish_newplatform:
       publisher = NewPlatformPublisher()
       publisher.publish_from_file(...)
   ```

### Custom Templates

Edit `blog_generator.py` to modify templates:
- Change post structure
- Add/remove sections
- Customize formatting
- Add custom metadata

### Custom Social Media

Edit `social_media_generator.py` to:
- Add new platforms
- Change post format
- Adjust character limits
- Modify hashtags

## Testing

### Manual Testing

```bash
# Test blog generation
python scripts/automation/blog_generator.py 0.3.0

# Test Dev.to publisher (dry run mode coming soon)
python scripts/automation/devto_publisher.py --file test.md

# Test full workflow
python scripts/automation/master_automation.py --marketing
```

### Integration Testing

```python
# In tests/automation/
import pytest
from scripts.automation.devto_publisher import DevToPublisher

def test_frontmatter_parsing():
    content = """---
title: Test
published: false
---
Body content
"""
    publisher = DevToPublisher(api_key="test")
    parsed = publisher.parse_frontmatter(content)
    assert parsed["frontmatter"]["title"] == "Test"
```

## Security

### Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all credentials
3. **Rotate tokens regularly**
4. **Use minimal permissions** (read/write only what's needed)
5. **Validate all inputs** before API calls
6. **Log without exposing** sensitive data

### API Key Management

```bash
# Use .env file (gitignored)
cp scripts/automation/.env.example scripts/automation/.env
# Edit .env with your keys

# Or export directly
export DEVTO_API_KEY="your_key"
export MEDIUM_API_TOKEN="your_token"

# For CI/CD
# Set as GitHub Secrets or environment variables
```

## Performance

### Optimization

- **Parallel publishing**: Publishers can run concurrently
- **Caching**: Generated content is cached in files
- **Rate limiting**: Built-in retry logic for API limits
- **Batch operations**: Directory publishing for multiple files

### Benchmarks

- Blog generation: ~1-2 seconds
- Single article publish: ~2-3 seconds
- Full marketing workflow: ~10-15 seconds
- Full release workflow: ~5-10 minutes (includes tests)

## Future Enhancements

- [ ] Add Hashnode publisher
- [ ] Add Substack publisher
- [ ] Implement retry logic with exponential backoff
- [ ] Add dry-run mode for testing
- [ ] Create GitHub Actions workflow
- [ ] Add article scheduling
- [ ] Implement content templates
- [ ] Add analytics tracking
- [ ] Create dashboard for monitoring
- [ ] Add webhook notifications
