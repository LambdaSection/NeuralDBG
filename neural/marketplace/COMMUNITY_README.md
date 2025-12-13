# Neural Marketplace - Community Features

This directory contains the community building features for Neural DSL's marketplace, designed to create engagement, foster collaboration, and drive organic growth through a flywheel effect.

## Components

### `community_features.py`
Core community engagement features including ratings, comments, bookmarks, contribution tracking, and gamification.

**Features:**
- ⭐ Model ratings and reviews
- 💬 Comments and threaded discussions
- 🔖 User bookmarks
- 🎮 Gamification with points and badges
- 🏆 Community leaderboard
- 🔥 Trending models algorithm
- 👤 User profiles and contributions

### `discord_bot.py`
Discord integration for real-time community engagement and notifications.

**Features:**
- 📢 Automated model announcements
- 🏆 Achievement notifications
- 📊 Weekly community highlights
- 📅 Event scheduling and management
- ❓ Help request system
- 🎪 Community events

### `education.py`
Educational resources and university program support.

**Features:**
- 📚 Course material repository
- 📝 Assignment templates
- 🎓 Tutorial library
- 🛤️ Structured learning paths
- 🏫 University registration
- 🎫 Academic license management

## Quick Start

### Basic Usage

```python
from neural.marketplace import CommunityFeatures

# Initialize community features
community = CommunityFeatures()

# Rate a model
community.rate_model(
    model_id="model_123",
    user_id="alice",
    rating=5,
    review="Excellent model!"
)

# Get leaderboard
leaderboard = community.get_leaderboard(limit=10)
for entry in leaderboard:
    print(f"{entry['username']}: {entry['points']} points")
```

### Discord Integration

```python
from neural.marketplace import DiscordWebhook

# Set up webhook
webhook = DiscordWebhook("YOUR_WEBHOOK_URL")

# Announce new model
webhook.announce_new_model(
    model_name="SuperCNN",
    author="Alice",
    description="State-of-the-art CNN for image classification",
    rating=4.8,
    tags=["computer-vision", "cnn"]
)
```

### Educational Resources

```python
from neural.marketplace import EducationalResources

# Initialize
edu = EducationalResources()

# Add course material
course_id = edu.add_course_material(
    title="Deep Learning with Neural DSL",
    instructor="Prof. Smith",
    university="MIT",
    level="intermediate",
    topics=["neural-networks", "cnn", "rnn"]
)

# Add tutorial
tutorial_id = edu.add_tutorial(
    title="Getting Started",
    author="Bob",
    difficulty="beginner",
    duration_minutes=30,
    content="# Tutorial content here..."
)
```

## CLI Commands

All community features are accessible via CLI:

```bash
# View leaderboard
neural community leaderboard --limit 20

# View user profile
neural community profile alice

# Rate a model
neural community rate model_123 --user alice --rating 5

# View trending models
neural community trending --days 7

# List tutorials
neural community tutorials --difficulty beginner

# View upcoming events
neural community events
```

## Gamification System

### Points

| Action | Points |
|--------|--------|
| Upload Model | 50 |
| Rate Model | 5 |
| Comment | 10 |
| Help Others | 20 |
| Documentation | 30 |
| Featured Model | 100 |

### Badges

Users automatically earn badges:
- 🎯 **First Model** - Upload your first model
- 📦 **Model Publisher** - Publish 5 models
- ⭐ **Prolific Creator** - Publish 10+ models
- 📝 **Model Reviewer** - Rate 10+ models
- 🤝 **Community Helper** - Make 20+ helpful comments
- 🌟 **Rising Star** - Earn 100+ points
- 🏆 **Community Champion** - Earn 500+ points
- 👑 **Legend** - Earn 1000+ points

## The Flywheel Effect

Community features create a self-reinforcing cycle:

1. **Quality Content** - Users upload valuable models
2. **Discovery** - Others find and use these models
3. **Engagement** - Users rate, review, and discuss
4. **Recognition** - Contributors earn points and badges
5. **Motivation** - Recognition drives more contributions
6. **Growth** - More content attracts more users
7. **Network Effects** - Each user increases platform value

The cycle repeats and accelerates, creating organic growth.

## Architecture

### Data Storage

Community features use JSON file storage by default:

```
marketplace_community/
├── ratings.json          # Model ratings and reviews
├── comments.json         # Comments and discussions
├── bookmarks.json        # User bookmarks
└── contributors.json     # User contributions and points

educational_resources/
├── courses.json          # Course materials
├── assignments.json      # Assignment templates
├── tutorials.json        # Tutorial library
└── universities.json     # University registrations

discord_community/
└── events.json          # Community events
```

### Extensibility

All modules are designed for easy extension:

```python
from neural.marketplace import CommunityFeatures

class CustomCommunity(CommunityFeatures):
    def custom_feature(self):
        # Add your custom logic
        pass
```

## Integration

### With Marketplace UI

```python
from neural.marketplace import MarketplaceUI

ui = MarketplaceUI(
    registry_dir="marketplace",
    discord_webhook="YOUR_WEBHOOK_URL"
)
ui.run(port=8052)
```

The web UI automatically includes:
- Community tab with leaderboard
- Model ratings and comments
- Trending models section
- Event calendar

### With Discord

Set up webhook URL:

```bash
export DISCORD_WEBHOOK_URL="your_webhook_url"
```

Notifications will be sent automatically for:
- New model uploads
- Achievement unlocks
- Community events
- Help requests

## Configuration

### Environment Variables

```bash
# Discord webhook for notifications
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Data directories (optional)
export COMMUNITY_DATA_DIR="marketplace_community"
export EDUCATION_DATA_DIR="educational_resources"
```

### Programmatic Configuration

```python
from neural.marketplace import CommunityFeatures

community = CommunityFeatures(
    data_dir="custom_community_dir"
)
```

## Examples

See `examples/community_example.py` for comprehensive examples covering:
- Rating and reviewing models
- Adding comments and discussions
- Bookmarking models
- Tracking contributions
- Discord integration
- Educational resources
- Event management

## Documentation

Full documentation available in:
- `docs/community_building.md` - Complete guide
- `docs/educators_guide.md` - University support
- `examples/showcase_submission.md` - Project submissions

## Testing

Run community feature tests:

```bash
pytest tests/test_marketplace.py -v -k community
```

## Contributing

We welcome contributions! To add features:

1. Fork the repository
2. Create a feature branch
3. Add your feature with tests
4. Submit a pull request

See `CONTRIBUTING.md` for details.

## Support

**Discord:** https://discord.gg/KFku4KvS
**GitHub:** https://github.com/Lemniscate-world/Neural/issues
**Email:** Lemniscate_zero@proton.me

## License

MIT License - see LICENSE.md for details

---

Built with ❤️ by the Neural DSL community
