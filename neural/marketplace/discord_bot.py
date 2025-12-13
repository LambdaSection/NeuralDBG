"""
Discord Bot - Community engagement through Discord integration.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DiscordWebhook:
    """Simple Discord webhook integration for notifications."""

    def __init__(self, webhook_url: str):
        """Initialize Discord webhook.

        Parameters
        ----------
        webhook_url : str
            Discord webhook URL
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required. Install with: pip install requests"
            )

        self.webhook_url = webhook_url

    def send_embed(
        self,
        title: str,
        description: str,
        color: int = 0x6366f1,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
        thumbnail: Optional[str] = None,
        image: Optional[str] = None
    ) -> bool:
        """Send an embed message.

        Parameters
        ----------
        title : str
            Embed title
        description : str
            Embed description
        color : int
            Color in decimal
        fields : List[Dict], optional
            Additional fields
        footer : str, optional
            Footer text
        thumbnail : str, optional
            Thumbnail URL
        image : str, optional
            Image URL

        Returns
        -------
        bool
            True if successful
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now().isoformat()
        }

        if fields:
            embed["fields"] = fields

        if footer:
            embed["footer"] = {"text": footer}

        if thumbnail:
            embed["thumbnail"] = {"url": thumbnail}

        if image:
            embed["image"] = {"url": image}

        payload = {"embeds": [embed]}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 204
        except Exception:
            return False

    def announce_new_model(
        self,
        model_name: str,
        author: str,
        description: str,
        downloads: int = 0,
        rating: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Announce a new model.

        Parameters
        ----------
        model_name : str
            Model name
        author : str
            Author name
        description : str
            Model description
        downloads : int
            Download count
        rating : float
            Average rating
        tags : List[str], optional
            Model tags

        Returns
        -------
        bool
            True if successful
        """
        fields = [
            {"name": "👤 Author", "value": author, "inline": True},
            {"name": "⭐ Rating", "value": f"{rating:.1f}/5.0", "inline": True},
            {"name": "📥 Downloads", "value": str(downloads), "inline": True},
        ]

        if tags:
            fields.append({
                "name": "🏷️ Tags",
                "value": ", ".join(tags),
                "inline": False
            })

        return self.send_embed(
            title=f"🚀 New Model: {model_name}",
            description=description,
            color=0x00ff00,
            fields=fields,
            footer="Neural DSL Marketplace"
        )

    def announce_achievement(
        self,
        username: str,
        achievement: str,
        description: str,
        points: int
    ) -> bool:
        """Announce a user achievement.

        Parameters
        ----------
        username : str
            Username
        achievement : str
            Achievement name
        description : str
            Achievement description
        points : int
            Points earned

        Returns
        -------
        bool
            True if successful
        """
        return self.send_embed(
            title="🏆 Achievement Unlocked!",
            description=f"**{username}** earned: **{achievement}**\n\n{description}",
            color=0xffd700,
            fields=[
                {"name": "Points", "value": f"+{points}", "inline": True}
            ],
            footer="Neural DSL Community"
        )

    def share_weekly_highlights(
        self,
        top_models: List[Dict[str, Any]],
        top_contributors: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> bool:
        """Share weekly community highlights.

        Parameters
        ----------
        top_models : List[Dict]
            Top models of the week
        top_contributors : List[Dict]
            Top contributors
        stats : Dict
            Community statistics

        Returns
        -------
        bool
            True if successful
        """
        description = "📊 **Community Stats**\n"
        description += f"• New Models: {stats.get('new_models', 0)}\n"
        description += f"• Total Downloads: {stats.get('total_downloads', 0)}\n"
        description += f"• Active Users: {stats.get('active_users', 0)}\n\n"

        fields = []

        if top_models:
            models_text = "\n".join([
                f"{i+1}. **{m['name']}** by {m['author']} ({m['downloads']} downloads)"
                for i, m in enumerate(top_models[:5])
            ])
            fields.append({
                "name": "🔥 Trending Models",
                "value": models_text,
                "inline": False
            })

        if top_contributors:
            contributors_text = "\n".join([
                f"{i+1}. **{c['username']}** ({c['points']} pts)"
                for i, c in enumerate(top_contributors[:5])
            ])
            fields.append({
                "name": "🌟 Top Contributors",
                "value": contributors_text,
                "inline": False
            })

        return self.send_embed(
            title="📈 Weekly Community Highlights",
            description=description,
            color=0x6366f1,
            fields=fields,
            footer="Neural DSL - Building the Future Together"
        )


class DiscordCommunityManager:
    """Manage Discord community engagement."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        data_dir: str = "discord_community"
    ):
        """Initialize community manager.

        Parameters
        ----------
        webhook_url : str, optional
            Discord webhook URL
        data_dir : str
            Data directory
        """
        self.webhook = DiscordWebhook(webhook_url) if webhook_url else None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.events_file = self.data_dir / "events.json"
        self._load_events()

    def _load_events(self):
        """Load events from disk."""
        if self.events_file.exists():
            with open(self.events_file, 'r') as f:
                self.events = json.load(f)
        else:
            self.events = []
            self._save_events()

    def _save_events(self):
        """Save events to disk."""
        with open(self.events_file, 'w') as f:
            json.dump(self.events, f, indent=2)

    def schedule_event(
        self,
        title: str,
        description: str,
        event_type: str,
        start_time: str,
        duration_minutes: int,
        host: str,
        max_participants: Optional[int] = None
    ) -> str:
        """Schedule a community event.

        Parameters
        ----------
        title : str
            Event title
        description : str
            Event description
        event_type : str
            Type: workshop, webinar, hackathon, office_hours
        start_time : str
            Start time (ISO format)
        duration_minutes : int
            Duration in minutes
        host : str
            Event host
        max_participants : int, optional
            Maximum participants

        Returns
        -------
        str
            Event ID
        """
        event_id = f"event_{len(self.events) + 1}_{datetime.now().strftime('%Y%m%d')}"

        event = {
            "id": event_id,
            "title": title,
            "description": description,
            "type": event_type,
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "host": host,
            "max_participants": max_participants,
            "participants": [],
            "created_at": datetime.now().isoformat(),
            "status": "scheduled"
        }

        self.events.append(event)
        self._save_events()

        if self.webhook:
            self._announce_event(event)

        return event_id

    def _announce_event(self, event: Dict[str, Any]) -> bool:
        """Announce event on Discord."""
        if not self.webhook:
            return False

        event_types_emoji = {
            "workshop": "🛠️",
            "webinar": "📺",
            "hackathon": "⚡",
            "office_hours": "💬"
        }

        emoji = event_types_emoji.get(event["type"], "📅")

        fields = [
            {"name": "🕒 Start Time", "value": event["start_time"], "inline": True},
            {"name": "⏱️ Duration", "value": f"{event['duration_minutes']} minutes", "inline": True},
            {"name": "👨‍🏫 Host", "value": event["host"], "inline": True},
        ]

        if event.get("max_participants"):
            fields.append({
                "name": "👥 Capacity",
                "value": f"{event['max_participants']} spots",
                "inline": True
            })

        return self.webhook.send_embed(
            title=f"{emoji} {event['title']}",
            description=event["description"],
            color=0xff6b6b,
            fields=fields,
            footer="React to register for this event!"
        )

    def register_participant(self, event_id: str, username: str) -> bool:
        """Register a participant for an event.

        Parameters
        ----------
        event_id : str
            Event ID
        username : str
            Username

        Returns
        -------
        bool
            True if successful
        """
        for event in self.events:
            if event["id"] == event_id:
                if event.get("max_participants"):
                    if len(event["participants"]) >= event["max_participants"]:
                        return False

                if username not in event["participants"]:
                    event["participants"].append(username)
                    self._save_events()
                    return True

        return False

    def get_upcoming_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get upcoming events.

        Parameters
        ----------
        limit : int
            Maximum number of events

        Returns
        -------
        List[Dict]
            Upcoming events
        """
        now = datetime.now()
        upcoming = []

        for event in self.events:
            try:
                start_time = datetime.fromisoformat(event["start_time"])
                if start_time > now and event["status"] == "scheduled":
                    upcoming.append(event)
            except (ValueError, KeyError):
                continue

        upcoming.sort(key=lambda e: e["start_time"])
        return upcoming[:limit]

    def create_help_request(
        self,
        username: str,
        title: str,
        description: str,
        category: str
    ) -> bool:
        """Create a help request on Discord.

        Parameters
        ----------
        username : str
            Username
        title : str
            Request title
        description : str
            Request description
        category : str
            Category: bug, question, feature_request

        Returns
        -------
        bool
            True if successful
        """
        if not self.webhook:
            return False

        category_colors = {
            "bug": 0xff0000,
            "question": 0x0099ff,
            "feature_request": 0x00ff00
        }

        category_emojis = {
            "bug": "🐛",
            "question": "❓",
            "feature_request": "💡"
        }

        color = category_colors.get(category, 0x6366f1)
        emoji = category_emojis.get(category, "📝")

        return self.webhook.send_embed(
            title=f"{emoji} {category.replace('_', ' ').title()}: {title}",
            description=description,
            color=color,
            fields=[
                {"name": "Posted by", "value": username, "inline": True}
            ],
            footer="Community help is on the way!"
        )
