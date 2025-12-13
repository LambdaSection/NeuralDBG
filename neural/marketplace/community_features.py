"""
Community Features for Marketplace - Creating a flywheel effect.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CommunityFeatures:
    """Community features for marketplace to drive engagement."""

    def __init__(self, data_dir: str = "marketplace_community"):
        """Initialize community features.

        Parameters
        ----------
        data_dir : str
            Directory to store community data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.ratings_file = self.data_dir / "ratings.json"
        self.comments_file = self.data_dir / "comments.json"
        self.bookmarks_file = self.data_dir / "bookmarks.json"
        self.contributors_file = self.data_dir / "contributors.json"

        self._load_data()

    def _load_data(self):
        """Load all community data."""
        self.ratings = self._load_json(self.ratings_file, {})
        self.comments = self._load_json(self.comments_file, {})
        self.bookmarks = self._load_json(self.bookmarks_file, {})
        self.contributors = self._load_json(self.contributors_file, {})

    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Load JSON from file."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default

    def _save_json(self, file_path: Path, data: Any):
        """Save JSON to file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def rate_model(
        self,
        model_id: str,
        user_id: str,
        rating: int,
        review: Optional[str] = None
    ) -> bool:
        """Rate a model.

        Parameters
        ----------
        model_id : str
            Model ID
        user_id : str
            User ID
        rating : int
            Rating (1-5)
        review : str, optional
            Review text

        Returns
        -------
        bool
            True if successful
        """
        if not 1 <= rating <= 5:
            return False

        if model_id not in self.ratings:
            self.ratings[model_id] = {
                "total_ratings": 0,
                "average_rating": 0.0,
                "ratings": []
            }

        rating_data = {
            "user_id": user_id,
            "rating": rating,
            "review": review,
            "timestamp": datetime.now().isoformat()
        }

        existing_idx = None
        for idx, r in enumerate(self.ratings[model_id]["ratings"]):
            if r["user_id"] == user_id:
                existing_idx = idx
                break

        if existing_idx is not None:
            self.ratings[model_id]["ratings"][existing_idx] = rating_data
        else:
            self.ratings[model_id]["ratings"].append(rating_data)

        total = sum(r["rating"] for r in self.ratings[model_id]["ratings"])
        count = len(self.ratings[model_id]["ratings"])

        self.ratings[model_id]["total_ratings"] = count
        self.ratings[model_id]["average_rating"] = total / count if count > 0 else 0.0

        self._save_json(self.ratings_file, self.ratings)
        return True

    def get_model_rating(self, model_id: str) -> Dict[str, Any]:
        """Get model rating statistics.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        Dict
            Rating statistics
        """
        if model_id not in self.ratings:
            return {
                "total_ratings": 0,
                "average_rating": 0.0,
                "ratings": []
            }

        return self.ratings[model_id]

    def add_comment(
        self,
        model_id: str,
        user_id: str,
        username: str,
        comment: str,
        parent_id: Optional[str] = None
    ) -> str:
        """Add a comment to a model.

        Parameters
        ----------
        model_id : str
            Model ID
        user_id : str
            User ID
        username : str
            Username
        comment : str
            Comment text
        parent_id : str, optional
            Parent comment ID for replies

        Returns
        -------
        str
            Comment ID
        """
        if model_id not in self.comments:
            self.comments[model_id] = []

        comment_id = (
            f"comment_{len(self.comments[model_id]) + 1}_"
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        comment_data = {
            "id": comment_id,
            "user_id": user_id,
            "username": username,
            "comment": comment,
            "parent_id": parent_id,
            "timestamp": datetime.now().isoformat(),
            "likes": 0,
            "replies": []
        }

        if parent_id:
            for c in self.comments[model_id]:
                if c["id"] == parent_id:
                    c["replies"].append(comment_id)
                    break

        self.comments[model_id].append(comment_data)
        self._save_json(self.comments_file, self.comments)

        return comment_id

    def get_model_comments(self, model_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get comments for a model.

        Parameters
        ----------
        model_id : str
            Model ID
        limit : int
            Maximum number of comments

        Returns
        -------
        List[Dict]
            List of comments
        """
        if model_id not in self.comments:
            return []

        comments = self.comments[model_id]
        comments.sort(key=lambda c: c["timestamp"], reverse=True)

        return comments[:limit]

    def bookmark_model(self, model_id: str, user_id: str) -> bool:
        """Bookmark a model.

        Parameters
        ----------
        model_id : str
            Model ID
        user_id : str
            User ID

        Returns
        -------
        bool
            True if successful
        """
        if user_id not in self.bookmarks:
            self.bookmarks[user_id] = []

        if model_id not in self.bookmarks[user_id]:
            self.bookmarks[user_id].append({
                "model_id": model_id,
                "timestamp": datetime.now().isoformat()
            })
            self._save_json(self.bookmarks_file, self.bookmarks)
            return True

        return False

    def remove_bookmark(self, model_id: str, user_id: str) -> bool:
        """Remove a bookmark.

        Parameters
        ----------
        model_id : str
            Model ID
        user_id : str
            User ID

        Returns
        -------
        bool
            True if successful
        """
        if user_id not in self.bookmarks:
            return False

        initial_len = len(self.bookmarks[user_id])
        self.bookmarks[user_id] = [
            b for b in self.bookmarks[user_id]
            if b["model_id"] != model_id
        ]

        if len(self.bookmarks[user_id]) < initial_len:
            self._save_json(self.bookmarks_file, self.bookmarks)
            return True

        return False

    def get_user_bookmarks(self, user_id: str) -> List[str]:
        """Get user's bookmarked models.

        Parameters
        ----------
        user_id : str
            User ID

        Returns
        -------
        List[str]
            List of model IDs
        """
        if user_id not in self.bookmarks:
            return []

        return [b["model_id"] for b in self.bookmarks[user_id]]

    def track_contribution(
        self,
        user_id: str,
        username: str,
        contribution_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Track user contributions for gamification.

        Parameters
        ----------
        user_id : str
            User ID
        username : str
            Username
        contribution_type : str
            Type: upload, rating, comment, help, documentation
        details : Dict, optional
            Additional details
        """
        if user_id not in self.contributors:
            self.contributors[user_id] = {
                "username": username,
                "contributions": [],
                "points": 0,
                "badges": []
            }

        contribution = {
            "type": contribution_type,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        self.contributors[user_id]["contributions"].append(contribution)

        points_map = {
            "upload": 50,
            "rating": 5,
            "comment": 10,
            "help": 20,
            "documentation": 30,
            "featured": 100
        }

        self.contributors[user_id]["points"] += points_map.get(contribution_type, 0)

        self._check_badges(user_id)
        self._save_json(self.contributors_file, self.contributors)

    def _check_badges(self, user_id: str):
        """Check and award badges to users."""
        contributor = self.contributors[user_id]
        current_badges = set(contributor.get("badges", []))

        uploads = len([c for c in contributor["contributions"] if c["type"] == "upload"])
        ratings = len([c for c in contributor["contributions"] if c["type"] == "rating"])
        comments = len([c for c in contributor["contributions"] if c["type"] == "comment"])
        points = contributor["points"]

        badge_criteria = {
            "First Model": uploads >= 1,
            "Model Publisher": uploads >= 5,
            "Prolific Creator": uploads >= 10,
            "Model Reviewer": ratings >= 10,
            "Community Helper": comments >= 20,
            "Rising Star": points >= 100,
            "Community Champion": points >= 500,
            "Legend": points >= 1000,
        }

        for badge, criteria in badge_criteria.items():
            if criteria and badge not in current_badges:
                contributor["badges"].append(badge)

    def get_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get community leaderboard.

        Parameters
        ----------
        limit : int
            Maximum number of entries

        Returns
        -------
        List[Dict]
            Sorted leaderboard
        """
        leaderboard = []

        for user_id, data in self.contributors.items():
            leaderboard.append({
                "user_id": user_id,
                "username": data["username"],
                "points": data["points"],
                "badges": data["badges"],
                "contribution_count": len(data["contributions"])
            })

        leaderboard.sort(key=lambda x: x["points"], reverse=True)

        return leaderboard[:limit]

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with contributions.

        Parameters
        ----------
        user_id : str
            User ID

        Returns
        -------
        Dict, optional
            User profile data
        """
        if user_id not in self.contributors:
            return None

        data = self.contributors[user_id]

        contribution_summary = {}
        for contrib in data["contributions"]:
            contrib_type = contrib["type"]
            contribution_summary[contrib_type] = contribution_summary.get(contrib_type, 0) + 1

        return {
            "user_id": user_id,
            "username": data["username"],
            "points": data["points"],
            "badges": data["badges"],
            "contribution_summary": contribution_summary,
            "total_contributions": len(data["contributions"])
        }

    def get_trending_models(
        self,
        time_window: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending models based on recent activity.

        Parameters
        ----------
        time_window : int
            Days to look back
        limit : int
            Maximum number of models

        Returns
        -------
        List[Dict]
            Trending models with scores
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=time_window)
        trending = {}

        for model_id, rating_data in self.ratings.items():
            recent_ratings = [
                r for r in rating_data["ratings"]
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]
            trending[model_id] = {
                "model_id": model_id,
                "recent_ratings": len(recent_ratings),
                "average_rating": rating_data["average_rating"],
                "score": len(recent_ratings) * rating_data["average_rating"]
            }

        for model_id, comment_list in self.comments.items():
            recent_comments = [
                c for c in comment_list
                if datetime.fromisoformat(c["timestamp"]) > cutoff
            ]

            if model_id in trending:
                trending[model_id]["recent_comments"] = len(recent_comments)
                trending[model_id]["score"] += len(recent_comments) * 2
            else:
                trending[model_id] = {
                    "model_id": model_id,
                    "recent_comments": len(recent_comments),
                    "score": len(recent_comments) * 2
                }

        trending_list = list(trending.values())
        trending_list.sort(key=lambda x: x["score"], reverse=True)

        return trending_list[:limit]
