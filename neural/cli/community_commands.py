"""
CLI commands for community features.
"""

from __future__ import annotations

import os
from typing import Optional

import click

from ..marketplace import (
    CommunityFeatures,
    DiscordCommunityManager,
    DiscordWebhook,
    EducationalResources,
)


@click.group(name='community')
def community_cli():
    """Community features and engagement."""
    pass


@community_cli.command(name='leaderboard')
@click.option('--limit', default=20, help='Number of entries to show')
def show_leaderboard(limit: int):
    """Show community leaderboard."""
    community = CommunityFeatures()
    leaderboard = community.get_leaderboard(limit=limit)

    click.echo(click.style("\n🏆 Community Leaderboard\n", fg='bright_yellow', bold=True))

    for i, entry in enumerate(leaderboard, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        click.echo(f"{medal} {entry['username']}: {entry['points']} points")

        if entry['badges']:
            click.echo(f"   Badges: {', '.join(entry['badges'])}")
        click.echo()


@community_cli.command(name='profile')
@click.argument('user_id')
def show_profile(user_id: str):
    """Show user profile."""
    community = CommunityFeatures()
    profile = community.get_user_profile(user_id)

    if not profile:
        click.echo(click.style(f"User '{user_id}' not found", fg='red'))
        return

    click.echo(click.style(f"\n👤 {profile['username']}\n", fg='bright_cyan', bold=True))
    click.echo(f"Points: {profile['points']}")
    click.echo(f"Total Contributions: {profile['total_contributions']}")

    if profile['badges']:
        click.echo(f"\nBadges: {', '.join(profile['badges'])}")

    click.echo("\nContribution Summary:")
    for contrib_type, count in profile['contribution_summary'].items():
        click.echo(f"  {contrib_type}: {count}")
    click.echo()


@community_cli.command(name='trending')
@click.option('--days', default=7, help='Number of days to look back')
@click.option('--limit', default=10, help='Number of models to show')
def show_trending(days: int, limit: int):
    """Show trending models."""
    community = CommunityFeatures()
    trending = community.get_trending_models(time_window=days, limit=limit)

    click.echo(
        click.style(f"\n🔥 Trending Models (last {days} days)\n", fg='bright_red', bold=True)
    )

    if not trending:
        click.echo("No trending models found")
        return

    for model in trending:
        click.echo(f"Model: {model['model_id']}")
        click.echo(f"  Score: {model['score']:.1f}")

        if 'recent_ratings' in model:
            click.echo(f"  Recent ratings: {model['recent_ratings']}")
        if 'recent_comments' in model:
            click.echo(f"  Recent comments: {model['recent_comments']}")
        click.echo()


@community_cli.command(name='rate')
@click.argument('model_id')
@click.option('--user', required=True, help='Your user ID')
@click.option('--rating', type=int, required=True, help='Rating (1-5)')
@click.option('--review', help='Review text')
def rate_model(model_id: str, user: str, rating: int, review: Optional[str]):
    """Rate a model."""
    if not 1 <= rating <= 5:
        click.echo(click.style("Rating must be between 1 and 5", fg='red'))
        return

    community = CommunityFeatures()
    success = community.rate_model(model_id, user, rating, review)

    if success:
        click.echo(click.style("✓ Rating submitted", fg='green'))

        community.track_contribution(
            user_id=user,
            username=user,
            contribution_type="rating"
        )
    else:
        click.echo(click.style("✗ Failed to submit rating", fg='red'))


@community_cli.command(name='comment')
@click.argument('model_id')
@click.option('--user', required=True, help='Your user ID')
@click.option('--username', required=True, help='Your username')
@click.option('--text', required=True, help='Comment text')
@click.option('--reply-to', help='Parent comment ID')
def add_comment(model_id: str, user: str, username: str, text: str, reply_to: Optional[str]):
    """Add a comment to a model."""
    community = CommunityFeatures()
    comment_id = community.add_comment(model_id, user, username, text, reply_to)

    click.echo(click.style(f"✓ Comment added: {comment_id}", fg='green'))

    community.track_contribution(
        user_id=user,
        username=username,
        contribution_type="comment"
    )


@community_cli.command(name='bookmark')
@click.argument('model_id')
@click.option('--user', required=True, help='Your user ID')
@click.option('--remove', is_flag=True, help='Remove bookmark')
def bookmark_model(model_id: str, user: str, remove: bool):
    """Bookmark or unbookmark a model."""
    community = CommunityFeatures()

    if remove:
        success = community.remove_bookmark(model_id, user)
        msg = "✓ Bookmark removed" if success else "✗ Bookmark not found"
    else:
        success = community.bookmark_model(model_id, user)
        msg = "✓ Model bookmarked" if success else "✗ Already bookmarked"

    click.echo(click.style(msg, fg='green' if success else 'yellow'))


@community_cli.command(name='bookmarks')
@click.option('--user', required=True, help='Your user ID')
def list_bookmarks(user: str):
    """List your bookmarked models."""
    community = CommunityFeatures()
    bookmarks = community.get_user_bookmarks(user)

    click.echo(
        click.style(f"\n📚 Your Bookmarks ({len(bookmarks)})\n", fg='bright_blue', bold=True)
    )

    if not bookmarks:
        click.echo("No bookmarks yet")
        return

    for model_id in bookmarks:
        click.echo(f"• {model_id}")
    click.echo()


@community_cli.command(name='discord-announce')
@click.argument('model_id')
@click.option('--name', required=True, help='Model name')
@click.option('--author', required=True, help='Author name')
@click.option('--description', required=True, help='Model description')
@click.option('--webhook', help='Discord webhook URL')
def discord_announce(
    model_id: str,
    name: str,
    author: str,
    description: str,
    webhook: Optional[str]
):
    """Announce a model on Discord."""
    webhook_url = webhook or os.getenv('DISCORD_WEBHOOK_URL')

    if not webhook_url:
        click.echo(
            click.style(
                "Discord webhook URL not provided. "
                "Set DISCORD_WEBHOOK_URL environment variable or use --webhook",
                fg='red'
            )
        )
        return

    try:
        discord = DiscordWebhook(webhook_url)
        success = discord.announce_new_model(
            model_name=name,
            author=author,
            description=description
        )

        if success:
            click.echo(click.style("✓ Announcement sent to Discord", fg='green'))
        else:
            click.echo(click.style("✗ Failed to send announcement", fg='red'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


@community_cli.command(name='events')
@click.option('--limit', default=10, help='Number of events to show')
def show_events(limit: int):
    """Show upcoming community events."""
    manager = DiscordCommunityManager()
    events = manager.get_upcoming_events(limit=limit)

    click.echo(click.style("\n📅 Upcoming Events\n", fg='bright_magenta', bold=True))

    if not events:
        click.echo("No upcoming events")
        return

    for event in events:
        click.echo(click.style(event['title'], fg='cyan', bold=True))
        click.echo(f"Type: {event['type']}")
        click.echo(f"Host: {event['host']}")
        click.echo(f"Start: {event['start_time']}")
        click.echo(f"Duration: {event['duration_minutes']} minutes")

        if event.get('max_participants'):
            spots_left = event['max_participants'] - len(event['participants'])
            click.echo(f"Spots left: {spots_left}/{event['max_participants']}")

        click.echo()


@community_cli.command(name='tutorials')
@click.option(
    '--difficulty',
    type=click.Choice(['beginner', 'intermediate', 'advanced']),
    help='Filter by difficulty'
)
@click.option('--limit', default=10, help='Number of tutorials to show')
def show_tutorials(difficulty: Optional[str], limit: int):
    """Show available tutorials."""
    edu = EducationalResources()
    tutorials = edu.get_tutorials(difficulty=difficulty, limit=limit)

    title = "📖 Tutorials"
    if difficulty:
        title += f" ({difficulty})"

    click.echo(click.style(f"\n{title}\n", fg='bright_green', bold=True))

    if not tutorials:
        click.echo("No tutorials found")
        return

    for tutorial in tutorials:
        click.echo(click.style(tutorial['title'], fg='green'))
        click.echo(f"Author: {tutorial['author']}")
        click.echo(f"Difficulty: {tutorial['difficulty']}")
        click.echo(f"Duration: {tutorial['duration_minutes']} minutes")
        click.echo(f"Views: {tutorial['views']}")

        if tutorial.get('tags'):
            click.echo(f"Tags: {', '.join(tutorial['tags'])}")

        click.echo()


@community_cli.command(name='courses')
@click.option(
    '--level',
    type=click.Choice(['beginner', 'intermediate', 'advanced']),
    help='Filter by level'
)
@click.option('--limit', default=10, help='Number of courses to show')
def show_courses(level: Optional[str], limit: int):
    """Show available courses."""
    edu = EducationalResources()
    courses = edu.get_courses(level=level, limit=limit)

    title = "🎓 Courses"
    if level:
        title += f" ({level})"

    click.echo(click.style(f"\n{title}\n", fg='bright_cyan', bold=True))

    if not courses:
        click.echo("No courses found")
        return

    for course in courses:
        click.echo(click.style(course['title'], fg='cyan', bold=True))
        click.echo(f"Instructor: {course['instructor']}")
        click.echo(f"University: {course['university']}")
        click.echo(f"Level: {course['level']}")
        click.echo(f"Topics: {', '.join(course['topics'])}")

        if course.get('resources'):
            click.echo("Resources available:")
            for key, url in course['resources'].items():
                click.echo(f"  - {key}: {url}")

        click.echo()


@community_cli.command(name='stats')
def show_stats():
    """Show community statistics."""
    from ..marketplace import ModelRegistry

    community = CommunityFeatures()
    registry = ModelRegistry()

    click.echo(click.style("\n📊 Community Statistics\n", fg='bright_yellow', bold=True))

    total_models = len(registry.metadata.get("models", {}))
    total_contributors = len(community.contributors)
    total_ratings = sum(len(r.get("ratings", [])) for r in community.ratings.values())
    total_comments = sum(len(c) for c in community.comments.values())

    click.echo(f"Total Models: {total_models}")
    click.echo(f"Total Contributors: {total_contributors}")
    click.echo(f"Total Ratings: {total_ratings}")
    click.echo(f"Total Comments: {total_comments}")

    if total_contributors > 0:
        active_contributors = len([
            c for c in community.contributors.values()
            if c.get("points", 0) > 0
        ])
        click.echo(f"Active Contributors: {active_contributors}")

        total_points = sum(c.get("points", 0) for c in community.contributors.values())
        avg_points = total_points / total_contributors
        click.echo(f"Average Points per User: {avg_points:.1f}")

    click.echo()


if __name__ == '__main__':
    community_cli()
