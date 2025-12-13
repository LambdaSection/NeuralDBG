"""
Example: Using Neural DSL Community Features

This example demonstrates how to use the community building features including
Discord integration, leaderboards, achievements, and educational resources.
"""

import os

from neural.marketplace import (
    CommunityFeatures,
    DiscordCommunityManager,
    DiscordWebhook,
    EducationalResources,
    UniversityLicenseManager,
)


def example_community_features():
    """Demonstrate community features."""
    print("=== Neural DSL Community Features Demo ===\n")

    community = CommunityFeatures()

    print("1. Rating a model...")
    community.rate_model(
        model_id="model_123",
        user_id="user_alice",
        rating=5,
        review="Excellent model! Very accurate and well-documented."
    )
    print("   ✓ Rating submitted\n")

    print("2. Adding a comment...")
    comment_id = community.add_comment(
        model_id="model_123",
        user_id="user_bob",
        username="Bob Smith",
        comment="How do I use this model for my dataset?"
    )
    print(f"   ✓ Comment added: {comment_id}\n")

    print("3. Replying to comment...")
    community.add_comment(
        model_id="model_123",
        user_id="user_alice",
        username="Alice Johnson",
        comment="Check the documentation at docs/usage.md",
        parent_id=comment_id
    )
    print("   ✓ Reply added\n")

    print("4. Bookmarking a model...")
    community.bookmark_model(model_id="model_123", user_id="user_bob")
    print("   ✓ Model bookmarked\n")

    print("5. Tracking contributions...")
    community.track_contribution(
        user_id="user_alice",
        username="Alice Johnson",
        contribution_type="upload",
        details={"model_id": "model_123"}
    )
    community.track_contribution(
        user_id="user_alice",
        username="Alice Johnson",
        contribution_type="help"
    )
    print("   ✓ Contributions tracked\n")

    print("6. Getting leaderboard...")
    leaderboard = community.get_leaderboard(limit=5)
    print("   Top Contributors:")
    for i, entry in enumerate(leaderboard, 1):
        print(f"   {i}. {entry['username']}: {entry['points']} points")
        if entry['badges']:
            print(f"      Badges: {', '.join(entry['badges'])}")
    print()

    print("7. Getting user profile...")
    profile = community.get_user_profile("user_alice")
    if profile:
        print(f"   User: {profile['username']}")
        print(f"   Points: {profile['points']}")
        print(f"   Badges: {', '.join(profile['badges'])}")
        print(f"   Contributions: {profile['contribution_summary']}")
    print()

    print("8. Getting trending models...")
    trending = community.get_trending_models(time_window=7, limit=5)
    print("   Trending Models (last 7 days):")
    for model in trending:
        print(f"   - Model {model['model_id']}: score {model['score']:.1f}")
    print()


def example_discord_integration():
    """Demonstrate Discord integration."""
    print("=== Discord Integration Demo ===\n")

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("⚠️  Set DISCORD_WEBHOOK_URL environment variable to test Discord integration")
        return

    discord = DiscordWebhook(webhook_url)

    print("1. Announcing a new model...")
    success = discord.announce_new_model(
        model_name="SuperCNN-v2",
        author="Alice Johnson",
        description="State-of-the-art CNN for image classification",
        downloads=150,
        rating=4.8,
        tags=["computer-vision", "cnn", "image-classification"]
    )
    print(f"   {'✓' if success else '✗'} Model announcement sent\n")

    print("2. Announcing an achievement...")
    success = discord.announce_achievement(
        username="Bob Smith",
        achievement="Model Publisher",
        description="Published 5 models to the marketplace",
        points=50
    )
    print(f"   {'✓' if success else '✗'} Achievement announcement sent\n")

    print("3. Sharing weekly highlights...")
    success = discord.share_weekly_highlights(
        top_models=[
            {"name": "SuperCNN-v2", "author": "Alice", "downloads": 150},
            {"name": "TransformerXL", "author": "Bob", "downloads": 120},
        ],
        top_contributors=[
            {"username": "Alice Johnson", "points": 250},
            {"username": "Bob Smith", "points": 180},
        ],
        stats={
            "new_models": 12,
            "total_downloads": 1500,
            "active_users": 45
        }
    )
    print(f"   {'✓' if success else '✗'} Weekly highlights sent\n")


def example_discord_community_manager():
    """Demonstrate Discord community manager."""
    print("=== Discord Community Manager Demo ===\n")

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    manager = DiscordCommunityManager(webhook_url=webhook_url)

    print("1. Scheduling a workshop...")
    event_id = manager.schedule_event(
        title="Neural DSL Workshop: Building Your First Model",
        description="Learn how to create and deploy neural networks with Neural DSL",
        event_type="workshop",
        start_time="2024-02-15T18:00:00",
        duration_minutes=120,
        host="Alice Johnson",
        max_participants=50
    )
    print(f"   ✓ Event scheduled: {event_id}\n")

    print("2. Registering participants...")
    manager.register_participant(event_id, "user_bob")
    manager.register_participant(event_id, "user_charlie")
    print("   ✓ 2 participants registered\n")

    print("3. Getting upcoming events...")
    events = manager.get_upcoming_events(limit=5)
    print(f"   Upcoming Events: {len(events)}")
    for event in events:
        print(f"   - {event['title']} ({event['type']}) - {event['start_time']}")
    print()

    print("4. Creating a help request...")
    success = manager.create_help_request(
        username="user_bob",
        title="How to implement custom layers?",
        description="I'm trying to create a custom attention layer but getting errors.",
        category="question"
    )
    print(f"   {'✓' if success else '✗'} Help request created\n")


def example_educational_resources():
    """Demonstrate educational resources."""
    print("=== Educational Resources Demo ===\n")

    edu = EducationalResources()

    print("1. Adding course material...")
    course_id = edu.add_course_material(
        title="Deep Learning with Neural DSL",
        instructor="Prof. Alice Johnson",
        university="MIT",
        description="Comprehensive introduction to neural networks using Neural DSL",
        level="intermediate",
        topics=["neural-networks", "deep-learning", "cnn", "rnn"],
        resources={
            "syllabus": "https://example.com/syllabus.pdf",
            "slides": "https://example.com/slides",
            "code": "https://github.com/example/course-repo"
        }
    )
    print(f"   ✓ Course added: {course_id}\n")

    print("2. Adding assignment template...")
    assignment_id = edu.add_assignment_template(
        title="Build a CNN for Image Classification",
        description="Implement a convolutional neural network to classify MNIST digits",
        difficulty="medium",
        learning_objectives=[
            "Understand CNN architecture",
            "Implement convolution and pooling layers",
            "Train and evaluate the model"
        ],
        starter_code="""
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    // TODO: Add your layers here
}
        """,
        grading_rubric={
            "architecture": 30,
            "training": 30,
            "accuracy": 30,
            "documentation": 10
        }
    )
    print(f"   ✓ Assignment added: {assignment_id}\n")

    print("3. Adding tutorial...")
    tutorial_id = edu.add_tutorial(
        title="Getting Started with Neural DSL",
        author="Bob Smith",
        description="A beginner-friendly tutorial covering the basics of Neural DSL",
        difficulty="beginner",
        duration_minutes=30,
        content="# Getting Started\n\nWelcome to Neural DSL...",
        code_examples=[
            "network SimpleNet { input: (784,) layers: Dense(128, 'relu') Output(10, 'softmax') }",
        ],
        prerequisites=["basic-python", "ml-fundamentals"],
        tags=["tutorial", "beginner", "introduction"]
    )
    print(f"   ✓ Tutorial added: {tutorial_id}\n")

    print("4. Getting courses...")
    courses = edu.get_courses(level="intermediate", limit=5)
    print(f"   Found {len(courses)} intermediate courses")
    for course in courses:
        print(f"   - {course['title']} by {course['instructor']} ({course['university']})")
    print()

    print("5. Creating learning path...")
    path_id = edu.create_learning_path(
        title="Neural DSL Mastery Track",
        description="Complete path from beginner to advanced Neural DSL user",
        modules=[
            {"title": "Basics", "tutorials": ["tutorial_1", "tutorial_2"]},
            {"title": "Intermediate", "tutorials": ["tutorial_3", "tutorial_4"]},
            {"title": "Advanced", "tutorials": ["tutorial_5", "tutorial_6"]},
        ],
        estimated_hours=40,
        target_audience="Students and professionals new to neural networks"
    )
    print(f"   ✓ Learning path created: {path_id}\n")


def example_university_licenses():
    """Demonstrate university license management."""
    print("=== University License Management Demo ===\n")

    license_manager = UniversityLicenseManager()

    print("1. Issuing academic license...")
    license_key = license_manager.issue_academic_license(
        university="MIT",
        department="Computer Science",
        instructor="Prof. Alice Johnson",
        email="alice@mit.edu",
        student_count=50,
        duration_months=12
    )
    print(f"   ✓ License issued: {license_key}\n")

    print("2. Verifying license...")
    is_valid = license_manager.verify_license(license_key)
    print(f"   License valid: {'✓' if is_valid else '✗'}\n")

    print("3. Getting license info...")
    info = license_manager.get_license_info(license_key)
    if info:
        print(f"   University: {info['university']}")
        print(f"   Department: {info['department']}")
        print(f"   Instructor: {info['instructor']}")
        print(f"   Students: {info['student_count']}")
        print(f"   Status: {info['status']}")
        print(f"   Expires: {info['expires_at']}")
    print()


if __name__ == "__main__":
    example_community_features()
    print("\n" + "="*60 + "\n")

    example_discord_integration()
    print("\n" + "="*60 + "\n")

    example_discord_community_manager()
    print("\n" + "="*60 + "\n")

    example_educational_resources()
    print("\n" + "="*60 + "\n")

    example_university_licenses()
