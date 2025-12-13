#!/usr/bin/env python3
"""
Example Usage of Blog Publishers

Demonstrates how to use the Dev.to and Medium publishers programmatically.
"""

import os
from pathlib import Path


def example_devto_usage():
    """Example: Publishing to Dev.to"""
    print("=" * 70)
    print("Dev.to Publisher Example")
    print("=" * 70)
    
    try:
        from devto_publisher import DevToPublisher
        
        # Initialize publisher (uses DEVTO_API_KEY from environment)
        publisher = DevToPublisher()
        
        # Example 1: Publish from file
        result = publisher.publish_from_file(
            "docs/blog/devto_v0.3.0_release.md",
            update_if_exists=True,  # Update if article exists
            force_publish=False      # Keep as draft
        )
        
        print(f"✓ Published to: {result['url']}")
        print(f"  Article ID: {result['id']}")
        print(f"  Published: {result['published']}")
        
        # Example 2: Publish multiple articles from directory
        results = publisher.publish_from_directory(
            "docs/blog",
            pattern="devto_*.md",
            update_if_exists=True
        )
        
        print(f"\n✓ Published {len(results)} articles")
        
        # Example 3: Find existing article
        article = publisher.find_article_by_title("Neural DSL v0.3.0 Release - What's New")
        if article:
            print(f"\n✓ Found existing article: {article['url']}")
        
    except ImportError:
        print("✗ requests library required: pip install requests")
    except ValueError as e:
        print(f"✗ Setup error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_medium_usage():
    """Example: Publishing to Medium"""
    print("\n" + "=" * 70)
    print("Medium Publisher Example")
    print("=" * 70)
    
    try:
        from medium_publisher import MediumPublisher
        
        # Initialize publisher (uses MEDIUM_API_TOKEN from environment)
        publisher = MediumPublisher()
        
        # Get user info
        user_info = publisher.get_user_info()
        print(f"✓ Authenticated as: {user_info['username']}")
        
        # Example 1: Publish as draft
        result = publisher.publish_from_file(
            "docs/blog/medium_v0.3.0_release.md",
            publish_status="draft"
        )
        
        print(f"\n✓ Published to: {result['url']}")
        print(f"  Article ID: {result['id']}")
        print(f"  Status: {result['publishStatus']}")
        print(f"  Tags: {', '.join(result.get('tags', []))}")
        
        # Example 2: List publications
        publications = publisher.get_user_publications()
        if publications:
            print(f"\n✓ Found {len(publications)} publications:")
            for pub in publications:
                print(f"  - {pub['name']}: {pub['id']}")
        
        # Example 3: Publish to a publication
        if publications:
            pub_id = publications[0]['id']
            result = publisher.publish_from_file(
                "docs/blog/medium_v0.3.0_release.md",
                publication_id=pub_id,
                publish_status="draft"
            )
            print(f"\n✓ Published to publication: {result['url']}")
        
    except ImportError:
        print("✗ requests library required: pip install requests")
    except ValueError as e:
        print(f"✗ Setup error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_create_article():
    """Example: Creating article with custom payload"""
    print("\n" + "=" * 70)
    print("Custom Article Creation Example")
    print("=" * 70)
    
    try:
        from devto_publisher import DevToPublisher
        
        publisher = DevToPublisher()
        
        # Create custom payload
        payload = publisher.create_article_payload(
            title="My Custom Article",
            body="""
# Introduction

This is a custom article created programmatically.

## Features

- Point 1
- Point 2
- Point 3

## Conclusion

Thank you for reading!
            """,
            published=False,
            tags=["python", "automation", "tutorial"],
            description="A custom article about automation"
        )
        
        # Create article
        result = publisher.create_article(payload)
        print(f"✓ Created article: {result['url']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_marketing_automation():
    """Example: Full marketing automation workflow"""
    print("\n" + "=" * 70)
    print("Marketing Automation Workflow")
    print("=" * 70)
    
    try:
        # Step 1: Generate blog posts
        from blog_generator import BlogGenerator
        
        print("\n[1/4] Generating blog posts...")
        generator = BlogGenerator(version="0.3.0")
        blog_paths = generator.save_blog_posts()
        
        for platform, path in blog_paths.items():
            print(f"  ✓ {platform}: {path}")
        
        # Step 2: Generate social media posts
        from social_media_generator import SocialMediaGenerator
        
        print("\n[2/4] Generating social media posts...")
        social_gen = SocialMediaGenerator(generator.version, generator.release_notes)
        social_paths = social_gen.save_posts()
        
        for platform, path in social_paths.items():
            print(f"  ✓ {platform}: {path}")
        
        # Step 3: Publish to Dev.to (optional)
        if os.getenv("DEVTO_API_KEY"):
            print("\n[3/4] Publishing to Dev.to...")
            from devto_publisher import DevToPublisher
            
            devto_publisher = DevToPublisher()
            devto_file = blog_paths.get("devto")
            
            if devto_file and Path(devto_file).exists():
                result = devto_publisher.publish_from_file(devto_file)
                print(f"  ✓ Published: {result['url']}")
            else:
                print("  ✗ Dev.to file not found")
        else:
            print("\n[3/4] Skipping Dev.to (no API key)")
        
        # Step 4: Publish to Medium (optional)
        if os.getenv("MEDIUM_API_TOKEN"):
            print("\n[4/4] Publishing to Medium...")
            from medium_publisher import MediumPublisher
            
            medium_publisher = MediumPublisher()
            medium_publisher.get_user_info()
            medium_file = blog_paths.get("medium")
            
            if medium_file and Path(medium_file).exists():
                result = medium_publisher.publish_from_file(medium_file)
                print(f"  ✓ Published: {result['url']}")
            else:
                print("  ✗ Medium file not found")
        else:
            print("\n[4/4] Skipping Medium (no API token)")
        
        print("\n✅ Marketing automation complete!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all examples"""
    print("""
Neural DSL Blog Publishers - Example Usage
==========================================

This script demonstrates how to use the blog publishers.

Note: Set environment variables before running:
  export DEVTO_API_KEY="your_key"
  export MEDIUM_API_TOKEN="your_token"

Or create a .env file from .env.example
    """)
    
    # Check if API keys are set
    has_devto = bool(os.getenv("DEVTO_API_KEY"))
    has_medium = bool(os.getenv("MEDIUM_API_TOKEN"))
    
    print(f"\nAPI Keys configured:")
    print(f"  Dev.to: {'✓' if has_devto else '✗'}")
    print(f"  Medium: {'✓' if has_medium else '✗'}")
    print()
    
    if not (has_devto or has_medium):
        print("⚠️  No API keys found. Set environment variables to test publishing.")
        print("   The examples below will show errors.")
        print()
    
    # Run examples (comment out ones you don't want to test)
    # example_devto_usage()
    # example_medium_usage()
    # example_create_article()
    example_marketing_automation()


if __name__ == "__main__":
    main()
