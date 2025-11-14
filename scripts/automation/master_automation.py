#!/usr/bin/env python3
"""
Master Automation Script

Orchestrates all automation tasks for Neural DSL.
Run this script to handle releases, blog posts, tests, and more.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.automation.blog_generator import BlogGenerator
from scripts.automation.release_automation import ReleaseAutomation
from scripts.automation.example_validator import ExampleValidator
from scripts.automation.test_automation import TestAutomation
from scripts.automation.social_media_generator import SocialMediaGenerator


def main():
    """Main automation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Neural DSL Master Automation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate blog posts for current version
  python master_automation.py --blog
  
  # Run tests and validate examples
  python master_automation.py --test --validate
  
  # Full release (bump version, test, release, blog, social)
  python master_automation.py --release --version-type patch
  
  # Daily maintenance tasks
  python master_automation.py --daily
        """
    )
    
    # Action flags
    parser.add_argument("--blog", action="store_true",
                       help="Generate blog posts")
    parser.add_argument("--social", action="store_true",
                       help="Generate social media posts")
    parser.add_argument("--test", action="store_true",
                       help="Run tests")
    parser.add_argument("--validate", action="store_true",
                       help="Validate examples")
    parser.add_argument("--release", action="store_true",
                       help="Run full release process")
    parser.add_argument("--daily", action="store_true",
                       help="Run daily maintenance tasks")
    
    # Release options
    parser.add_argument("--version-type", choices=["major", "minor", "patch"],
                       default="patch", help="Version bump type (for releases)")
    parser.add_argument("--version", type=str,
                       help="Specific version (for blog/social generation)")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip tests during release")
    parser.add_argument("--draft", action="store_true",
                       help="Create draft release")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Publish to TestPyPI")
    
    # Test options
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    
    args = parser.parse_args()
    
    # If no specific action, show help
    if not any([args.blog, args.social, args.test, args.validate, 
                args.release, args.daily]):
        parser.print_help()
        return
    
    version = args.version
    
    print("=" * 70)
    print("Neural DSL Automation")
    print("=" * 70)
    print()
    
    # Daily tasks
    if args.daily:
        print("Running daily maintenance tasks...")
        print()
        
        # Run tests
        print("1. Running tests...")
        test_auto = TestAutomation()
        test_auto.run_and_report(coverage=args.coverage)
        print()
        
        # Validate examples
        print("2. Validating examples...")
        validator = ExampleValidator()
        validator.generate_report()
        print()
        
        print("✅ Daily tasks complete!")
        return
    
    # Full release
    if args.release:
        print("Starting release process...")
        print()
        
        automation = ReleaseAutomation(version=version)
        automation.full_release(
            version_type=args.version_type,
            skip_tests=args.skip_tests,
            draft=args.draft,
            test_pypi=args.test_pypi
        )
        return
    
    # Individual tasks
    if args.test:
        print("Running tests...")
        test_auto = TestAutomation()
        test_auto.run_and_report(coverage=args.coverage)
        print()
    
    if args.validate:
        print("Validating examples...")
        validator = ExampleValidator()
        validator.generate_report()
        print()
    
    if args.blog:
        print("Generating blog posts...")
        generator = BlogGenerator(version=version)
        generator.save_blog_posts()
        print()
    
    if args.social:
        print("Generating social media posts...")
        generator = BlogGenerator(version=version)
        social_gen = SocialMediaGenerator(version or generator.version, generator.release_notes)
        social_gen.save_posts()
        print()
    
    print("✅ Automation complete!")


if __name__ == "__main__":
    main()

