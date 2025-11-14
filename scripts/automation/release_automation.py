"""
Automated Release Script

Handles version bumping, changelog updates, GitHub releases, and PyPI publishing.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class ReleaseAutomation:
    """Automate release process."""
    
    def __init__(self, version: Optional[str] = None):
        """Initialize release automation."""
        self.version = version
        self.repo_root = Path(__file__).parent.parent.parent
    
    def bump_version(self, version_type: str = "patch") -> str:
        """
        Bump version number.
        
        Args:
            version_type: 'major', 'minor', or 'patch'
            
        Returns:
            New version string
        """
        current_version = self._get_current_version()
        new_version = self._increment_version(current_version, version_type)
        
        # Update version in files
        self._update_version_in_file("setup.py", new_version)
        self._update_version_in_file("neural/__init__.py", new_version)
        
        print(f"✓ Bumped version: {current_version} -> {new_version}")
        return new_version
    
    def _get_current_version(self) -> str:
        """Get current version from setup.py."""
        try:
            with open(self.repo_root / "setup.py", "r") as f:
                content = f.read()
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except:
            pass
        return "0.3.0-dev"
    
    def _increment_version(self, version: str, version_type: str) -> str:
        """Increment version number."""
        # Remove dev suffix if present
        version = re.sub(r'\.dev\d+$', '', version)
        version = re.sub(r'-dev$', '', version)
        
        parts = version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _update_version_in_file(self, filepath: str, new_version: str):
        """Update version in a file."""
        file_path = self.repo_root / filepath
        if not file_path.exists():
            return
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update version string
        content = re.sub(
            r'(__version__|version)\s*=\s*["\']([^"\']+)["\']',
            f'\\1 = "{new_version}"',
            content
        )
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def create_github_release(self, version: str, release_notes: str, draft: bool = False):
        """Create GitHub release using gh CLI."""
        tag = f"v{version}"
        
        # Check if gh CLI is available
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except:
            print("⚠ GitHub CLI (gh) not found. Install from https://cli.github.com/")
            print(f"  Would create release: {tag}")
            return
        
        # Create release
        cmd = ["gh", "release", "create", tag, "--title", f"Neural DSL {tag}"]
        
        if release_notes:
            # Write release notes to temp file
            notes_file = self.repo_root / f"release_notes_{version}.md"
            with open(notes_file, "w", encoding="utf-8") as f:
                f.write(release_notes)
            cmd.extend(["--notes-file", str(notes_file)])
        
        if draft:
            cmd.append("--draft")
        
        try:
            subprocess.run(cmd, check=True, cwd=self.repo_root)
            print(f"✓ Created GitHub release: {tag}")
            
            # Clean up temp file
            if release_notes and notes_file.exists():
                notes_file.unlink()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create GitHub release: {e}")
    
    def build_and_publish_pypi(self, test: bool = False):
        """Build and publish to PyPI."""
        print("Building package...")
        
        # Build
        subprocess.run([sys.executable, "-m", "build"], check=True, cwd=self.repo_root)
        
        if test:
            # Upload to TestPyPI
            print("Uploading to TestPyPI...")
            subprocess.run([
                sys.executable, "-m", "twine", "upload",
                "--repository", "testpypi",
                "dist/*"
            ], check=True, cwd=self.repo_root)
            print("✓ Uploaded to TestPyPI")
        else:
            # Upload to PyPI
            print("Uploading to PyPI...")
            subprocess.run([
                sys.executable, "-m", "twine", "upload",
                "dist/*"
            ], check=True, cwd=self.repo_root)
            print("✓ Uploaded to PyPI")
    
    def run_tests(self) -> bool:
        """Run test suite."""
        print("Running tests...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ All tests passed")
                return True
            else:
                print("✗ Some tests failed")
                print(result.stdout)
                print(result.stderr)
                return False
        except Exception as e:
            print(f"✗ Error running tests: {e}")
            return False
    
    def generate_release_notes(self) -> str:
        """Generate release notes from changelog."""
        from .blog_generator import BlogGenerator
        
        generator = BlogGenerator(version=self.version)
        return generator.generate_github_release_notes()
    
    def full_release(self, version_type: str = "patch", skip_tests: bool = False, 
                     draft: bool = False, test_pypi: bool = False):
        """Run full release process."""
        print("=" * 70)
        print("Neural DSL Release Automation")
        print("=" * 70)
        
        # 1. Run tests
        if not skip_tests:
            if not self.run_tests():
                print("\n✗ Tests failed. Aborting release.")
                return False
        
        # 2. Bump version
        new_version = self.bump_version(version_type)
        self.version = new_version
        
        # 3. Generate release notes
        print("\nGenerating release notes...")
        release_notes = self.generate_release_notes()
        
        # 4. Create GitHub release
        print("\nCreating GitHub release...")
        self.create_github_release(new_version, release_notes, draft=draft)
        
        # 5. Build and publish to PyPI
        print("\nPublishing to PyPI...")
        try:
            self.build_and_publish_pypi(test=test_pypi)
        except Exception as e:
            print(f"⚠ PyPI publishing failed: {e}")
            print("  You can publish manually later with: twine upload dist/*")
        
        # 6. Generate blog posts
        print("\nGenerating blog posts...")
        from .blog_generator import BlogGenerator
        generator = BlogGenerator(version=new_version)
        generator.save_blog_posts()
        
        print("\n" + "=" * 70)
        print("✅ Release complete!")
        print(f"Version: {new_version}")
        print("=" * 70)
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automate Neural DSL releases")
    parser.add_argument("--version-type", choices=["major", "minor", "patch"], 
                       default="patch", help="Version bump type")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--draft", action="store_true", 
                       help="Create draft GitHub release")
    parser.add_argument("--test-pypi", action="store_true", 
                       help="Publish to TestPyPI instead of PyPI")
    
    args = parser.parse_args()
    
    automation = ReleaseAutomation()
    automation.full_release(
        version_type=args.version_type,
        skip_tests=args.skip_tests,
        draft=args.draft,
        test_pypi=args.test_pypi
    )

