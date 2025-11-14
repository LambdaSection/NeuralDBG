"""
Automated Blog Post Generator

Generates blog posts from CHANGELOG.md and version information.
Supports multiple platforms: Medium, Dev.to, Hashnode, etc.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BlogGenerator:
    """Generate blog posts from changelog and version info."""
    
    def __init__(self, changelog_path: str = "CHANGELOG.md", version: Optional[str] = None):
        """
        Initialize blog generator.
        
        Args:
            changelog_path: Path to CHANGELOG.md
            version: Version number (auto-detected if None)
        """
        self.changelog_path = changelog_path
        self.version = version or self._detect_version()
        self.changelog_content = self._read_changelog()
        self.release_notes = self._extract_release_notes()
    
    def _detect_version(self) -> str:
        """Detect current version from setup.py or __init__.py."""
        # Try setup.py first
        try:
            with open("setup.py", "r") as f:
                content = f.read()
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except:
            pass
        
        # Try __init__.py
        try:
            with open("neural/__init__.py", "r") as f:
                content = f.read()
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
        except:
            pass
        
        return "0.3.0-dev"
    
    def _read_changelog(self) -> str:
        """Read changelog file."""
        try:
            with open(self.changelog_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading changelog: {e}")
            return ""
    
    def _extract_release_notes(self) -> Dict[str, any]:
        """Extract release notes for current version."""
        if not self.changelog_content:
            return {}
        
        # Find the section for current version
        version_pattern = rf"##\s*\[{re.escape(self.version)}\].*?(?=##|$)"
        match = re.search(version_pattern, self.changelog_content, re.DOTALL)
        
        if not match:
            # Try without brackets
            version_pattern = rf"##\s*{re.escape(self.version)}.*?(?=##|$)"
            match = re.search(version_pattern, self.changelog_content, re.DOTALL)
        
        if not match:
            return {"content": "", "date": datetime.now().strftime("%Y-%m-%d")}
        
        content = match.group(0)
        
        # Extract date
        date_match = re.search(r"-?\s*(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})", content)
        date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
        
        # Extract sections
        sections = {
            "Added": self._extract_section(content, "### Added"),
            "Fixed": self._extract_section(content, "### Fixed"),
            "Changed": self._extract_section(content, "### Changed"),
            "Improved": self._extract_section(content, "### Improved"),
            "Removed": self._extract_section(content, "### Removed"),
            "Deprecated": self._extract_section(content, "### Deprecated"),
            "Security": self._extract_section(content, "### Security"),
        }
        
        return {
            "content": content,
            "date": date,
            "sections": sections,
            "full_text": content
        }
    
    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """Extract items from a changelog section."""
        pattern = rf"{re.escape(section_name)}\s*\n(.*?)(?=\n###|\n---|$)"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return []
        
        section_text = match.group(1)
        # Extract list items
        items = re.findall(r'^-\s*(.+)$', section_text, re.MULTILINE)
        return [item.strip() for item in items if item.strip()]
    
    def generate_medium_post(self) -> str:
        """Generate blog post for Medium."""
        template = f"""# Neural DSL v{self.version} Release: What's New

*Published on {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}*

We're excited to announce the release of **Neural DSL v{self.version}**! This release brings significant improvements, new features, and bug fixes to make neural network development easier than ever.

## 🎉 What's New

"""
        
        if self.release_notes.get("sections", {}).get("Added"):
            template += "### ✨ New Features\n\n"
            for item in self.release_notes["sections"]["Added"]:
                template += f"- {item}\n\n"
        
        if self.release_notes.get("sections", {}).get("Improved"):
            template += "\n### 🚀 Improvements\n\n"
            for item in self.release_notes["sections"]["Improved"]:
                template += f"- {item}\n\n"
        
        if self.release_notes.get("sections", {}).get("Fixed"):
            template += "\n### 🐛 Bug Fixes\n\n"
            for item in self.release_notes["sections"]["Fixed"]:
                template += f"- {item}\n\n"
        
        template += f"""
## 📦 Installation

Get the latest version:

```bash
pip install --upgrade neural-dsl
```

## 🔗 Resources

- **GitHub**: https://github.com/Lemniscate-SHA-256/Neural
- **Documentation**: https://github.com/Lemniscate-SHA-256/Neural#readme
- **PyPI**: https://pypi.org/project/neural-dsl/

## 🙏 Thank You

Thank you to all contributors and users who helped make this release possible!

---

*For the full changelog, visit our [GitHub repository](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md).*
"""
        
        return template
    
    def generate_devto_post(self) -> str:
        """Generate blog post for Dev.to."""
        template = f"""---
title: Neural DSL v{self.version} Release - What's New
published: true
description: Announcing Neural DSL v{self.version} with new features, improvements, and bug fixes
tags: neuralnetworks, python, machinelearning, deeplearning
---

# Neural DSL v{self.version} Release: What's New 🚀

We're thrilled to announce **Neural DSL v{self.version}**! This release includes exciting new features, improvements, and bug fixes.

"""
        
        if self.release_notes.get("sections", {}).get("Added"):
            template += "## ✨ New Features\n\n"
            for item in self.release_notes["sections"]["Added"]:
                template += f"- {item}\n\n"
        
        if self.release_notes.get("sections", {}).get("Improved"):
            template += "\n## 🚀 Improvements\n\n"
            for item in self.release_notes["sections"]["Improved"]:
                template += f"- {item}\n\n"
        
        if self.release_notes.get("sections", {}).get("Fixed"):
            template += "\n## 🐛 Bug Fixes\n\n"
            for item in self.release_notes["sections"]["Fixed"]:
                template += f"- {item}\n\n"
        
        template += f"""
## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 🔗 Links

- GitHub: https://github.com/Lemniscate-SHA-256/Neural
- Documentation: https://github.com/Lemniscate-SHA-256/Neural#readme
- PyPI: https://pypi.org/project/neural-dsl/

---

*Full changelog: [GitHub](https://github.com/Lemniscate-SHA-256/Neural/blob/main/CHANGELOG.md)*
"""
        
        return template
    
    def generate_github_release_notes(self) -> str:
        """Generate GitHub release notes."""
        template = f"""# Neural DSL v{self.version}

Release date: {self.release_notes.get('date', datetime.now().strftime('%Y-%m-%d'))}

"""
        
        if self.release_notes.get("sections", {}).get("Added"):
            template += "## ✨ What's New\n\n"
            for item in self.release_notes["sections"]["Added"]:
                template += f"- {item}\n"
            template += "\n"
        
        if self.release_notes.get("sections", {}).get("Improved"):
            template += "## 🚀 Improvements\n\n"
            for item in self.release_notes["sections"]["Improved"]:
                template += f"- {item}\n"
            template += "\n"
        
        if self.release_notes.get("sections", {}).get("Fixed"):
            template += "## 🐛 Bug Fixes\n\n"
            for item in self.release_notes["sections"]["Fixed"]:
                template += f"- {item}\n"
            template += "\n"
        
        template += f"""## 📦 Installation

```bash
pip install --upgrade neural-dsl
```

## 📚 Documentation

- [Full Changelog](CHANGELOG.md)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural#readme)
- [Examples](examples/)

## 🙏 Contributors

Thank you to everyone who contributed to this release!
"""
        
        return template
    
    def save_blog_posts(self, output_dir: str = "docs/blog"):
        """Save generated blog posts to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Medium
        medium_post = self.generate_medium_post()
        medium_path = os.path.join(output_dir, f"medium_v{self.version}_release.md")
        with open(medium_path, "w", encoding="utf-8") as f:
            f.write(medium_post)
        print(f"✓ Generated Medium post: {medium_path}")
        
        # Dev.to
        devto_post = self.generate_devto_post()
        devto_path = os.path.join(output_dir, f"devto_v{self.version}_release.md")
        with open(devto_path, "w", encoding="utf-8") as f:
            f.write(devto_post)
        print(f"✓ Generated Dev.to post: {devto_path}")
        
        # GitHub Release Notes
        github_notes = self.generate_github_release_notes()
        github_path = os.path.join(output_dir, f"github_v{self.version}_release.md")
        with open(github_path, "w", encoding="utf-8") as f:
            f.write(github_notes)
        print(f"✓ Generated GitHub release notes: {github_path}")
        
        return {
            "medium": medium_path,
            "devto": devto_path,
            "github": github_path
        }


if __name__ == "__main__":
    import sys
    
    version = sys.argv[1] if len(sys.argv) > 1 else None
    generator = BlogGenerator(version=version)
    
    print(f"Generating blog posts for version {generator.version}...")
    paths = generator.save_blog_posts()
    
    print("\n✅ Blog posts generated successfully!")
    print(f"\nFiles created:")
    for platform, path in paths.items():
        print(f"  - {platform}: {path}")

