"""
Medium Article Publisher

Automates publishing of blog posts to Medium via API.
Handles article creation, tags, canonical URLs, and error handling.
"""

import os
import re
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediumPublisher:
    """Publish articles to Medium via API."""
    
    API_BASE_URL = "https://api.medium.com/v1"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Medium publisher.
        
        Args:
            api_token: Medium API token (defaults to MEDIUM_API_TOKEN env var)
        """
        if requests is None:
            raise ImportError("requests library required. Install with: pip install requests")
        
        self.api_token = api_token or os.environ.get("MEDIUM_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Medium API token required. Set MEDIUM_API_TOKEN environment variable "
                "or pass api_token parameter"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self.user_id = None
    
    def get_user_info(self) -> Dict:
        """
        Get authenticated user information.
        
        Returns:
            User info dict
        """
        url = f"{self.API_BASE_URL}/me"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            user_data = result.get("data", {})
            self.user_id = user_data.get("id")
            
            logger.info(f"✓ Authenticated as: {user_data.get('username')}")
            logger.info(f"  Name: {user_data.get('name')}")
            logger.info(f"  User ID: {self.user_id}")
            
            return user_data
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ HTTP error getting user info: {e}")
            logger.error(f"  Response: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"✗ Error getting user info: {e}")
            raise
    
    def parse_frontmatter(self, content: str) -> Dict:
        """
        Parse frontmatter from markdown file.
        
        Args:
            content: Markdown content with optional frontmatter
            
        Returns:
            Dict with 'frontmatter' and 'body' keys
        """
        # Match YAML frontmatter
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content.strip(), re.DOTALL)
        
        if not match:
            return {"frontmatter": {}, "body": content}
        
        frontmatter_text = match.group(1)
        body = match.group(2).strip()
        
        # Parse YAML frontmatter (simple parser)
        frontmatter = {}
        for line in frontmatter_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle booleans
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Handle lists (tags)
                elif ',' in value:
                    value = [tag.strip() for tag in value.split(',')]
                
                frontmatter[key] = value
        
        return {"frontmatter": frontmatter, "body": body}
    
    def extract_title_from_markdown(self, markdown: str) -> tuple:
        """
        Extract title from markdown content.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Tuple of (title, remaining_content)
        """
        # Look for first H1 heading
        lines = markdown.split('\n')
        title = None
        start_index = 0
        
        for i, line in enumerate(lines):
            if line.startswith('# '):
                title = line[2:].strip()
                start_index = i + 1
                break
        
        if title:
            remaining = '\n'.join(lines[start_index:]).strip()
            return title, remaining
        
        return "Untitled", markdown
    
    def create_article_payload(self, title: str, content: str,
                              content_format: str = "markdown",
                              publish_status: str = "draft",
                              tags: Optional[List[str]] = None,
                              canonical_url: Optional[str] = None,
                              notify_followers: bool = False,
                              license: str = "all-rights-reserved") -> Dict:
        """
        Create article payload for Medium API.
        
        Args:
            title: Article title
            content: Article content (markdown or HTML)
            content_format: 'markdown' or 'html'
            publish_status: 'draft', 'public', or 'unlisted'
            tags: List of tags (max 5)
            canonical_url: Original URL if cross-posting
            notify_followers: Whether to notify followers
            license: License type
            
        Returns:
            API payload dict
        """
        payload = {
            "title": title,
            "contentFormat": content_format,
            "content": content,
            "publishStatus": publish_status,
            "license": license,
            "notifyFollowers": notify_followers
        }
        
        if tags:
            # Medium allows max 5 tags
            payload["tags"] = tags[:5]
        
        if canonical_url:
            payload["canonicalUrl"] = canonical_url
        
        return payload
    
    def create_article(self, payload: Dict, user_id: Optional[str] = None) -> Dict:
        """
        Create a new article on Medium.
        
        Args:
            payload: Article payload
            user_id: User ID (uses authenticated user if None)
            
        Returns:
            API response dict
        """
        # Get user ID if not provided
        if not user_id:
            if not self.user_id:
                self.get_user_info()
            user_id = self.user_id
        
        url = f"{self.API_BASE_URL}/users/{user_id}/posts"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            post_data = result.get("data", {})
            
            logger.info(f"✓ Article created successfully: {post_data.get('url')}")
            logger.info(f"  Article ID: {post_data.get('id')}")
            logger.info(f"  Status: {post_data.get('publishStatus')}")
            logger.info(f"  Tags: {', '.join(post_data.get('tags', []))}")
            
            return post_data
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ HTTP error creating article: {e}")
            try:
                error_data = e.response.json()
                logger.error(f"  Error: {error_data}")
            except:
                logger.error(f"  Response: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"✗ Error creating article: {e}")
            raise
    
    def get_user_publications(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Get user's publications.
        
        Args:
            user_id: User ID (uses authenticated user if None)
            
        Returns:
            List of publications
        """
        if not user_id:
            if not self.user_id:
                self.get_user_info()
            user_id = self.user_id
        
        url = f"{self.API_BASE_URL}/users/{user_id}/publications"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            return result.get("data", [])
        
        except Exception as e:
            logger.error(f"✗ Error fetching publications: {e}")
            return []
    
    def create_publication_post(self, publication_id: str, payload: Dict) -> Dict:
        """
        Create a post under a publication.
        
        Args:
            publication_id: Publication ID
            payload: Article payload
            
        Returns:
            API response dict
        """
        url = f"{self.API_BASE_URL}/publications/{publication_id}/posts"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            post_data = result.get("data", {})
            
            logger.info(f"✓ Publication post created: {post_data.get('url')}")
            return post_data
        
        except Exception as e:
            logger.error(f"✗ Error creating publication post: {e}")
            raise
    
    def publish_from_file(self, file_path: str,
                         publish_status: str = "draft",
                         publication_id: Optional[str] = None,
                         force_publish: bool = False,
                         notify_followers: bool = False) -> Dict:
        """
        Publish article from markdown file.
        
        Args:
            file_path: Path to markdown file
            publish_status: 'draft', 'public', or 'unlisted'
            publication_id: Publish to specific publication
            force_publish: Override frontmatter publish status
            notify_followers: Notify followers on publish
            
        Returns:
            API response dict
        """
        logger.info(f"Publishing article from: {file_path}")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"✗ Error reading file: {e}")
            raise
        
        # Parse frontmatter
        parsed = self.parse_frontmatter(content)
        frontmatter = parsed["frontmatter"]
        body = parsed["body"]
        
        # Extract title from body if not in frontmatter
        title = frontmatter.get("title")
        if not title:
            title, body = self.extract_title_from_markdown(body)
        
        # Extract metadata
        if not force_publish:
            status = frontmatter.get("status") or frontmatter.get("publish_status", publish_status)
        else:
            status = publish_status
        
        tags = frontmatter.get("tags", [])
        
        # Handle tags as string or list
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',')]
        
        canonical_url = frontmatter.get("canonical_url")
        license_type = frontmatter.get("license", "all-rights-reserved")
        
        # Create payload
        payload = self.create_article_payload(
            title=title,
            content=body,
            content_format="markdown",
            publish_status=status,
            tags=tags,
            canonical_url=canonical_url,
            notify_followers=notify_followers,
            license=license_type
        )
        
        # Create article
        if publication_id:
            return self.create_publication_post(publication_id, payload)
        else:
            return self.create_article(payload)
    
    def publish_from_directory(self, directory: str,
                               pattern: str = "*.md",
                               publish_status: str = "draft",
                               publication_id: Optional[str] = None) -> Dict[str, Dict]:
        """
        Publish all markdown files in directory.
        
        Args:
            directory: Directory containing markdown files
            pattern: File pattern to match
            publish_status: Default publish status
            publication_id: Publish to specific publication
            
        Returns:
            Dict mapping filenames to API responses
        """
        results = {}
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"✗ Directory not found: {directory}")
            return results
        
        files = list(directory_path.glob(pattern))
        
        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return results
        
        logger.info(f"Found {len(files)} file(s) to publish")
        
        for file_path in files:
            try:
                result = self.publish_from_file(
                    str(file_path),
                    publish_status=publish_status,
                    publication_id=publication_id
                )
                results[file_path.name] = result
                
            except Exception as e:
                logger.error(f"✗ Failed to publish {file_path.name}: {e}")
                results[file_path.name] = {"error": str(e)}
        
        return results


def main():
    """CLI for Medium publisher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Publish articles to Medium",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish single article as draft
  python medium_publisher.py --file article.md
  
  # Publish as public post
  python medium_publisher.py --file article.md --status public
  
  # Publish all articles in directory
  python medium_publisher.py --directory docs/blog
  
  # Publish to a publication
  python medium_publisher.py --file article.md --publication-id abc123
  
  # List publications
  python medium_publisher.py --list-publications

Environment Variables:
  MEDIUM_API_TOKEN: Medium API token (required)
        """
    )
    
    parser.add_argument("--file", type=str,
                       help="Markdown file to publish")
    parser.add_argument("--directory", type=str,
                       help="Directory containing markdown files")
    parser.add_argument("--pattern", type=str, default="medium_*.md",
                       help="File pattern for directory mode (default: medium_*.md)")
    parser.add_argument("--status", type=str, 
                       choices=["draft", "public", "unlisted"],
                       default="draft",
                       help="Publish status (default: draft)")
    parser.add_argument("--publication-id", type=str,
                       help="Publish to specific publication")
    parser.add_argument("--notify", action="store_true",
                       help="Notify followers on publish")
    parser.add_argument("--list-publications", action="store_true",
                       help="List available publications")
    parser.add_argument("--api-token", type=str,
                       help="Medium API token (or use MEDIUM_API_TOKEN env var)")
    
    args = parser.parse_args()
    
    if not any([args.file, args.directory, args.list_publications]):
        parser.print_help()
        return
    
    try:
        publisher = MediumPublisher(api_token=args.api_token)
        
        # Get user info first
        user_info = publisher.get_user_info()
        
        if args.list_publications:
            publications = publisher.get_user_publications()
            
            print("\n" + "=" * 70)
            print("Publications")
            print("=" * 70)
            
            if publications:
                for pub in publications:
                    print(f"\nName: {pub.get('name')}")
                    print(f"ID: {pub.get('id')}")
                    print(f"URL: {pub.get('url')}")
                    print(f"Image: {pub.get('imageUrl')}")
            else:
                print("No publications found")
            
            return
        
        if args.file:
            result = publisher.publish_from_file(
                args.file,
                publish_status=args.status,
                publication_id=args.publication_id,
                notify_followers=args.notify
            )
            
            print("\n" + "=" * 70)
            print("✅ Success!")
            print("=" * 70)
            print(f"URL: {result.get('url')}")
            print(f"ID: {result.get('id')}")
            print(f"Status: {result.get('publishStatus')}")
            print(f"Tags: {', '.join(result.get('tags', []))}")
        
        elif args.directory:
            results = publisher.publish_from_directory(
                args.directory,
                pattern=args.pattern,
                publish_status=args.status,
                publication_id=args.publication_id
            )
            
            print("\n" + "=" * 70)
            print(f"✅ Published {len(results)} article(s)")
            print("=" * 70)
            
            for filename, result in results.items():
                if "error" in result:
                    print(f"✗ {filename}: {result['error']}")
                else:
                    print(f"✓ {filename}: {result.get('url')}")
    
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        raise


if __name__ == "__main__":
    main()
