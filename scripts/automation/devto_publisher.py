"""
Dev.to Article Publisher

Automates publishing of blog posts to Dev.to via API.
Handles frontmatter, article creation/update logic, and error handling.
"""

import os
import re
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DevToPublisher:
    """Publish articles to Dev.to via API."""
    
    API_BASE_URL = "https://dev.to/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Dev.to publisher.
        
        Args:
            api_key: Dev.to API key (defaults to DEVTO_API_KEY env var)
        """
        if requests is None:
            raise ImportError("requests library required. Install with: pip install requests")
        
        self.api_key = api_key or os.environ.get("DEVTO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Dev.to API key required. Set DEVTO_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def parse_frontmatter(self, content: str) -> Dict:
        """
        Parse frontmatter from markdown file.
        
        Args:
            content: Markdown content with frontmatter
            
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
    
    def create_article_payload(self, title: str, body: str, 
                              published: bool = False,
                              tags: Optional[list] = None,
                              description: Optional[str] = None,
                              canonical_url: Optional[str] = None,
                              series: Optional[str] = None,
                              main_image: Optional[str] = None,
                              organization_id: Optional[int] = None) -> Dict:
        """
        Create article payload for Dev.to API.
        
        Args:
            title: Article title
            body: Article body (markdown)
            published: Whether to publish immediately
            tags: List of tags (max 4)
            description: Article description
            canonical_url: Canonical URL
            series: Series name
            main_image: Main image URL
            organization_id: Organization ID
            
        Returns:
            API payload dict
        """
        payload = {
            "article": {
                "title": title,
                "body_markdown": body,
                "published": published
            }
        }
        
        if tags:
            # Dev.to allows max 4 tags
            payload["article"]["tags"] = tags[:4]
        
        if description:
            payload["article"]["description"] = description
        
        if canonical_url:
            payload["article"]["canonical_url"] = canonical_url
        
        if series:
            payload["article"]["series"] = series
        
        if main_image:
            payload["article"]["main_image"] = main_image
        
        if organization_id:
            payload["article"]["organization_id"] = organization_id
        
        return payload
    
    def create_article(self, payload: Dict) -> Dict:
        """
        Create a new article on Dev.to.
        
        Args:
            payload: Article payload
            
        Returns:
            API response dict
        """
        url = f"{self.API_BASE_URL}/articles"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ Article created successfully: {result.get('url')}")
            logger.info(f"  Article ID: {result.get('id')}")
            logger.info(f"  Published: {result.get('published')}")
            
            return result
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ HTTP error creating article: {e}")
            logger.error(f"  Response: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"✗ Error creating article: {e}")
            raise
    
    def update_article(self, article_id: int, payload: Dict) -> Dict:
        """
        Update an existing article on Dev.to.
        
        Args:
            article_id: Article ID to update
            payload: Article payload
            
        Returns:
            API response dict
        """
        url = f"{self.API_BASE_URL}/articles/{article_id}"
        
        try:
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ Article updated successfully: {result.get('url')}")
            logger.info(f"  Published: {result.get('published')}")
            
            return result
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ HTTP error updating article: {e}")
            logger.error(f"  Response: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"✗ Error updating article: {e}")
            raise
    
    def get_my_articles(self, page: int = 1, per_page: int = 30) -> list:
        """
        Get user's published articles.
        
        Args:
            page: Page number
            per_page: Articles per page
            
        Returns:
            List of articles
        """
        url = f"{self.API_BASE_URL}/articles/me"
        params = {"page": page, "per_page": per_page}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"✗ Error fetching articles: {e}")
            raise
    
    def find_article_by_title(self, title: str) -> Optional[Dict]:
        """
        Find article by title.
        
        Args:
            title: Article title to search for
            
        Returns:
            Article dict or None if not found
        """
        try:
            articles = self.get_my_articles()
            
            for article in articles:
                if article.get("title") == title:
                    return article
            
            return None
        
        except Exception as e:
            logger.error(f"✗ Error finding article: {e}")
            return None
    
    def publish_from_file(self, file_path: str, 
                         update_if_exists: bool = True,
                         force_publish: bool = False) -> Dict:
        """
        Publish article from markdown file.
        
        Args:
            file_path: Path to markdown file
            update_if_exists: Update if article with same title exists
            force_publish: Publish immediately (override frontmatter)
            
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
        
        # Extract metadata
        title = frontmatter.get("title", "Untitled")
        published = force_publish or frontmatter.get("published", False)
        tags = frontmatter.get("tags", [])
        
        # Handle tags as string or list
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',')]
        
        description = frontmatter.get("description")
        canonical_url = frontmatter.get("canonical_url")
        series = frontmatter.get("series")
        main_image = frontmatter.get("main_image") or frontmatter.get("cover_image")
        
        # Create payload
        payload = self.create_article_payload(
            title=title,
            body=body,
            published=published,
            tags=tags,
            description=description,
            canonical_url=canonical_url,
            series=series,
            main_image=main_image
        )
        
        # Check if article exists
        if update_if_exists:
            existing = self.find_article_by_title(title)
            
            if existing:
                article_id = existing["id"]
                logger.info(f"Article exists (ID: {article_id}), updating...")
                return self.update_article(article_id, payload)
        
        # Create new article
        return self.create_article(payload)
    
    def publish_from_directory(self, directory: str, 
                               pattern: str = "*.md",
                               update_if_exists: bool = True) -> Dict[str, Dict]:
        """
        Publish all markdown files in directory.
        
        Args:
            directory: Directory containing markdown files
            pattern: File pattern to match
            update_if_exists: Update existing articles
            
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
                    update_if_exists=update_if_exists
                )
                results[file_path.name] = result
                
            except Exception as e:
                logger.error(f"✗ Failed to publish {file_path.name}: {e}")
                results[file_path.name] = {"error": str(e)}
        
        return results


def main():
    """CLI for Dev.to publisher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Publish articles to Dev.to",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish single article
  python devto_publisher.py --file article.md
  
  # Publish and immediately make public
  python devto_publisher.py --file article.md --publish
  
  # Publish all articles in directory
  python devto_publisher.py --directory docs/blog
  
  # Update existing articles
  python devto_publisher.py --directory docs/blog --update

Environment Variables:
  DEVTO_API_KEY: Dev.to API key (required)
        """
    )
    
    parser.add_argument("--file", type=str,
                       help="Markdown file to publish")
    parser.add_argument("--directory", type=str,
                       help="Directory containing markdown files")
    parser.add_argument("--pattern", type=str, default="devto_*.md",
                       help="File pattern for directory mode (default: devto_*.md)")
    parser.add_argument("--publish", action="store_true",
                       help="Publish immediately (override frontmatter)")
    parser.add_argument("--update", action="store_true", default=True,
                       help="Update existing articles (default: True)")
    parser.add_argument("--no-update", action="store_true",
                       help="Don't update existing articles")
    parser.add_argument("--api-key", type=str,
                       help="Dev.to API key (or use DEVTO_API_KEY env var)")
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.print_help()
        return
    
    update_if_exists = not args.no_update and args.update
    
    try:
        publisher = DevToPublisher(api_key=args.api_key)
        
        if args.file:
            result = publisher.publish_from_file(
                args.file,
                update_if_exists=update_if_exists,
                force_publish=args.publish
            )
            
            print("\n" + "=" * 70)
            print("✅ Success!")
            print("=" * 70)
            print(f"URL: {result.get('url')}")
            print(f"ID: {result.get('id')}")
            print(f"Published: {result.get('published')}")
        
        elif args.directory:
            results = publisher.publish_from_directory(
                args.directory,
                pattern=args.pattern,
                update_if_exists=update_if_exists
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
