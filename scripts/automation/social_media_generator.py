"""
Social Media Post Generator

Generates posts for Twitter, LinkedIn, etc. from release information.
"""

import os
from datetime import datetime
from typing import Dict


class SocialMediaGenerator:
    """Generate social media posts."""
    
    def __init__(self, version: str, release_notes: Dict):
        """Initialize generator."""
        self.version = version
        self.release_notes = release_notes
    
    def generate_twitter_post(self) -> str:
        """Generate Twitter/X post."""
        post = f"""🚀 Neural DSL v{self.version} is here!

"""
        
        if self.release_notes.get("sections", {}).get("Added"):
            features = self.release_notes["sections"]["Added"][:2]  # Top 2 features
            for feature in features:
                # Truncate if too long
                feature_short = feature[:100] + "..." if len(feature) > 100 else feature
                post += f"✨ {feature_short}\n"
        
        post += f"""
📦 pip install --upgrade neural-dsl

🔗 GitHub: https://github.com/Lemniscate-SHA-256/Neural

#NeuralNetworks #Python #MachineLearning #DeepLearning #OpenSource
"""
        
        # Twitter limit is 280 chars
        if len(post) > 280:
            post = post[:277] + "..."
        
        return post
    
    def generate_linkedin_post(self) -> str:
        """Generate LinkedIn post."""
        post = f"""🎉 Exciting News: Neural DSL v{self.version} Release!

We're thrilled to announce the latest release of Neural DSL, making neural network development easier than ever.

"""
        
        if self.release_notes.get("sections", {}).get("Added"):
            post += "✨ What's New:\n\n"
            for feature in self.release_notes["sections"]["Added"][:5]:
                post += f"• {feature}\n"
            post += "\n"
        
        post += f"""🚀 Get Started:

pip install --upgrade neural-dsl

📚 Learn more: https://github.com/Lemniscate-SHA-256/Neural

#NeuralNetworks #Python #MachineLearning #DeepLearning #AI #OpenSource #SoftwareDevelopment
"""
        
        return post
    
    def save_posts(self, output_dir: str = "docs/social"):
        """Save social media posts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Twitter
        twitter_post = self.generate_twitter_post()
        twitter_path = os.path.join(output_dir, f"twitter_v{self.version}.txt")
        with open(twitter_path, "w", encoding="utf-8") as f:
            f.write(twitter_post)
        print(f"✓ Generated Twitter post: {twitter_path}")
        
        # LinkedIn
        linkedin_post = self.generate_linkedin_post()
        linkedin_path = os.path.join(output_dir, f"linkedin_v{self.version}.txt")
        with open(linkedin_path, "w", encoding="utf-8") as f:
            f.write(linkedin_post)
        print(f"✓ Generated LinkedIn post: {linkedin_path}")
        
        return {
            "twitter": twitter_path,
            "linkedin": linkedin_path
        }


if __name__ == "__main__":
    from blog_generator import BlogGenerator
    
    version = "0.3.0"
    generator = BlogGenerator(version=version)
    
    social_gen = SocialMediaGenerator(version, generator.release_notes)
    social_gen.save_posts()

