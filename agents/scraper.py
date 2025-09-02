import praw
from typing import List, Dict
from dotenv import load_dotenv
import warnings
import os
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

client_id = os.getenv("reddit_id")
client_secret = os.getenv("reddit_secret")

# Reddit Post Scraper
class RedditScraperAgent:
    """Enhanced Data Harvester with realistic social media samples"""
        
    async def scrape_posts(self, subreddit: str, post_limit: int) -> List[Dict]:
        """
        Scrape posts from a subreddit using Reddit API (PRAW).
        
        Args:
            subreddit_name (str): Name of the subreddit (e.g. 'python')
            limit (int): Number of posts to fetch
        
        Returns:
            List[Dict]: List of posts with title, text, score, url, and author
        """

        # Reddit API credentials (get from https://www.reddit.com/prefs/apps)
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="RedditScraperAgent/1.0"
        )

        posts_data = []
        subreddit = reddit.subreddit(subreddit)

        for post in subreddit.hot(limit=post_limit):  # hot/new/top available
            posts_data.append({
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "url": post.url,
                "author": str(post.author),
                "created_utc": post.created_utc
            })
        print(posts_data)
        return posts_data
