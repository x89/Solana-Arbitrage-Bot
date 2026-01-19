#!/usr/bin/env python3
"""
Sentiment Collector Module
Comprehensive sentiment data collection system including:
- News article collection (RSS feeds, web scraping)
- Social media sentiment (Twitter, Reddit, Telegram)
- Forum sentiment collection
- Real-time sentiment monitoring
- Multi-source aggregation
- Data validation and cleaning
"""

import requests
import feedparser
import tweepy
from bs4 import BeautifulSoup
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentCollector:
    """Comprehensive sentiment data collector"""
    
    def __init__(
        self,
        twitter_api_key: Optional[str] = None,
        twitter_api_secret: Optional[str] = None,
        twitter_access_token: Optional[str] = None,
        twitter_access_token_secret: Optional[str] = None
    ):
        """
        Initialize sentiment collector
        
        Args:
            twitter_api_key: Twitter API key
            twitter_api_secret: Twitter API secret
            twitter_access_token: Twitter access token
            twitter_access_token_secret: Twitter access token secret
        """
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret
        
        # Twitter client
        self.twitter_api = None
        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret]):
            try:
                auth = tweepy.OAuth1UserHandler(
                    twitter_api_key, twitter_api_secret,
                    twitter_access_token, twitter_access_token_secret
                )
                self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API initialized")
            except Exception as e:
                logger.warning(f"Twitter API initialization failed: {e}")
        
        # Collection sources
        self.rss_feeds = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'decrypt': 'https://decrypt.co/feed',
            'the_block': 'https://www.theblockcrypto.com/rss.xml'
        }
        
        self.collection_stats = {
            'total_collected': 0,
            'news_articles': 0,
            'social_posts': 0,
            'forum_posts': 0
        }
        
        logger.info("Sentiment collector initialized")
    
    def collect_news_articles(self, keywords: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect news articles from RSS feeds
        
        Args:
            keywords: Keywords to filter articles
            limit: Maximum number of articles to collect
            
        Returns:
            List of collected articles
        """
        try:
            logger.info(f"Collecting news articles for keywords: {keywords}")
            
            articles = []
            
            for source_name, feed_url in self.rss_feeds.items():
                try:
                    logger.info(f"Fetching from {source_name}...")
                    
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        published = entry.get('published', '')
                        
                        # Check if relevant to keywords
                        full_text = f"{title} {summary}".lower()
                        if any(keyword.lower() in full_text for keyword in keywords):
                            article = {
                                'source': source_name,
                                'title': title,
                                'link': link,
                                'summary': summary,
                                'published': published,
                                'collected_at': datetime.now().isoformat(),
                                'type': 'news'
                            }
                            articles.append(article)
                        
                        if len(articles) >= limit:
                            break
                    
                except Exception as e:
                    logger.error(f"Error fetching from {source_name}: {e}")
                    continue
            
            self.collection_stats['news_articles'] += len(articles)
            self.collection_stats['total_collected'] += len(articles)
            
            logger.info(f"Collected {len(articles)} news articles")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting news articles: {e}")
            return []
    
    def collect_twitter_sentiment(self, keywords: List[str], count: int = 100) -> List[Dict[str, Any]]:
        """
        Collect Twitter sentiment
        
        Args:
            keywords: Search keywords
            count: Number of tweets to collect
            
        Returns:
            List of collected tweets
        """
        try:
            if not self.twitter_api:
                logger.warning("Twitter API not initialized")
                return []
            
            logger.info(f"Collecting Twitter sentiment for keywords: {keywords}")
            
            tweets = []
            query = " OR ".join(keywords) + " -filter:retweets"
            
            try:
                for tweet in tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=query,
                    lang='en',
                    result_type='recent',
                    tweet_mode='extended'
                ).items(count):
                    tweet_data = {
                        'id': tweet.id_str,
                        'text': tweet.full_text,
                        'user': tweet.user.screen_name,
                        'created_at': tweet.created_at.isoformat(),
                        'retweet_count': tweet.retweet_count,
                        'favorite_count': tweet.favorite_count,
                        'collected_at': datetime.now().isoformat(),
                        'type': 'twitter'
                    }
                    tweets.append(tweet_data)
                
                self.collection_stats['social_posts'] += len(tweets)
                self.collection_stats['total_collected'] += len(tweets)
                
                logger.info(f"Collected {len(tweets)} tweets")
                
            except Exception as e:
                logger.error(f"Error collecting Twitter data: {e}")
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error in Twitter collection: {e}")
            return []
    
    async def collect_from_url(self, url: str, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """
        Collect sentiment data from a specific URL
        
        Args:
            url: URL to scrape
            keywords: Keywords to search for
            
        Returns:
            Collected data or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract text
                        text = soup.get_text()
                        
                        # Check if contains keywords
                        text_lower = text.lower()
                        if any(keyword.lower() in text_lower for keyword in keywords):
                            return {
                                'url': url,
                                'text': text[:1000],  # First 1000 chars
                                'collected_at': datetime.now().isoformat(),
                                'type': 'web'
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Error collecting from URL {url}: {e}")
            return None
    
    def collect_reddit_sentiment(self, subreddit: str, keywords: List[str], limit: int = 25) -> List[Dict[str, Any]]:
        """
        Collect Reddit sentiment (using requests, no PRAW dependency)
        
        Args:
            subreddit: Subreddit name
            keywords: Keywords to filter posts
            limit: Number of posts to collect
            
        Returns:
            List of collected Reddit posts
        """
        try:
            logger.info(f"Collecting Reddit sentiment from r/{subreddit}")
            
            posts = []
            
            # Reddit JSON API (no authentication required for reading)
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', {}).get('children', []):
                    post_data = item.get('data', {})
                    
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    url_link = post_data.get('url', '')
                    upvotes = post_data.get('ups', 0)
                    created = post_data.get('created_utc', 0)
                    
                    # Check if relevant
                    full_text = f"{title} {selftext}".lower()
                    if any(keyword.lower() in full_text for keyword in keywords):
                        post = {
                            'source': f'r/{subreddit}',
                            'title': title,
                            'text': selftext,
                            'url': url_link,
                            'upvotes': upvotes,
                            'created_at': datetime.fromtimestamp(created).isoformat(),
                            'collected_at': datetime.now().isoformat(),
                            'type': 'reddit'
                        }
                        posts.append(post)
            
            self.collection_stats['forum_posts'] += len(posts)
            self.collection_stats['total_collected'] += len(posts)
            
            logger.info(f"Collected {len(posts)} Reddit posts")
            
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment: {e}")
            return []
    
    def collect_all_sources(self, keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect sentiment from all sources
        
        Args:
            keywords: Keywords to search for
            
        Returns:
            Dictionary of collected data by source
        """
        try:
            logger.info("Collecting sentiment from all sources...")
            
            collected_data = {
                'news': [],
                'twitter': [],
                'reddit': [],
                'web': []
            }
            
            # News articles
            collected_data['news'] = self.collect_news_articles(keywords, limit=30)
            
            # Twitter
            collected_data['twitter'] = self.collect_twitter_sentiment(keywords, count=50)
            
            # Reddit
            subreddits = ['CryptoCurrency', 'Bitcoin', 'ethereum', 'solana']
            for subreddit in subreddits:
                try:
                    reddit_posts = self.collect_reddit_sentiment(subreddit, keywords, limit=10)
                    collected_data['reddit'].extend(reddit_posts)
                except Exception as e:
                    logger.error(f"Error collecting from r/{subreddit}: {e}")
            
            total_collected = sum(len(data) for data in collected_data.values())
            logger.info(f"Total collected: {total_collected} items")
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Error collecting from all sources: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics"""
        return self.collection_stats.copy()

def main():
    """Main function to demonstrate sentiment collection"""
    try:
        logger.info("=" * 60)
        logger.info("Sentiment Collector Demo")
        logger.info("=" * 60)
        
        # Initialize collector
        collector = SentimentCollector()
        
        # Collect sentiment data
        keywords = ['bitcoin', 'crypto', 'trading', 'market']
        
        logger.info(f"Collecting sentiment for keywords: {keywords}")
        
        collected_data = collector.collect_all_sources(keywords)
        
        # Display results
        for source, data in collected_data.items():
            logger.info(f"{source}: {len(data)} items collected")
        
        # Display stats
        stats = collector.get_collection_stats()
        logger.info(f"\nCollection Statistics:")
        logger.info(f"  Total: {stats['total_collected']}")
        logger.info(f"  News: {stats['news_articles']}")
        logger.info(f"  Social: {stats['social_posts']}")
        logger.info(f"  Forum: {stats['forum_posts']}")
        
        logger.info("=" * 60)
        logger.info("Sentiment Collector Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

