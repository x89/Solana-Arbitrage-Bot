#!/usr/bin/env python3
"""
News Collecting System
Comprehensive news collection and processing system including:
- Multi-source news aggregation
- Real-time news monitoring
- Sentiment analysis
- Relevance scoring
- News impact assessment
"""

import asyncio
import aiohttp
import requests
import feedparser
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import re
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article structure"""
    title: str
    content: str
    summary: str
    source: str
    url: str
    published_time: datetime
    sentiment_score: float
    relevance_score: float
    impact_score: float
    keywords: List[str]
    category: str
    language: str
    metadata: Dict[str, Any]

@dataclass
class NewsSource:
    """News source configuration"""
    name: str
    url: str
    type: str  # 'rss', 'api', 'scraper'
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, str]] = None
    enabled: bool = True
    priority: int = 1  # 1 = highest priority

class SentimentAnalyzer:
    """Advanced sentiment analysis"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial sentiment keywords
        self.bullish_keywords = {
            'bullish', 'rise', 'gain', 'up', 'positive', 'growth', 'profit', 'surge',
            'rally', 'boom', 'breakthrough', 'milestone', 'record', 'high', 'moon',
            'pump', 'buy', 'long', 'optimistic', 'strong', 'robust', 'excellent'
        }
        
        self.bearish_keywords = {
            'bearish', 'fall', 'drop', 'down', 'negative', 'loss', 'crash', 'decline',
            'plunge', 'slump', 'weak', 'poor', 'disappointing', 'concern', 'risk',
            'dump', 'sell', 'short', 'pessimistic', 'weak', 'struggling', 'crisis'
        }
        
        self.neutral_keywords = {
            'stable', 'unchanged', 'flat', 'sideways', 'consolidation', 'range',
            'wait', 'hold', 'neutral', 'mixed', 'uncertain', 'volatile'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            # Clean text
            text = self._clean_text(text)
            
            # Get VADER sentiment scores
            vader_scores = self.sia.polarity_scores(text)
            
            # Get keyword-based sentiment
            keyword_sentiment = self._analyze_keyword_sentiment(text)
            
            # Combine scores
            combined_score = self._combine_sentiment_scores(vader_scores, keyword_sentiment)
            
            return {
                'compound': combined_score,
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'keyword_sentiment': keyword_sentiment,
                'confidence': abs(combined_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'confidence': 0.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _analyze_keyword_sentiment(self, text: str) -> float:
        """Analyze sentiment based on financial keywords"""
        try:
            words = word_tokenize(text.lower())
            
            bullish_count = sum(1 for word in words if word in self.bullish_keywords)
            bearish_count = sum(1 for word in words if word in self.bearish_keywords)
            neutral_count = sum(1 for word in words if word in self.neutral_keywords)
            
            total_keywords = bullish_count + bearish_count + neutral_count
            
            if total_keywords == 0:
                return 0.0
            
            # Calculate weighted sentiment
            sentiment = (bullish_count - bearish_count) / total_keywords
            
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing keyword sentiment: {e}")
            return 0.0
    
    def _combine_sentiment_scores(self, vader_scores: Dict[str, float], keyword_sentiment: float) -> float:
        """Combine VADER and keyword sentiment scores"""
        try:
            # Weighted combination (70% VADER, 30% keywords)
            vader_weight = 0.7
            keyword_weight = 0.3
            
            combined = (vader_scores['compound'] * vader_weight + 
                       keyword_sentiment * keyword_weight)
            
            return max(-1.0, min(1.0, combined))
            
        except Exception as e:
            logger.error(f"Error combining sentiment scores: {e}")
            return vader_scores.get('compound', 0.0)

class RelevanceScorer:
    """News relevance scoring"""
    
    def __init__(self):
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'cardano', 'ada',
            'crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'trading',
            'exchange', 'wallet', 'mining', 'staking', 'yield', 'liquidity'
        }
        
        self.market_keywords = {
            'market', 'price', 'volume', 'trading', 'bull', 'bear', 'rally', 'crash',
            'volatility', 'trend', 'support', 'resistance', 'breakout', 'breakdown'
        }
        
        self.impact_keywords = {
            'regulation', 'ban', 'approve', 'adoption', 'partnership', 'merger',
            'acquisition', 'funding', 'investment', 'launch', 'upgrade', 'fork'
        }
    
    def calculate_relevance(self, title: str, content: str, keywords: List[str]) -> float:
        """Calculate news relevance score"""
        try:
            text = f"{title} {content}".lower()
            
            # Keyword matching
            crypto_score = self._calculate_keyword_score(text, self.crypto_keywords)
            market_score = self._calculate_keyword_score(text, self.market_keywords)
            impact_score = self._calculate_keyword_score(text, self.impact_keywords)
            
            # Title importance (titles are more important)
            title_crypto_score = self._calculate_keyword_score(title.lower(), self.crypto_keywords)
            title_market_score = self._calculate_keyword_score(title.lower(), self.market_keywords)
            
            # Calculate weighted relevance
            relevance = (
                crypto_score * 0.4 +
                market_score * 0.3 +
                impact_score * 0.2 +
                title_crypto_score * 0.05 +
                title_market_score * 0.05
            )
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, text: str, keywords: set) -> float:
        """Calculate score based on keyword matches"""
        try:
            words = set(text.split())
            matches = len(words.intersection(keywords))
            
            if len(keywords) == 0:
                return 0.0
            
            return min(1.0, matches / len(keywords) * 10)  # Scale to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating keyword score: {e}")
            return 0.0
    
    def calculate_impact(self, title: str, content: str, source: str) -> float:
        """Calculate news impact score"""
        try:
            text = f"{title} {content}".lower()
            
            # Source credibility
            source_credibility = self._get_source_credibility(source)
            
            # Impact keywords
            impact_score = self._calculate_keyword_score(text, self.impact_keywords)
            
            # Text length (longer articles often have more impact)
            length_score = min(1.0, len(content) / 1000)
            
            # Calculate weighted impact
            impact = (
                source_credibility * 0.4 +
                impact_score * 0.4 +
                length_score * 0.2
            )
            
            return min(1.0, impact)
            
        except Exception as e:
            logger.error(f"Error calculating impact: {e}")
            return 0.0
    
    def _get_source_credibility(self, source: str) -> float:
        """Get source credibility score"""
        credibility_scores = {
            'reuters': 0.9,
            'bloomberg': 0.9,
            'coindesk': 0.8,
            'cointelegraph': 0.8,
            'yahoo_finance': 0.7,
            'marketwatch': 0.7,
            'cnbc': 0.8,
            'wsj': 0.9,
            'ft': 0.9,
            'forbes': 0.7,
            'techcrunch': 0.6,
            'reddit': 0.3,
            'twitter': 0.4,
            'telegram': 0.3
        }
        
        return credibility_scores.get(source.lower(), 0.5)

class RSSNewsCollector:
    """RSS news collector"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def collect_from_rss(self, source: NewsSource) -> List[NewsArticle]:
        """Collect news from RSS feed"""
        try:
            articles = []
            
            # Parse RSS feed
            feed = feedparser.parse(source.url)
            
            if feed.bozo:
                logger.warning(f"RSS feed parsing error for {source.name}")
                return articles
            
            for entry in feed.entries[:20]:  # Limit to 20 articles
                try:
                    # Extract article data
                    title = entry.get('title', '')
                    content = entry.get('summary', '') or entry.get('description', '')
                    url = entry.get('link', '')
                    
                    # Parse published time
                    published_time = self._parse_published_time(entry)
                    
                    # Extract content if needed
                    if len(content) < 100:
                        content = await self._extract_full_content(url)
                    
                    # Create article
                    article = NewsArticle(
                        title=title,
                        content=content,
                        summary=content[:200] + '...' if len(content) > 200 else content,
                        source=source.name,
                        url=url,
                        published_time=published_time,
                        sentiment_score=0.0,  # Will be calculated later
                        relevance_score=0.0,   # Will be calculated later
                        impact_score=0.0,     # Will be calculated later
                        keywords=[],
                        category='general',
                        language='en',
                        metadata={'rss_entry': entry}
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error processing RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting RSS news from {source.name}: {e}")
            return []
    
    def _parse_published_time(self, entry) -> datetime:
        """Parse published time from RSS entry"""
        try:
            if 'published_parsed' in entry:
                return datetime(*entry.published_parsed[:6])
            elif 'updated_parsed' in entry:
                return datetime(*entry.updated_parsed[:6])
            else:
                return datetime.now()
        except Exception as e:
            logger.error(f"Error parsing published time: {e}")
            return datetime.now()
    
    async def _extract_full_content(self, url: str) -> str:
        """Extract full content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text
                        text = soup.get_text()
                        
                        # Clean text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text[:2000]  # Limit content length
                    
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
        
        return ""

class APINewsCollector:
    """API-based news collector"""
    
    def __init__(self):
        self.session = requests.Session()
    
    async def collect_from_api(self, source: NewsSource) -> List[NewsArticle]:
        """Collect news from API"""
        try:
            articles = []
            
            # Prepare request
            headers = source.headers or {}
            params = source.params or {}
            
            if source.api_key:
                params['api_key'] = source.api_key
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(source.url, headers=headers, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process API response (format depends on API)
                        articles = self._process_api_response(data, source)
                    else:
                        logger.error(f"API request failed for {source.name}: {response.status}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting API news from {source.name}: {e}")
            return []
    
    def _process_api_response(self, data: Dict[str, Any], source: NewsSource) -> List[NewsArticle]:
        """Process API response data"""
        try:
            articles = []
            
            # This is a generic implementation - adjust based on specific API format
            if 'articles' in data:
                items = data['articles']
            elif 'results' in data:
                items = data['results']
            elif isinstance(data, list):
                items = data
            else:
                logger.warning(f"Unknown API response format for {source.name}")
                return articles
            
            for item in items[:20]:  # Limit to 20 articles
                try:
                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('content', '') or item.get('description', ''),
                        summary=item.get('summary', ''),
                        source=source.name,
                        url=item.get('url', ''),
                        published_time=self._parse_api_time(item.get('publishedAt', '')),
                        sentiment_score=0.0,
                        relevance_score=0.0,
                        impact_score=0.0,
                        keywords=item.get('keywords', []),
                        category=item.get('category', 'general'),
                        language=item.get('language', 'en'),
                        metadata={'api_data': item}
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error processing API item: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error processing API response: {e}")
            return []
    
    def _parse_api_time(self, time_str: str) -> datetime:
        """Parse time from API response"""
        try:
            if not time_str:
                return datetime.now()
            
            # Try different time formats
            formats = [
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            
            return datetime.now()
            
        except Exception as e:
            logger.error(f"Error parsing API time: {e}")
            return datetime.now()

class NewsDatabase:
    """News database management"""
    
    def __init__(self, db_path: str = "news_collection.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize news database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                source TEXT NOT NULL,
                url TEXT UNIQUE,
                published_time DATETIME NOT NULL,
                sentiment_score REAL NOT NULL,
                relevance_score REAL NOT NULL,
                impact_score REAL NOT NULL,
                keywords TEXT,
                category TEXT,
                language TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                type TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 1,
                last_collected DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_article(self, article: NewsArticle) -> bool:
        """Save article to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO news_articles 
                (title, content, summary, source, url, published_time, sentiment_score,
                 relevance_score, impact_score, keywords, category, language, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.title,
                article.content,
                article.summary,
                article.source,
                article.url,
                article.published_time,
                article.sentiment_score,
                article.relevance_score,
                article.impact_score,
                json.dumps(article.keywords),
                article.category,
                article.language,
                json.dumps(article.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving article: {e}")
            return False
    
    def get_recent_articles(self, hours: int = 24, limit: int = 100) -> List[NewsArticle]:
        """Get recent articles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, content, summary, source, url, published_time, sentiment_score,
                       relevance_score, impact_score, keywords, category, language, metadata
                FROM news_articles
                WHERE published_time >= datetime('now', '-{} hours')
                ORDER BY published_time DESC
                LIMIT ?
            '''.format(hours), (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            articles = []
            for row in rows:
                articles.append(NewsArticle(
                    title=row[0],
                    content=row[1],
                    summary=row[2],
                    source=row[3],
                    url=row[4],
                    published_time=datetime.fromisoformat(row[5]),
                    sentiment_score=row[6],
                    relevance_score=row[7],
                    impact_score=row[8],
                    keywords=json.loads(row[9]) if row[9] else [],
                    category=row[10],
                    language=row[11],
                    metadata=json.loads(row[12]) if row[12] else {}
                ))
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []
    
    def get_articles_by_sentiment(self, sentiment_range: Tuple[float, float], 
                                hours: int = 24) -> List[NewsArticle]:
        """Get articles by sentiment range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, content, summary, source, url, published_time, sentiment_score,
                       relevance_score, impact_score, keywords, category, language, metadata
                FROM news_articles
                WHERE published_time >= datetime('now', '-{} hours')
                AND sentiment_score BETWEEN ? AND ?
                ORDER BY sentiment_score DESC
            '''.format(hours), sentiment_range)
            
            rows = cursor.fetchall()
            conn.close()
            
            articles = []
            for row in rows:
                articles.append(NewsArticle(
                    title=row[0],
                    content=row[1],
                    summary=row[2],
                    source=row[3],
                    url=row[4],
                    published_time=datetime.fromisoformat(row[5]),
                    sentiment_score=row[6],
                    relevance_score=row[7],
                    impact_score=row[8],
                    keywords=json.loads(row[9]) if row[9] else [],
                    category=row[10],
                    language=row[11],
                    metadata=json.loads(row[12]) if row[12] else {}
                ))
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles by sentiment: {e}")
            return []

class NewsCollectionManager:
    """Main news collection manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.relevance_scorer = RelevanceScorer()
        self.rss_collector = RSSNewsCollector()
        self.api_collector = APINewsCollector()
        self.database = NewsDatabase(self.config.get('db_path', 'news_collection.db'))
        
        # Initialize news sources
        self.sources = self._initialize_sources()
        
        # Collection settings
        self.collection_interval = self.config.get('collection_interval', 300)  # 5 minutes
        self.max_articles_per_source = self.config.get('max_articles_per_source', 20)
        self.running = False
    
    def _initialize_sources(self) -> List[NewsSource]:
        """Initialize news sources"""
        sources = [
            # RSS Sources
            NewsSource(
                name='CoinDesk',
                url='https://www.coindesk.com/arc/outboundfeeds/rss/',
                type='rss',
                priority=1
            ),
            NewsSource(
                name='CoinTelegraph',
                url='https://cointelegraph.com/rss',
                type='rss',
                priority=1
            ),
            NewsSource(
                name='Yahoo Finance Crypto',
                url='https://feeds.finance.yahoo.com/rss/2.0/headline',
                type='rss',
                priority=2
            ),
            NewsSource(
                name='MarketWatch',
                url='https://feeds.marketwatch.com/marketwatch/topstories/',
                type='rss',
                priority=2
            ),
            NewsSource(
                name='Bloomberg',
                url='https://feeds.bloomberg.com/markets/news.rss',
                type='rss',
                priority=1
            ),
            NewsSource(
                name='Reuters',
                url='https://feeds.reuters.com/reuters/businessNews',
                type='rss',
                priority=1
            ),
            
            # API Sources (add your API keys)
            # NewsSource(
            #     name='NewsAPI',
            #     url='https://newsapi.org/v2/everything',
            #     type='api',
            #     api_key='your_api_key_here',
            #     params={'q': 'cryptocurrency', 'sortBy': 'publishedAt'},
            #     priority=1
            # ),
        ]
        
        return sources
    
    async def start_collection(self):
        """Start news collection process"""
        self.running = True
        logger.info("Starting news collection...")
        
        while self.running:
            try:
                await self._collect_all_sources()
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in news collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_all_sources(self):
        """Collect news from all enabled sources"""
        try:
            tasks = []
            
            for source in self.sources:
                if source.enabled:
                    if source.type == 'rss':
                        task = self._collect_rss_source(source)
                    elif source.type == 'api':
                        task = self._collect_api_source(source)
                    else:
                        logger.warning(f"Unknown source type: {source.type}")
                        continue
                    
                    tasks.append(task)
            
            # Collect from all sources concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            total_articles = 0
            for result in results:
                if isinstance(result, list):
                    total_articles += len(result)
                elif isinstance(result, Exception):
                    logger.error(f"Collection error: {result}")
            
            logger.info(f"Collected {total_articles} articles from {len(tasks)} sources")
            
        except Exception as e:
            logger.error(f"Error collecting from all sources: {e}")
    
    async def _collect_rss_source(self, source: NewsSource) -> List[NewsArticle]:
        """Collect from RSS source"""
        try:
            articles = await self.rss_collector.collect_from_rss(source)
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_article = await self._process_article(article)
                if processed_article:
                    processed_articles.append(processed_article)
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error collecting from RSS source {source.name}: {e}")
            return []
    
    async def _collect_api_source(self, source: NewsSource) -> List[NewsArticle]:
        """Collect from API source"""
        try:
            articles = await self.api_collector.collect_from_api(source)
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_article = await self._process_article(article)
                if processed_article:
                    processed_articles.append(processed_article)
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error collecting from API source {source.name}: {e}")
            return []
    
    async def _process_article(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Process article with sentiment analysis and scoring"""
        try:
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                f"{article.title} {article.content}"
            )
            article.sentiment_score = sentiment_result['compound']
            
            # Calculate relevance
            article.relevance_score = self.relevance_scorer.calculate_relevance(
                article.title, article.content, article.keywords
            )
            
            # Calculate impact
            article.impact_score = self.relevance_scorer.calculate_impact(
                article.title, article.content, article.source
            )
            
            # Extract keywords
            article.keywords = self._extract_keywords(article.title, article.content)
            
            # Save to database
            if self.database.save_article(article):
                return article
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract keywords from title and content"""
        try:
            text = f"{title} {content}".lower()
            
            # Simple keyword extraction
            crypto_keywords = self.relevance_scorer.crypto_keywords
            market_keywords = self.relevance_scorer.market_keywords
            impact_keywords = self.relevance_scorer.impact_keywords
            
            all_keywords = crypto_keywords.union(market_keywords).union(impact_keywords)
            
            # Find keywords in text
            found_keywords = []
            for keyword in all_keywords:
                if keyword in text:
                    found_keywords.append(keyword)
            
            return found_keywords[:10]  # Limit to 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def get_market_sentiment(self, hours: int = 24) -> Dict[str, float]:
        """Get overall market sentiment"""
        try:
            articles = self.database.get_recent_articles(hours)
            
            if not articles:
                return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0
            
            for article in articles:
                weight = article.relevance_score * article.impact_score
                weighted_sentiment += article.sentiment_score * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_sentiment = weighted_sentiment / total_weight
            else:
                avg_sentiment = 0.0
            
            # Calculate confidence based on article count and relevance
            avg_relevance = np.mean([a.relevance_score for a in articles])
            confidence = min(1.0, len(articles) / 50 * avg_relevance)
            
            return {
                'sentiment': avg_sentiment,
                'confidence': confidence,
                'article_count': len(articles),
                'avg_relevance': avg_relevance
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
    
    def get_top_news(self, hours: int = 24, limit: int = 10) -> List[NewsArticle]:
        """Get top news by impact and relevance"""
        try:
            articles = self.database.get_recent_articles(hours, limit * 2)
            
            # Sort by combined score
            articles.sort(key=lambda x: x.impact_score * x.relevance_score, reverse=True)
            
            return articles[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top news: {e}")
            return []
    
    def stop_collection(self):
        """Stop news collection"""
        self.running = False
        logger.info("News collection stopped")

async def main():
    """Main function to run news collection"""
    try:
        # Configuration
        config = {
            'collection_interval': 300,  # 5 minutes
            'max_articles_per_source': 20,
            'db_path': 'news_collection.db'
        }
        
        # Initialize news collection manager
        manager = NewsCollectionManager(config)
        
        # Start collection
        await manager.start_collection()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main news collection function: {e}")
    finally:
        if 'manager' in locals():
            manager.stop_collection()

if __name__ == "__main__":
    asyncio.run(main())
