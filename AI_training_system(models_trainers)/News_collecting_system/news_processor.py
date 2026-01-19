#!/usr/bin/env python3
"""
News Processor Module
Comprehensive news processing system including:
- Text cleaning and normalization
- Entity extraction
- Sentiment analysis integration
- Article categorization
- Keyword extraction
- Content summarization
"""

import re
import string
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsProcessor:
    """Comprehensive news processing system"""
    
    def __init__(self, remove_stopwords: bool = True, min_word_length: int = 3):
        """
        Initialize news processor
        
        Args:
            remove_stopwords: Whether to remove stop words
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        
        # Common stop words
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'from', 'as', 'it',
            'its', 'it\'s', 'its\'', 'they', 'them', 'their', 'there'
        }
        
        # Trading-related keywords
        self.trading_keywords = {
            'bullish', 'bearish', 'bull', 'bear', 'pump', 'dump', 'rally',
            'crash', 'surge', 'plunge', 'soar', 'tumble', 'pump', 'dip',
            'rally', 'fomo', 'hodl', 'moonshot', 'lambo', 'rekt', 'fud',
            'adoption', 'institutional', 'whale', 'retail', 'deficit',
            'surplus', 'leverage', 'margin', 'liquidation', 'short squeeze'
        }
        
        logger.info("News processor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        try:
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^\w\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Split into words
            words = cleaned_text.split()
            
            # Filter by length
            words = [w for w in words if len(w) >= self.min_word_length]
            
            # Remove stop words if enabled
            if self.remove_stopwords:
                words = [w for w in words if w not in self.stop_words]
            
            # Count frequencies
            word_counts = Counter(words)
            
            # Get top keywords
            top_keywords = [word for word, count in word_counts.most_common(num_keywords)]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        try:
            sentiment_scores = {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral'
            }
            
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                sentiment_scores['polarity'] = float(blob.sentiment.polarity)
                sentiment_scores['subjectivity'] = float(blob.sentiment.subjectivity)
                
                # Determine sentiment label
                if sentiment_scores['polarity'] > 0.1:
                    sentiment_scores['sentiment'] = 'positive'
                elif sentiment_scores['polarity'] < -0.1:
                    sentiment_scores['sentiment'] = 'negative'
                else:
                    sentiment_scores['sentiment'] = 'neutral'
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text (cryptocurrency names, stock symbols, etc.)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and entities
        """
        try:
            entities = {
                'cryptocurrencies': [],
                'tickers': [],
                'orgs': [],
                'people': []
            }
            
            # Common cryptocurrency names
            crypto_names = {
                'bitcoin', 'btc', 'ethereum', 'eth', 'ripple', 'xrp',
                'binance', 'bnb', 'solana', 'sol', 'cardano', 'ada',
                'polkadot', 'dot', 'dogecoin', 'doge', 'litecoin', 'ltc',
                'chainlink', 'link', 'uniswap', 'uni', 'avalanche', 'avax'
            }
            
            # Extract words from text
            words = self.clean_text(text).split()
            
            # Check for cryptocurrency names
            text_lower = text.lower()
            for crypto in crypto_names:
                if crypto in text_lower:
                    entities['cryptocurrencies'].append(crypto.upper())
            
            # Extract ticker symbols (3-5 uppercase letters with $ prefix or standalone)
            ticker_pattern = r'\$?([A-Z]{2,5})\b'
            tickers = re.findall(ticker_pattern, text)
            entities['tickers'].extend(tickers)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {'cryptocurrencies': [], 'tickers': [], 'orgs': [], 'people': []}
    
    def categorize_article(self, text: str, title: str = "") -> str:
        """
        Categorize article based on content
        
        Args:
            text: Article text
            title: Article title
            
        Returns:
            Article category
        """
        try:
            combined_text = f"{title} {text}".lower()
            
            categories = {
                'price_action': ['price', 'increase', 'decrease', 'surge', 'drop', 'rally', 'crash'],
                'adoption': ['adoption', 'partnership', 'collaboration', 'integration'],
                'regulation': ['regulation', 'regulatory', 'sec', 'compliant', 'compliance'],
                'technology': ['upgrade', 'update', 'hard fork', 'soft fork', 'blockchain', 'scalability'],
                'market_analysis': ['analysis', 'forecast', 'prediction', 'trend', 'outlook'],
                'security': ['hack', 'breach', 'security', 'audit', 'vulnerability'],
                'general': []
            }
            
            for category, keywords in categories.items():
                if any(keyword in combined_text for keyword in keywords):
                    return category
            
            return 'general'
            
        except Exception as e:
            logger.error(f"Error categorizing article: {e}")
            return 'general'
    
    def calculate_relevance_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keywords
        
        Args:
            text: Input text
            keywords: List of keywords to match
            
        Returns:
            Relevance score (0-1)
        """
        try:
            if not keywords:
                return 0.0
            
            cleaned_text = self.clean_text(text)
            word_list = set(cleaned_text.split())
            
            matches = sum(1 for keyword in keywords if keyword in word_list)
            
            relevance = matches / len(keywords)
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def process_article(
        self,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete article
        
        Args:
            title: Article title
            content: Article content
            metadata: Optional metadata
            
        Returns:
            Processed article data
        """
        try:
            # Combine title and content
            full_text = f"{title} {content}"
            
            # Process
            keywords = self.extract_keywords(full_text, num_keywords=10)
            sentiment = self.analyze_sentiment(full_text)
            entities = self.extract_entities(full_text)
            category = self.categorize_article(content, title)
            
            # Build result
            processed = {
                'title': title,
                'content': content,
                'cleaned_content': self.clean_text(content),
                'keywords': keywords,
                'sentiment': sentiment,
                'entities': entities,
                'category': category,
                'trading_relevance': self.calculate_relevance_score(
                    full_text,
                    list(self.trading_keywords)
                ),
                'processed_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return {}
    
    def process_batch(
        self,
        articles: List[Dict[str, str]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple articles
        
        Args:
            articles: List of article dicts with 'title' and 'content'
            progress_callback: Optional callback function
            
        Returns:
            List of processed articles
        """
        try:
            processed_articles = []
            
            for idx, article in enumerate(articles):
                processed = self.process_article(
                    article.get('title', ''),
                    article.get('content', ''),
                    article.get('metadata', {})
                )
                processed_articles.append(processed)
                
                if progress_callback:
                    progress_callback(idx + 1, len(articles))
            
            logger.info(f"Processed {len(processed_articles)} articles")
            
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []

def main():
    """Main function to demonstrate news processing"""
    try:
        logger.info("=" * 60)
        logger.info("News Processor Demo")
        logger.info("=" * 60)
        
        # Initialize processor
        processor = NewsProcessor()
        
        # Sample article
        sample_article = {
            'title': 'Bitcoin Surges Past $50,000 as Institutional Adoption Grows',
            'content': 'Bitcoin has experienced a significant price increase, reaching new highs above $50,000. Institutional investors are showing increasing interest in cryptocurrency adoption. Analysts are bullish on the future outlook for digital assets.',
            'metadata': {'source': 'crypto_news', 'date': '2024-01-15'}
        }
        
        # Process article
        result = processor.process_article(**sample_article)
        
        logger.info("\nProcessing Results:")
        logger.info(f"Category: {result.get('category')}")
        logger.info(f"Sentiment: {result.get('sentiment', {})}")
        logger.info(f"Keywords: {result.get('keywords', [])[:5]}")
        logger.info(f"Trading Relevance: {result.get('trading_relevance', 0):.2f}")
        
        logger.info("=" * 60)
        logger.info("News Processor Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

