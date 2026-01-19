#!/usr/bin/env python3
"""
AI Sentiment Training & Analyzing System
Advanced sentiment analysis system for trading including:
- Sentiment model training
- Real-time sentiment analysis
- Multi-source sentiment aggregation
- Sentiment-based trading signals
- Sentiment performance tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser
import tweepy
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Sentiment data structure"""
    text: str
    sentiment_label: int  # -1: negative, 0: neutral, 1: positive
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class SentimentPrediction:
    """Sentiment prediction result"""
    text: str
    predicted_sentiment: int
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    timestamp: datetime

class SentimentDataset(Dataset):
    """PyTorch dataset for sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentTransformerModel(nn.Module):
    """Transformer-based sentiment analysis model"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 3):
        super(SentimentTransformerModel, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class SentimentTrainer:
    """Sentiment model trainer"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
    
    def prepare_data(self, df: pd.DataFrame, text_column: str, label_column: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training"""
        try:
            # Clean and prepare texts
            texts = df[text_column].fillna('').astype(str).tolist()
            labels = df[label_column].astype(int).tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create datasets
            train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
            test_dataset = SentimentDataset(X_test, y_test, self.tokenizer)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error preparing sentiment data: {e}")
            return None, None
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, 
                   epochs: int = 5) -> Dict[str, float]:
        """Train sentiment model"""
        try:
            # Initialize model
            self.model = SentimentTransformerModel(self.model_name).to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
            
            # Training loop
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                scheduler.step()
                
                # Validation
                val_metrics = self.evaluate_model(test_loader)
                
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': total_loss / len(train_loader),
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_score']
                }
                
                self.training_history.append(epoch_info)
                logger.info(f"Epoch {epoch + 1}: Loss={epoch_info['train_loss']:.4f}, "
                           f"Accuracy={epoch_info['val_accuracy']:.4f}")
            
            # Final evaluation
            final_metrics = self.evaluate_model(test_loader)
            
            logger.info(f"Training completed. Final accuracy: {final_metrics['accuracy']:.4f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            return {}
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            self.model.eval()
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    preds = torch.argmax(outputs, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sentiment model: {e}")
            return {}
    
    def predict_sentiment(self, text: str) -> SentimentPrediction:
        """Predict sentiment for a single text"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            self.model.eval()
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Convert to sentiment labels
            sentiment_map = {0: -1, 1: 0, 2: 1}  # negative, neutral, positive
            predicted_sentiment = sentiment_map[predicted_class]
            
            prob_dict = {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            }
            
            return SentimentPrediction(
                text=text,
                predicted_sentiment=predicted_sentiment,
                confidence=confidence,
                probabilities=prob_dict,
                model_name=self.model_name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'tokenizer': self.tokenizer,
                'training_history': self.training_history
            }
            
            torch.save(model_data, filepath)
            logger.info(f"Sentiment model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment model: {e}")

class SentimentAnalyzer:
    """Advanced sentiment analyzer"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial sentiment keywords
        self.bullish_keywords = {
            'bullish', 'rise', 'gain', 'up', 'positive', 'growth', 'profit', 'surge',
            'rally', 'boom', 'breakthrough', 'milestone', 'record', 'high', 'moon',
            'pump', 'buy', 'long', 'optimistic', 'strong', 'robust', 'excellent',
            'outperform', 'beat', 'exceed', 'surpass', 'soar', 'climb', 'advance'
        }
        
        self.bearish_keywords = {
            'bearish', 'fall', 'drop', 'down', 'negative', 'loss', 'crash', 'decline',
            'plunge', 'slump', 'weak', 'poor', 'disappointing', 'concern', 'risk',
            'dump', 'sell', 'short', 'pessimistic', 'weak', 'struggling', 'crisis',
            'underperform', 'miss', 'disappoint', 'plummet', 'tumble', 'retreat'
        }
        
        self.neutral_keywords = {
            'stable', 'unchanged', 'flat', 'sideways', 'consolidation', 'range',
            'wait', 'hold', 'neutral', 'mixed', 'uncertain', 'volatile'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods"""
        try:
            # Clean text
            text = self._clean_text(text)
            
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # Keyword-based sentiment
            keyword_sentiment = self._analyze_keyword_sentiment(text)
            
            # Financial sentiment analysis
            financial_sentiment = self._analyze_financial_sentiment(text)
            
            # Combine scores
            combined_score = self._combine_sentiment_scores(
                vader_scores, keyword_sentiment, financial_sentiment
            )
            
            return {
                'compound': combined_score,
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'keyword_sentiment': keyword_sentiment,
                'financial_sentiment': financial_sentiment,
                'confidence': abs(combined_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'confidence': 0.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _analyze_keyword_sentiment(self, text: str) -> float:
        """Analyze sentiment based on keywords"""
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
    
    def _analyze_financial_sentiment(self, text: str) -> float:
        """Analyze financial-specific sentiment"""
        try:
            # Look for financial indicators
            price_indicators = ['price', 'cost', 'value', 'worth', 'valuation']
            market_indicators = ['market', 'trading', 'volume', 'liquidity']
            performance_indicators = ['performance', 'returns', 'yield', 'earnings']
            
            text_lower = text.lower()
            
            # Check for price-related sentiment
            price_sentiment = 0
            for indicator in price_indicators:
                if indicator in text_lower:
                    # Look for positive/negative words around the indicator
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if word == indicator:
                            # Check surrounding words
                            context_words = words[max(0, i-3):i+4]
                            bullish_context = sum(1 for w in context_words if w in self.bullish_keywords)
                            bearish_context = sum(1 for w in context_words if w in self.bearish_keywords)
                            price_sentiment += (bullish_context - bearish_context) / len(context_words)
            
            return max(-1.0, min(1.0, price_sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing financial sentiment: {e}")
            return 0.0
    
    def _combine_sentiment_scores(self, vader_scores: Dict, keyword_sentiment: float, 
                                 financial_sentiment: float) -> float:
        """Combine different sentiment scores"""
        try:
            # Weighted combination
            vader_weight = 0.4
            keyword_weight = 0.3
            financial_weight = 0.3
            
            combined = (
                vader_scores['compound'] * vader_weight +
                keyword_sentiment * keyword_weight +
                financial_sentiment * financial_weight
            )
            
            return max(-1.0, min(1.0, combined))
            
        except Exception as e:
            logger.error(f"Error combining sentiment scores: {e}")
            return vader_scores.get('compound', 0.0)

class SentimentDataCollector:
    """Collect sentiment data from various sources"""
    
    def __init__(self):
        self.sources = {
            'news': ['coindesk', 'cointelegraph', 'bloomberg', 'reuters'],
            'social': ['twitter', 'reddit', 'telegram'],
            'forums': ['bitcointalk', 'cryptocurrency_forums']
        }
    
    async def collect_news_sentiment(self, keywords: List[str]) -> List[SentimentData]:
        """Collect sentiment from news sources"""
        try:
            sentiment_data = []
            
            # RSS feeds for news
            rss_feeds = {
                'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'cointelegraph': 'https://cointelegraph.com/rss',
                'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
                'reuters': 'https://feeds.reuters.com/reuters/businessNews'
            }
            
            for source, url in rss_feeds.items():
                try:
                    feed = feedparser.parse(url)
                    
                    for entry in feed.entries[:10]:  # Limit to 10 entries per source
                        title = entry.get('title', '')
                        content = entry.get('summary', '') or entry.get('description', '')
                        
                        # Check if relevant to keywords
                        if any(keyword.lower() in (title + content).lower() for keyword in keywords):
                            text = f"{title} {content}"
                            
                            # Analyze sentiment
                            analyzer = SentimentAnalyzer()
                            sentiment_result = analyzer.analyze_sentiment(text)
                            
                            # Convert to sentiment label
                            compound_score = sentiment_result['compound']
                            if compound_score > 0.1:
                                sentiment_label = 1  # positive
                            elif compound_score < -0.1:
                                sentiment_label = -1  # negative
                            else:
                                sentiment_label = 0  # neutral
                            
                            sentiment_data.append(SentimentData(
                                text=text,
                                sentiment_label=sentiment_label,
                                confidence=sentiment_result['confidence'],
                                source=source,
                                timestamp=datetime.now(),
                                metadata={
                                    'url': entry.get('link', ''),
                                    'published': entry.get('published', ''),
                                    'sentiment_scores': sentiment_result
                                }
                            ))
                            
                except Exception as e:
                    logger.error(f"Error collecting from {source}: {e}")
                    continue
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting news sentiment: {e}")
            return []
    
    async def collect_social_sentiment(self, keywords: List[str]) -> List[SentimentData]:
        """Collect sentiment from social media"""
        try:
            sentiment_data = []
            
            # This would integrate with actual social media APIs
            # For now, simulate social media sentiment collection
            
            sample_social_posts = [
                "Bitcoin is going to the moon! 🚀",
                "Market crash incoming, sell everything!",
                "Crypto market looks stable today",
                "Great news for Ethereum holders",
                "Another day, another dip in crypto"
            ]
            
            analyzer = SentimentAnalyzer()
            
            for post in sample_social_posts:
                if any(keyword.lower() in post.lower() for keyword in keywords):
                    sentiment_result = analyzer.analyze_sentiment(post)
                    
                    compound_score = sentiment_result['compound']
                    if compound_score > 0.1:
                        sentiment_label = 1
                    elif compound_score < -0.1:
                        sentiment_label = -1
                    else:
                        sentiment_label = 0
                    
                    sentiment_data.append(SentimentData(
                        text=post,
                        sentiment_label=sentiment_label,
                        confidence=sentiment_result['confidence'],
                        source='social_media',
                        timestamp=datetime.now(),
                        metadata={'sentiment_scores': sentiment_result}
                    ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting social sentiment: {e}")
            return []

class SentimentDatabase:
    """Database management for sentiment data"""
    
    def __init__(self, db_path: str = "sentiment_analysis.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize sentiment database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                predicted_sentiment INTEGER NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                model_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_path TEXT NOT NULL,
                accuracy REAL NOT NULL,
                f1_score REAL NOT NULL,
                training_date DATETIME NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_sentiment_data(self, data: SentimentData) -> bool:
        """Save sentiment data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sentiment_data (text, sentiment_label, confidence, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.text,
                data.sentiment_label,
                data.confidence,
                data.source,
                data.timestamp,
                json.dumps(data.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            return False
    
    def save_prediction(self, prediction: SentimentPrediction) -> bool:
        """Save sentiment prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sentiment_predictions (text, predicted_sentiment, confidence, probabilities, model_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction.text,
                prediction.predicted_sentiment,
                prediction.confidence,
                json.dumps(prediction.probabilities),
                prediction.model_name,
                prediction.timestamp
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sentiment prediction: {e}")
            return False
    
    def get_sentiment_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary for recent period"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_count,
                    AVG(sentiment_label) as avg_sentiment,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN sentiment_label = 1 THEN 1 END) as positive_count,
                    COUNT(CASE WHEN sentiment_label = 0 THEN 1 END) as neutral_count,
                    COUNT(CASE WHEN sentiment_label = -1 THEN 1 END) as negative_count
                FROM sentiment_data
                WHERE timestamp >= datetime('now', '-{} hours')
            '''.format(hours))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                total_count, avg_sentiment, avg_confidence, positive_count, neutral_count, negative_count = result
                
                return {
                    'total_count': total_count,
                    'avg_sentiment': avg_sentiment,
                    'avg_confidence': avg_confidence,
                    'positive_percentage': positive_count / total_count * 100,
                    'neutral_percentage': neutral_count / total_count * 100,
                    'negative_percentage': negative_count / total_count * 100,
                    'sentiment_score': avg_sentiment  # -1 to 1 scale
                }
            else:
                return {
                    'total_count': 0,
                    'avg_sentiment': 0.0,
                    'avg_confidence': 0.0,
                    'positive_percentage': 0.0,
                    'neutral_percentage': 0.0,
                    'negative_percentage': 0.0,
                    'sentiment_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}

class SentimentManager:
    """Main sentiment analysis manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.trainer = SentimentTrainer()
        self.analyzer = SentimentAnalyzer()
        self.data_collector = SentimentDataCollector()
        self.database = SentimentDatabase()
        
        # Model management
        self.active_model = None
        self.model_performance = {}
        
        # Collection settings
        self.collection_keywords = self.config.get('keywords', ['crypto', 'bitcoin', 'ethereum', 'trading'])
        self.collection_interval = self.config.get('collection_interval', 300)  # 5 minutes
        
        # Running state
        self.running = False
    
    def train_sentiment_model(self, training_data: pd.DataFrame, 
                            text_column: str = 'text', 
                            label_column: str = 'sentiment') -> Dict[str, float]:
        """Train sentiment analysis model"""
        try:
            logger.info("Starting sentiment model training...")
            
            # Prepare data
            train_loader, test_loader = self.trainer.prepare_data(
                training_data, text_column, label_column
            )
            
            if train_loader is None or test_loader is None:
                raise ValueError("Failed to prepare training data")
            
            # Train model
            metrics = self.trainer.train_model(train_loader, test_loader, epochs=5)
            
            # Save model
            model_path = f"models/sentiment_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            self.trainer.save_model(model_path)
            
            # Update active model
            self.active_model = self.trainer
            
            # Store performance metrics
            self.model_performance[model_path] = metrics
            
            logger.info(f"Sentiment model training completed. Accuracy: {metrics.get('accuracy', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            return {}
    
    async def start_sentiment_collection(self):
        """Start sentiment data collection"""
        self.running = True
        logger.info("Starting sentiment data collection...")
        
        while self.running:
            try:
                # Collect news sentiment
                news_sentiment = await self.data_collector.collect_news_sentiment(self.collection_keywords)
                
                # Collect social sentiment
                social_sentiment = await self.data_collector.collect_social_sentiment(self.collection_keywords)
                
                # Save to database
                all_sentiment = news_sentiment + social_sentiment
                for data in all_sentiment:
                    self.database.save_sentiment_data(data)
                
                logger.info(f"Collected {len(all_sentiment)} sentiment samples")
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in sentiment collection: {e}")
                await asyncio.sleep(60)
    
    def analyze_text_sentiment(self, text: str) -> Optional[SentimentPrediction]:
        """Analyze sentiment of a single text"""
        try:
            if self.active_model:
                # Use trained model
                prediction = self.active_model.predict_sentiment(text)
            else:
                # Use rule-based analyzer
                sentiment_result = self.analyzer.analyze_sentiment(text)
                
                compound_score = sentiment_result['compound']
                if compound_score > 0.1:
                    predicted_sentiment = 1
                elif compound_score < -0.1:
                    predicted_sentiment = -1
                else:
                    predicted_sentiment = 0
                
                prediction = SentimentPrediction(
                    text=text,
                    predicted_sentiment=predicted_sentiment,
                    confidence=sentiment_result['confidence'],
                    probabilities={
                        'negative': sentiment_result['negative'],
                        'neutral': sentiment_result['neutral'],
                        'positive': sentiment_result['positive']
                    },
                    model_name='rule_based',
                    timestamp=datetime.now()
                )
            
            # Save prediction
            if prediction:
                self.database.save_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return None
    
    def get_market_sentiment(self, hours: int = 24) -> Dict[str, Any]:
        """Get overall market sentiment"""
        try:
            summary = self.database.get_sentiment_summary(hours)
            
            # Determine market sentiment
            sentiment_score = summary.get('sentiment_score', 0)
            if sentiment_score > 0.2:
                market_sentiment = 'bullish'
            elif sentiment_score < -0.2:
                market_sentiment = 'bearish'
            else:
                market_sentiment = 'neutral'
            
            return {
                'market_sentiment': market_sentiment,
                'sentiment_score': sentiment_score,
                'confidence': summary.get('avg_confidence', 0),
                'sample_count': summary.get('total_count', 0),
                'positive_percentage': summary.get('positive_percentage', 0),
                'negative_percentage': summary.get('negative_percentage', 0),
                'neutral_percentage': summary.get('neutral_percentage', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}
    
    def stop_collection(self):
        """Stop sentiment data collection"""
        self.running = False
        logger.info("Sentiment data collection stopped")

def main():
    """Main function to demonstrate sentiment analysis system"""
    try:
        # Initialize sentiment manager
        config = {
            'keywords': ['crypto', 'bitcoin', 'ethereum', 'solana', 'trading'],
            'collection_interval': 300
        }
        
        manager = SentimentManager(config)
        
        # Create sample training data
        sample_data = pd.DataFrame({
            'text': [
                'Bitcoin is going to the moon!',
                'Market crash incoming, sell everything!',
                'Crypto market looks stable today',
                'Great news for Ethereum holders',
                'Another day, another dip in crypto',
                'Bullish sentiment in the market',
                'Bearish outlook for altcoins',
                'Neutral market conditions',
                'Strong performance from DeFi tokens',
                'Weak trading volume today'
            ],
            'sentiment': [1, -1, 0, 1, -1, 1, -1, 0, 1, -1]
        })
        
        # Train model
        logger.info("Training sentiment model...")
        metrics = manager.train_sentiment_model(sample_data)
        
        if metrics:
            logger.info(f"Model training completed: {metrics}")
        
        # Test sentiment analysis
        test_texts = [
            "Bitcoin price is surging!",
            "Market looks bearish today",
            "Neutral market conditions"
        ]
        
        for text in test_texts:
            prediction = manager.analyze_text_sentiment(text)
            if prediction:
                logger.info(f"Text: {text}")
                logger.info(f"Sentiment: {prediction.predicted_sentiment}, Confidence: {prediction.confidence:.3f}")
        
        # Get market sentiment
        market_sentiment = manager.get_market_sentiment()
        logger.info(f"Market sentiment: {market_sentiment}")
        
        logger.info("Sentiment analysis system test completed!")
        
    except Exception as e:
        logger.error(f"Error in main sentiment analysis function: {e}")

if __name__ == "__main__":
    main()
