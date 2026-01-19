#!/usr/bin/env python3
"""
Data Collection System - Bitget Integration
Comprehensive data collection infrastructure including:
- Bitget API integration
- Real-time data streaming
- Historical data collection
- Database management
- Data validation and cleaning
- Multi-symbol support
"""

import asyncio
import aiohttp
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sqlite3
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    symbol: str = "SOLUSDT"
    timeframe: str = "15m"
    exchange: str = "bitget"
    batch_size: int = 1000
    storage_path: str = "Datas/auto_realtime_data"
    auto_save: bool = True

class BitgetClient:
    """Bitget API client"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('BITGET_API_KEY')
        self.api_secret = api_secret or os.getenv('BITGET_API_SECRET')
        self.base_url = "https://api.bitget.com"
        self.session = None
    
    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch candle data from Bitget"""
        try:
            url = f"{self.base_url}/api/v2/mix/market/candles"
            
            params = {
                "symbol": symbol,
                "granularity": timeframe,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('code') == '00000':
                        df = pd.DataFrame(data['data'], columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                        return df
                    else:
                        logger.error(f"API error: {data.get('msg')}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return pd.DataFrame()
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest ticker data"""
        try:
            url = f"{self.base_url}/api/v2/mix/market/ticker"
            params = {"productType": "crypto", "symbol": symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('data', {})
                    
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}

class DataValidator:
    """Data validation and quality checks"""
    
    @staticmethod
    def validate_candle_data(df: pd.DataFrame) -> bool:
        """Validate candle data quality"""
        try:
            if df.empty:
                return False
            
            # Check required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for NaN values
            if df[required_cols].isnull().any().any():
                logger.warning("Data contains NaN values")
                return False
            
            # Check OHLC logic
            ohlc_valid = ((df['high'] >= df['low']) & 
                         (df['high'] >= df['open']) & 
                         (df['high'] >= df['close']) &
                         (df['low'] <= df['open']) & 
                         (df['low'] <= df['close']))
            
            if not ohlc_valid.all():
                logger.warning("OHLC data contains logical errors")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data"""
        try:
            df = df.copy()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Remove outliers (beyond 3 standard deviations)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df

class DataStorage:
    """Data storage and retrieval"""
    
    def __init__(self, db_path: str = "Datas/market_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    last_price REAL NOT NULL,
                    bid_price REAL NOT NULL,
                    ask_price REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    high_24h REAL NOT NULL,
                    low_24h REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_candles(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Save candle data to database"""
        try:
            if df.empty:
                return False
            
            conn = sqlite3.connect(self.db_path)
            
            for _, row in df.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO candles 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, timeframe, row['timestamp'], row['open'], row['high'],
                        row['low'], row['close'], row['volume'], row.get('quote_volume', 0)
                    ))
                except sqlite3.IntegrityError:
                    continue  # Skip duplicates
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved {len(df)} candles to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving candles: {e}")
            return False
    
    def load_candles(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Load candle data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT timestamp, open, high, low, close, volume, quote_volume
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', conn, params=(symbol, timeframe, limit))
            
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading candles: {e}")
            return pd.DataFrame()

class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.client = BitgetClient()
        self.validator = DataValidator()
        self.storage = DataStorage()
        self.running = False
    
    async def collect_historical_data(
        self,
        days: int = 365
    ) -> pd.DataFrame:
        """Collect historical data"""
        try:
            logger.info(f"Collecting {days} days of historical data for {self.config.symbol}")
            
            all_data = []
            end_time = datetime.now()
            
            for i in range(days):
                start_time = end_time - timedelta(days=1)
                
                df = await self.client.fetch_candles(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    limit=1000,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not df.empty:
                    all_data.append(df)
                
                end_time = start_time
                
                if i % 10 == 0:
                    logger.info(f"Collected {i+1}/{days} days")
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')
                
                # Validate data
                if self.validator.validate_candle_data(combined_df):
                    # Save to database
                    if self.config.auto_save:
                        self.storage.save_candles(combined_df, self.config.symbol, self.config.timeframe)
                    
                    logger.info(f"Historical data collection completed: {len(combined_df)} records")
                    return combined_df
                else:
                    logger.error("Data validation failed")
                    return pd.DataFrame()
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return pd.DataFrame()
    
    async def stream_realtime_data(self, callback=None):
        """Stream real-time market data"""
        try:
            self.running = True
            logger.info(f"Starting real-time data stream for {self.config.symbol}")
            
            while self.running:
                # Fetch ticker data
                ticker_data = await self.client.fetch_ticker(self.config.symbol)
                
                if ticker_data and callback:
                    callback(ticker_data)
                
                # Fetch latest candle
                df = await self.client.fetch_candles(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    limit=1
                )
                
                if not df.empty:
                    # Validate
                    if self.validator.validate_candle_data(df):
                        # Save to database
                        if self.config.auto_save:
                            self.storage.save_candles(df, self.config.symbol, self.config.timeframe)
                    
                    if callback:
                        callback(df)
                
                # Wait before next fetch
                await asyncio.sleep(60)  # Fetch every minute
                
        except Exception as e:
            logger.error(f"Error in real-time streaming: {e}")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.running = False
        logger.info("Real-time data streaming stopped")

def main():
    """Main function"""
    try:
        # Configure data collection
        config = DataCollectionConfig(
            symbol="SOLUSDT",
            timeframe="15m",
            batch_size=1000,
            auto_save=True
        )
        
        # Initialize collector
        collector = DataCollector(config)
        
        # Collect historical data
        historical_data = asyncio.run(collector.collect_historical_data(days=30))
        
        if not historical_data.empty:
            logger.info(f"Successfully collected {len(historical_data)} records")
            logger.info(f"Date range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
            
            # Save to file
            output_file = f"Datas/auto_realtime_data/{config.symbol}_historical.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            historical_data.to_json(output_file, orient='records', date_format='iso')
            
            logger.info(f"Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

