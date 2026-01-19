"""
Real-time OHLCV candle generation from tick data
Optimized for incremental computation
"""

from datetime import datetime
from typing import Optional, Dict
from collections import deque
import numpy as np


class CandleGenerator:
    """
    Generate OHLCV candles from tick data in real-time
    Supports multiple timeframes: 1s, 1m, 5m, 15m, 1h
    """
    
    def __init__(self, symbol: str, timeframe: str = "1m", max_candles: int = 1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_candles = max_candles
        
        # Current candle state
        self.current_candle = None
        self.candle_start_time = None
        
        # Historical candles buffer
        self.candles = deque(maxlen=max_candles)
        
        # Timeframe in seconds
        self.timeframe_seconds = self._get_timeframe_seconds()
    
    def _get_timeframe_seconds(self) -> int:
        """Convert timeframe string to seconds"""
        mapping = {
            "1s": 1,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        return mapping.get(self.timeframe, 60)
    
    def process_tick(self, tick: Dict) -> Optional[Dict]:
        """
        Process a tick and return completed candle if ready
        
        Args:
            tick: Dictionary with 'timestamp', 'price', 'volume'
        
        Returns:
            Completed candle dict or None if still forming
        """
        tick_timestamp = tick['timestamp']
        price = tick['price']
        volume = tick.get('volume', 0.0)
        
        # Initialize first candle
        if self.current_candle is None:
            self._start_new_candle(tick_timestamp, price, volume)
            return None
        
        # Check if we need to close current candle
        elapsed = (tick_timestamp - self.candle_start_time).total_seconds()
        
        if elapsed >= self.timeframe_seconds:
            # Close and save completed candle
            completed_candle = self._finalize_candle()
            
            # Start new candle
            self._start_new_candle(tick_timestamp, price, volume)
            
            return completed_candle
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += volume
            self.current_candle['tick_count'] += 1
            
            return None
    
    def _start_new_candle(self, timestamp: datetime, price: float, volume: float):
        """Start a new candle"""
        self.current_candle = {
            'timestamp': timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
            'tick_count': 1
        }
        self.candle_start_time = timestamp
    
    def _finalize_candle(self) -> Dict:
        """Finalize and return the current candle"""
        completed = self.current_candle.copy()
        
        # Add metadata
        completed['duration'] = (completed['timestamp'] - self.candle_start_time).total_seconds()
        
        # Store in buffer
        self.candles.append(completed)
        
        return completed
    
    def get_latest_candles(self, n: int = 100) -> list:
        """Get latest N candles"""
        return list(self.candles)[-n:]
    
    def get_candle_array(self, n: int = 100) -> np.ndarray:
        """Get candles as numpy array for model input"""
        candles_list = self.get_latest_candles(n)
        
        # Convert to OHLCV array: (timesteps, features=5)
        arr = []
        for candle in candles_list:
            arr.append([
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume']
            ])
        
        return np.array(arr, dtype=np.float32)
    
    def get_current_state(self) -> Optional[Dict]:
        """Get current candle being formed"""
        return self.current_candle
    
    def reset(self):
        """Reset generator"""
        self.current_candle = None
        self.candle_start_time = None
        self.candles.clear()


class MultiTimeframeGenerator:
    """
    Generate candles for multiple timeframes simultaneously
    Efficient for multi-model predictions
    """
    
    def __init__(self, symbol: str, timeframes: list = None):
        self.symbol = symbol
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h"]
        
        # Create generator for each timeframe
        self.generators = {
            tf: CandleGenerator(symbol, tf) for tf in self.timeframes
        }
    
    def process_tick(self, tick: Dict) -> Dict[str, Optional[Dict]]:
        """
        Process tick for all timeframes
        
        Returns:
            Dict mapping timeframe -> completed candle (or None)
        """
        results = {}
        
        for tf, generator in self.generators.items():
            completed = generator.process_tick(tick)
            if completed:
                results[tf] = completed
        
        return results
    
    def get_candles(self, timeframe: str, n: int = 100) -> list:
        """Get candles for a specific timeframe"""
        return self.generators[timeframe].get_latest_candles(n)
    
    def get_all_candles(self, n: int = 100) -> Dict[str, list]:
        """Get candles for all timeframes"""
        return {tf: gen.get_latest_candles(n) for tf, gen in self.generators.items()}
    
    def reset(self):
        """Reset all generators"""
        for generator in self.generators.values():
            generator.reset()

