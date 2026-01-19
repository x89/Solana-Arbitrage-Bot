"""
Real-time Feature Engine
Computes features incrementally for low-latency inference
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for computed features"""
    timestamp: pd.Timestamp
    symbol: str
    features: np.ndarray
    feature_names: List[str]


class IncrementalFeatureEngine:
    """
    Compute features incrementally without recomputing entire windows
    Optimized for real-time inference with minimal latency
    """
    
    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        
        # Circular buffers for incremental computation
        self.price_buffer = deque(maxlen=window_size)
        self.volume_buffer = deque(maxlen=window_size)
        self.returns_buffer = deque(maxlen=window_size)
        
        # Running statistics (updated incrementally)
        self.running_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
        }
        
        self.feature_cache = {}
    
    def update(self, candle: Dict) -> Dict[str, float]:
        """
        Update with new candle and return features
        
        Args:
            candle: Dict with OHLCV data
        
        Returns:
            Dict of feature_name -> feature_value
        """
        # Extract data
        close = candle['close']
        high = candle['high']
        low = candle['low']
        volume = candle['volume']
        
        # Compute return
        if len(self.price_buffer) > 0:
            return_val = np.log(close / self.price_buffer[-1])
        else:
            return_val = 0.0
        
        # Update buffers
        self.price_buffer.append(close)
        self.volume_buffer.append(volume)
        self.returns_buffer.append(return_val)
        
        # Compute features
        features = self._compute_features(candle, return_val)
        
        return features
    
    def _compute_features(self, candle: Dict, return_val: float) -> Dict[str, float]:
        """Compute all features"""
        features = {}
        
        # Basic OHLCV features
        features.update(self._ohlc_features(candle))
        
        # Return-based features
        features.update(self._return_features())
        
        # Volume features
        features.update(self._volume_features())
        
        # Time features
        features.update(self._time_features(candle))
        
        # Normalized features (z-scores)
        features.update(self._normalized_features())
        
        return features
    
    def _ohlc_features(self, candle: Dict) -> Dict[str, float]:
        """OHLC-based features"""
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        features = {
            'high': high,
            'low': low,
            'close': close,
            'hl_range': high - low,
            'body': abs(candle['close'] - candle.get('open', close)),
            'upper_shadow': high - max(candle.get('open', close), close),
            'lower_shadow': min(candle.get('open', close), close) - low,
        }
        
        return features
    
    def _return_features(self) -> Dict[str, float]:
        """Return-based features"""
        if len(self.returns_buffer) < 2:
            return {'return': 0.0, 'return_ma5': 0.0, 'return_ma20': 0.0}
        
        returns = np.array(self.returns_buffer)
        
        features = {
            'return': returns[-1],
            'return_ma5': np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
            'return_ma20': np.mean(returns[-20:]) if len(returns) >= 20 else 0.0,
            'return_std5': np.std(returns[-5:]) if len(returns) >= 5 else 0.0,
            'return_std20': np.std(returns[-20:]) if len(returns) >= 20 else 0.0,
        }
        
        return features
    
    def _volume_features(self) -> Dict[str, float]:
        """Volume-based features"""
        if len(self.volume_buffer) < 2:
            return {'volume': 0.0, 'volume_ma': 0.0, 'volume_ratio': 0.0}
        
        volumes = np.array(self.volume_buffer)
        
        features = {
            'volume': volumes[-1],
            'volume_ma10': np.mean(volumes[-10:]) if len(volumes) >= 10 else 0.0,
            'volume_ratio': volumes[-1] / (np.mean(volumes[-10:]) + 1e-6),
            'volume_zscore': (volumes[-1] - np.mean(volumes[-10:])) / (np.std(volumes[-10:]) + 1e-6),
        }
        
        return features
    
    def _time_features(self, candle: Dict) -> Dict[str, float]:
        """Time-based cyclical features"""
        timestamp = candle['timestamp']
        
        # Minute of hour (0-59)
        minute = timestamp.minute
        hour = timestamp.hour
        
        # Cyclical encoding (sin/cos)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        features = {
            'minute_sin': minute_sin,
            'minute_cos': minute_cos,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
        }
        
        return features
    
    def _normalized_features(self) -> Dict[str, float]:
        """Z-score normalized features"""
        features = {}
        
        if len(self.returns_buffer) < 10:
            return features
        
        returns = np.array(self.returns_buffer)
        mean_ret = np.mean(returns[-20:])
        std_ret = np.std(returns[-20:])
        
        features['return_zscore'] = (returns[-1] - mean_ret) / (std_ret + 1e-6)
        
        return features
    
    def get_feature_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array for model input"""
        # Ensure consistent order
        ordered_features = [
            'close', 'high', 'low', 'hl_range', 'body',
            'return', 'return_ma5', 'return_ma20',
            'volume', 'volume_ratio', 'volume_zscore',
            'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
            'return_zscore'
        ]
        
        vector = []
        for feat_name in ordered_features:
            if feat_name in feature_dict:
                vector.append(feature_dict[feat_name])
            else:
                vector.append(0.0)
        
        return np.array(vector, dtype=np.float32)
    
    def reset(self):
        """Reset the engine"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.returns_buffer.clear()
        self.feature_cache.clear()

