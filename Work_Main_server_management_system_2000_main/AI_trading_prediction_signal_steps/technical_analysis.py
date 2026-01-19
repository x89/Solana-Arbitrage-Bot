#!/usr/bin/env python3
"""
AI Trading Prediction Signal Bot - Technical Analysis Module
Comprehensive technical analysis and indicator calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

# Import existing technical indicators
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from Main_server_management_system.technical_indicators import calculate_all_indicators
except ImportError:
    print("Warning: Could not import existing technical indicators. Using built-in calculations.")
    calculate_all_indicators = None

from config import Config, SignalStrength

class TechnicalSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TechnicalAnalysis:
    """Technical analysis results"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_position: float  # Position within Bollinger Bands (0-1)
    bb_width: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    volume_ratio: float
    price_trend: float  # Short-term price trend
    signal: TechnicalSignal
    confidence: float
    strength: SignalStrength

class TechnicalAnalyzer:
    """Technical analysis and signal generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Technical indicator parameters
        self.rsi_period = config.ai.RSI_PERIOD
        self.rsi_overbought = config.ai.RSI_OVERBOUGHT
        self.rsi_oversold = config.ai.RSI_OVERSOLD
        
        self.macd_fast = config.ai.MACD_FAST
        self.macd_slow = config.ai.MACD_SLOW
        self.macd_signal = config.ai.MACD_SIGNAL
        
        self.bb_period = config.ai.BB_PERIOD
        self.bb_std = config.ai.BB_STD
        
        # Signal thresholds
        self.signal_thresholds = {
            'rsi_extreme': 0.8,  # RSI > 80 or < 20
            'macd_crossover': 0.6,  # MACD crossover strength
            'bb_extreme': 0.7,  # Price at BB extremes
            'trend_strength': 0.5  # Price trend strength
        }
    
    def calculate_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            df = market_data.copy()
            
            # Use existing technical indicators if available
            if calculate_all_indicators:
                prices = np.array(df['close'].values, dtype=float)
                volumes = np.array(df['volume'].values, dtype=float)
                
                indicators = calculate_all_indicators(prices, volumes)
                
                for indicator_name, values in indicators.items():
                    if len(values) == len(df):
                        df[f'{indicator_name}'] = values
            else:
                # Calculate indicators manually
                df = self._calculate_manual_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return market_data.copy()
    
    def _calculate_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators manually"""
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Price trends
            df['price_change'] = df['close'].pct_change()
            df['price_trend_5'] = df['close'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating manual indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Default to neutral RSI
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=self.macd_fast).mean()
            ema_slow = prices.ewm(span=self.macd_slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=self.macd_signal).mean()
            macd_histogram = macd - macd_signal
            
            return macd, macd_signal, macd_histogram
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            bb_middle = prices.rolling(self.bb_period).mean()
            bb_std = prices.rolling(self.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.bb_std)
            bb_lower = bb_middle - (bb_std * self.bb_std)
            
            return bb_upper, bb_middle, bb_lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    def generate_signal(self, technical_data: pd.DataFrame) -> Optional[TechnicalSignal]:
        """Generate technical trading signal"""
        try:
            if technical_data.empty or len(technical_data) < 50:
                return None
            
            latest = technical_data.iloc[-1]
            prev = technical_data.iloc[-2]
            
            # Calculate signal components
            rsi_signal = self._analyze_rsi_signal(latest['rsi'])
            macd_signal = self._analyze_macd_signal(latest['macd'], latest['macd_signal'], 
                                                   latest['macd_histogram'], prev['macd_histogram'])
            bb_signal = self._analyze_bollinger_signal(latest['close'], latest['bb_upper'], 
                                                      latest['bb_lower'], latest['bb_middle'])
            trend_signal = self._analyze_trend_signal(latest['close'], latest['sma_20'], 
                                                    latest['sma_50'], latest['ema_12'], latest['ema_26'])
            volume_signal = self._analyze_volume_signal(latest['volume_ratio'])
            
            # Combine signals
            combined_signal = self._combine_technical_signals(
                rsi_signal, macd_signal, bb_signal, trend_signal, volume_signal
            )
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error generating technical signal: {e}")
            return None
    
    def _analyze_rsi_signal(self, rsi: float) -> int:
        """Analyze RSI signal (-2 to 2 scale)"""
        if rsi > self.rsi_overbought:
            return -2  # Strong sell
        elif rsi > 60:
            return -1  # Sell
        elif rsi < self.rsi_oversold:
            return 2   # Strong buy
        elif rsi < 40:
            return 1   # Buy
        else:
            return 0   # Neutral
    
    def _analyze_macd_signal(self, macd: float, macd_signal: float, 
                           macd_hist: float, prev_macd_hist: float) -> int:
        """Analyze MACD signal (-2 to 2 scale)"""
        # MACD crossover analysis
        if macd_hist > 0 and prev_macd_hist <= 0:
            return 2   # Strong buy (bullish crossover)
        elif macd_hist < 0 and prev_macd_hist >= 0:
            return -2  # Strong sell (bearish crossover)
        elif macd > macd_signal and macd_hist > 0:
            return 1   # Buy
        elif macd < macd_signal and macd_hist < 0:
            return -1  # Sell
        else:
            return 0   # Neutral
    
    def _analyze_bollinger_signal(self, price: float, bb_upper: float, 
                                bb_lower: float, bb_middle: float) -> int:
        """Analyze Bollinger Bands signal (-2 to 2 scale)"""
        if bb_upper == bb_lower:  # Avoid division by zero
            return 0
        
        bb_position = (price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position > 0.95:
            return -2  # Strong sell (price at upper band)
        elif bb_position > 0.8:
            return -1  # Sell
        elif bb_position < 0.05:
            return 2   # Strong buy (price at lower band)
        elif bb_position < 0.2:
            return 1   # Buy
        else:
            return 0   # Neutral
    
    def _analyze_trend_signal(self, price: float, sma_20: float, sma_50: float, 
                            ema_12: float, ema_26: float) -> int:
        """Analyze trend signal (-2 to 2 scale)"""
        signals = []
        
        # SMA trend
        if sma_20 > sma_50:
            signals.append(1)
        elif sma_20 < sma_50:
            signals.append(-1)
        else:
            signals.append(0)
        
        # EMA trend
        if ema_12 > ema_26:
            signals.append(1)
        elif ema_12 < ema_26:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Price vs moving averages
        if price > sma_20 and price > sma_50:
            signals.append(1)
        elif price < sma_20 and price < sma_50:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Average the signals
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.5:
            return 2   # Strong buy
        elif avg_signal > 0.2:
            return 1   # Buy
        elif avg_signal < -0.5:
            return -2  # Strong sell
        elif avg_signal < -0.2:
            return -1 # Sell
        else:
            return 0   # Neutral
    
    def _analyze_volume_signal(self, volume_ratio: float) -> int:
        """Analyze volume signal (-1 to 1 scale)"""
        if volume_ratio > 1.5:
            return 1   # High volume (confirms signal)
        elif volume_ratio < 0.5:
            return -1  # Low volume (weakens signal)
        else:
            return 0   # Normal volume
    
    def _combine_technical_signals(self, rsi_signal: int, macd_signal: int, 
                                 bb_signal: int, trend_signal: int, volume_signal: int) -> TechnicalSignal:
        """Combine all technical signals"""
        # Weight the signals
        weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bb': 0.2,
            'trend': 0.25,
            'volume': 0.1
        }
        
        combined_score = (
            rsi_signal * weights['rsi'] +
            macd_signal * weights['macd'] +
            bb_signal * weights['bb'] +
            trend_signal * weights['trend'] +
            volume_signal * weights['volume']
        )
        
        # Convert to TechnicalSignal enum
        if combined_score > 1.5:
            return TechnicalSignal.STRONG_BUY
        elif combined_score > 0.5:
            return TechnicalSignal.BUY
        elif combined_score < -1.5:
            return TechnicalSignal.STRONG_SELL
        elif combined_score < -0.5:
            return TechnicalSignal.SELL
        else:
            return TechnicalSignal.NEUTRAL
    
    def calculate_signal_strength(self, technical_data: pd.DataFrame) -> SignalStrength:
        """Calculate signal strength based on technical analysis"""
        try:
            if technical_data.empty:
                return SignalStrength.WEAK
            
            latest = technical_data.iloc[-1]
            
            # Calculate strength based on indicator extremes
            strength_score = 0
            
            # RSI strength
            rsi_extreme = abs(latest['rsi'] - 50) / 50
            strength_score += rsi_extreme * 0.3
            
            # MACD strength
            macd_strength = abs(latest['macd_histogram']) / (latest['close'] * 0.01)
            strength_score += min(macd_strength, 1.0) * 0.3
            
            # Bollinger Bands strength
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            bb_strength = abs(bb_position - 0.5) * 2
            strength_score += bb_strength * 0.2
            
            # Volume strength
            volume_strength = min(latest['volume_ratio'], 2.0) / 2.0
            strength_score += volume_strength * 0.2
            
            # Convert to SignalStrength enum
            if strength_score > 0.8:
                return SignalStrength.VERY_STRONG
            elif strength_score > 0.6:
                return SignalStrength.STRONG
            elif strength_score > 0.4:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
                
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return SignalStrength.WEAK
    
    def calculate_signal_confidence(self, technical_data: pd.DataFrame) -> float:
        """Calculate signal confidence based on technical analysis"""
        try:
            if technical_data.empty:
                return 0.0
            
            latest = technical_data.iloc[-1]
            
            # Calculate confidence based on indicator agreement
            confidence_factors = []
            
            # RSI confidence
            rsi_conf = 1.0 - abs(latest['rsi'] - 50) / 50
            confidence_factors.append(rsi_conf)
            
            # MACD confidence
            macd_conf = 1.0 - abs(latest['macd_histogram']) / (latest['close'] * 0.01)
            confidence_factors.append(max(0, min(1, macd_conf)))
            
            # Bollinger Bands confidence
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            bb_conf = 1.0 - abs(bb_position - 0.5) * 2
            confidence_factors.append(max(0, min(1, bb_conf)))
            
            # Volume confidence
            volume_conf = min(latest['volume_ratio'], 2.0) / 2.0
            confidence_factors.append(volume_conf)
            
            # Average confidence
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            
            return max(0.0, min(1.0, avg_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.0
    
    def get_technical_analysis(self, market_data: pd.DataFrame) -> Optional[TechnicalAnalysis]:
        """Get comprehensive technical analysis"""
        try:
            # Calculate indicators
            technical_data = self.calculate_indicators(market_data)
            
            if technical_data.empty:
                return None
            
            latest = technical_data.iloc[-1]
            
            # Generate signal
            signal = self.generate_signal(technical_data)
            if not signal:
                return None
            
            # Calculate Bollinger Band position
            bb_position = 0.5
            if latest['bb_upper'] != latest['bb_lower']:
                bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            
            # Calculate Bollinger Band width
            bb_width = 0.0
            if latest['bb_middle'] != 0:
                bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
            
            # Calculate price trend
            price_trend = latest.get('price_trend_5', 0.0)
            
            return TechnicalAnalysis(
                rsi=latest['rsi'],
                macd=latest['macd'],
                macd_signal=latest['macd_signal'],
                macd_histogram=latest['macd_histogram'],
                bb_position=bb_position,
                bb_width=bb_width,
                sma_20=latest['sma_20'],
                sma_50=latest['sma_50'],
                ema_12=latest['ema_12'],
                ema_26=latest['ema_26'],
                volume_ratio=latest['volume_ratio'],
                price_trend=price_trend,
                signal=signal,
                confidence=self.calculate_signal_confidence(technical_data),
                strength=self.calculate_signal_strength(technical_data)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting technical analysis: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    from config import config
    
    # Initialize technical analyzer
    analyzer = TechnicalAnalyzer(config)
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    # Get technical analysis
    analysis = analyzer.get_technical_analysis(sample_data)
    
    if analysis:
        print("Technical Analysis Results:")
        print(f"RSI: {analysis.rsi:.2f}")
        print(f"MACD: {analysis.macd:.4f}")
        print(f"Signal: {analysis.signal.value}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Strength: {analysis.strength.name}")
    else:
        print("No technical analysis available")
