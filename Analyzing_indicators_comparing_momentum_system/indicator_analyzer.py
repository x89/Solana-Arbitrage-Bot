#!/usr/bin/env python3
"""
Analyzing Indicators & Comparing Momentum System
Advanced technical indicator analysis and momentum comparison system including:
- Comprehensive technical indicator calculation
- Momentum analysis and comparison
- Multi-timeframe indicator analysis
- Indicator correlation analysis
- Momentum-based trading signals
- Performance comparison and optimization
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Some technical indicators will be unavailable.")
    talib = None

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta not available. Some indicators will use fallbacks.")

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas_ta not available. Some indicators will use fallbacks.")

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some features will be unavailable.")

import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class IndicatorValue:
    """Indicator value structure"""
    name: str
    value: float
    timestamp: datetime
    timeframe: str
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    metadata: Dict[str, Any]

@dataclass
class MomentumAnalysis:
    """Momentum analysis result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    momentum_score: float
    trend_direction: str
    momentum_strength: str
    indicators: Dict[str, IndicatorValue]
    correlation_matrix: Dict[str, Dict[str, float]]
    recommendations: List[str]

class TechnicalIndicators:
    """Comprehensive technical indicators calculator"""
    
    def __init__(self):
        self.indicators = {}
        self.calculated_indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for indicator calculation")
                return df
            
            df_copy = df.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_copy.columns:
                    logger.error(f"Required column {col} not found")
                    return df
            
            # Convert to numpy arrays for talib
            open_prices = df_copy['open'].values
            high_prices = df_copy['high'].values
            low_prices = df_copy['low'].values
            close_prices = df_copy['close'].values
            volume = df_copy['volume'].values
            
            # Moving Averages
            df_copy = self._calculate_moving_averages(df_copy, close_prices)
            
            # Oscillators
            df_copy = self._calculate_oscillators(df_copy, high_prices, low_prices, close_prices)
            
            # Momentum Indicators
            df_copy = self._calculate_momentum_indicators(df_copy, close_prices)
            
            # Volatility Indicators
            df_copy = self._calculate_volatility_indicators(df_copy, high_prices, low_prices, close_prices)
            
            # Volume Indicators
            df_copy = self._calculate_volume_indicators(df_copy, high_prices, low_prices, close_prices, volume)
            
            # Trend Indicators
            df_copy = self._calculate_trend_indicators(df_copy, high_prices, low_prices, close_prices)
            
            # Custom Indicators
            df_copy = self._calculate_custom_indicators(df_copy)
            
            logger.info(f"Calculated {len([col for col in df_copy.columns if col not in df.columns])} technical indicators")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate moving averages"""
        try:
            # Simple Moving Averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
            
            # Exponential Moving Averages
            ema_periods = [12, 21, 26, 50]
            for period in ema_periods:
                df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
            # Weighted Moving Average
            df['wma_20'] = talib.WMA(close_prices, timeperiod=20)
            
            # Hull Moving Average
            df['hma_20'] = self._calculate_hull_moving_average(close_prices, 20)
            
            # Kaufman's Adaptive Moving Average
            df['kama_20'] = talib.KAMA(close_prices, timeperiod=20)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return df
    
    def _calculate_oscillators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                             low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate oscillators"""
        try:
            # RSI
            df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            df['rsi_21'] = talib.RSI(close_prices, timeperiod=21)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
            
            # Commodity Channel Index
            df['cci_20'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)
            
            # Rate of Change
            df['roc_10'] = talib.ROC(close_prices, timeperiod=10)
            df['roc_20'] = talib.ROC(close_prices, timeperiod=20)
            
            # Money Flow Index
            df['mfi_14'] = talib.MFI(high_prices, low_prices, close_prices, df['volume'].values, timeperiod=14)
            
            # Ultimate Oscillator
            df['ult_osc'] = talib.ULTOSC(high_prices, low_prices, close_prices)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating oscillators: {e}")
            return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate momentum indicators"""
        try:
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # MACD with different parameters
            macd_fast, macd_signal_fast, macd_hist_fast = talib.MACD(close_prices, fastperiod=5, slowperiod=13, signalperiod=9)
            df['macd_fast'] = macd_fast
            df['macd_signal_fast'] = macd_signal_fast
            
            # Momentum
            df['momentum_10'] = talib.MOM(close_prices, timeperiod=10)
            df['momentum_20'] = talib.MOM(close_prices, timeperiod=20)
            
            # Average Directional Index
            df['adx_14'] = talib.ADX(df['high'].values, df['low'].values, close_prices, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, close_prices, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, close_prices, timeperiod=14)
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(df['high'].values, df['low'].values, timeperiod=14)
            df['aroon_down'] = aroon_down
            df['aroon_up'] = aroon_up
            df['aroon_oscillator'] = aroon_up - aroon_down
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                       low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate volatility indicators"""
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # Average True Range
            df['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df['atr_20'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)
            
            # True Range
            df['true_range'] = talib.TRANGE(high_prices, low_prices, close_prices)
            
            # Normalized Average True Range
            df['natr_14'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Keltner Channels
            df = self._calculate_keltner_channels(df, high_prices, low_prices, close_prices)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                   low_prices: np.ndarray, close_prices: np.ndarray, 
                                   volume: np.ndarray) -> pd.DataFrame:
        """Calculate volume indicators"""
        try:
            # On Balance Volume
            df['obv'] = talib.OBV(close_prices, volume)
            
            # Accumulation/Distribution Line
            df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
            
            # Chaikin A/D Oscillator
            df['adosc'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
            
            # Volume Rate of Change
            df['volume_roc'] = talib.ROC(volume, timeperiod=10)
            
            # Volume Moving Average
            df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma_20']
            
            # Price Volume Trend
            df['pvt'] = self._calculate_price_volume_trend(close_prices, volume)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                  low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate trend indicators"""
        try:
            # Parabolic SAR
            df['sar'] = talib.SAR(high_prices, low_prices)
            
            # Ichimoku Cloud
            df = self._calculate_ichimoku_cloud(df, high_prices, low_prices, close_prices)
            
            # Linear Regression
            df['linear_reg'] = talib.LINEARREG(close_prices, timeperiod=14)
            df['linear_reg_slope'] = talib.LINEARREG_SLOPE(close_prices, timeperiod=14)
            df['linear_reg_angle'] = talib.LINEARREG_ANGLE(close_prices, timeperiod=14)
            
            # Standard Deviation
            df['stddev_20'] = talib.STDDEV(close_prices, timeperiod=20)
            
            # Variance
            df['var_20'] = talib.VAR(close_prices, timeperiod=20)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return df
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom indicators"""
        try:
            # Fisher Transform
            df = self._calculate_fisher_transform(df)
            
            # Supertrend
            df = self._calculate_supertrend(df)
            
            # Volume Profile
            df = self._calculate_volume_profile(df)
            
            # Market Structure
            df = self._calculate_market_structure(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
            return df
    
    def _calculate_hull_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Hull Moving Average"""
        try:
            wma_half = talib.WMA(prices, timeperiod=period//2)
            wma_full = talib.WMA(prices, timeperiod=period)
            wma_diff = 2 * wma_half - wma_full
            hma = talib.WMA(wma_diff, timeperiod=int(np.sqrt(period)))
            return hma
        except Exception as e:
            logger.error(f"Error calculating Hull Moving Average: {e}")
            return np.full(len(prices), np.nan)
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                  low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        try:
            ema_20 = talib.EMA(close_prices, timeperiod=20)
            atr_20 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)
            
            df['kc_upper'] = ema_20 + (2 * atr_20)
            df['kc_middle'] = ema_20
            df['kc_lower'] = ema_20 - (2 * atr_20)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {e}")
            return df
    
    def _calculate_ichimoku_cloud(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                 low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        try:
            # Tenkan-sen (9-period high + low) / 2
            tenkan_high = talib.MAX(high_prices, timeperiod=9)
            tenkan_low = talib.MIN(low_prices, timeperiod=9)
            df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (26-period high + low) / 2
            kijun_high = talib.MAX(high_prices, timeperiod=26)
            kijun_low = talib.MIN(low_prices, timeperiod=26)
            df['kijun_sen'] = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Tenkan + Kijun) / 2, shifted 26 periods
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (52-period high + low) / 2, shifted 26 periods
            senkou_high = talib.MAX(high_prices, timeperiod=52)
            senkou_low = talib.MIN(low_prices, timeperiod=52)
            df['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(26)
            
            # Chikou Span (Close, shifted -26 periods)
            df['chikou_span'] = close_prices.shift(-26)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {e}")
            return df
    
    def _calculate_fisher_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fisher Transform"""
        try:
            high_10 = talib.MAX(df['high'].values, timeperiod=10)
            low_10 = talib.MIN(df['low'].values, timeperiod=10)
            
            # Raw value
            raw_value = 2 * ((df['close'] - low_10) / (high_10 - low_10) - 0.5)
            raw_value = np.clip(raw_value, -0.9999, 0.9999)
            
            # Smoothed raw value
            smoothed_raw = talib.EMA(raw_value.values, timeperiod=5)
            
            # Fisher Transform
            fisher_transform = 0.5 * np.log((1 + smoothed_raw) / (1 - smoothed_raw))
            
            df['fisher_transform'] = fisher_transform
            df['fisher_signal'] = talib.EMA(fisher_transform, timeperiod=3)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Fisher Transform: {e}")
            return df
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend indicator"""
        try:
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=10)
            hl2 = (df['high'] + df['low']) / 2
            
            upper_band = hl2 + (3 * atr)
            lower_band = hl2 - (3 * atr)
            
            # Calculate Supertrend
            supertrend = np.zeros(len(df))
            direction = np.zeros(len(df))
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] <= lower_band[i-1]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
                elif df['close'].iloc[i] >= upper_band[i-1]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
            
            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            return df
    
    def _calculate_price_volume_trend(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Price Volume Trend"""
        try:
            pvt = np.zeros(len(close_prices))
            pvt[0] = volume[0]
            
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    pvt[i] = pvt[i-1] + volume[i]
                elif close_prices[i] < close_prices[i-1]:
                    pvt[i] = pvt[i-1] - volume[i]
                else:
                    pvt[i] = pvt[i-1]
            
            return pvt
        except Exception as e:
            logger.error(f"Error calculating Price Volume Trend: {e}")
            return np.zeros(len(close_prices))
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Profile"""
        try:
            # Price levels
            price_range = df['high'].max() - df['low'].min()
            num_levels = 20
            level_size = price_range / num_levels
            
            # Volume at each price level
            volume_profile = np.zeros(num_levels)
            
            for i in range(len(df)):
                price_level = int((df['close'].iloc[i] - df['low'].min()) / level_size)
                price_level = min(price_level, num_levels - 1)
                volume_profile[price_level] += df['volume'].iloc[i]
            
            # Find POC (Point of Control)
            poc_level = np.argmax(volume_profile)
            poc_price = df['low'].min() + (poc_level * level_size)
            
            df['volume_poc'] = poc_price
            df['volume_profile'] = volume_profile[poc_level] / volume_profile.sum()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Volume Profile: {e}")
            return df
    
    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Market Structure"""
        try:
            # Higher Highs and Lower Lows
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
            
            # Market structure score
            structure_score = np.zeros(len(df))
            for i in range(1, len(df)):
                if df['higher_high'].iloc[i]:
                    structure_score[i] = structure_score[i-1] + 1
                elif df['lower_low'].iloc[i]:
                    structure_score[i] = structure_score[i-1] - 1
                else:
                    structure_score[i] = structure_score[i-1]
            
            df['market_structure_score'] = structure_score
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Market Structure: {e}")
            return df

class MomentumAnalyzer:
    """Momentum analysis and comparison"""
    
    def __init__(self):
        self.momentum_indicators = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r', 'cci_20', 'roc_10', 'roc_20',
            'momentum_10', 'momentum_20', 'adx_14', 'plus_di', 'minus_di',
            'aroon_up', 'aroon_down', 'aroon_oscillator'
        ]
    
    def analyze_momentum(self, df: pd.DataFrame, symbol: str, timeframe: str) -> MomentumAnalysis:
        """Analyze momentum across multiple indicators"""
        try:
            if df.empty:
                return None
            
            latest_data = df.iloc[-1]
            indicators = {}
            
            # Analyze each momentum indicator
            for indicator in self.momentum_indicators:
                if indicator in df.columns:
                    value = latest_data[indicator]
                    if not pd.isna(value):
                        signal, strength = self._analyze_indicator_signal(df, indicator)
                        
                        indicators[indicator] = IndicatorValue(
                            name=indicator,
                            value=float(value),
                            timestamp=datetime.now(),
                            timeframe=timeframe,
                            signal=signal,
                            strength=strength,
                            metadata={'raw_value': float(value)}
                        )
            
            # Calculate overall momentum score
            momentum_score = self._calculate_momentum_score(indicators)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(df)
            
            # Determine momentum strength
            momentum_strength = self._determine_momentum_strength(momentum_score)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(df)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(indicators, momentum_score, trend_direction)
            
            return MomentumAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                momentum_score=momentum_score,
                trend_direction=trend_direction,
                momentum_strength=momentum_strength,
                indicators=indicators,
                correlation_matrix=correlation_matrix,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return None
    
    def _analyze_indicator_signal(self, df: pd.DataFrame, indicator: str) -> Tuple[str, float]:
        """Analyze signal for a specific indicator"""
        try:
            current_value = df[indicator].iloc[-1]
            prev_value = df[indicator].iloc[-2] if len(df) > 1 else current_value
            
            # RSI analysis
            if indicator.startswith('rsi'):
                if current_value > 70:
                    return 'sell', min(1.0, (current_value - 70) / 30)
                elif current_value < 30:
                    return 'buy', min(1.0, (30 - current_value) / 30)
                else:
                    return 'hold', 0.5
            
            # MACD analysis
            elif indicator == 'macd':
                if current_value > prev_value and current_value > 0:
                    return 'buy', min(1.0, abs(current_value) / 10)
                elif current_value < prev_value and current_value < 0:
                    return 'sell', min(1.0, abs(current_value) / 10)
                else:
                    return 'hold', 0.5
            
            # Stochastic analysis
            elif indicator.startswith('stoch'):
                if current_value > 80:
                    return 'sell', min(1.0, (current_value - 80) / 20)
                elif current_value < 20:
                    return 'buy', min(1.0, (20 - current_value) / 20)
                else:
                    return 'hold', 0.5
            
            # Williams %R analysis
            elif indicator == 'williams_r':
                if current_value > -20:
                    return 'sell', min(1.0, (current_value + 20) / 20)
                elif current_value < -80:
                    return 'buy', min(1.0, (-80 - current_value) / 20)
                else:
                    return 'hold', 0.5
            
            # CCI analysis
            elif indicator == 'cci_20':
                if current_value > 100:
                    return 'buy', min(1.0, current_value / 200)
                elif current_value < -100:
                    return 'sell', min(1.0, abs(current_value) / 200)
                else:
                    return 'hold', 0.5
            
            # ROC analysis
            elif indicator.startswith('roc'):
                if current_value > 0:
                    return 'buy', min(1.0, current_value / 10)
                elif current_value < 0:
                    return 'sell', min(1.0, abs(current_value) / 10)
                else:
                    return 'hold', 0.5
            
            # ADX analysis
            elif indicator == 'adx_14':
                if current_value > 25:
                    return 'strong_trend', min(1.0, current_value / 50)
                elif current_value > 20:
                    return 'moderate_trend', min(1.0, current_value / 25)
                else:
                    return 'weak_trend', current_value / 20
            
            # Default analysis
            else:
                if current_value > prev_value:
                    return 'buy', 0.6
                elif current_value < prev_value:
                    return 'sell', 0.6
                else:
                    return 'hold', 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing indicator signal: {e}")
            return 'hold', 0.5
    
    def _calculate_momentum_score(self, indicators: Dict[str, IndicatorValue]) -> float:
        """Calculate overall momentum score"""
        try:
            if not indicators:
                return 0.0
            
            buy_signals = 0
            sell_signals = 0
            total_strength = 0
            
            for indicator in indicators.values():
                if indicator.signal == 'buy':
                    buy_signals += indicator.strength
                elif indicator.signal == 'sell':
                    sell_signals += indicator.strength
                
                total_strength += indicator.strength
            
            if total_strength == 0:
                return 0.0
            
            # Normalize momentum score (-1 to 1)
            momentum_score = (buy_signals - sell_signals) / total_strength
            
            return max(-1.0, min(1.0, momentum_score))
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        try:
            # Use multiple trend indicators
            trend_signals = []
            
            # Moving average trend
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                if sma_20 > sma_50:
                    trend_signals.append('bullish')
                else:
                    trend_signals.append('bearish')
            
            # MACD trend
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                if macd > macd_signal:
                    trend_signals.append('bullish')
                else:
                    trend_signals.append('bearish')
            
            # ADX trend
            if 'adx_14' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
                adx = df['adx_14'].iloc[-1]
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]
                
                if adx > 20:  # Strong trend
                    if plus_di > minus_di:
                        trend_signals.append('bullish')
                    else:
                        trend_signals.append('bearish')
            
            # Determine overall trend
            bullish_count = trend_signals.count('bullish')
            bearish_count = trend_signals.count('bearish')
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
            
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return 'neutral'
    
    def _determine_momentum_strength(self, momentum_score: float) -> str:
        """Determine momentum strength"""
        try:
            abs_score = abs(momentum_score)
            
            if abs_score > 0.7:
                return 'very_strong'
            elif abs_score > 0.5:
                return 'strong'
            elif abs_score > 0.3:
                return 'moderate'
            elif abs_score > 0.1:
                return 'weak'
            else:
                return 'very_weak'
            
        except Exception as e:
            logger.error(f"Error determining momentum strength: {e}")
            return 'weak'
    
    def _calculate_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between indicators"""
        try:
            # Select numeric columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            indicator_columns = [col for col in numeric_columns if col in self.momentum_indicators]
            
            if len(indicator_columns) < 2:
                return {}
            
            # Calculate correlation matrix
            correlation_df = df[indicator_columns].corr()
            
            # Convert to dictionary
            correlation_matrix = {}
            for col1 in correlation_df.columns:
                correlation_matrix[col1] = {}
                for col2 in correlation_df.columns:
                    correlation_matrix[col1][col2] = float(correlation_df.loc[col1, col2])
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def _generate_recommendations(self, indicators: Dict[str, IndicatorValue], 
                                 momentum_score: float, trend_direction: str) -> List[str]:
        """Generate trading recommendations based on momentum analysis"""
        try:
            recommendations = []
            
            # Overall momentum recommendation
            if momentum_score > 0.5:
                recommendations.append("Strong bullish momentum detected - consider long positions")
            elif momentum_score < -0.5:
                recommendations.append("Strong bearish momentum detected - consider short positions")
            else:
                recommendations.append("Neutral momentum - wait for clearer signals")
            
            # Trend-based recommendations
            if trend_direction == 'bullish':
                recommendations.append("Overall trend is bullish - favor long positions")
            elif trend_direction == 'bearish':
                recommendations.append("Overall trend is bearish - favor short positions")
            else:
                recommendations.append("Trend is neutral - use range trading strategies")
            
            # Individual indicator recommendations
            strong_signals = [ind for ind in indicators.values() if ind.strength > 0.7]
            
            for signal in strong_signals:
                if signal.signal == 'buy':
                    recommendations.append(f"Strong buy signal from {signal.name}")
                elif signal.signal == 'sell':
                    recommendations.append(f"Strong sell signal from {signal.name}")
            
            # Risk management recommendations
            if abs(momentum_score) > 0.8:
                recommendations.append("High momentum detected - use smaller position sizes")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

class IndicatorDatabase:
    """Database management for indicator data"""
    
    def __init__(self, db_path: str = "indicators_analysis.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize indicator database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicator_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                signal TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS momentum_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                momentum_score REAL NOT NULL,
                trend_direction TEXT NOT NULL,
                momentum_strength TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                indicators_data TEXT NOT NULL,
                correlation_matrix TEXT,
                recommendations TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_indicator_values(self, symbol: str, timeframe: str, 
                            indicators: Dict[str, IndicatorValue]) -> bool:
        """Save indicator values to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for indicator in indicators.values():
                cursor.execute('''
                    INSERT INTO indicator_values 
                    (symbol, timeframe, indicator_name, value, signal, strength, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timeframe,
                    indicator.name,
                    indicator.value,
                    indicator.signal,
                    indicator.strength,
                    indicator.timestamp,
                    json.dumps(indicator.metadata)
                ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving indicator values: {e}")
            return False
    
    def save_momentum_analysis(self, analysis: MomentumAnalysis) -> bool:
        """Save momentum analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO momentum_analysis 
                (symbol, timeframe, momentum_score, trend_direction, momentum_strength, 
                 timestamp, indicators_data, correlation_matrix, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.symbol,
                analysis.timeframe,
                analysis.momentum_score,
                analysis.trend_direction,
                analysis.momentum_strength,
                analysis.timestamp,
                json.dumps({k: asdict(v) for k, v in analysis.indicators.items()}),
                json.dumps(analysis.correlation_matrix),
                json.dumps(analysis.recommendations)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving momentum analysis: {e}")
            return False

class IndicatorManager:
    """Main indicator analysis manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.indicators_calculator = TechnicalIndicators()
        self.momentum_analyzer = MomentumAnalyzer()
        self.database = IndicatorDatabase()
        
        # Analysis settings
        self.symbols = self.config.get('symbols', ['SOLUSDT', 'BTCUSDT', 'ETHUSDT'])
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h'])
        
        # Running state
        self.running = False
    
    def analyze_symbol(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[MomentumAnalysis]:
        """Analyze indicators and momentum for a symbol"""
        try:
            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None
            
            # Calculate all technical indicators
            df_with_indicators = self.indicators_calculator.calculate_all_indicators(df)
            
            # Analyze momentum
            momentum_analysis = self.momentum_analyzer.analyze_momentum(
                df_with_indicators, symbol, timeframe
            )
            
            if momentum_analysis:
                # Save to database
                self.database.save_momentum_analysis(momentum_analysis)
                
                logger.info(f"Momentum analysis completed for {symbol} {timeframe}: "
                           f"Score={momentum_analysis.momentum_score:.3f}, "
                           f"Trend={momentum_analysis.trend_direction}")
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
    
    def compare_momentum(self, symbol: str, timeframes: List[str]) -> Dict[str, Any]:
        """Compare momentum across multiple timeframes"""
        try:
            momentum_comparison = {}
            
            for timeframe in timeframes:
                # This would fetch data for the specific timeframe
                # For now, create sample data
                sample_data = self._create_sample_data()
                
                analysis = self.analyze_symbol(sample_data, symbol, timeframe)
                
                if analysis:
                    momentum_comparison[timeframe] = {
                        'momentum_score': analysis.momentum_score,
                        'trend_direction': analysis.trend_direction,
                        'momentum_strength': analysis.momentum_strength,
                        'recommendations': analysis.recommendations
                    }
            
            return momentum_comparison
            
        except Exception as e:
            logger.error(f"Error comparing momentum: {e}")
            return {}
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        try:
            # Generate sample OHLCV data
            np.random.seed(42)
            n_samples = 200
            
            base_price = 100
            prices = [base_price]
            
            for i in range(n_samples - 1):
                change = np.random.normal(0, 0.02)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # Create OHLCV data
            data = []
            for i, price in enumerate(prices):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.index = pd.date_range(start=datetime.now() - timedelta(days=n_samples), periods=n_samples, freq='1H')
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return pd.DataFrame()

def main():
    """Main function to demonstrate indicator analysis system"""
    try:
        # Initialize indicator manager
        config = {
            'symbols': ['SOLUSDT', 'BTCUSDT', 'ETHUSDT'],
            'timeframes': ['1m', '5m', '15m', '1h']
        }
        
        manager = IndicatorManager(config)
        
        # Test with sample data
        sample_data = manager._create_sample_data()
        
        # Analyze symbols
        for symbol in config['symbols']:
            logger.info(f"Analyzing {symbol}...")
            
            analysis = manager.analyze_symbol(sample_data, symbol, '1h')
            
            if analysis:
                logger.info(f"Symbol: {analysis.symbol}")
                logger.info(f"Momentum Score: {analysis.momentum_score:.3f}")
                logger.info(f"Trend Direction: {analysis.trend_direction}")
                logger.info(f"Momentum Strength: {analysis.momentum_strength}")
                logger.info(f"Recommendations: {analysis.recommendations[:2]}")  # Show first 2
        
        # Compare momentum across timeframes
        logger.info("Comparing momentum across timeframes...")
        momentum_comparison = manager.compare_momentum('SOLUSDT', ['1m', '5m', '15m', '1h'])
        
        for timeframe, data in momentum_comparison.items():
            logger.info(f"{timeframe}: Score={data['momentum_score']:.3f}, "
                       f"Trend={data['trend_direction']}")
        
        logger.info("Indicator analysis system test completed!")
        
    except Exception as e:
        logger.error(f"Error in main indicator analysis function: {e}")

if __name__ == "__main__":
    main()
