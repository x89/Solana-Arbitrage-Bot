#!/usr/bin/env python3
"""
Indicator Calculator Module
Comprehensive technical indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Some technical indicators will use fallback calculations.")
    talib = None

class IndicatorCalculator:
    """Calculate comprehensive technical indicators"""
    
    def __init__(self):
        self.calculated_indicators = {}
        self.advanced_indicators = None
    
    def get_advanced_indicators(self):
        """Get advanced indicators instance"""
        if self.advanced_indicators is None:
            try:
                from advanced_indicators import AdvancedIndicators
                self.advanced_indicators = AdvancedIndicators()
            except ImportError:
                logger.warning("Advanced indicators module not available")
        return self.advanced_indicators
    
    def calculate_all(self, df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
        """Calculate all available indicators"""
        try:
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for indicator calculation")
                return df
            
            df_copy = df.copy()
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_copy.columns for col in required):
                logger.error("Missing required columns")
                return df
            
            # Convert to numpy arrays
            open_prices = df_copy['open'].values
            high_prices = df_copy['high'].values
            low_prices = df_copy['low'].values
            close_prices = df_copy['close'].values
            volume = df_copy['volume'].values
            
            # Calculate indicators
            df_copy = self._calculate_trend_indicators(df_copy, open_prices, high_prices, low_prices, close_prices)
            df_copy = self._calculate_oscillators(df_copy, high_prices, low_prices, close_prices, volume)
            df_copy = self._calculate_momentum_indicators(df_copy, high_prices, low_prices, close_prices)
            df_copy = self._calculate_volatility_indicators(df_copy, high_prices, low_prices, close_prices)
            df_copy = self._calculate_volume_indicators(df_copy, high_prices, low_prices, close_prices, volume)
            
            # Calculate advanced indicators
            if include_advanced:
                try:
                    df_copy = self.calculate_advanced_indicators(df_copy)
                except Exception as e:
                    logger.warning(f"Could not calculate advanced indicators: {e}")
            
            logger.info(f"Calculated {len([col for col in df_copy.columns if col not in df.columns])} indicators")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        try:
            if not TALIB_AVAILABLE:
                logger.warning("TA-Lib not available. Using fallback RSI calculation.")
                return self._calculate_rsi_fallback(df, period)
            return talib.RSI(df['close'].values, timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return self._calculate_rsi_fallback(df, period)
    
    def _calculate_rsi_fallback(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Fallback RSI calculation"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
        except Exception as e:
            logger.error(f"Error in fallback RSI calculation: {e}")
            return np.full(len(df), 50.0)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if not TALIB_AVAILABLE:
                logger.warning("TA-Lib not available. Using fallback MACD calculation.")
                return self._calculate_macd_fallback(df, fast, slow, signal)
            macd, signal_line, histogram = talib.MACD(
                df['close'].values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return self._calculate_macd_fallback(df, fast, slow, signal)
    
    def _calculate_macd_fallback(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """Fallback MACD calculation"""
        try:
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd.fillna(0).values,
                'signal': signal_line.fillna(0).values,
                'histogram': histogram.fillna(0).values
            }
        except Exception as e:
            logger.error(f"Error in fallback MACD calculation: {e}")
            return {'macd': np.zeros(len(df)), 'signal': np.zeros(len(df)), 'histogram': np.zeros(len(df))}
    
    def _calculate_trend_indicators(self, df: pd.DataFrame, open_prices: np.ndarray, 
                                   high_prices: np.ndarray, low_prices: np.ndarray, 
                                   close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate trend indicators"""
        try:
            if not TALIB_AVAILABLE:
                # Use pandas-based calculations
                for period in [5, 10, 20, 50, 100, 200]:
                    df[f'sma_{period}'] = pd.Series(close_prices).rolling(window=period).mean().values
                    df[f'ema_{period}'] = pd.Series(close_prices).ewm(span=period, adjust=False).mean().values
                
                logger.info("Using fallback moving averages calculations")
                return df
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
            # WMA (Weighted Moving Average)
            for period in [10, 20, 50]:
                df[f'wma_{period}'] = talib.WMA(close_prices, timeperiod=period)
            
            # DEMA (Double Exponential Moving Average)
            for period in [10, 20]:
                df[f'dema_{period}'] = talib.DEMA(close_prices, timeperiod=period)
            
            # TEMA (Triple Exponential Moving Average)
            for period in [10, 20]:
                df[f'tema_{period}'] = talib.TEMA(close_prices, timeperiod=period)
            
            # KAMA (Kaufman Adaptive Moving Average)
            df['kama_20'] = talib.KAMA(close_prices, timeperiod=20)
            
            # TRIMA (Triangular Moving Average)
            df['trima_20'] = talib.TRIMA(close_prices, timeperiod=20)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(high_prices, low_prices)
            
            # Directional Movement Index
            df['adx_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
            df['aroon_down'] = aroon_down
            df['aroon_up'] = aroon_up
            df['aroon_oscillator'] = aroon_up - aroon_down
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return df
    
    def _calculate_oscillators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                              low_prices: np.ndarray, close_prices: np.ndarray, 
                              volume: np.ndarray) -> pd.DataFrame:
        """Calculate oscillators"""
        try:
            # RSI
            for period in [9, 14, 21]:
                df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            df['stoch_signal'] = slowk - slowd
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
            
            # CCI (Commodity Channel Index)
            df['cci_20'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)
            
            # ROC (Rate of Change)
            for period in [10, 20]:
                df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
            
            # MFI (Money Flow Index)
            df['mfi_14'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # Ultimate Oscillator
            df['ult_osc'] = talib.ULTOSC(high_prices, low_prices, close_prices)
            
            # Awesome Oscillator
            df['ao'] = self._calculate_awesome_oscillator(high_prices, low_prices)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating oscillators: {e}")
            return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                      low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate momentum indicators"""
        try:
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # Momentum
            for period in [10, 20]:
                df[f'momentum_{period}'] = talib.MOM(close_prices, timeperiod=period)
            
            # ADX
            df['adx_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Balance of Power
            df['bop'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)
            
            # CMO (Chande Momentum Oscillator)
            df['cmo_14'] = talib.CMO(close_prices, timeperiod=14)
            
            # TRIX
            df['trix'] = talib.TRIX(close_prices)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                         low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate volatility indicators"""
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # ATR (Average True Range)
            for period in [14, 20]:
                df[f'atr_{period}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            
            # True Range
            df['true_range'] = talib.TRANGE(high_prices, low_prices, close_prices)
            
            # NATR (Normalized ATR)
            df['natr_14'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Keltner Channels
            df = self._calculate_keltner_channels(df, high_prices, low_prices, close_prices)
            
            # Donchian Channels
            df = self._calculate_donchian_channels(df, high_prices, low_prices, close_prices)
            
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
            df['volume_ratio'] = volume / df['volume_sma_20'].values
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = (close_prices * volume).cumsum() / volume.cumsum()
            
            # Price Volume Trend
            df['pvt'] = self._calculate_price_volume_trend(close_prices, volume)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return df
    
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
    
    def _calculate_donchian_channels(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate Donchian Channels"""
        try:
            period = 20
            df['dc_upper'] = talib.MAX(high_prices, timeperiod=period)
            df['dc_lower'] = talib.MIN(low_prices, timeperiod=period)
            df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Donchian Channels: {e}")
            return df
    
    def _calculate_awesome_oscillator(self, high_prices: np.ndarray, low_prices: np.ndarray) -> np.ndarray:
        """Calculate Awesome Oscillator"""
        try:
            ao = talib.SMA((high_prices + low_prices) / 2, timeperiod=5) - talib.SMA((high_prices + low_prices) / 2, timeperiod=34)
            return ao
        except Exception as e:
            logger.error(f"Error calculating Awesome Oscillator: {e}")
            return np.zeros(len(high_prices))
    
    def _calculate_price_volume_trend(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Price Volume Trend"""
        try:
            pvt = np.zeros(len(close_prices))
            pvt[0] = volume[0]
            
            for i in range(1, len(close_prices)):
                price_change = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                pvt[i] = pvt[i-1] + (price_change * volume[i])
            
            return pvt
        except Exception as e:
            logger.error(f"Error calculating Price Volume Trend: {e}")
            return np.zeros(len(close_prices))
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced indicators including CCI, Bollinger, Supertrend"""
        try:
            df_copy = df.copy()
            advanced = self.get_advanced_indicators()
            
            if advanced is None:
                logger.warning("Advanced indicators not available")
                return df_copy
            
            # CCI
            logger.info("Calculating CCI...")
            cci = advanced.calculate_cci(df_copy, period=20)
            df_copy['cci'] = cci
            
            # Bollinger Bands
            logger.info("Calculating Bollinger Bands...")
            bb = advanced.calculate_bollinger_bands(df_copy, period=20)
            df_copy['bb_upper_advanced'] = bb['upper']
            df_copy['bb_middle_advanced'] = bb['middle']
            df_copy['bb_lower_advanced'] = bb['lower']
            df_copy['bb_width_advanced'] = bb['width']
            df_copy['bb_position_advanced'] = bb['position']
            
            # Supertrend
            logger.info("Calculating Supertrend...")
            st = advanced.calculate_supertrend(df_copy, period=10, multiplier=3.0)
            df_copy['supertrend'] = st['supertrend']
            df_copy['supertrend_direction'] = st['direction']
            df_copy['supertrend_upper'] = st['upper_band']
            df_copy['supertrend_lower'] = st['lower_band']
            
            logger.info("Advanced indicators calculated successfully")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {e}")
            return df
    
    def get_dtfx_zones(self, df: pd.DataFrame, structure_len: int = 10) -> List:
        """Get DTFX Algo Zones"""
        try:
            advanced = self.get_advanced_indicators()
            if advanced is None:
                return []
            
            return advanced.calculate_dtfx_zones(df, structure_len=structure_len)
            
        except Exception as e:
            logger.error(f"Error getting DTFX zones: {e}")
            return []
    
    def get_zigzag_points(self, df: pd.DataFrame, momentum_type: str = 'macd') -> List:
        """Get ZigZag points"""
        try:
            advanced = self.get_advanced_indicators()
            if advanced is None:
                return []
            
            return advanced.calculate_zigzag_momentum(df, momentum_type=momentum_type)
            
        except Exception as e:
            logger.error(f"Error getting ZigZag points: {e}")
            return []

