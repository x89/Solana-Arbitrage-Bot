#!/usr/bin/env python3
"""
Advanced Indicators Module
Implements advanced trading strategies including:
- DTFX Algo Zones with Fibonacci levels
- Momentum-based ZigZag
- CCI (Commodity Channel Index)
- Bollinger Bands
- Supertrend
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using fallback calculations.")
    talib = None

@dataclass
class Zone:
    """Zone structure for DTFX Algo Zones"""
    top: float
    bottom: float
    direction: int  # 1 for bullish, -1 for bearish
    start_bar: int
    end_bar: int
    fib_levels: Dict[str, float]

@dataclass
class ZigZagPoint:
    """ZigZag point structure"""
    price: float
    bar_index: int
    direction: int  # 1 for up, -1 for down
    momentum_signal: str  # 'macd', 'ma', 'qqe'
    
class AdvancedIndicators:
    """Advanced indicator implementations"""
    
    def __init__(self):
        self.zones = []
        self.zigzag_points = []
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)"""
        try:
            if TALIB_AVAILABLE:
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                cci = talib.CCI(high, low, close, timeperiod=period)
                return pd.Series(cci, index=df.index)
            else:
                return self._calculate_cci_fallback(df, period)
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return self._calculate_cci_fallback(df, period)
    
    def _calculate_cci_fallback(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Fallback CCI calculation"""
        try:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - sma) / (0.015 * mad)
            return cci.fillna(0)
        except Exception as e:
            logger.error(f"Error in CCI fallback: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            if TALIB_AVAILABLE:
                close = df['close'].values
                upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
                return {
                    'upper': pd.Series(upper, index=df.index),
                    'middle': pd.Series(middle, index=df.index),
                    'lower': pd.Series(lower, index=df.index),
                    'width': pd.Series((upper - lower) / middle, index=df.index),
                    'position': pd.Series((close - lower) / (upper - lower), index=df.index)
                }
            else:
                return self._calculate_bollinger_fallback(df, period, std_dev)
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return self._calculate_bollinger_fallback(df, period, std_dev)
    
    def _calculate_bollinger_fallback(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Fallback Bollinger Bands calculation"""
        try:
            close = df['close']
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            
            return {
                'upper': upper,
                'middle': sma,
                'lower': lower,
                'width': (upper - lower) / sma,
                'position': (close - lower) / (upper - lower)
            }
        except Exception as e:
            logger.error(f"Error in Bollinger fallback: {e}")
            empty = pd.Series([np.nan] * len(df), index=df.index)
            return {'upper': empty, 'middle': empty, 'lower': empty, 'width': empty, 'position': empty}
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """Calculate Supertrend indicator"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if TALIB_AVAILABLE:
                atr = talib.ATR(high, low, close, timeperiod=period)
            else:
                # Calculate ATR manually
                tr = pd.DataFrame({
                    'hl': df['high'] - df['low'],
                    'hc': np.abs(df['high'] - df['close'].shift(1)),
                    'lc': np.abs(df['low'] - df['close'].shift(1))
                }).max(axis=1)
                atr = tr.rolling(window=period).mean().values
            
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Calculate SuperTrend
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
            
            return {
                'supertrend': pd.Series(supertrend, index=df.index),
                'direction': pd.Series(direction, index=df.index),
                'upper_band': pd.Series(upper_band, index=df.index),
                'lower_band': pd.Series(lower_band, index=df.index)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            empty = pd.Series([np.nan] * len(df), index=df.index)
            return {'supertrend': empty, 'direction': empty, 'upper_band': empty, 'lower_band': empty}
    
    def calculate_dtfx_zones(self, df: pd.DataFrame, structure_len: int = 10,
                            fib_levels: List[float] = [0, 0.3, 0.5, 0.7, 1.0]) -> List[Zone]:
        """Calculate DTFX Algo Zones based on swing structure"""
        try:
            zones = []
            dir_var = 0
            
            if len(df) < structure_len * 2:
                return zones
            
            for i in range(structure_len * 2, len(df)):
                # Get structure high/low
                window_high = df['high'].iloc[i-structure_len:i].max()
                window_low = df['low'].iloc[i-structure_len:i].min()
                
                current_high = df['high'].iloc[i-structure_len]
                current_low = df['low'].iloc[i-structure_len]
                
                # Detect structure changes
                if dir_var >= 0 and current_high > window_high:
                    # Potential bearish structure
                    top = current_high
                    bot = df['low'].iloc[i-structure_len+1:i].min()
                    
                    if bot is not None and top is not None:
                        # Create bullish zone (expectation of bounce)
                        zone = self._create_zone(top, bot, 1, fib_levels, i - structure_len, i)
                        zones.append(zone)
                    
                    dir_var = -1
                
                elif dir_var <= 0 and current_low < window_low:
                    # Potential bullish structure  
                    top = df['high'].iloc[i-structure_len+1:i].max()
                    bot = current_low
                    
                    if bot is not None and top is not None:
                        # Create bearish zone (expectation of rejection)
                        zone = self._create_zone(top, bot, -1, fib_levels, i - structure_len, i)
                        zones.append(zone)
                    
                    dir_var = 1
            
            self.zones = zones
            return zones
            
        except Exception as e:
            logger.error(f"Error calculating DTFX zones: {e}")
            return []
    
    def _create_zone(self, top: float, bottom: float, direction: int, 
                    fib_levels: List[float], start_bar: int, end_bar: int) -> Zone:
        """Create a zone with Fibonacci levels"""
        try:
            rng = abs(top - bottom)
            anchor = bottom if direction == 1 else top
            
            fibs = {}
            for level in fib_levels:
                fib_price = anchor + (rng * level) if direction == 1 else anchor - (rng * level)
                fibs[f'fib_{level}'] = float(fib_price)
            
            return Zone(
                top=top,
                bottom=bottom,
                direction=direction,
                start_bar=start_bar,
                end_bar=end_bar,
                fib_levels=fibs
            )
        except Exception as e:
            logger.error(f"Error creating zone: {e}")
            return None
    
    def calculate_zigzag_momentum(self, df: pd.DataFrame, momentum_type: str = 'macd') -> List[ZigZagPoint]:
        """Calculate momentum-based ZigZag"""
        try:
            zigzag_points = []
            
            if momentum_type == 'macd':
                momentum_direction = self._get_macd_momentum(df)
            elif momentum_type == 'ma':
                momentum_direction = self._get_ma_momentum(df)
            elif momentum_type == 'qqe':
                momentum_direction = self._get_qqe_momentum(df)
            else:
                momentum_direction = self._get_macd_momentum(df)
            
            # Detect ZigZag points based on momentum direction
            var_high = df['high'].iloc[0]
            var_low = df['low'].iloc[0]
            var_direction = momentum_direction[0] if len(momentum_direction) > 0 else 0
            
            for i in range(1, len(df)):
                current_dir = momentum_direction[i-1] if i < len(momentum_direction) else 0
                prev_dir = momentum_direction[i-2] if i > 1 else 0
                
                # Check for direction change
                if current_dir != prev_dir and prev_dir != 0:
                    if prev_dir == 1:
                        # Up to down transition
                        zigzag_points.append(ZigZagPoint(
                            price=var_high,
                            bar_index=i-1,
                            direction=-1,
                            momentum_signal=momentum_type
                        ))
                        var_low = df['low'].iloc[i]
                    elif prev_dir == -1:
                        # Down to up transition
                        zigzag_points.append(ZigZagPoint(
                            price=var_low,
                            bar_index=i-1,
                            direction=1,
                            momentum_signal=momentum_type
                        ))
                        var_high = df['high'].iloc[i]
                
                # Track highs and lows
                if current_dir == 1:
                    var_high = max(var_high, df['high'].iloc[i])
                elif current_dir == -1:
                    var_low = min(var_low, df['low'].iloc[i])
            
            self.zigzag_points = zigzag_points
            return zigzag_points
            
        except Exception as e:
            logger.error(f"Error calculating ZigZag: {e}")
            return []
    
    def _get_macd_momentum(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
        """Get momentum direction from MACD"""
        try:
            close = df['close']
            
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            momentum = np.zeros(len(df))
            for i in range(1, len(df)):
                if macd.iloc[i] > signal_line.iloc[i]:
                    momentum[i] = 1
                elif macd.iloc[i] < signal_line.iloc[i]:
                    momentum[i] = -1
                else:
                    momentum[i] = momentum[i-1]
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error getting MACD momentum: {e}")
            return np.zeros(len(df))
    
    def _get_ma_momentum(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """Get momentum direction from Moving Average"""
        try:
            close = df['close']
            ma = close.rolling(window=period).mean()
            
            momentum = np.zeros(len(df))
            for i in range(period, len(df)):
                if ma.iloc[i] > ma.iloc[i-1] and ma.iloc[i-1] > ma.iloc[i-2]:
                    momentum[i] = 1
                elif ma.iloc[i] < ma.iloc[i-1] and ma.iloc[i-1] < ma.iloc[i-2]:
                    momentum[i] = -1
                else:
                    momentum[i] = momentum[i-1]
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error getting MA momentum: {e}")
            return np.zeros(len(df))
    
    def _get_qqe_momentum(self, df: pd.DataFrame, rsi_period: int = 14, factor: float = 4.238) -> np.ndarray:
        """Get momentum direction from QQE (Quantitative Qualitative Estimation)"""
        try:
            # Simplified QQE implementation
            close = df['close']
            rsi = self._calculate_rsi_simple(close, rsi_period)
            
            # RSI smoothing
            rsi_ma = rsi.rolling(window=5).mean()
            
            # ATR of RSI
            atr_rsi = abs(rsi_ma - rsi_ma.shift(1))
            ma_atr_rsi = atr_rsi.rolling(window=rsi_period * 2 - 1).mean()
            dar = ma_atr_rsi.rolling(window=rsi_period * 2 - 1).mean() * factor
            
            # Long and short bands
            longband = rsi_ma - dar
            shortband = rsi_ma + dar
            
            # Trend detection
            momentum = np.zeros(len(df))
            for i in range(rsi_period * 4, len(df)):
                if rsi_ma.iloc[i] > shortband.iloc[i-1]:
                    momentum[i] = 1
                elif rsi_ma.iloc[i] < longband.iloc[i-1]:
                    momentum[i] = -1
                else:
                    momentum[i] = momentum[i-1]
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error getting QQE momentum: {e}")
            return np.zeros(len(df))
    
    def _calculate_rsi_simple(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI calculation"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(close), index=close.index)
    
    def get_zone_signals(self, price: float, zones: List[Zone] = None) -> Dict[str, Any]:
        """Get signals from current zones"""
        try:
            if zones is None:
                zones = self.zones
            
            if not zones:
                return {'signal': 'none', 'zone': None, 'distance': None}
            
            # Check price against most recent zones
            latest_zone = zones[-1]
            
            # Check if price is in the zone
            if latest_zone.bottom <= price <= latest_zone.top:
                # Check Fibonacci levels
                for fib_name, fib_price in latest_zone.fib_levels.items():
                    if abs(price - fib_price) / price < 0.005:  # Within 0.5%
                        if latest_zone.direction == 1:
                            return {
                                'signal': 'bullish_zone',
                                'zone': latest_zone,
                                'fib_level': fib_name,
                                'distance': 0
                            }
                        else:
                            return {
                                'signal': 'bearish_zone',
                                'zone': latest_zone,
                                'fib_level': fib_name,
                                'distance': 0
                            }
            
            # Calculate distance to zone
            distance = (price - latest_zone.top) if price > latest_zone.top else (latest_zone.bottom - price)
            
            return {
                'signal': 'none',
                'zone': latest_zone,
                'distance': abs(distance)
            }
            
        except Exception as e:
            logger.error(f"Error getting zone signals: {e}")
            return {'signal': 'none', 'zone': None, 'distance': None}

def main():
    """Demonstrate advanced indicators"""
    try:
        logger.info("=" * 70)
        logger.info("Advanced Indicators Demo")
        logger.info("=" * 70)
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        data.index = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
        
        # Initialize
        indicators = AdvancedIndicators()
        
        logger.info("\n[1] Calculating CCI...")
        cci = indicators.calculate_cci(data, period=20)
        logger.info(f"CCI calculated. Last value: {cci.iloc[-1]:.2f}")
        
        logger.info("\n[2] Calculating Bollinger Bands...")
        bb = indicators.calculate_bollinger_bands(data, period=20)
        logger.info(f"Bollinger Bands calculated.")
        logger.info(f"  Upper: {bb['upper'].iloc[-1]:.2f}")
        logger.info(f"  Middle: {bb['middle'].iloc[-1]:.2f}")
        logger.info(f"  Lower: {bb['lower'].iloc[-1]:.2f}")
        logger.info(f"  Position: {bb['position'].iloc[-1]:.2f}")
        
        logger.info("\n[3] Calculating Supertrend...")
        st = indicators.calculate_supertrend(data, period=10, multiplier=3.0)
        logger.info(f"Supertrend calculated.")
        logger.info(f"  Direction: {st['direction'].iloc[-1]}")
        logger.info(f"  Value: {st['supertrend'].iloc[-1]:.2f}")
        
        logger.info("\n[4] Calculating DTFX Zones...")
        zones = indicators.calculate_dtfx_zones(data, structure_len=10)
        logger.info(f"DTFX Zones calculated. Found {len(zones)} zones")
        
        if zones:
            zone = zones[-1]
            logger.info(f"  Latest zone: {zone.direction} from {zone.bottom:.2f} to {zone.top:.2f}")
        
        logger.info("\n[5] Calculating ZigZag...")
        zigzag = indicators.calculate_zigzag_momentum(data, momentum_type='macd')
        logger.info(f"ZigZag calculated. Found {len(zigzag)} points")
        
        if zigzag:
            point = zigzag[-1]
            logger.info(f"  Latest point: {point.direction} at {point.price:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Advanced indicators demo completed!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in advanced indicators demo: {e}")

if __name__ == "__main__":
    main()

