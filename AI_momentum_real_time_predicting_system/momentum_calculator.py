#!/usr/bin/env python3
"""
Momentum Calculator Module
Comprehensive momentum calculation system including:
- Rate of Change (ROC)
- Momentum indicators
- True Strength Index (TSI)
- Williams %R
- Stochastic Oscillator
- Commodity Channel Index (CCI)
- Advanced momentum metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import logging
    logging.warning("TA-Lib not installed. Some technical indicators will be unavailable.")
    logging.warning("To install on Windows, download wheel from: https://github.com/cgohlke/talib-build/releases")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumCalculator:
    """Comprehensive momentum indicators calculator"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize momentum calculator
        
        Args:
            data: Input dataframe with OHLC data
        """
        self.data = data
        logger.info("Momentum calculator initialized")
    
    @staticmethod
    def calculate_roc(series: pd.Series, period: int = 10) -> np.ndarray:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            ROC values
        """
        try:
            if TALIB_AVAILABLE:
                return talib.ROC(series.values, timeperiod=period)
            else:
                # Fallback implementation
                return ((series / series.shift(period)) - 1) * 100
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return np.array([])
    
    @staticmethod
    def calculate_momentum(series: pd.Series, period: int = 10) -> np.ndarray:
        """
        Calculate Momentum indicator
        
        Args:
            series: Price series
            period: Lookback period
            
        Returns:
            Momentum values
        """
        try:
            if TALIB_AVAILABLE:
                return talib.MOM(series.values, timeperiod=period)
            else:
                # Fallback implementation
                return series - series.shift(period)
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return np.array([])
    
    @staticmethod
    def calculate_tsi(series: pd.Series, fast: int = 25, slow: int = 13) -> pd.Series:
        """
        Calculate True Strength Index (TSI)
        
        Args:
            series: Price series
            fast: Fast period
            slow: Slow period
            
        Returns:
            TSI values
        """
        try:
            pc = series.diff()
            smoothed_pc = pc.ewm(span=fast, adjust=False).mean()
            abs_pc = pc.abs()
            smoothed_abs_pc = abs_pc.ewm(span=slow, adjust=False).mean()
            
            tsi = pd.Series(100 * (smoothed_pc / smoothed_abs_pc), index=series.index)
            
            return tsi.fillna(0)
        except Exception as e:
            logger.error(f"Error calculating TSI: {e}")
            return pd.Series([0] * len(series), index=series.index)
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> np.ndarray:
        """
        Calculate Williams %R
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            
        Returns:
            Williams %R values
        """
        try:
            if TALIB_AVAILABLE:
                return talib.WILLR(high.values, low.values, close.values, timeperiod=period)
            else:
                # Fallback implementation
                highest_high = high.rolling(window=period).max()
                lowest_low = low.rolling(window=period).min()
                return -100 * (highest_high - close) / (highest_high - lowest_low)
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return np.array([])
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowd_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            fastk_period: Fast K period
            slowk_period: Slow K period
            slowd_period: Slow D period
            
        Returns:
            Tuple of (K line, D line)
        """
        try:
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(
                    high.values, low.values, close.values,
                    fastk_period=fastk_period,
                    slowk_period=slowk_period,
                    slowd_period=slowd_period
                )
                return slowk, slowd
            else:
                # Fallback implementation
                lowest_low = low.rolling(window=fastk_period).min()
                highest_high = high.rolling(window=fastk_period).max()
                slowk = 100 * (close - lowest_low) / (highest_high - lowest_low)
                slowd = slowk.rolling(window=slowk_period).mean()
                return slowk.values, slowd.values
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return np.array([]), np.array([])
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> np.ndarray:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            
        Returns:
            CCI values
        """
        try:
            if TALIB_AVAILABLE:
                return talib.CCI(high.values, low.values, close.values, timeperiod=period)
            else:
                # Fallback implementation
                tp = (high + low + close) / 3  # Typical Price
                sma = tp.rolling(window=period).mean()
                mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
                return (tp - sma) / (0.015 * mad)
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return np.array([])
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate all momentum indicators
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary of indicators
        """
        try:
            logger.info("Calculating all momentum indicators...")
            
            indicators = {}
            
            close = df['close']
            high = df['high']
            low = df['low']
            
            # ROC
            indicators['roc_10'] = self.calculate_roc(close, period=10)
            indicators['roc_20'] = self.calculate_roc(close, period=20)
            
            # Momentum
            indicators['momentum_10'] = self.calculate_momentum(close, period=10)
            indicators['momentum_20'] = self.calculate_momentum(close, period=20)
            
            # TSI
            indicators['tsi'] = self.calculate_tsi(close)
            
            # Williams %R
            indicators['williams_r'] = self.calculate_williams_r(high, low, close)
            
            # Stochastic
            slowk, slowd = self.calculate_stochastic(high, low, close)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # CCI
            indicators['cci'] = self.calculate_cci(high, low, close)
            
            logger.info(f"Calculated {len(indicators)} momentum indicators")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return {}
    
    def get_momentum_signals(self, indicators: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Generate momentum signals from indicators
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            Dictionary of signals
        """
        try:
            signals = {}
            
            # ROC signals
            if 'roc_10' in indicators:
                roc = indicators['roc_10'][-1]
                signals['roc_signal'] = 'bullish' if roc > 0 else 'bearish'
            
            # TSI signals
            if 'tsi' in indicators:
                tsi_value = indicators['tsi'][-1] if isinstance(indicators['tsi'], np.ndarray) else indicators['tsi']
                signals['tsi_signal'] = 'bullish' if tsi_value > 25 else 'bearish' if tsi_value < -25 else 'neutral'
            
            # Williams %R signals
            if 'williams_r' in indicators:
                wr = indicators['williams_r'][-1]
                signals['williams_signal'] = 'bullish' if wr > -20 else 'bearish' if wr < -80 else 'neutral'
            
            # Stochastic signals
            if 'stoch_k' in indicators and 'stoch_d' in indicators:
                k = indicators['stoch_k'][-1]
                d = indicators['stoch_d'][-1]
                signals['stoch_signal'] = 'bullish' if k > 80 else 'bearish' if k < 20 else 'neutral'
            
            # CCI signals
            if 'cci' in indicators:
                cci = indicators['cci'][-1]
                signals['cci_signal'] = 'bullish' if cci > 100 else 'bearish' if cci < -100 else 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {}

def main():
    """Main function to demonstrate momentum calculator"""
    try:
        logger.info("=" * 60)
        logger.info("Momentum Calculator Demo")
        logger.info("=" * 60)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        df = pd.DataFrame(data)
        
        # Initialize calculator
        calculator = MomentumCalculator(df)
        
        # Calculate all indicators
        indicators = calculator.calculate_all_indicators(df)
        
        logger.info(f"\nCalculated {len(indicators)} indicators")
        
        # Generate signals
        signals = calculator.get_momentum_signals(indicators)
        logger.info(f"\nMomentum signals:")
        for signal, value in signals.items():
            logger.info(f"  {signal}: {value}")
        
        logger.info("=" * 60)
        logger.info("Momentum Calculator Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

