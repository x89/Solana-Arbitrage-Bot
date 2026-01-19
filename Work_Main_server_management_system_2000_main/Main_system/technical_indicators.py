"""
Custom Technical Indicators Module
Pure Python implementation without TA-Lib dependency
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    sma = np.convolve(prices, np.ones(period)/period, mode='valid')
    # Pad the beginning with NaN values
    padding = np.full(period - 1, np.nan)
    return np.concatenate([padding, sma])

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    alpha = 2.0 / (period + 1.0)
    ema = np.full_like(prices, np.nan)
    ema[period - 1] = np.mean(prices[:period])
    
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return np.full_like(prices, np.nan)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = calculate_sma(gains, period)
    avg_losses = calculate_sma(losses, period)
    
    # Calculate RSI
    rs = avg_gains / np.where(avg_losses == 0, 1e-10, avg_losses)
    rsi = 100 - (100 / (1 + rs))
    
    # Pad the beginning with NaN
    padding = np.full(1, np.nan)
    return np.concatenate([padding, rsi])

def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow_period:
        return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)
    
    # Calculate EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = calculate_ema(macd_line[~np.isnan(macd_line)], signal_period)
    
    # Pad signal line with NaN values
    signal_padded = np.full_like(macd_line, np.nan)
    signal_padded[~np.isnan(macd_line)][len(signal_padded[~np.isnan(macd_line)]) - len(signal_line):] = signal_line
    
    # Calculate histogram
    histogram = macd_line - signal_padded
    
    return macd_line, signal_padded, histogram

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return np.full_like(prices, np.nan), np.full_like(prices, np.nan), np.full_like(prices, np.nan)
    
    # Calculate SMA
    sma = calculate_sma(prices, period)
    
    # Calculate standard deviation
    upper_band = np.full_like(prices, np.nan)
    lower_band = np.full_like(prices, np.nan)
    
    for i in range(period - 1, len(prices)):
        if not np.isnan(sma[i]):
            window = prices[i - period + 1:i + 1]
            std = np.std(window)
            upper_band[i] = sma[i] + (std_dev * std)
            lower_band[i] = sma[i] - (std_dev * std)
    
    return upper_band, sma, lower_band

def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic Oscillator"""
    if len(close) < k_period:
        return np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    k_percent = np.full_like(close, np.nan)
    
    for i in range(k_period - 1, len(close)):
        high_window = high[i - k_period + 1:i + 1]
        low_window = low[i - k_period + 1:i + 1]
        close_current = close[i]
        
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        
        if highest_high != lowest_low:
            k_percent[i] = ((close_current - lowest_low) / (highest_high - lowest_low)) * 100
        else:
            k_percent[i] = 50  # Neutral when high == low
    
    # Calculate %D (SMA of %K)
    d_percent = calculate_sma(k_percent, d_period)
    
    return k_percent, d_percent

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    
    # Calculate True Range
    tr = np.full_like(close, np.nan)
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close_prev = abs(high[i] - close[i - 1])
        low_close_prev = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_close_prev, low_close_prev)
    
    # Calculate ATR as SMA of TR
    atr = calculate_sma(tr, period)
    
    return atr

def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Volume Simple Moving Average"""
    return calculate_sma(volume, period)

def calculate_price_momentum(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Price Momentum"""
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    momentum = np.full_like(prices, np.nan)
    
    for i in range(period - 1, len(prices)):
        if i >= period - 1:
            momentum[i] = (prices[i] / prices[i - period + 1] - 1) * 100
    
    return momentum

def calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Williams %R"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    williams_r = np.full_like(close, np.nan)
    
    for i in range(period - 1, len(close)):
        high_window = high[i - period + 1:i + 1]
        low_window = low[i - period + 1:i + 1]
        close_current = close[i]
        
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_current) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # Neutral when high == low
    
    return williams_r

def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Commodity Channel Index"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    # Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate SMA of Typical Price
    sma_tp = calculate_sma(typical_price, period)
    
    # Calculate Mean Deviation
    cci = np.full_like(close, np.nan)
    
    for i in range(period - 1, len(close)):
        if not np.isnan(sma_tp[i]):
            window = typical_price[i - period + 1:i + 1]
            mean_deviation = np.mean(np.abs(window - sma_tp[i]))
            
            if mean_deviation != 0:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)
            else:
                cci[i] = 0
    
    return cci

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average Directional Index (simplified version)"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.full_like(close, np.nan)
    dm_plus = np.full_like(close, np.nan)
    dm_minus = np.full_like(close, np.nan)
    
    for i in range(1, len(close)):
        # True Range
        high_low = high[i] - low[i]
        high_close_prev = abs(high[i] - close[i - 1])
        low_close_prev = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_close_prev, low_close_prev)
        
        # Directional Movement
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
            
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Calculate smoothed values
    tr_smooth = calculate_sma(tr, period)
    dm_plus_smooth = calculate_sma(dm_plus, period)
    dm_minus_smooth = calculate_sma(dm_minus, period)
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(close, np.nan)
    di_minus = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / tr_smooth[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / tr_smooth[i]) * 100
    
    # Calculate DX and ADX
    dx = np.full_like(close, np.nan)
    adx = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = (abs(di_plus[i] - di_minus[i]) / di_sum) * 100
    
    # Calculate ADX as SMA of DX
    adx = calculate_sma(dx, period)
    
    return adx

def calculate_all_indicators(prices: np.ndarray, volumes: Optional[np.ndarray] = None, 
                           high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None) -> dict:
    """Calculate all technical indicators for a price series"""
    indicators = {}
    
    # Basic indicators that only need prices
    indicators['sma_20'] = calculate_sma(prices, 20)
    indicators['sma_50'] = calculate_sma(prices, 50)
    indicators['ema_12'] = calculate_ema(prices, 12)
    indicators['ema_26'] = calculate_ema(prices, 26)
    indicators['rsi'] = calculate_rsi(prices, 14)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(prices)
    indicators['macd'] = macd_line
    indicators['macd_signal'] = signal_line
    indicators['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    indicators['bb_upper'] = bb_upper
    indicators['bb_middle'] = bb_middle
    indicators['bb_lower'] = bb_lower
    
    # Price momentum
    indicators['price_momentum'] = calculate_price_momentum(prices, 20)
    
    # Volume indicators (if volume data available)
    if volumes is not None:
        indicators['volume_sma'] = calculate_volume_sma(volumes, 20)
        indicators['volume_ratio'] = volumes / np.where(indicators['volume_sma'] == 0, 1e-10, indicators['volume_sma'])
    
    # Additional indicators (if high/low data available)
    if high is not None and low is not None:
        indicators['stochastic_k'], indicators['stochastic_d'] = calculate_stochastic(high, low, prices)
        indicators['williams_r'] = calculate_williams_r(high, low, prices)
        indicators['cci'] = calculate_cci(high, low, prices)
        indicators['atr'] = calculate_atr(high, low, prices)
        indicators['adx'] = calculate_adx(high, low, prices)
    
    return indicators 