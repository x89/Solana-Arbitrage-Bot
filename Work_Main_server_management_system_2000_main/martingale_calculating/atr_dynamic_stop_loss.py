#!/usr/bin/env python3
"""
ATR Dynamic Stop-Loss with AI Optimization
Advanced stop-loss system using ATR (Average True Range) with AI-powered optimization
Based on Dynamic Supertrend MA Cross Quantitative Trading Strategy
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from .martingale_calculator import MartingaleCalculator, MartingaleType
    from .config import MARTINGALE_CONFIG
except ImportError:
    from martingale_calculator import MartingaleCalculator, MartingaleType
    from config import MARTINGALE_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ATRConfig:
    """ATR Dynamic Stop-Loss Configuration"""
    atr_period: int = 10
    atr_multiplier: float = 1.8  # Stop-loss at 1.8x ATR
    risk_reward_ratio: float = 3.0  # 3:1 default
    position_size_pct: float = 15.0  # 15% of account
    supertrend_factor: float = 2.8
    supertrend_period: int = 10
    ma_length: int = 20
    use_close_filter: bool = True  # Conservative vs aggressive crossover
    
@dataclass
class StopLossMetrics:
    """Stop-loss and take-profit metrics"""
    atr_value: float
    stop_distance: float
    take_profit_distance: float
    stop_price: float
    take_profit_price: float
    position_size: float
    risk_amount: float
    potential_reward: float

class ATRDynamicStopLoss:
    """ATR-based dynamic stop-loss calculator"""
    
    def __init__(self, config: ATRConfig = None):
        self.config = config or ATRConfig()
        self.atr_history = []
        self.stop_loss_history = []
        
        logger.info(f"ATRDynamicStopLoss initialized: ATR={self.config.atr_period}, Multiplier={self.config.atr_multiplier}, R:R={self.config.risk_reward_ratio}")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period (defaults to config)
            
        Returns:
            Series with ATR values
        """
        try:
            period = period or self.config.atr_period
            
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR (exponential moving average)
            atr = tr.ewm(span=period, adjust=False).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(dtype=float)
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, 
                          current_atr: float, account_equity: float = 10000.0) -> StopLossMetrics:
        """
        Calculate dynamic stop-loss based on ATR
        
        Args:
            entry_price: Entry price of the position
            position_type: 'long' or 'short'
            current_atr: Current ATR value
            account_equity: Current account equity
            
        Returns:
            StopLossMetrics with all calculated values
        """
        try:
            # Calculate stop distance from ATR
            stop_distance = current_atr * self.config.atr_multiplier
            
            # Calculate take-profit distance based on risk-reward ratio
            take_profit_distance = stop_distance * self.config.risk_reward_ratio
            
            # Calculate stop and take-profit prices
            if position_type == 'long':
                stop_price = entry_price - stop_distance
                take_profit_price = entry_price + take_profit_distance
            else:  # short
                stop_price = entry_price + stop_distance
                take_profit_price = entry_price - take_profit_distance
            
            # Calculate position size based on risk
            # Risk: Position size should be such that stop_distance represents max loss
            risk_pct = (stop_distance / entry_price) * 100
            risk_amount = account_equity * (risk_pct / 100)  # Total risk for this trade
            
            # Calculate position size in units
            position_size = risk_amount / stop_distance
            
            # Calculate potential reward
            potential_reward = position_size * take_profit_distance
            
            metrics = StopLossMetrics(
                atr_value=current_atr,
                stop_distance=stop_distance,
                take_profit_distance=take_profit_distance,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                position_size=position_size,
                risk_amount=risk_amount,
                potential_reward=potential_reward
            )
            
            # Store for history
            self.stop_loss_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None
    
    def update_stop_loss(self, entry_price: float, position_type: str, 
                        new_atr: float, trail_stop: bool = True,
                        trail_multiplier: float = 0.5) -> Tuple[float, float]:
        """
        Update stop-loss dynamically (for trailing stops)
        
        Args:
            entry_price: Original entry price
            position_type: 'long' or 'short'
            new_atr: Updated ATR value
            trail_stop: Whether to use trailing stop
            trail_multiplier: Trailing stop multiplier
            
        Returns:
            Tuple of (new_stop_price, new_take_profit_price)
        """
        try:
            stop_distance = new_atr * self.config.atr_multiplier
            take_profit_distance = stop_distance * self.config.risk_reward_ratio
            
            if position_type == 'long':
                new_stop = entry_price - stop_distance
                new_tp = entry_price + take_profit_distance
                
                if trail_stop:
                    # Trailing stop for long: move up, never down
                    # This would be implemented with live price tracking
                    trail_distance = new_atr * trail_multiplier
                    new_stop = entry_price - trail_distance
            
            else:  # short
                new_stop = entry_price + stop_distance
                new_tp = entry_price - take_profit_distance
                
                if trail_stop:
                    # Trailing stop for short: move down, never up
                    trail_distance = new_atr * trail_multiplier
                    new_stop = entry_price + trail_distance
            
            return new_stop, new_tp
            
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return entry_price, entry_price
    
    def calculate_supertrend_signal(self, df: pd.DataFrame, 
                                   factor: float = None, 
                                   period: int = None) -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        
        Args:
            df: DataFrame with OHLC data
            factor: Supertrend factor (defaults to config)
            period: Supertrend period (defaults to config)
            
        Returns:
            DataFrame with Supertrend values and direction
        """
        try:
            factor = factor or self.config.supertrend_factor
            period = period or self.config.supertrend_period
            
            # Calculate ATR
            atr = self.calculate_atr(df, period)
            
            # Calculate basic bands
            hl_avg = (df['high'] + df['low']) / 2
            
            upper_band = hl_avg + (factor * atr)
            lower_band = hl_avg - (factor * atr)
            
            # Initialize arrays
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            # Calculate Supertrend
            for i in range(len(df)):
                if i == 0:
                    supertrend.iloc[i] = hl_avg.iloc[i]
                    direction.iloc[i] = 1
                else:
                    # Upper band
                    if hl_avg.iloc[i] > supertrend.iloc[i-1]:
                        supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                        direction.iloc[i] = -1  # Uptrend
                    else:
                        supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                        direction.iloc[i] = 1  # Downtrend
                    
                    # Lower band
                    if hl_avg.iloc[i] < supertrend.iloc[i-1]:
                        supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                        direction.iloc[i] = 1  # Downtrend
                    else:
                        supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                        direction.iloc[i] = -1  # Uptrend
            
            df['supertrend'] = supertrend
            df['supertrend_dir'] = direction
            df['supertrend_signal'] = df['supertrend_dir'] < 0  # True for uptrend
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            return df
    
    def calculate_ma_crossover_signal(self, df: pd.DataFrame, ma_length: int = None) -> pd.DataFrame:
        """
        Calculate Moving Average crossover signals
        
        Args:
            df: DataFrame with OHLC data
            ma_length: MA period (defaults to config)
            
        Returns:
            DataFrame with MA signals
        """
        try:
            ma_length = ma_length or self.config.ma_length
            
            # Calculate Simple Moving Average
            df['ma'] = df['close'].rolling(window=ma_length).mean()
            
            # Detect crossovers
            df['ma_cross_up'] = (df['close'] > df['ma']) & (df['close'].shift(1) <= df['ma'].shift(1))
            df['ma_cross_down'] = (df['close'] < df['ma']) & (df['close'].shift(1) >= df['ma'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating MA crossover: {e}")
            return df
    
    def generate_entry_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate entry signals using MA + Supertrend confirmation
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Calculate indicators
            df = self.calculate_ma_crossover_signal(df)
            df = self.calculate_supertrend_signal(df)
            atr = self.calculate_atr(df)
            
            # Get latest values
            latest_atr = atr.iloc[-1]
            is_uptrend = df['supertrend_signal'].iloc[-1]
            ma_cross_up = df['ma_cross_up'].iloc[-1]
            ma_cross_down = df['ma_cross_down'].iloc[-1]
            
            signal = {
                'timestamp': df.index[-1],
                'long_signal': False,
                'short_signal': False,
                'atr': latest_atr,
                'signal_type': None,
                'confidence': 0.0
            }
            
            # Long signal: MA cross UP + Supertrend uptrend
            if ma_cross_up and is_uptrend:
                signal['long_signal'] = True
                signal['signal_type'] = 'long'
                signal['confidence'] = 0.75
            
            # Short signal: MA cross DOWN + Supertrend downtrend
            elif ma_cross_down and not is_uptrend:
                signal['short_signal'] = True
                signal['signal_type'] = 'short'
                signal['confidence'] = 0.75
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return {
                'long_signal': False,
                'short_signal': False,
                'signal_type': None
            }
    
    def optimize_atr_parameters(self, historical_data: pd.DataFrame,
                               optimization_range: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Optimize ATR parameters using grid search
        
        Args:
            historical_data: Historical OHLC data
            optimization_range: Parameter ranges for optimization
            
        Returns:
            Best parameters found
        """
        try:
            best_score = -np.inf
            best_params = {}
            
            # Generate parameter combinations
            import itertools
            keys = list(optimization_range.keys())
            values = list(optimization_range.values())
            combinations = list(itertools.product(*values))
            
            logger.info(f"Testing {len(combinations)} parameter combinations")
            
            for i, combination in enumerate(combinations):
                params = dict(zip(keys, combination))
                
                # Test these parameters
                score = self._test_parameters(historical_data, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Tested {i+1}/{len(combinations)} combinations")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'total_combinations': len(combinations)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
    
    def _test_parameters(self, df: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Test parameter combination and return score"""
        try:
            # Extract params
            atr_mult = params.get('atr_multiplier', 1.8)
            rr_ratio = params.get('risk_reward_ratio', 3.0)
            
            # Simulate trading
            trades = []
            for i in range(len(df) - 10):
                signal = self.generate_entry_signal(df.iloc[:i+10])
                
                if signal['long_signal'] or signal['short_signal']:
                    # Simplified scoring based on ATR and RR
                    score = (atr_mult * rr_ratio) / 10.0  # Simplified
                    trades.append(score)
            
            if len(trades) == 0:
                return 0.0
            
            return np.mean(trades)
            
        except Exception as e:
            logger.error(f"Error testing parameters: {e}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ATR and stop-loss statistics"""
        try:
            if not self.stop_loss_history:
                return {}
            
            # Calculate statistics
            atr_values = [h['metrics'].atr_value for h in self.stop_loss_history]
            stop_distances = [h['metrics'].stop_distance for h in self.stop_loss_history]
            
            return {
                'total_stops_calculated': len(self.stop_loss_history),
                'avg_atr': np.mean(atr_values) if atr_values else 0.0,
                'avg_stop_distance': np.mean(stop_distances) if stop_distances else 0.0,
                'recent_atr': atr_values[-1] if atr_values else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

def main():
    """Example usage"""
    # Create ATR stop-loss system
    atr_config = ATRConfig(
        atr_period=10,
        atr_multiplier=1.8,
        risk_reward_ratio=3.0,
        position_size_pct=15.0
    )
    
    stop_loss_system = ATRDynamicStopLoss(atr_config)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Calculate stop-loss for a long position
    entry_price = 100.0
    current_atr = sample_data['close'].std()  # Simplified ATR
    account_equity = 10000.0
    
    metrics = stop_loss_system.calculate_stop_loss(
        entry_price=entry_price,
        position_type='long',
        current_atr=current_atr,
        account_equity=account_equity
    )
    
    print(f"\nATR Dynamic Stop-Loss Metrics:")
    print(f"Entry Price: {entry_price:.2f}")
    print(f"Stop Price: {metrics.stop_price:.2f}")
    print(f"Take Profit: {metrics.take_profit_price:.2f}")
    print(f"Risk Amount: ${metrics.risk_amount:.2f}")
    print(f"Potential Reward: ${metrics.potential_reward:.2f}")
    print(f"Position Size: {metrics.position_size:.2f} units")
    
    # Generate signals
    signal = stop_loss_system.generate_entry_signal(sample_data)
    print(f"\nSignal Generated: {signal['signal_type']}")
    
    # Get statistics
    stats = stop_loss_system.get_statistics()
    print(f"\nStatistics: {stats}")

if __name__ == "__main__":
    main()

