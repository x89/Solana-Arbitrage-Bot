#!/usr/bin/env python3
"""
Martingale Calculator
Calculate optimal martingale betting strategies for trading
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MartingaleType(Enum):
    """Types of martingale strategies"""
    CLASSIC = "classic"
    FIBONACCI = "fibonacci"
    D_ALEMBERT = "d_alembert"
    OSCARS_GRIND = "oscars_grind"
    REVERSE = "reverse"
    ADAPTIVE = "adaptive"

@dataclass
class MartingaleConfig:
    """Martingale strategy configuration"""
    strategy_type: MartingaleType
    initial_bet: float
    multiplier: float = 2.0
    max_bet: Optional[float] = None
    max_loss: Optional[float] = None
    recovery_threshold: float = 0.5
    max_consecutive_losses: int = 10
    win_target: Optional[float] = None

@dataclass
class TradingPosition:
    """Trading position information"""
    position_id: str
    entry_price: float
    current_price: float
    position_size: float
    position_type: str  # 'long' or 'short'
    leverage: float
    unrealized_pnl: float

class MartingaleCalculator:
    """Calculate martingale strategy parameters for trading"""
    
    def __init__(self, config: MartingaleConfig):
        self.config = config
        self.bet_history = []
        self.current_bet = config.initial_bet
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_bets = 0
        self.total_wins = 0
        self.total_losses = 0
        self.net_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_profit = 0.0
        
        logger.info(f"MartingaleCalculator initialized with strategy: {config.strategy_type.value}")
    
    def calculate_next_bet(self, last_result: Optional[bool] = None) -> float:
        """Calculate next bet based on last result"""
        try:
            if last_result is not None:
                if last_result:
                    self._handle_win()
                else:
                    self._handle_loss()
            
            if self._should_stop():
                return 0.0
            
            next_bet = self._calculate_bet_amount()
            
            # Apply constraints
            if self.config.max_bet and next_bet > self.config.max_bet:
                next_bet = self.config.max_bet
            
            self.current_bet = next_bet
            return next_bet
            
        except Exception as e:
            logger.error(f"Error calculating next bet: {e}")
            return 0.0
    
    def _calculate_bet_amount(self) -> float:
        """Calculate bet amount based on strategy"""
        if self.config.strategy_type == MartingaleType.CLASSIC:
            return self._classic_martingale()
        elif self.config.strategy_type == MartingaleType.FIBONACCI:
            return self._fibonacci_martingale()
        elif self.config.strategy_type == MartingaleType.D_ALEMBERT:
            return self._d_alembert_martingale()
        elif self.config.strategy_type == MartingaleType.OSCARS_GRIND:
            return self._oscars_grind_martingale()
        elif self.config.strategy_type == MartingaleType.REVERSE:
            return self._reverse_martingale()
        elif self.config.strategy_type == MartingaleType.ADAPTIVE:
            return self._adaptive_martingale()
        else:
            return self.config.initial_bet
    
    def _classic_martingale(self) -> float:
        """Classic martingale: double bet after loss"""
        if self.consecutive_losses == 0:
            return self.config.initial_bet
        return self.config.initial_bet * (self.config.multiplier ** self.consecutive_losses)
    
    def _fibonacci_martingale(self) -> float:
        """Fibonacci sequence betting"""
        if self.consecutive_losses == 0:
            return self.config.initial_bet
        
        # Fibonacci sequence
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        fib_index = min(self.consecutive_losses, len(fib_sequence) - 1)
        return self.config.initial_bet * fib_sequence[fib_index]
    
    def _d_alembert_martingale(self) -> float:
        """D'Alembert betting strategy"""
        if self.consecutive_losses == 0:
            return self.config.initial_bet
        
        # Increase bet by initial bet amount after loss
        return self.config.initial_bet + (self.consecutive_losses * self.config.initial_bet)
    
    def _oscars_grind_martingale(self) -> float:
        """Oscar's Grind betting strategy"""
        if self.consecutive_losses == 0:
            return self.config.initial_bet
        
        # Oscars grind: increase by 1 unit after loss
        return self.config.initial_bet * (1 + self.consecutive_losses)
    
    def _reverse_martingale(self) -> float:
        """Reverse martingale: increase bet after win"""
        if self.consecutive_wins == 0:
            return self.config.initial_bet
        
        # Double bet after win
        return self.config.initial_bet * (self.config.multiplier ** self.consecutive_wins)
    
    def _adaptive_martingale(self) -> float:
        """Adaptive martingale based on win rate"""
        win_rate = self._calculate_win_rate()
        
        if win_rate < self.config.recovery_threshold:
            # Below threshold, use conservative betting
            return self.config.initial_bet
        else:
            # Above threshold, use aggressive betting
            return self._classic_martingale()
    
    def _handle_win(self):
        """Handle a win"""
        self.consecutive_wins += 1
        self.consecutive_losses = 0
        self.total_wins += 1
        self.net_profit += self.current_bet
        
        if self.net_profit > self.peak_profit:
            self.peak_profit = self.net_profit
        
        logger.debug(f"Win recorded. Consecutive wins: {self.consecutive_wins}")
    
    def _handle_loss(self):
        """Handle a loss"""
        self.consecutive_losses += 1
        self.consecutive_wins = 0
        self.total_losses += 1
        self.net_profit -= self.current_bet
        
        if self.net_profit < self.peak_profit:
            drawdown = self.peak_profit - self.net_profit
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        logger.debug(f"Loss recorded. Consecutive losses: {self.consecutive_losses}")
    
    def _should_stop(self) -> bool:
        """Check if we should stop betting"""
        # Check max consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"Stopping: Max consecutive losses reached ({self.consecutive_losses})")
            return True
        
        # Check max loss
        if self.config.max_loss and abs(self.net_profit) >= self.config.max_loss:
            logger.warning(f"Stopping: Max loss reached ({abs(self.net_profit)})")
            return True
        
        # Check win target
        if self.config.win_target and self.net_profit >= self.config.win_target:
            logger.info(f"Stopping: Win target reached ({self.net_profit})")
            return True
        
        return False
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        if self.total_bets == 0:
            return 0.0
        return self.total_wins / self.total_bets
    
    def record_bet(self, bet_amount: float, result: bool):
        """Record a bet and its result"""
        self.bet_history.append({
            'bet_amount': bet_amount,
            'result': result,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'net_profit': self.net_profit
        })
        
        self.total_bets += 1
        
        if result:
            self._handle_win()
        else:
            self._handle_loss()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'total_bets': self.total_bets,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': self._calculate_win_rate(),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'net_profit': self.net_profit,
            'max_drawdown': self.max_drawdown,
            'peak_profit': self.peak_profit,
            'current_bet': self.current_bet,
            'strategy_type': self.config.strategy_type.value
        }
    
    def reset(self):
        """Reset calculator to initial state"""
        self.bet_history = []
        self.current_bet = self.config.initial_bet
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_bets = 0
        self.total_wins = 0
        self.total_losses = 0
        self.net_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_profit = 0.0
        
        logger.info("MartingaleCalculator reset")

class MartingalePositionManager:
    """Manage martingale strategy for trading positions"""
    
    def __init__(self, calculator: MartingaleCalculator):
        self.calculator = calculator
        self.positions: Dict[str, TradingPosition] = {}
        self.position_history = []
        self.max_position_size = 1000.0
        
    def calculate_position_size(self, base_size: float, previous_result: Optional[bool] = None) -> float:
        """Calculate position size using martingale strategy"""
        # Get next bet amount
        bet_multiplier = self.calculator.calculate_next_bet(previous_result)
        
        if bet_multiplier == 0:
            return 0
        
        # Calculate position size
        position_size = base_size * (bet_multiplier / self.calculator.config.initial_bet)
        
        # Apply constraints
        if position_size > self.max_position_size:
            position_size = self.max_position_size
        
        logger.info(f"Calculated position size: {position_size} (multiplier: {bet_multiplier})")
        
        return position_size
    
    def open_position(self, position_id: str, entry_price: float, position_size: float, 
                     position_type: str = 'long', leverage: float = 1.0) -> bool:
        """Open a new position"""
        try:
            position = TradingPosition(
                position_id=position_id,
                entry_price=entry_price,
                current_price=entry_price,
                position_size=position_size,
                position_type=position_type,
                leverage=leverage,
                unrealized_pnl=0.0
            )
            
            self.positions[position_id] = position
            self.position_history.append({
                'action': 'open',
                'position_id': position_id,
                'timestamp': pd.Timestamp.now(),
                'entry_price': entry_price,
                'position_size': position_size
            })
            
            logger.info(f"Position opened: {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, position_id: str, exit_price: float) -> bool:
        """Close a position"""
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return False
        
        try:
            position = self.positions[position_id]
            position.current_price = exit_price
            
            # Calculate PnL
            if position.position_type == 'long':
                pnl = (exit_price - position.entry_price) * position.position_size * position.leverage
            else:  # short
                pnl = (position.entry_price - exit_price) * position.position_size * position.leverage
            
            position.unrealized_pnl = pnl
            
            # Record with calculator
            result = pnl > 0
            self.calculator.record_bet(
                bet_amount=position.position_size,
                result=result
            )
            
            self.position_history.append({
                'action': 'close',
                'position_id': position_id,
                'timestamp': pd.Timestamp.now(),
                'exit_price': exit_price,
                'pnl': pnl
            })
            
            # Remove from active positions
            del self.positions[position_id]
            
            logger.info(f"Position closed: {position_id}, PnL: {pnl}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        return {
            'active_positions': len(self.positions),
            'total_unrealized_pnl': total_pnl,
            'positions': [
                {
                    'position_id': p.position_id,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'position_size': p.position_size,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in self.positions.values()
            ]
        }

class MartingaleBacktester:
    """Backtest martingale strategies"""
    
    def __init__(self, calculator: MartingaleCalculator):
        self.calculator = calculator
    
    def backtest_strategy(self, price_data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """Backtest martingale strategy on historical data"""
        try:
            self.calculator.reset()
            
            capital = initial_capital
            trades = []
            equity_curve = []
            
            for i, row in price_data.iterrows():
                # Calculate position size based on strategy
                base_size = initial_capital * 0.01  # 1% of capital
                
                # Entry signal (simplified)
                signal = self._generate_signal(row)
                
                if signal == 'long':
                    entry_price = row['close']
                    position_size = self.calculator.calculate_next_bet()
                    
                    # Check if we have enough capital
                    if position_size > capital:
                        position_size = capital
                    
                    # Calculate potential profit/loss
                    if i < len(price_data) - 1:
                        exit_price = price_data.iloc[i+1]['close']
                        pnl = (exit_price - entry_price) * position_size
                        
                        capital += pnl
                        result = pnl > 0
                        
                        self.calculator.record_bet(position_size, result)
                        
                        trades.append({
                            'timestamp': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'capital': capital
                        })
                
                equity_curve.append(capital)
                
                # Check if we should stop
                if self.calculator._should_stop():
                    break
            
            # Calculate metrics
            if trades:
                total_return = (capital - initial_capital) / initial_capital
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                max_drawdown = self._calculate_max_drawdown(equity_curve)
                
                return {
                    'total_trades': len(trades),
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'final_capital': capital,
                    'max_drawdown': max_drawdown,
                    'equity_curve': equity_curve,
                    'trades': trades[-10:]  # Last 10 trades
                }
            else:
                return {'error': 'No trades executed'}
                
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e)}
    
    def _generate_signal(self, row: pd.Series) -> str:
        """Generate trading signal (simplified for demo)"""
        # Simple moving average crossover
        if 'ma_short' in row and 'ma_long' in row:
            if row['ma_short'] > row['ma_long']:
                return 'long'
        return 'hold'
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

def main():
    """Example usage"""
    # Configuration
    config = MartingaleConfig(
        strategy_type=MartingaleType.CLASSIC,
        initial_bet=100.0,
        multiplier=2.0,
        max_bet=1000.0,
        max_consecutive_losses=5
    )
    
    # Initialize calculator
    calculator = MartingaleCalculator(config)
    
    # Example: Simulate some bets
    for i in range(10):
        bet = calculator.calculate_next_bet()
        result = i % 3 != 0  # 66% win rate
        calculator.record_bet(bet, result)
        print(f"Bet {i+1}: {bet:.2f}, Result: {'Win' if result else 'Loss'}")
    
    # Print statistics
    stats = calculator.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

