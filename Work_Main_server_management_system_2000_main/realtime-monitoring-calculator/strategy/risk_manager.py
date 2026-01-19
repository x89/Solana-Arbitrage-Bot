"""
Risk Management Module
Position sizing, stop-loss, take-profit, drawdown protection
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    timestamp: float


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    portfolio_value: float
    daily_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    current_exposure: float


class RiskManager:
    """
    Risk management for trading system
    Handles position sizing, stop-loss, take-profit, drawdown protection
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.1,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.15
    ):
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        
        # Portfolio state
        self.positions = []
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.peak_equity = initial_capital
        
    def calculate_position_size(
        self,
        signal: Dict,
        current_price: float,
        stop_loss_price: float,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk
        
        Uses fixed-fraction sizing or Kelly Criterion
        """
        # Calculate risk amount
        risk_amount = self.portfolio_value * self.risk_per_trade
        
        # Calculate risk per unit
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk == 0:
            return 0.0
        
        # Calculate position size
        position_size = risk_amount / price_risk
        
        # Adjust for volatility
        volatility_factor = min(0.02 / volatility, 2.0)
        position_size *= volatility_factor
        
        # Limit max position
        max_position_value = self.portfolio_value * self.max_position_size
        max_position_quantity = max_position_value / current_price
        position_size = min(position_size, max_position_quantity)
        
        return position_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        signal_type: str,
        atr: float = 0.02  # Average True Range
    ) -> float:
        """
        Calculate stop loss price
        
        Uses ATR-based or fixed percentage
        """
        if signal_type == 'long' or signal_type == 'buy':
            stop_loss = entry_price * (1 - atr)
        else:  # short or sell
            stop_loss = entry_price * (1 + atr)
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        signal_type: str,
        risk_reward_ratio: float = 2.5
    ) -> float:
        """
        Calculate take profit price
        
        Uses risk-reward ratio (e.g., 2.5:1)
        """
        # First calculate stop loss
        stop_loss = self.calculate_stop_loss(entry_price, signal_type)
        
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Calculate reward
        reward = risk * risk_reward_ratio
        
        if signal_type == 'long' or signal_type == 'buy':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit
    
    def open_position(
        self,
        symbol: str,
        signal: Dict,
        current_price: float
    ) -> Optional[Position]:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price
        
        Returns:
            Position object or None if rejected
        """
        # Check daily loss limit
        if self.daily_pnl < -abs(self.max_daily_loss * self.initial_capital):
            logger.warning("Daily loss limit reached - rejecting position")
            return None
        
        # Check drawdown
        current_drawdown = (self.peak_equity - self.portfolio_value) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            logger.warning("Max drawdown exceeded - rejecting position")
            return None
        
        # Determine side
        signal_type = signal['signal_type']
        side = 'long' if signal_type.value == 1 else 'short'
        
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(
            current_price,
            side,
            atr=abs(signal.get('prediction_return', 0.02))
        )
        
        # Calculate take profit
        take_profit = self.calculate_take_profit(current_price, side)
        
        # Calculate position size
        volatility = abs(signal.get('prediction_return', 0.02))
        quantity = self.calculate_position_size(signal, current_price, stop_loss, volatility)
        
        # Check if position size is valid
        if quantity <= 0:
            logger.warning("Invalid position size")
            return None
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.get('timestamp', 0)
        )
        
        self.positions.append(position)
        logger.info(f"Opened {side} position: {quantity:.4f} @ {current_price:.2f}")
        
        return position
    
    def update_position(
        self,
        position: Position,
        current_price: float
    ) -> Dict:
        """
        Update position and check stop-loss/take-profit
        
        Returns:
            Dict with status, pnl, action
        """
        # Calculate unrealized P&L
        if position.side == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        unrealized_pnl = pnl_pct * position.quantity * position.entry_price
        
        # Check stop loss
        if position.side == 'long' and current_price <= position.stop_loss:
            return {
                'action': 'close',
                'reason': 'stop_loss',
                'pnl': unrealized_pnl
            }
        
        if position.side == 'short' and current_price >= position.stop_loss:
            return {
                'action': 'close',
                'reason': 'stop_loss',
                'pnl': unrealized_pnl
            }
        
        # Check take profit
        if position.side == 'long' and current_price >= position.take_profit:
            return {
                'action': 'close',
                'reason': 'take_profit',
                'pnl': unrealized_pnl
            }
        
        if position.side == 'short' and current_price <= position.take_profit:
            return {
                'action': 'close',
                'reason': 'take_profit',
                'pnl': unrealized_pnl
            }
        
        return {
            'action': 'hold',
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': pnl_pct
        }
    
    def close_position(self, position: Position, current_price: float) -> float:
        """Close position and realize P&L"""
        # Calculate P&L
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity
        
        # Update portfolio
        self.portfolio_value += pnl
        self.daily_pnl += pnl
        
        # Update peak equity
        if self.portfolio_value > self.peak_equity:
            self.peak_equity = self.portfolio_value
        
        # Remove from positions
        self.positions.remove(position)
        self.closed_trades.append({
            'position': position,
            'exit_price': current_price,
            'pnl': pnl
        })
        
        logger.info(f"Closed position: P&L = ${pnl:.2f}")
        
        return pnl
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        # Calculate total exposure
        total_exposure = sum(
            pos.quantity * pos.entry_price for pos in self.positions
        )
        exposure_pct = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate win rate
        if len(self.closed_trades) > 0:
            winning_trades = sum(1 for t in self.closed_trades if t['pnl'] > 0)
            win_rate = winning_trades / len(self.closed_trades)
        else:
            win_rate = 0.0
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for pos in self.positions:
            # Would need current prices - simplified
            unrealized_pnl += 0  # TODO: implement
        
        return RiskMetrics(
            portfolio_value=self.portfolio_value,
            daily_pnl=self.daily_pnl,
            unrealized_pnl=unrealized_pnl,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=0.0,  # TODO: calculate
            win_rate=win_rate,
            current_exposure=exposure_pct
        )

