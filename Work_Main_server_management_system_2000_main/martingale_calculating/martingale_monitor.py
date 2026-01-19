#!/usr/bin/env python3
"""
Martingale Strategy Monitor
Monitor and track martingale strategy performance in real-time
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque

from martingale_calculator import MartingaleCalculator, MartingaleConfig, MartingalePositionManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for martingale strategy"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    current_drawdown: float
    max_drawdown: float
    peak_profit: float
    risk_reward_ratio: float
    sharpe_ratio: float
    consecutive_wins: int
    consecutive_losses: int

class MartingaleMonitor:
    """Monitor martingale strategy performance"""
    
    def __init__(self, calculator: MartingaleCalculator, lookback_period: int = 100):
        self.calculator = calculator
        self.lookback_period = lookback_period
        self.metrics_history: deque = deque(maxlen=lookback_period)
        self.performance_alerts = []
        
        logger.info("MartingaleMonitor initialized")
    
    def update_metrics(self) -> PerformanceMetrics:
        """Update and return current performance metrics"""
        stats = self.calculator.get_statistics()
        
        # Calculate additional metrics
        risk_reward = self._calculate_risk_reward()
        sharpe = self._calculate_sharpe_ratio()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            total_trades=stats['total_bets'],
            winning_trades=stats['total_wins'],
            losing_trades=stats['total_losses'],
            win_rate=stats['win_rate'],
            total_pnl=stats['net_profit'],
            current_drawdown=self._calculate_current_drawdown(stats),
            max_drawdown=stats['max_drawdown'],
            peak_profit=stats['peak_profit'],
            risk_reward_ratio=risk_reward,
            sharpe_ratio=sharpe,
            consecutive_wins=stats['consecutive_wins'],
            consecutive_losses=stats['consecutive_losses']
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_current_drawdown(self, stats: Dict[str, Any]) -> float:
        """Calculate current drawdown"""
        if stats['peak_profit'] == 0:
            return 0.0
        return (stats['peak_profit'] - stats['net_profit']) / stats['peak_profit']
    
    def _calculate_risk_reward(self) -> float:
        """Calculate risk/reward ratio"""
        if self.calculator.total_losses == 0:
            return 0.0
        
        avg_loss = self.calculator.config.initial_bet
        avg_win = self.calculator.net_profit / max(self.calculator.total_wins, 1)
        
        if avg_loss == 0:
            return 0.0
        
        return avg_win / avg_loss
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent performance"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        returns = [m.total_pnl for m in self.metrics_history]
        returns = pd.Series(returns).diff().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
    
    def check_risk_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Check for risk alerts"""
        alerts = []
        
        # Check consecutive losses
        if metrics.consecutive_losses >= self.calculator.config.max_consecutive_losses:
            alerts.append({
                'type': 'warning',
                'title': 'Max Consecutive Losses',
                'message': f'Reached {metrics.consecutive_losses} consecutive losses',
                'severity': 'high'
            })
        
        # Check drawdown
        if metrics.current_drawdown > 0.5:
            alerts.append({
                'type': 'warning',
                'title': 'High Drawdown',
                'message': f'Current drawdown: {metrics.current_drawdown:.2%}',
                'severity': 'medium'
            })
        
        # Check win rate
        if metrics.win_rate < 0.4 and metrics.total_trades > 10:
            alerts.append({
                'type': 'warning',
                'title': 'Low Win Rate',
                'message': f'Win rate: {metrics.win_rate:.2%}',
                'severity': 'medium'
            })
        
        self.performance_alerts.extend(alerts)
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate trends
        if len(self.metrics_history) > 1:
            recent = list(self.metrics_history)[-10:]
            
            trend = 'stable'
            if latest.total_pnl > recent[0].total_pnl:
                trend = 'improving'
            elif latest.total_pnl < recent[0].total_pnl:
                trend = 'declining'
        else:
            trend = 'unknown'
        
        summary = {
            'timestamp': latest.timestamp.isoformat(),
            'total_trades': latest.total_trades,
            'win_rate': latest.win_rate,
            'total_pnl': latest.total_pnl,
            'current_drawdown': latest.current_drawdown,
            'max_drawdown': latest.max_drawdown,
            'sharpe_ratio': latest.sharpe_ratio,
            'risk_reward_ratio': latest.risk_reward_ratio,
            'peak_profit': latest.peak_profit,
            'consecutive_losses': latest.consecutive_losses,
            'consecutive_wins': latest.consecutive_wins,
            'trend': trend,
            'alerts': len(self.performance_alerts)
        }
        
        return summary
    
    def export_metrics_to_dataframe(self) -> pd.DataFrame:
        """Export metrics history to DataFrame"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        data = [asdict(m) for m in self.metrics_history]
        df = pd.DataFrame(data)
        
        return df
    
    def plot_performance_chart(self, save_path: Optional[str] = None):
        """Plot performance chart"""
        if not self.metrics_history:
            logger.warning("No metrics history to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            timestamps = [m.timestamp for m in self.metrics_history]
            pnl = [m.total_pnl for m in self.metrics_history]
            drawdown = [m.current_drawdown for m in self.metrics_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot PnL
            ax1.plot(timestamps, pnl, label='Total PnL')
            ax1.axhline(y=0, color='r', linestyle='--', label='Break-even')
            ax1.set_title('Martingale Strategy Performance')
            ax1.set_ylabel('PnL')
            ax1.legend()
            ax1.grid(True)
            
            # Plot drawdown
            ax2.plot(timestamps, drawdown, label='Drawdown', color='r')
            ax2.set_title('Current Drawdown')
            ax2.set_ylabel('Drawdown')
            ax2.set_xlabel('Time')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Performance chart saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
        except Exception as e:
            logger.error(f"Error plotting performance chart: {e}")

class RealTimePositionTracker:
    """Track positions in real-time"""
    
    def __init__(self, position_manager: MartingalePositionManager):
        self.position_manager = position_manager
        self.update_interval = 1.0  # seconds
        self.last_update = datetime.now()
        
    def update_positions(self, market_data: Dict[str, float]):
        """Update positions with latest market data"""
        try:
            positions = self.position_manager.get_positions_summary()
            
            for position in positions['positions']:
                position_id = position['position_id']
                
                if position_id in self.position_manager.positions:
                    pos = self.position_manager.positions[position_id]
                    
                    # Update current price
                    symbol = self._extract_symbol_from_id(position_id)
                    if symbol in market_data:
                        pos.current_price = market_data[symbol]
                        
                        # Recalculate unrealized PnL
                        if pos.position_type == 'long':
                            pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.position_size * pos.leverage
                        else:
                            pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.position_size * pos.leverage
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _extract_symbol_from_id(self, position_id: str) -> str:
        """Extract symbol from position ID"""
        # Simplified - assumes format like 'BTCUSDT_long_12345'
        parts = position_id.split('_')
        if len(parts) > 0:
            return parts[0]
        return position_id

def main():
    """Example usage"""
    # Create configuration
    config = MartingaleConfig(
        strategy_type=MartingaleType.CLASSIC,
        initial_bet=100.0,
        multiplier=2.0,
        max_consecutive_losses=5
    )
    
    # Create calculator and monitor
    calculator = MartingaleCalculator(config)
    monitor = MartingaleMonitor(calculator)
    
    # Simulate some trades
    for i in range(20):
        bet = calculator.calculate_next_bet()
        result = np.random.random() > 0.4  # 60% win rate
        calculator.record_bet(bet, result)
        
        # Update metrics
        metrics = monitor.update_metrics()
        
        if i % 5 == 0:
            alerts = monitor.check_risk_alerts(metrics)
            if alerts:
                print(f"Alerts at trade {i+1}:")
                for alert in alerts:
                    print(f"  - {alert['title']}: {alert['message']}")
    
    # Get summary
    summary = monitor.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

