#!/usr/bin/env python3
"""
AI Trading Prediction Signal Bot - Risk Management Module
Comprehensive risk management and position sizing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

from config import Config, SignalStrength

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class PositionSize(Enum):
    MINIMAL = "MINIMAL"      # 1% of portfolio
    SMALL = "SMALL"          # 2-3% of portfolio
    MEDIUM = "MEDIUM"        # 4-6% of portfolio
    LARGE = "LARGE"          # 7-10% of portfolio
    MAXIMUM = "MAXIMUM"      # 10%+ of portfolio (emergency stop)

@dataclass
class RiskAssessment:
    """Risk assessment results"""
    risk_level: RiskLevel
    portfolio_risk: float  # Current portfolio risk percentage
    position_risk: float    # Risk for new position
    max_position_size: float # Maximum position size allowed
    stop_loss_price: float  # Calculated stop loss price
    take_profit_price: float # Calculated take profit price
    risk_reward_ratio: float # Risk to reward ratio
    volatility: float       # Current market volatility
    correlation_risk: float # Correlation with existing positions
    liquidity_risk: float   # Liquidity risk assessment
    recommendations: List[str] # Risk management recommendations

@dataclass
class PortfolioInfo:
    """Portfolio information"""
    total_value: float
    cash_available: float
    positions: Dict[str, Dict]  # symbol -> position info
    daily_pnl: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    last_updated: datetime

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = config.trading.MAX_POSITION_SIZE
        self.min_position_size = config.trading.MIN_POSITION_SIZE
        self.stop_loss_pct = config.trading.STOP_LOSS_PCT
        self.take_profit_pct = config.trading.TAKE_PROFIT_PCT
        self.max_daily_loss = config.trading.MAX_DAILY_LOSS
        self.max_drawdown = config.trading.MAX_DRAWDOWN
        self.max_concurrent_positions = config.bot.MAX_CONCURRENT_POSITIONS
        
        # Risk tracking
        self.portfolio_history: List[PortfolioInfo] = []
        self.trade_history: List[Dict] = []
        self.risk_alerts: List[str] = []
        
        # Volatility calculation
        self.volatility_window = 20  # Days for volatility calculation
        self.correlation_window = 30  # Days for correlation calculation
    
    def assess_risk(self, market_data: pd.DataFrame, 
                   portfolio_info: Optional[PortfolioInfo] = None,
                   symbol: str = None) -> Optional[RiskAssessment]:
        """Comprehensive risk assessment"""
        try:
            if market_data.empty:
                return None
            
            current_price = market_data['close'].iloc[-1]
            
            # Calculate market volatility
            volatility = self._calculate_volatility(market_data)
            
            # Assess portfolio risk
            portfolio_risk = self._assess_portfolio_risk(portfolio_info)
            
            # Calculate position risk
            position_risk = self._calculate_position_risk(market_data, current_price)
            
            # Determine risk level
            risk_level = self._determine_risk_level(portfolio_risk, position_risk, volatility)
            
            # Calculate position sizing
            max_position_size = self._calculate_max_position_size(
                risk_level, portfolio_risk, volatility, portfolio_info
            )
            
            # Calculate stop loss and take profit
            stop_loss_price, take_profit_price = self._calculate_stop_take_prices(
                current_price, volatility
            )
            
            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                current_price, stop_loss_price, take_profit_price
            )
            
            # Assess correlation risk
            correlation_risk = self._assess_correlation_risk(symbol, portfolio_info)
            
            # Assess liquidity risk
            liquidity_risk = self._assess_liquidity_risk(market_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_level, portfolio_risk, position_risk, volatility,
                max_position_size, risk_reward_ratio
            )
            
            return RiskAssessment(
                risk_level=risk_level,
                portfolio_risk=portfolio_risk,
                position_risk=position_risk,
                max_position_size=max_position_size,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_reward_ratio=risk_reward_ratio,
                volatility=volatility,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing risk: {e}")
            return None
    
    def generate_risk_signal(self, risk_assessment: RiskAssessment) -> Optional[str]:
        """Generate risk-based trading signal"""
        try:
            # Emergency stop conditions
            if risk_assessment.risk_level == RiskLevel.CRITICAL:
                return "EXIT_ALL"  # Emergency exit all positions
            
            # High risk conditions
            if risk_assessment.risk_level == RiskLevel.HIGH:
                if risk_assessment.portfolio_risk > self.max_daily_loss:
                    return "EXIT_ALL"  # Daily loss limit exceeded
                elif risk_assessment.max_position_size < self.min_position_size:
                    return "HOLD"  # Position too risky
            
            # Medium risk conditions
            if risk_assessment.risk_level == RiskLevel.MEDIUM:
                if risk_assessment.risk_reward_ratio < 1.5:
                    return "HOLD"  # Poor risk-reward ratio
                elif risk_assessment.correlation_risk > 0.7:
                    return "HOLD"  # High correlation risk
            
            # Low risk - allow trading
            if risk_assessment.risk_level == RiskLevel.LOW:
                return None  # No risk-based restrictions
            
            return "HOLD"  # Default to hold for medium/high risk
            
        except Exception as e:
            self.logger.error(f"Error generating risk signal: {e}")
            return "HOLD"
    
    def calculate_position_size(self, risk_assessment: RiskAssessment, 
                              signal_confidence: float, 
                              portfolio_value: float) -> float:
        """Calculate optimal position size based on risk assessment"""
        try:
            # Base position size from risk assessment
            base_size = risk_assessment.max_position_size
            
            # Adjust based on signal confidence
            confidence_multiplier = min(signal_confidence * 1.5, 1.0)
            adjusted_size = base_size * confidence_multiplier
            
            # Adjust based on risk level
            risk_multiplier = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.7,
                RiskLevel.HIGH: 0.4,
                RiskLevel.CRITICAL: 0.0
            }.get(risk_assessment.risk_level, 0.0)
            
            final_size = adjusted_size * risk_multiplier
            
            # Ensure within bounds
            final_size = max(self.min_position_size, min(final_size, self.max_position_size))
            
            # Calculate actual position value
            position_value = portfolio_value * final_size
            
            return position_value
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return portfolio_value * self.min_position_size
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        try:
            if len(market_data) < 20:
                return 0.02  # Default 2% volatility
            
            # Calculate daily returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
            
            # Annualize volatility (assuming daily data)
            annualized_volatility = volatility * np.sqrt(252)
            
            return max(0.01, min(annualized_volatility, 1.0))  # Cap between 1% and 100%
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _assess_portfolio_risk(self, portfolio_info: Optional[PortfolioInfo]) -> float:
        """Assess current portfolio risk"""
        try:
            if not portfolio_info:
                return 0.0
            
            # Calculate portfolio risk as percentage of total value
            portfolio_risk = abs(portfolio_info.daily_pnl) / portfolio_info.total_value
            
            # Add drawdown risk
            drawdown_risk = abs(portfolio_info.max_drawdown) / portfolio_info.total_value
            
            # Combined risk
            total_risk = portfolio_risk + drawdown_risk
            
            return min(total_risk, 1.0)  # Cap at 100%
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            return 0.0
    
    def _calculate_position_risk(self, market_data: pd.DataFrame, current_price: float) -> float:
        """Calculate risk for a new position"""
        try:
            # Calculate potential loss based on stop loss
            stop_loss_distance = current_price * self.stop_loss_pct
            position_risk = stop_loss_distance / current_price
            
            return min(position_risk, 0.1)  # Cap at 10%
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            return 0.05
    
    def _determine_risk_level(self, portfolio_risk: float, position_risk: float, 
                            volatility: float) -> RiskLevel:
        """Determine overall risk level"""
        try:
            # Risk scoring
            risk_score = 0
            
            # Portfolio risk component
            if portfolio_risk > self.max_daily_loss:
                risk_score += 3
            elif portfolio_risk > self.max_daily_loss * 0.5:
                risk_score += 2
            elif portfolio_risk > 0:
                risk_score += 1
            
            # Position risk component
            if position_risk > 0.05:
                risk_score += 2
            elif position_risk > 0.03:
                risk_score += 1
            
            # Volatility component
            if volatility > 0.3:
                risk_score += 2
            elif volatility > 0.2:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                return RiskLevel.CRITICAL
            elif risk_score >= 4:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _calculate_max_position_size(self, risk_level: RiskLevel, portfolio_risk: float,
                                   volatility: float, portfolio_info: Optional[PortfolioInfo]) -> float:
        """Calculate maximum position size based on risk"""
        try:
            # Base position sizes by risk level
            base_sizes = {
                RiskLevel.LOW: self.max_position_size,
                RiskLevel.MEDIUM: self.max_position_size * 0.7,
                RiskLevel.HIGH: self.max_position_size * 0.4,
                RiskLevel.CRITICAL: 0.0
            }
            
            base_size = base_sizes.get(risk_level, self.max_position_size * 0.5)
            
            # Adjust for portfolio risk
            if portfolio_risk > self.max_daily_loss * 0.5:
                base_size *= 0.5
            
            # Adjust for volatility
            if volatility > 0.2:
                base_size *= (0.2 / volatility)
            
            # Adjust for number of positions
            if portfolio_info and len(portfolio_info.positions) >= self.max_concurrent_positions:
                base_size *= 0.5
            
            return max(self.min_position_size, min(base_size, self.max_position_size))
            
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}")
            return self.min_position_size
    
    def _calculate_stop_take_prices(self, current_price: float, volatility: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices"""
        try:
            # Adjust stop loss based on volatility
            adjusted_stop_loss = self.stop_loss_pct * (1 + volatility)
            adjusted_take_profit = self.take_profit_pct * (1 - volatility * 0.5)
            
            stop_loss_price = current_price * (1 - adjusted_stop_loss)
            take_profit_price = current_price * (1 + adjusted_take_profit)
            
            return stop_loss_price, take_profit_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop/take prices: {e}")
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            return stop_loss, take_profit
    
    def _calculate_risk_reward_ratio(self, current_price: float, stop_loss_price: float,
                                   take_profit_price: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            risk = current_price - stop_loss_price
            reward = take_profit_price - current_price
            
            if risk <= 0:
                return 0.0
            
            return reward / risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return 1.0
    
    def _assess_correlation_risk(self, symbol: str, portfolio_info: Optional[PortfolioInfo]) -> float:
        """Assess correlation risk with existing positions"""
        try:
            if not portfolio_info or not portfolio_info.positions:
                return 0.0
            
            # For now, return a simple correlation estimate
            # In a real implementation, you would calculate actual correlations
            num_positions = len(portfolio_info.positions)
            
            if num_positions >= self.max_concurrent_positions:
                return 0.8  # High correlation risk
            elif num_positions >= self.max_concurrent_positions * 0.7:
                return 0.5  # Medium correlation risk
            else:
                return 0.2  # Low correlation risk
                
        except Exception as e:
            self.logger.error(f"Error assessing correlation risk: {e}")
            return 0.5
    
    def _assess_liquidity_risk(self, market_data: pd.DataFrame) -> float:
        """Assess liquidity risk based on volume"""
        try:
            if market_data.empty:
                return 0.5
            
            # Calculate average volume
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Convert to liquidity risk (inverse relationship)
            if volume_ratio > 2.0:
                return 0.1  # High liquidity
            elif volume_ratio > 1.0:
                return 0.3  # Good liquidity
            elif volume_ratio > 0.5:
                return 0.6  # Moderate liquidity
            else:
                return 0.9  # Low liquidity
                
        except Exception as e:
            self.logger.error(f"Error assessing liquidity risk: {e}")
            return 0.5
    
    def _generate_recommendations(self, risk_level: RiskLevel, portfolio_risk: float,
                               position_risk: float, volatility: float,
                               max_position_size: float, risk_reward_ratio: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Risk level recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL RISK: Exit all positions immediately")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("HIGH RISK: Reduce position sizes significantly")
        
        # Portfolio risk recommendations
        if portfolio_risk > self.max_daily_loss:
            recommendations.append(f"Daily loss limit exceeded ({portfolio_risk:.1%})")
        elif portfolio_risk > self.max_daily_loss * 0.8:
            recommendations.append("Approaching daily loss limit")
        
        # Position risk recommendations
        if position_risk > 0.05:
            recommendations.append("High position risk - consider smaller position")
        
        # Volatility recommendations
        if volatility > 0.3:
            recommendations.append("High market volatility - use wider stops")
        elif volatility < 0.1:
            recommendations.append("Low volatility - consider tighter stops")
        
        # Risk-reward recommendations
        if risk_reward_ratio < 1.5:
            recommendations.append("Poor risk-reward ratio - reconsider trade")
        elif risk_reward_ratio > 3.0:
            recommendations.append("Excellent risk-reward ratio")
        
        # Position size recommendations
        if max_position_size < self.min_position_size:
            recommendations.append("Position too risky - skip trade")
        elif max_position_size > self.max_position_size * 0.8:
            recommendations.append("Consider maximum position size")
        
        return recommendations
    
    def update_portfolio_info(self, portfolio_info: PortfolioInfo):
        """Update portfolio information"""
        try:
            self.portfolio_history.append(portfolio_info)
            
            # Keep only recent history
            if len(self.portfolio_history) > 100:
                self.portfolio_history = self.portfolio_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio info: {e}")
    
    def add_trade_record(self, trade_info: Dict):
        """Add trade record for risk tracking"""
        try:
            trade_info['timestamp'] = datetime.now()
            self.trade_history.append(trade_info)
            
            # Keep only recent trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error adding trade record: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        try:
            summary = {
                'total_trades': len(self.trade_history),
                'portfolio_history_length': len(self.portfolio_history),
                'risk_alerts': len(self.risk_alerts),
                'current_risk_level': 'UNKNOWN',
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
            if self.portfolio_history:
                latest_portfolio = self.portfolio_history[-1]
                summary['current_risk_level'] = 'CALCULATED'
                summary['max_drawdown'] = latest_portfolio.max_drawdown
                summary['sharpe_ratio'] = latest_portfolio.sharpe_ratio
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    from config import config
    
    # Initialize risk manager
    risk_manager = RiskManager(config)
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    # Create sample portfolio info
    portfolio_info = PortfolioInfo(
        total_value=10000.0,
        cash_available=5000.0,
        positions={'SOLUSDT': {'size': 100, 'value': 5000}},
        daily_pnl=-100.0,
        total_pnl=500.0,
        max_drawdown=-200.0,
        sharpe_ratio=1.2,
        last_updated=datetime.now()
    )
    
    # Assess risk
    risk_assessment = risk_manager.assess_risk(sample_data, portfolio_info, 'SOLUSDT')
    
    if risk_assessment:
        print("Risk Assessment Results:")
        print(f"Risk Level: {risk_assessment.risk_level.value}")
        print(f"Portfolio Risk: {risk_assessment.portfolio_risk:.2%}")
        print(f"Max Position Size: {risk_assessment.max_position_size:.2%}")
        print(f"Risk-Reward Ratio: {risk_assessment.risk_reward_ratio:.2f}")
        print(f"Volatility: {risk_assessment.volatility:.2%}")
        print("\nRecommendations:")
        for rec in risk_assessment.recommendations:
            print(f"  - {rec}")
    else:
        print("No risk assessment available")
