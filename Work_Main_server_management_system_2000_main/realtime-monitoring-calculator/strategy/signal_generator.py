"""
Trading Signal Generator
Generates buy/sell/hold signals from model predictions
Includes risk management and position sizing
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    signal_type: SignalType
    confidence: float
    price: float
    prediction_return: float
    timestamp: float
    metadata: Dict


class SignalGenerator:
    """
    Generate trading signals from model predictions
    Implements risk management and signal confirmation
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_prediction_return: float = 0.002,  # 0.2%
        confirmation_periods: int = 3,
        risk_per_trade: float = 0.02  # 2% of portfolio
    ):
        self.min_confidence = min_confidence
        self.min_prediction_return = min_prediction_return
        self.confirmation_periods = confirmation_periods
        self.risk_per_trade = risk_per_trade
        
        # Recent signals for confirmation
        self.recent_signals = []
        
    def generate_signal(
        self,
        prediction: Dict,
        current_price: float,
        metadata: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from prediction
        
        Args:
            prediction: Model prediction dict
            current_price: Current market price
            metadata: Additional metadata
        
        Returns:
            TradingSignal or None if no signal
        """
        # Extract prediction
        pred_value = prediction['prediction']
        horizon = prediction.get('horizon', 6)
        
        # Compute expected return over prediction horizon
        if len(pred_value.shape) > 0:
            # Use mean or last value of prediction
            expected_return = float(np.mean(pred_value))
        else:
            expected_return = float(pred_value)
        
        # Compute confidence (based on prediction strength)
        confidence = self._compute_confidence(prediction, expected_return)
        
        # Determine signal type
        signal_type = self._determine_signal_type(expected_return, confidence)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Check confirmation if needed
        if not self._confirm_signal(signal_type, confidence):
            return None
        
        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            prediction_return=expected_return,
            timestamp=prediction.get('timestamp', 0),
            metadata=metadata or {}
        )
        
        # Store for confirmation
        self.recent_signals.append(signal)
        if len(self.recent_signals) > self.confirmation_periods * 2:
            self.recent_signals.pop(0)
        
        return signal
    
    def _compute_confidence(
        self,
        prediction: Dict,
        expected_return: float
    ) -> float:
        """Compute signal confidence"""
        # Base confidence on prediction magnitude
        base_confidence = min(abs(expected_return) * 100, 1.0)
        
        # Boost confidence if ensemble (multiple models agree)
        if 'individual_predictions' in prediction:
            individual_preds = prediction['individual_predictions']
            
            # Check consensus
            signs = [np.sign(np.mean(p)) for p in individual_preds]
            consensus = len(set(signs)) == 1  # All same sign
            
            if consensus:
                base_confidence *= 1.2
        
        return min(base_confidence, 0.99)
    
    def _determine_signal_type(
        self,
        expected_return: float,
        confidence: float
    ) -> SignalType:
        """Determine signal type from expected return"""
        
        if confidence < self.min_confidence:
            return SignalType.HOLD
        
        if abs(expected_return) < self.min_prediction_return:
            return SignalType.HOLD
        
        if expected_return > 0:
            return SignalType.BUY
        else:
            return SignalType.SELL
    
    def _confirm_signal(self, signal_type: SignalType, confidence: float) -> bool:
        """Check if signal is confirmed by recent signals"""
        if len(self.recent_signals) < self.confirmation_periods:
            return False
        
        # Count recent signals of same type
        recent_same = sum(
            1 for s in self.recent_signals[-self.confirmation_periods:]
            if s.signal_type == signal_type
        )
        
        # Require majority confirmation
        return recent_same >= (self.confirmation_periods // 2)
    
    def get_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        volatility: float = 0.02  # 2% daily volatility
    ) -> float:
        """
        Calculate position size based on signal and risk
        
        Uses Kelly Criterion or fixed-fraction sizing
        """
        # Base position size
        base_size = portfolio_value * self.risk_per_trade
        
        # Adjust by confidence
        confidence_multiplier = signal.confidence
        
        # Adjust by volatility
        volatility_multiplier = min(0.02 / volatility, 2.0)
        
        # Final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        # Limit max position
        max_position = portfolio_value * 0.1  # Max 10% of portfolio
        position_size = min(position_size, max_position)
        
        return position_size


class EWMA_SignalSmoothing:
    """
    Exponential Weighted Moving Average smoothing for noisy signals
    Reduces false signals by smoothing predictions
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.smoothed_value = None
    
    def smooth(self, value: float) -> float:
        """Apply EWMA smoothing"""
        if self.smoothed_value is None:
            self.smoothed_value = value
        else:
            self.smoothed_value = self.alpha * value + (1 - self.alpha) * self.smoothed_value
        
        return self.smoothed_value
    
    def reset(self):
        """Reset smoothed value"""
        self.smoothed_value = None


class SignalValidator:
    """
    Validate signals before execution
    Checks risk limits, portfolio constraints, etc.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.15
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
    
    def validate(
        self,
        signal: TradingSignal,
        portfolio: Dict,
        daily_pnl: float
    ) -> bool:
        """
        Validate signal against risk limits
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            daily_pnl: Current daily P&L
        
        Returns:
            True if signal is valid
        """
        # Check daily loss limit
        if daily_pnl < -abs(self.max_daily_loss):
            logger.warning(f"Daily loss limit reached: {daily_pnl:.2%}")
            return False
        
        # Check position size
        proposed_position = portfolio.get('current_positions', 0) + abs(signal.prediction_return)
        if proposed_position > self.max_position_size:
            logger.warning(f"Position size too large: {proposed_position:.2%}")
            return False
        
        return True

