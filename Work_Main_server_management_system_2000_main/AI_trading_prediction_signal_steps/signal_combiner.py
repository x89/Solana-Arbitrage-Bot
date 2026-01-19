#!/usr/bin/env python3
"""
AI Trading Prediction Signal Bot - Core Signal Generation System
Unified signal generation combining multiple AI models and indicators
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config, SignalStrength
from AI_trading_prediction_signal_bot.ai_models import AIModelManager
from AI_trading_prediction_signal_bot.technical_analysis import TechnicalAnalyzer
from AI_trading_prediction_signal_bot.risk_manager import RiskManager

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"

class SignalSource(Enum):
    FORECAST = "forecast"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    ML = "ml"
    RISK = "risk"

@dataclass
class Signal:
    """Individual signal from a specific source"""
    signal_type: SignalType
    source: SignalSource
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    price: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CombinedSignal:
    """Combined signal from multiple sources"""
    signal_type: SignalType
    confidence: float
    strength: SignalStrength
    timestamp: datetime
    price: float
    individual_signals: List[Signal]
    weights: Dict[str, float]
    reasoning: str
    
class SignalGenerator:
    """Main signal generation class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ai_manager = AIModelManager(config)
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.risk_manager = RiskManager(config)
        
        # Signal history for tracking
        self.signal_history: List[CombinedSignal] = []
        self.last_signals: Dict[SignalSource, Signal] = {}
        
        # Performance tracking
        self.signal_performance = {
            'total_signals': 0,
            'correct_signals': 0,
            'accuracy': 0.0
        }
    
    def generate_signals(self, market_data: pd.DataFrame, 
                        portfolio_info: Dict = None) -> CombinedSignal:
        """
        Generate trading signals from all available sources
        
        Args:
            market_data: OHLCV data with technical indicators
            portfolio_info: Current portfolio information
            
        Returns:
            CombinedSignal: Final trading signal
        """
        try:
            current_price = market_data['close'].iloc[-1]
            timestamp = datetime.now()
            
            # Generate signals from each source
            signals = []
            
            # 1. AI Forecasting Signals
            if self.config.signal.ENABLE_FORECAST_SIGNALS:
                forecast_signal = self._generate_forecast_signal(market_data)
                if forecast_signal:
                    signals.append(forecast_signal)
            
            # 2. Pattern Detection Signals
            if self.config.signal.ENABLE_PATTERN_SIGNALS:
                pattern_signal = self._generate_pattern_signal(market_data)
                if pattern_signal:
                    signals.append(pattern_signal)
            
            # 3. Sentiment Analysis Signals
            if self.config.signal.ENABLE_SENTIMENT_SIGNALS:
                sentiment_signal = self._generate_sentiment_signal(market_data)
                if sentiment_signal:
                    signals.append(sentiment_signal)
            
            # 4. Technical Analysis Signals
            if self.config.signal.ENABLE_TECHNICAL_SIGNALS:
                technical_signal = self._generate_technical_signal(market_data)
                if technical_signal:
                    signals.append(technical_signal)
            
            # 5. Machine Learning Signals
            if self.config.signal.ENABLE_ML_SIGNALS:
                ml_signal = self._generate_ml_signal(market_data)
                if ml_signal:
                    signals.append(ml_signal)
            
            # 6. Risk Management Signals
            risk_signal = self._generate_risk_signal(market_data, portfolio_info)
            if risk_signal:
                signals.append(risk_signal)
            
            # Combine signals
            combined_signal = self._combine_signals(signals, current_price, timestamp)
            
            # Update signal history
            self.signal_history.append(combined_signal)
            self._update_signal_performance()
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return self._create_default_signal(current_price, timestamp)
    
    def _generate_forecast_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from AI forecasting models"""
        try:
            # Get forecasts from AI models
            forecasts = self.ai_manager.get_forecasts(market_data)
            
            if not forecasts:
                return None
            
            # Analyze forecast trends
            forecast_signal = self._analyze_forecast_trends(forecasts)
            
            if forecast_signal:
                return Signal(
                    signal_type=forecast_signal,
                    source=SignalSource.FORECAST,
                    strength=self._calculate_forecast_strength(forecasts),
                    confidence=self._calculate_forecast_confidence(forecasts),
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'forecasts': forecasts}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating forecast signal: {e}")
            return None
    
    def _generate_pattern_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from chart pattern detection"""
        try:
            # Detect chart patterns
            patterns = self.ai_manager.detect_patterns(market_data)
            
            if not patterns:
                return None
            
            # Analyze pattern signals
            pattern_signal = self._analyze_pattern_signals(patterns)
            
            if pattern_signal:
                return Signal(
                    signal_type=pattern_signal,
                    source=SignalSource.PATTERN,
                    strength=self._calculate_pattern_strength(patterns),
                    confidence=self._calculate_pattern_confidence(patterns),
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'patterns': patterns}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating pattern signal: {e}")
            return None
    
    def _generate_sentiment_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from sentiment analysis"""
        try:
            # Get sentiment analysis
            sentiment_data = self.ai_manager.analyze_sentiment(market_data)
            
            if not sentiment_data:
                return None
            
            # Analyze sentiment signals
            sentiment_signal = self._analyze_sentiment_signals(sentiment_data)
            
            if sentiment_signal:
                return Signal(
                    signal_type=sentiment_signal,
                    source=SignalSource.SENTIMENT,
                    strength=self._calculate_sentiment_strength(sentiment_data),
                    confidence=self._calculate_sentiment_confidence(sentiment_data),
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'sentiment': sentiment_data}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signal: {e}")
            return None
    
    def _generate_technical_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from technical analysis"""
        try:
            # Calculate technical indicators
            technical_data = self.technical_analyzer.calculate_indicators(market_data)
            
            # Analyze technical signals
            technical_signal = self.technical_analyzer.generate_signal(technical_data)
            
            if technical_signal:
                return Signal(
                    signal_type=technical_signal,
                    source=SignalSource.TECHNICAL,
                    strength=self.technical_analyzer.calculate_signal_strength(technical_data),
                    confidence=self.technical_analyzer.calculate_signal_confidence(technical_data),
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'technical': technical_data}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating technical signal: {e}")
            return None
    
    def _generate_ml_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate signal from machine learning models"""
        try:
            # Get ML prediction
            ml_prediction = self.ai_manager.get_ml_prediction(market_data)
            
            if not ml_prediction:
                return None
            
            # Convert ML prediction to signal
            ml_signal = self._convert_ml_prediction_to_signal(ml_prediction)
            
            if ml_signal:
                return Signal(
                    signal_type=ml_signal,
                    source=SignalSource.ML,
                    strength=self._calculate_ml_strength(ml_prediction),
                    confidence=self._calculate_ml_confidence(ml_prediction),
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'ml_prediction': ml_prediction}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating ML signal: {e}")
            return None
    
    def _generate_risk_signal(self, market_data: pd.DataFrame, 
                             portfolio_info: Dict = None) -> Optional[Signal]:
        """Generate risk management signals"""
        try:
            # Check risk conditions
            risk_assessment = self.risk_manager.assess_risk(market_data, portfolio_info)
            
            if not risk_assessment:
                return None
            
            # Generate risk-based signals
            risk_signal = self.risk_manager.generate_risk_signal(risk_assessment)
            
            if risk_signal:
                return Signal(
                    signal_type=risk_signal,
                    source=SignalSource.RISK,
                    strength=SignalStrength.VERY_STRONG,  # Risk signals are always strong
                    confidence=1.0,  # Risk signals have maximum confidence
                    timestamp=datetime.now(),
                    price=market_data['close'].iloc[-1],
                    metadata={'risk_assessment': risk_assessment}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating risk signal: {e}")
            return None
    
    def _combine_signals(self, signals: List[Signal], 
                        current_price: float, timestamp: datetime) -> CombinedSignal:
        """Combine multiple signals into a final trading signal"""
        if not signals:
            return self._create_default_signal(current_price, timestamp)
        
        # Filter signals by minimum confidence
        valid_signals = [
            s for s in signals 
            if s.confidence >= self.config.signal.MIN_SIGNAL_CONFIDENCE
        ]
        
        if not valid_signals:
            return self._create_default_signal(current_price, timestamp)
        
        # Calculate weighted signal
        signal_weights = self.config.signal.WEIGHTS
        weighted_score = 0.0
        total_weight = 0.0
        
        signal_summary = []
        
        for signal in valid_signals:
            weight = signal_weights.get(signal.source.value, 0.0)
            signal_value = self._signal_to_numeric(signal.signal_type)
            weighted_score += signal_value * weight * signal.confidence
            total_weight += weight
            
            signal_summary.append(f"{signal.source.value}: {signal.signal_type.value} "
                                f"(conf: {signal.confidence:.2f}, strength: {signal.strength.name})")
        
        # Normalize score
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0
        
        # Determine final signal
        final_signal_type = self._numeric_to_signal(normalized_score)
        final_confidence = min(abs(normalized_score), 1.0)
        final_strength = self._calculate_combined_strength(valid_signals)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(valid_signals, signal_summary, normalized_score)
        
        return CombinedSignal(
            signal_type=final_signal_type,
            confidence=final_confidence,
            strength=final_strength,
            timestamp=timestamp,
            price=current_price,
            individual_signals=valid_signals,
            weights=signal_weights,
            reasoning=reasoning
        )
    
    def _signal_to_numeric(self, signal_type: SignalType) -> float:
        """Convert signal type to numeric value"""
        mapping = {
            SignalType.SELL: -1.0,
            SignalType.HOLD: 0.0,
            SignalType.BUY: 1.0,
            SignalType.EXIT: 0.0
        }
        return mapping.get(signal_type, 0.0)
    
    def _numeric_to_signal(self, score: float) -> SignalType:
        """Convert numeric score to signal type"""
        if score > 0.3:
            return SignalType.BUY
        elif score < -0.3:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_combined_strength(self, signals: List[Signal]) -> SignalStrength:
        """Calculate combined signal strength"""
        if not signals:
            return SignalStrength.WEAK
        
        # Use the strongest signal as base
        max_strength = max(signal.strength for signal in signals)
        
        # Boost strength if multiple sources agree
        buy_signals = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_signals = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        
        if buy_signals >= 2 or sell_signals >= 2:
            # Multiple sources agree - boost strength
            strength_values = [SignalStrength.WEAK, SignalStrength.MODERATE, 
                            SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            current_index = strength_values.index(max_strength)
            if current_index < len(strength_values) - 1:
                return strength_values[current_index + 1]
        
        return max_strength
    
    def _generate_reasoning(self, signals: List[Signal], 
                          signal_summary: List[str], score: float) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning_parts = []
        
        # Add signal summary
        reasoning_parts.append("Signal Analysis:")
        for summary in signal_summary:
            reasoning_parts.append(f"  - {summary}")
        
        # Add combined score interpretation
        reasoning_parts.append(f"\nCombined Score: {score:.3f}")
        
        if score > 0.5:
            reasoning_parts.append("Strong bullish consensus across multiple indicators")
        elif score > 0.3:
            reasoning_parts.append("Moderate bullish sentiment")
        elif score < -0.5:
            reasoning_parts.append("Strong bearish consensus across multiple indicators")
        elif score < -0.3:
            reasoning_parts.append("Moderate bearish sentiment")
        else:
            reasoning_parts.append("Mixed signals - maintaining neutral position")
        
        return "\n".join(reasoning_parts)
    
    def _create_default_signal(self, price: float, timestamp: datetime) -> CombinedSignal:
        """Create default HOLD signal when no valid signals are available"""
        return CombinedSignal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strength=SignalStrength.WEAK,
            timestamp=timestamp,
            price=price,
            individual_signals=[],
            weights=self.config.signal.WEIGHTS,
            reasoning="No valid signals available - maintaining HOLD position"
        )
    
    def _update_signal_performance(self):
        """Update signal performance tracking"""
        # This would be implemented to track signal accuracy over time
        # For now, just increment total signals
        self.signal_performance['total_signals'] += 1
    
    def get_signal_history(self, limit: int = 100) -> List[CombinedSignal]:
        """Get recent signal history"""
        return self.signal_history[-limit:]
    
    def get_performance_stats(self) -> Dict:
        """Get signal performance statistics"""
        return self.signal_performance.copy()
    
    # Helper methods for individual signal analysis
    def _analyze_forecast_trends(self, forecasts: Dict) -> Optional[SignalType]:
        """Analyze forecast trends to determine signal"""
        # Implementation would analyze multiple forecast models
        # and determine overall trend direction
        return None
    
    def _analyze_pattern_signals(self, patterns: Dict) -> Optional[SignalType]:
        """Analyze detected patterns to determine signal"""
        # Implementation would analyze chart patterns
        # and determine signal direction
        return None
    
    def _analyze_sentiment_signals(self, sentiment: Dict) -> Optional[SignalType]:
        """Analyze sentiment data to determine signal"""
        # Implementation would analyze sentiment scores
        # and determine signal direction
        return None
    
    def _convert_ml_prediction_to_signal(self, prediction: Dict) -> Optional[SignalType]:
        """Convert ML model prediction to signal type"""
        # Implementation would convert ML model output
        # to trading signal
        return None
    
    # Strength and confidence calculation methods
    def _calculate_forecast_strength(self, forecasts: Dict) -> SignalStrength:
        """Calculate forecast signal strength"""
        return SignalStrength.MODERATE
    
    def _calculate_forecast_confidence(self, forecasts: Dict) -> float:
        """Calculate forecast signal confidence"""
        return 0.7
    
    def _calculate_pattern_strength(self, patterns: Dict) -> SignalStrength:
        """Calculate pattern signal strength"""
        return SignalStrength.STRONG
    
    def _calculate_pattern_confidence(self, patterns: Dict) -> float:
        """Calculate pattern signal confidence"""
        return 0.8
    
    def _calculate_sentiment_strength(self, sentiment: Dict) -> SignalStrength:
        """Calculate sentiment signal strength"""
        return SignalStrength.MODERATE
    
    def _calculate_sentiment_confidence(self, sentiment: Dict) -> float:
        """Calculate sentiment signal confidence"""
        return 0.6
    
    def _calculate_ml_strength(self, prediction: Dict) -> SignalStrength:
        """Calculate ML signal strength"""
        return SignalStrength.STRONG
    
    def _calculate_ml_confidence(self, prediction: Dict) -> float:
        """Calculate ML signal confidence"""
        return 0.85

if __name__ == "__main__":
    # Example usage
    from config import config
    
    # Initialize signal generator
    signal_generator = SignalGenerator(config)
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Generate signals
    signal = signal_generator.generate_signals(sample_data)
    
    print("Generated Signal:")
    print(f"Type: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Strength: {signal.strength.name}")
    print(f"Reasoning: {signal.reasoning}")
