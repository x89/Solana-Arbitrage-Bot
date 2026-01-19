#!/usr/bin/env python3
"""
Predictive Signal Analyzer
Analyze AI prediction signals in real-time
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class PredictionSignal:
    """AI prediction signal"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price_prediction: float
    predicted_change: float
    features: Dict[str, float]
    model_version: str

@dataclass
class SignalAnalysis:
    """Signal analysis result"""
    signal: PredictionSignal
    analysis_score: float
    recommendation: str
    risk_level: str
    expected_return: float
    confidence_adjusted: float
    reasoning: List[str]

class PredictiveSignalAnalyzer:
    """Analyze AI prediction signals in real-time"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.signal_history = deque(maxlen=1000)
        self.performance_tracking = {}
        self.model_ensemble = []
        
        logger.info("PredictiveSignalAnalyzer initialized")
    
    def analyze_signal(self, signal: PredictionSignal) -> SignalAnalysis:
        """Analyze a prediction signal"""
        try:
            # Calculate analysis score
            analysis_score = self._calculate_analysis_score(signal)
            
            # Get recommendation
            recommendation = self._get_recommendation(signal, analysis_score)
            
            # Assess risk level
            risk_level = self._assess_risk_level(signal, analysis_score)
            
            # Estimate expected return
            expected_return = self._estimate_expected_return(signal)
            
            # Adjust confidence based on historical performance
            confidence_adjusted = self._adjust_confidence(signal)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(signal, analysis_score)
            
            analysis = SignalAnalysis(
                signal=signal,
                analysis_score=analysis_score,
                recommendation=recommendation,
                risk_level=risk_level,
                expected_return=expected_return,
                confidence_adjusted=confidence_adjusted,
                reasoning=reasoning
            )
            
            self.signal_history.append(signal)
            
            logger.info(f"Signal analyzed: {signal.signal_id}, Score: {analysis_score:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing signal: {e}")
            return None
    
    def _calculate_analysis_score(self, signal: PredictionSignal) -> float:
        """Calculate overall analysis score"""
        score = 0.0
        
        # Base confidence (40%)
        score += signal.confidence * 0.4
        
        # Historical performance (30%)
        hist_score = self._get_historical_performance(signal)
        score += hist_score * 0.3
        
        # Feature analysis (20%)
        feature_score = self._analyze_features(signal.features)
        score += feature_score * 0.2
        
        # Signal strength (10%)
        strength = abs(signal.predicted_change)
        strength_score = min(strength / 0.1, 1.0)  # Normalize
        score += strength_score * 0.1
        
        return min(score, 1.0)
    
    def _get_recommendation(self, signal: PredictionSignal, analysis_score: float) -> str:
        """Get trading recommendation"""
        if analysis_score < 0.3:
            return "REJECT"
        elif analysis_score < 0.5:
            return "HOLD"
        elif analysis_score < 0.7:
            if signal.signal_type == 'buy':
                return "BUY (WEAK)"
            elif signal.signal_type == 'sell':
                return "SELL (WEAK)"
        else:
            if signal.signal_type == 'buy':
                return "BUY (STRONG)"
            elif signal.signal_type == 'sell':
                return "SELL (STRONG)"
            else:
                return "HOLD"
    
    def _assess_risk_level(self, signal: PredictionSignal, analysis_score: float) -> str:
        """Assess risk level"""
        # Lower confidence = higher risk
        if signal.confidence < 0.5:
            return "HIGH"
        elif signal.confidence < 0.7:
            return "MEDIUM"
        else:
            if abs(signal.predicted_change) > 0.05:
                return "MEDIUM"  # Large predicted change
            return "LOW"
    
    def _estimate_expected_return(self, signal: PredictionSignal) -> float:
        """Estimate expected return"""
        # Expected return = confidence * predicted change
        return signal.confidence * signal.predicted_change
    
    def _adjust_confidence(self, signal: PredictionSignal) -> float:
        """Adjust confidence based on historical performance"""
        # Get model performance
        model_perf = self._get_model_performance(signal.model_version)
        
        # Adjust confidence based on historical accuracy
        adjusted = signal.confidence * model_perf
        
        return max(0.0, min(adjusted, 1.0))
    
    def _generate_reasoning(self, signal: PredictionSignal, analysis_score: float) -> List[str]:
        """Generate reasoning for the analysis"""
        reasons = []
        
        if signal.confidence > self.confidence_threshold:
            reasons.append(f"High confidence ({signal.confidence:.2%})")
        
        if analysis_score > 0.7:
            reasons.append("Strong signal score")
        elif analysis_score < 0.3:
            reasons.append("Weak signal score")
        
        if abs(signal.predicted_change) > 0.03:
            reasons.append(f"Significant predicted change ({signal.predicted_change:.2%})")
        
        # Add feature-based reasoning
        if 'volume' in signal.features:
            vol = signal.features['volume']
            if vol > 1.5:
                reasons.append("High volume detected")
            elif vol < 0.5:
                reasons.append("Low volume - caution advised")
        
        return reasons
    
    def _get_historical_performance(self, signal: PredictionSignal) -> float:
        """Get historical performance for similar signals"""
        # Simplified: average performance of recent similar signals
        if len(self.signal_history) == 0:
            return 0.5
        
        similar_signals = [
            s for s in self.signal_history 
            if s.signal_type == signal.signal_type and s.symbol == signal.symbol
        ]
        
        if len(similar_signals) == 0:
            return 0.5
        
        # Average confidence of similar signals
        avg_confidence = np.mean([s.confidence for s in similar_signals])
        return avg_confidence
    
    def _analyze_features(self, features: Dict[str, float]) -> float:
        """Analyze signal features"""
        score = 0.5  # Neutral
        
        # Volume analysis
        if 'volume' in features:
            vol = features['volume']
            if 0.8 <= vol <= 1.2:
                score += 0.2  # Normal volume
            elif vol > 1.5:
                score += 0.1  # High volume
        
        # RSI analysis
        if 'rsi' in features:
            rsi = features['rsi']
            if rsi < 30:
                score += 0.1  # Oversold
            elif rsi > 70:
                score -= 0.1  # Overbought
        
        # Trend analysis
        if 'trend_strength' in features:
            trend = features['trend_strength']
            score += trend * 0.2
        
        return max(0.0, min(score, 1.0))
    
    def _get_model_performance(self, model_version: str) -> float:
        """Get historical performance of the model"""
        if model_version not in self.performance_tracking:
            return 0.7  # Default performance
        
        return self.performance_tracking[model_version]
    
    def update_performance(self, signal: PredictionSignal, actual_result: float):
        """Update model performance tracking"""
        # Calculate accuracy
        predicted_return = signal.predicted_change
        accuracy = 1.0 - abs(predicted_return - actual_result) / abs(actual_result) if actual_result != 0 else 0.0
        
        # Update tracking
        if signal.model_version not in self.performance_tracking:
            self.performance_tracking[signal.model_version] = []
        
        self.performance_tracking[signal.model_version].append(accuracy)
        
        # Keep only recent performance
        if len(self.performance_tracking[signal.model_version]) > 100:
            self.performance_tracking[signal.model_version] = \
                self.performance_tracking[signal.model_version][-100:]
        
        logger.debug(f"Performance updated for {signal.model_version}: {accuracy:.2f}")
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed signals"""
        if len(self.signal_history) == 0:
            return {}
        
        signals = list(self.signal_history)
        
        stats = {
            'total_signals': len(signals),
            'buy_signals': sum(1 for s in signals if s.signal_type == 'buy'),
            'sell_signals': sum(1 for s in signals if s.signal_type == 'sell'),
            'hold_signals': sum(1 for s in signals if s.signal_type == 'hold'),
            'avg_confidence': np.mean([s.confidence for s in signals]),
            'avg_predicted_change': np.mean([s.predicted_change for s in signals]),
            'high_confidence_signals': sum(1 for s in signals if s.confidence > self.confidence_threshold)
        }
        
        return stats

class SignalValidator:
    """Validate prediction signals"""
    
    def __init__(self):
        self.validation_rules = []
        logger.info("SignalValidator initialized")
    
    def validate_signal(self, signal: PredictionSignal) -> Tuple[bool, List[str]]:
        """Validate a signal"""
        is_valid = True
        issues = []
        
        # Check confidence
        if signal.confidence < 0.3:
            is_valid = False
            issues.append("Confidence too low")
        
        # Check timestamp
        if signal.timestamp < datetime.now() - timedelta(minutes=5):
            is_valid = False
            issues.append("Signal too old")
        
        # Check required fields
        if not signal.symbol or not signal.signal_type:
            is_valid = False
            issues.append("Missing required fields")
        
        # Check features
        if not signal.features or len(signal.features) == 0:
            is_valid = False
            issues.append("No features provided")
        
        return is_valid, issues
    
    def add_validation_rule(self, rule_name: str, rule_func: callable):
        """Add custom validation rule"""
        self.validation_rules.append((rule_name, rule_func))

class EnsembleAnalyzer:
    """Ensemble analysis for multiple AI models"""
    
    def __init__(self):
        self.models = {}
        logger.info("EnsembleAnalyzer initialized")
    
    def add_model(self, model_id: str, model: Any, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[model_id] = {
            'model': model,
            'weight': weight
        }
        logger.info(f"Added model {model_id} to ensemble with weight {weight}")
    
    def ensemble_predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get ensemble prediction"""
        predictions = {}
        total_weight = sum(m['weight'] for m in self.models.values())
        
        for model_id, model_data in self.models.items():
            try:
                # Get prediction from model (simplified)
                prediction = model_data['model'].predict(features)
                weight = model_data['weight']
                
                # Weighted average
                for key, value in prediction.items():
                    if key not in predictions:
                        predictions[key] = 0
                    predictions[key] += value * (weight / total_weight)
                    
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_id}: {e}")
        
        return predictions
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Get contribution of each model"""
        return {model_id: model_data['weight'] 
                for model_id, model_data in self.models.items()}

def main():
    """Example usage"""
    analyzer = PredictiveSignalAnalyzer()
    
    # Create sample signal
    signal = PredictionSignal(
        signal_id="test_001",
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        signal_type="buy",
        confidence=0.85,
        price_prediction=50000.0,
        predicted_change=0.03,
        features={
            'volume': 1.2,
            'rsi': 45.0,
            'trend_strength': 0.7
        },
        model_version="model_v1"
    )
    
    # Analyze signal
    analysis = analyzer.analyze_signal(signal)
    
    print(f"Signal Analysis:")
    print(f"  Score: {analysis.analysis_score:.2f}")
    print(f"  Recommendation: {analysis.recommendation}")
    print(f"  Risk Level: {analysis.risk_level}")
    print(f"  Expected Return: {analysis.expected_return:.2%}")
    print(f"  Reasoning: {analysis.reasoning}")

if __name__ == "__main__":
    main()

