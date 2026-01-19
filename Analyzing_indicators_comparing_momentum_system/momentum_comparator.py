#!/usr/bin/env python3
"""
Momentum Comparator Module
Comprehensive momentum comparison across timeframes, symbols, and indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MomentumComparison:
    """Momentum comparison result"""
    symbol: str
    comparison_type: str
    timeframes: List[str]
    symbols: List[str]
    timeframes_data: Dict[str, Any]
    symbols_data: Dict[str, Any]
    correlation_scores: Dict[str, float]
    divergence_signals: List[str]
    recommendations: List[str]
    timestamp: datetime

class MomentumComparator:
    """Compare momentum across different dimensions"""
    
    def __init__(self):
        self.comparisons = []
    
    def compare_timeframes(self, momentum_data: Dict[str, Dict[str, Any]], 
                          symbol: str) -> MomentumComparison:
        """
        Compare momentum across multiple timeframes
        
        Args:
            momentum_data: Dictionary with timeframe as key and momentum data as value
            symbol: Trading symbol
            
        Returns:
            MomentumComparison object
        """
        try:
            timeframes = list(momentum_data.keys())
            timeframes_data = {}
            momentum_scores = []
            trend_directions = []
            
            for tf, data in momentum_data.items():
                timeframes_data[tf] = {
                    'momentum_score': data.get('momentum_score', 0),
                    'trend_direction': data.get('trend_direction', 'neutral'),
                    'momentum_strength': data.get('momentum_strength', 'weak'),
                    'indicators_count': len(data.get('indicators', {}))
                }
                momentum_scores.append(data.get('momentum_score', 0))
                trend_directions.append(data.get('trend_direction', 'neutral'))
            
            # Calculate correlation
            correlation_scores = self._calculate_timeframe_correlation(momentum_scores)
            
            # Detect divergence
            divergence_signals = self._detect_timeframe_divergence(timeframes, trend_directions, momentum_scores)
            
            # Generate recommendations
            recommendations = self._generate_timeframe_recommendations(
                timeframes_data, momentum_scores, divergence_signals
            )
            
            return MomentumComparison(
                symbol=symbol,
                comparison_type='timeframes',
                timeframes=timeframes,
                symbols=[symbol],
                timeframes_data=timeframes_data,
                symbols_data={},
                correlation_scores=correlation_scores,
                divergence_signals=divergence_signals,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error comparing timeframes: {e}")
            return None
    
    def compare_symbols(self, momentum_data: Dict[str, Dict[str, Any]], 
                       timeframe: str) -> MomentumComparison:
        """
        Compare momentum across multiple symbols
        
        Args:
            momentum_data: Dictionary with symbol as key and momentum data as value
            timeframe: Trading timeframe
            
        Returns:
            MomentumComparison object
        """
        try:
            symbols = list(momentum_data.keys())
            symbols_data = {}
            momentum_scores = []
            trend_directions = []
            
            for symbol, data in momentum_data.items():
                symbols_data[symbol] = {
                    'momentum_score': data.get('momentum_score', 0),
                    'trend_direction': data.get('trend_direction', 'neutral'),
                    'momentum_strength': data.get('momentum_strength', 'weak'),
                    'indicators_count': len(data.get('indicators', {}))
                }
                momentum_scores.append(data.get('momentum_score', 0))
                trend_directions.append(data.get('trend_direction', 'neutral'))
            
            # Calculate correlation
            correlation_scores = self._calculate_symbol_correlation(momentum_scores)
            
            # Detect divergence
            divergence_signals = self._detect_symbol_divergence(symbols, trend_directions, momentum_scores)
            
            # Generate recommendations
            recommendations = self._generate_symbol_recommendations(
                symbols_data, momentum_scores, divergence_signals
            )
            
            return MomentumComparison(
                symbol=', '.join(symbols),
                comparison_type='symbols',
                timeframes=[timeframe],
                symbols=symbols,
                timeframes_data={},
                symbols_data=symbols_data,
                correlation_scores=correlation_scores,
                divergence_signals=divergence_signals,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error comparing symbols: {e}")
            return None
    
    def compare_indicators(self, indicator_data: Dict[str, Any], 
                          timeframe: str = '1h') -> Dict[str, Any]:
        """
        Compare momentum signals from different indicators
        
        Args:
            indicator_data: Dictionary with indicator data
            timeframe: Trading timeframe
            
        Returns:
            Comparison results
        """
        try:
            buy_signals = []
            sell_signals = []
            strong_signals = []
            
            for indicator, data in indicator_data.items():
                signal = data.get('signal', 'hold')
                strength = data.get('strength', 0)
                
                if signal == 'buy':
                    buy_signals.append(indicator)
                elif signal == 'sell':
                    sell_signals.append(indicator)
                
                if strength > 0.7:
                    strong_signals.append(indicator)
            
            # Calculate agreement
            total_signals = len(buy_signals) + len(sell_signals)
            agreement_ratio = abs(len(buy_signals) - len(sell_signals)) / total_signals if total_signals > 0 else 0
            
            # Determine consensus
            if len(buy_signals) > len(sell_signals) * 1.5:
                consensus = 'strong_buy'
            elif len(buy_signals) > len(sell_signals):
                consensus = 'buy'
            elif len(sell_signals) > len(buy_signals) * 1.5:
                consensus = 'strong_sell'
            elif len(sell_signals) > len(buy_signals):
                consensus = 'sell'
            else:
                consensus = 'neutral'
            
            return {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'strong_signals': strong_signals,
                'agreement_ratio': agreement_ratio,
                'consensus': consensus,
                'total_indicators': len(indicator_data),
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Error comparing indicators: {e}")
            return {}
    
    def _calculate_timeframe_correlation(self, momentum_scores: List[float]) -> Dict[str, float]:
        """Calculate correlation between timeframes"""
        try:
            # Create DataFrame from scores
            df = pd.DataFrame(momentum_scores)
            
            # Calculate correlation if we have multiple data points
            correlation_scores = {}
            
            if len(momentum_scores) > 1:
                # Simple correlation between adjacent timeframes
                for i in range(len(momentum_scores) - 1):
                    correlation_scores[f"tf_{i}_to_tf_{i+1}"] = np.corrcoef(
                        [i, i+1], momentum_scores
                    )[0, 1] if len(momentum_scores) > 1 else 0
                
                # Overall correlation (all timeframes)
                if len(momentum_scores) > 2:
                    correlation_scores['overall_correlation'] = np.std(momentum_scores) / np.mean(np.abs(momentum_scores)) if np.mean(np.abs(momentum_scores)) != 0 else 0
            
            return correlation_scores
            
        except Exception as e:
            logger.error(f"Error calculating timeframe correlation: {e}")
            return {}
    
    def _calculate_symbol_correlation(self, momentum_scores: List[float]) -> Dict[str, float]:
        """Calculate correlation between symbols"""
        try:
            correlation_scores = {
                'score_range': float(np.max(momentum_scores) - np.min(momentum_scores)),
                'average_score': float(np.mean(momentum_scores)),
                'std_deviation': float(np.std(momentum_scores)),
                'consistency': float(1 - (np.std(momentum_scores) / np.max(np.abs(momentum_scores)))) if np.max(np.abs(momentum_scores)) != 0 else 0
            }
            
            return correlation_scores
            
        except Exception as e:
            logger.error(f"Error calculating symbol correlation: {e}")
            return {}
    
    def _detect_timeframe_divergence(self, timeframes: List[str], 
                                     trend_directions: List[str], 
                                     momentum_scores: List[float]) -> List[str]:
        """Detect divergence signals across timeframes"""
        try:
            divergence_signals = []
            
            # Check for trend divergence
            bullish_count = trend_directions.count('bullish')
            bearish_count = trend_directions.count('bearish')
            
            if bullish_count > 0 and bearish_count > 0:
                divergence_signals.append(f"Trend divergence detected: {bullish_count} bullish, {bearish_count} bearish timeframes")
            
            # Check for momentum divergence
            positive_scores = sum(1 for score in momentum_scores if score > 0)
            negative_scores = sum(1 for score in momentum_scores if score < 0)
            
            if positive_scores > 0 and negative_scores > 0:
                divergence_signals.append(f"Momentum divergence: {positive_scores} positive, {negative_scores} negative")
            
            # Check for score extremes
            max_score = max(momentum_scores) if momentum_scores else 0
            min_score = min(momentum_scores) if momentum_scores else 0
            
            if abs(max_score - min_score) > 0.5:
                divergence_signals.append(f"High momentum divergence: range {min_score:.3f} to {max_score:.3f}")
            
            return divergence_signals
            
        except Exception as e:
            logger.error(f"Error detecting timeframe divergence: {e}")
            return []
    
    def _detect_symbol_divergence(self, symbols: List[str], 
                                  trend_directions: List[str], 
                                  momentum_scores: List[float]) -> List[str]:
        """Detect divergence signals across symbols"""
        try:
            divergence_signals = []
            
            # Check for trend divergence
            unique_trends = len(set(trend_directions))
            if unique_trends > 1:
                divergence_signals.append(f"Mixed trend signals: {unique_trends} different trend directions")
            
            # Check for momentum divergence
            positive_scores = sum(1 for score in momentum_scores if score > 0)
            negative_scores = sum(1 for score in momentum_scores if score < 0)
            
            if positive_scores > 0 and negative_scores > 0:
                divergence_signals.append(f"Momentum split: {positive_scores} positive, {negative_scores} negative")
            
            return divergence_signals
            
        except Exception as e:
            logger.error(f"Error detecting symbol divergence: {e}")
            return []
    
    def _generate_timeframe_recommendations(self, timeframes_data: Dict[str, Any], 
                                           momentum_scores: List[float], 
                                           divergence_signals: List[str]) -> List[str]:
        """Generate recommendations based on timeframe comparison"""
        try:
            recommendations = []
            
            # Check for alignment
            positive_count = sum(1 for s in momentum_scores if s > 0)
            negative_count = sum(1 for s in momentum_scores if s < 0)
            
            if positive_count == len(momentum_scores):
                recommendations.append("All timeframes show bullish momentum - consider long positions")
            elif negative_count == len(momentum_scores):
                recommendations.append("All timeframes show bearish momentum - consider short positions")
            
            # Check for divergence
            if divergence_signals:
                recommendations.append(f"Divergence detected across timeframes - use caution")
                recommendations.append(f"Consider trading the dominant timeframe signal")
            
            # Check for strength
            avg_momentum = np.mean(momentum_scores)
            if abs(avg_momentum) > 0.6:
                recommendations.append(f"Strong overall momentum ({avg_momentum:.3f}) - reduce position size")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating timeframe recommendations: {e}")
            return []
    
    def _generate_symbol_recommendations(self, symbols_data: Dict[str, Any], 
                                        momentum_scores: List[float], 
                                        divergence_signals: List[str]) -> List[str]:
        """Generate recommendations based on symbol comparison"""
        try:
            recommendations = []
            
            # Find strongest momentum
            if momentum_scores:
                max_momentum_idx = momentum_scores.index(max(momentum_scores, key=abs))
                symbols = list(symbols_data.keys())
                strongest_symbol = symbols[max_momentum_idx]
                recommendations.append(f"{strongest_symbol} shows strongest momentum")
            
            # Check for consistency
            score_range = max(momentum_scores) - min(momentum_scores)
            if score_range < 0.3:
                recommendations.append("All symbols show consistent momentum - market-wide trend")
            else:
                recommendations.append("Symbols show divergent momentum - selective trading")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating symbol recommendations: {e}")
            return []
    
    def get_summary(self, comparison: MomentumComparison) -> str:
        """Get text summary of comparison"""
        try:
            summary = f"""
            Momentum Comparison Summary
            ===========================
            
            Type: {comparison.comparison_type}
            Symbol(s): {comparison.symbol}
            Timeframe(s): {', '.join(comparison.timeframes)}
            
            Signals:
            {len(comparison.divergence_signals)} divergence signals detected
            
            Recommendations:
            """
            
            for i, rec in enumerate(comparison.recommendations, 1):
                summary += f"\n{i}. {rec}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

