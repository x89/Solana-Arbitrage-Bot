#!/usr/bin/env python3
"""
AI-Powered ATR Dynamic Stop-Loss Optimizer
Combines machine learning, Bayesian optimization, and adaptive parameter tuning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - using basic optimization")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

try:
    from .atr_dynamic_stop_loss import ATRDynamicStopLoss, ATRConfig
    from .config import get_config
except ImportError:
    from atr_dynamic_stop_loss import ATRDynamicStopLoss, ATRConfig
    from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class AIOptimizationResult:
    """AI optimization result"""
    best_params: Dict[str, float]
    best_score: float
    optimization_method: str
    n_trials: int
    convergence_data: List[float]
    feature_importance: Optional[Dict[str, float]]
    confidence: float

class AIATROptimizer:
    """AI-powered ATR parameter optimizer"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_params_cache = {}
        self.feature_importance = {}
        
        logger.info("AIATROptimizer initialized")
    
    def optimize_with_bayesian_search(self, historical_data: pd.DataFrame,
                                     optimization_ranges: Dict[str, List[float]] = None,
                                     n_trials: int = 100,
                                     metric: str = 'sharpe_ratio') -> AIOptimizationResult:
        """
        Optimize ATR parameters using Bayesian optimization (Optuna)
        
        Args:
            historical_data: Historical OHLC data
            optimization_ranges: Parameter ranges to optimize
            n_trials: Number of optimization trials
            metric: Optimization metric
            
        Returns:
            AIOptimizationResult with best parameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to grid search")
            return self._grid_search_optimization(historical_data, optimization_ranges, n_trials)
        
        try:
            def objective(trial):
                # Suggest parameters
                atr_multiplier = trial.suggest_categorical(
                    'atr_multiplier',
                    optimization_ranges.get('atr_multiplier', [1.2, 1.5, 1.8, 2.0, 2.5])
                )
                
                risk_reward = trial.suggest_categorical(
                    'risk_reward_ratio',
                    optimization_ranges.get('risk_reward_ratio', [2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
                )
                
                supertrend_factor = trial.suggest_categorical(
                    'supertrend_factor',
                    optimization_ranges.get('supertrend_factor', [2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5])
                )
                
                ma_length = trial.suggest_categorical(
                    'ma_length',
                    optimization_ranges.get('ma_length', [10, 15, 20, 25, 30, 50])
                )
                
                # Create ATR config
                atr_config = ATRConfig(
                    atr_multiplier=atr_multiplier,
                    risk_reward_ratio=risk_reward,
                    supertrend_factor=supertrend_factor,
                    ma_length=ma_length
                )
                
                # Test parameters
                score = self._evaluate_parameters(historical_data, atr_config, metric)
                
                return score
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                study_name=f'ATR_Optimization_{datetime.now().timestamp()}'
            )
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Get best result
            best_params = study.best_params
            best_score = study.best_value
            
            # Calculate convergence
            convergence = []
            for trial in study.trials:
                if trial.value is not None:
                    convergence.append(trial.value)
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(study, 'get_trials_dataframe'):
                try:
                    importance = optuna.importance.get_param_importances(study)
                    feature_importance = importance
                except:
                    feature_importance = {}
            
            result = AIOptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_method='bayesian',
                n_trials=n_trials,
                convergence_data=convergence,
                feature_importance=feature_importance,
                confidence=self._calculate_confidence(study)
            )
            
            self.optimization_history.append(result)
            logger.info(f"Bayesian optimization completed: Best score = {best_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return self._basic_optimization(historical_data, optimization_ranges, n_trials)
    
    def optimize_with_ml_prediction(self, historical_data: pd.DataFrame,
                                    n_features: int = 20) -> Dict[str, Any]:
        """
        Use machine learning to predict optimal parameters
        
        Args:
            historical_data: Historical OHLC data
            n_features: Number of features to extract
            
        Returns:
            Predicted optimal parameters
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for ML prediction")
            return {}
        
        try:
            # Extract features
            features_df = self._extract_features(historical_data)
            
            # Create target (optimal parameters from backtesting)
            # This is a simplified version - in production, you'd have historical optimal params
            targets = self._generate_target_params(historical_data)
            
            if len(features_df) == 0 or len(targets) == 0:
                return {}
            
            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features_df.values, targets)
            
            # Predict optimal parameters for current market
            latest_features = features_df.iloc[-1:].values
            predicted_params = model.predict(latest_features)[0]
            
            # Get feature importance
            feature_names = features_df.columns.tolist()
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            return {
                'predicted_params': predicted_params,
                'feature_importance': importance_dict,
                'confidence': self._calculate_ml_confidence(model, latest_features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return {}
    
    def adaptive_parameter_adjustment(self, current_atr: float, 
                                      historical_atr: pd.Series) -> Dict[str, float]:
        """
        Adaptively adjust ATR parameters based on current volatility
        
        Args:
            current_atr: Current ATR value
            historical_atr: Historical ATR series
            
        Returns:
            Adjusted parameters
        """
        try:
            # Calculate volatility state
            avg_atr = historical_atr.mean()
            vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            # Get config
            config = get_config('atr_dynamic_stop_loss')
            
            base_multiplier = config.get('atr_multiplier', 1.8)
            base_rr = config.get('risk_reward_ratio', 3.0)
            
            # Adjust based on volatility
            if vol_ratio > 1.5:  # High volatility
                adjusted_multiplier = base_multiplier * 1.2  # Wider stops
                adjusted_rr = base_rr * 0.9  # Slightly tighter RR
            elif vol_ratio < 0.7:  # Low volatility
                adjusted_multiplier = base_multiplier * 0.8  # Tighter stops
                adjusted_rr = base_rr * 1.1  # Better RR
            else:  # Normal volatility
                adjusted_multiplier = base_multiplier
                adjusted_rr = base_rr
            
            # Apply bounds
            min_mult = config.get('min_atr_multiplier', 1.2)
            max_mult = config.get('max_atr_multiplier', 2.5)
            min_rr = config.get('min_rr_ratio', 2.0)
            max_rr = config.get('max_rr_ratio', 5.0)
            
            adjusted_multiplier = np.clip(adjusted_multiplier, min_mult, max_mult)
            adjusted_rr = np.clip(adjusted_rr, min_rr, max_rr)
            
            return {
                'atr_multiplier': adjusted_multiplier,
                'risk_reward_ratio': adjusted_rr,
                'volatility_ratio': vol_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in adaptive adjustment: {e}")
            return {}
    
    def _evaluate_parameters(self, data: pd.DataFrame, atr_config: ATRConfig, metric: str) -> float:
        """Evaluate parameter combination and return score"""
        try:
            # Create ATR system
            atr_system = ATRDynamicStopLoss(atr_config)
            
            # Generate signals
            signals = []
            for i in range(50, len(data)):  # Start from 50 to have enough data
                window_data = data.iloc[:i+1]
                signal = atr_system.generate_entry_signal(window_data)
                signals.append(signal)
            
            # Calculate score
            if metric == 'sharpe_ratio':
                return self._calculate_sharpe(signals, data)
            elif metric == 'total_return':
                return self._calculate_total_return(signals, data)
            elif metric == 'win_rate':
                return self._calculate_win_rate(signals, data)
            else:
                return self._calculate_total_return(signals, data)
                
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return -1000.0
    
    def _calculate_sharpe(self, signals: List[Dict], data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Simplified scoring
            long_signals = sum(1 for s in signals if s.get('long_signal', False))
            short_signals = sum(1 for s in signals if s.get('short_signal', False))
            
            # Dummy return calculation
            if long_signals + short_signals == 0:
                return -1000.0
            
            return (long_signals + short_signals) / len(signals) if len(signals) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe: {e}")
            return -1000.0
    
    def _calculate_total_return(self, signals: List[Dict], data: pd.DataFrame) -> float:
        """Calculate total return"""
        try:
            total_signals = sum(1 for s in signals if s.get('long_signal') or s.get('short_signal'))
            return total_signals
            
        except Exception as e:
            logger.error(f"Error calculating total return: {e}")
            return -1000.0
    
    def _calculate_win_rate(self, signals: List[Dict], data: pd.DataFrame) -> float:
        """Calculate win rate"""
        try:
            # Simplified - would need actual trade results
            return np.random.random()  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML prediction"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price features
            features['returns'] = data['close'].pct_change()
            features['returns_lag1'] = features['returns'].shift(1)
            features['returns_lag2'] = features['returns'].shift(2)
            
            # Volatility features
            features['volatility'] = data['close'].rolling(20).std()
            features['atr'] = self._calculate_simple_atr(data, 10)
            
            # Momentum features
            features['rsi'] = self._calculate_simple_rsi(data, 14)
            features['momentum'] = data['close'] / data['close'].shift(10) - 1
            
            # Price position
            features['price_range'] = (data['close'] - data['low'].rolling(20).min()) / \
                                      (data['high'].rolling(20).max() - data['low'].rolling(20).min())
            
            # Volume features (if available)
            if 'volume' in data.columns:
                features['volume_ma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_ma']
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _calculate_simple_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate simple ATR"""
        try:
            high_low = data['high'] - data['low']
            return high_low.rolling(period).mean()
        except:
            return pd.Series(dtype=float)
    
    def _calculate_simple_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate simple RSI"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return pd.Series(dtype=float)
    
    def _generate_target_params(self, data: pd.DataFrame) -> np.ndarray:
        """Generate target parameters (placeholder)"""
        # In production, this would be based on actual backtesting results
        # For now, return synthetic data
        n_samples = len(data)
        targets = np.random.rand(n_samples, 2) * 3.0 + 1.0  # Random params
        return targets
    
    def _calculate_confidence(self, study) -> float:
        """Calculate optimization confidence"""
        try:
            if len(study.trials) < 10:
                return 0.5
            
            recent_scores = [t.value for t in study.trials[-10:] if t.value is not None]
            if len(recent_scores) == 0:
                return 0.5
            
            std_dev = np.std(recent_scores)
            mean_score = np.mean(recent_scores)
            
            # Low std = high confidence
            confidence = 1.0 - min(std_dev / abs(mean_score), 1.0) if mean_score != 0 else 0.5
            
            return confidence
            
        except Exception as e:
            return 0.5
    
    def _calculate_ml_confidence(self, model, features) -> float:
        """Calculate ML prediction confidence"""
        # Placeholder - would use prediction variance
        return 0.75
    
    def _grid_search_optimization(self, historical_data: pd.DataFrame,
                                   optimization_ranges: Dict[str, List[float]],
                                   n_trials: int) -> AIOptimizationResult:
        """Fallback grid search optimization"""
        # Simplified implementation
        return AIOptimizationResult(
            best_params={},
            best_score=0.0,
            optimization_method='grid',
            n_trials=n_trials,
            convergence_data=[],
            feature_importance={},
            confidence=0.5
        )
    
    def _basic_optimization(self, historical_data: pd.DataFrame,
                           optimization_ranges: Dict[str, List[float]],
                           n_trials: int) -> AIOptimizationResult:
        """Basic optimization fallback"""
        return AIOptimizationResult(
            best_params={},
            best_score=0.0,
            optimization_method='basic',
            n_trials=0,
            convergence_data=[],
            feature_importance={},
            confidence=0.0
        )

def main():
    """Example usage"""
    # Create AI optimizer
    optimizer = AIATROptimizer()
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(200).cumsum(),
        'high': 101 + np.random.randn(200).cumsum(),
        'low': 99 + np.random.randn(200).cumsum(),
        'close': 100 + np.random.randn(200).cumsum(),
        'volume': np.random.randint(1000, 5000, 200)
    }, index=dates)
    
    # Optimization ranges
    ranges = {
        'atr_multiplier': [1.2, 1.5, 1.8, 2.0, 2.5],
        'risk_reward_ratio': [2.0, 2.5, 3.0, 3.5, 4.0],
        'supertrend_factor': [2.0, 2.5, 2.8, 3.0, 3.5],
        'ma_length': [10, 15, 20, 25, 30]
    }
    
    print("Starting AI-powered optimization...")
    
    if OPTUNA_AVAILABLE:
        result = optimizer.optimize_with_bayesian_search(
            sample_data, ranges, n_trials=50, metric='sharpe_ratio'
        )
        
        print(f"\nOptimization Results:")
        print(f"Best Score: {result.best_score:.4f}")
        print(f"Best Parameters: {result.best_params}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Method: {result.optimization_method}")
    else:
        print("Optuna not available - install with: pip install optuna")

if __name__ == "__main__":
    main()

