#!/usr/bin/env python3
"""
Ensemble Manager Module
Comprehensive ensemble management system for combining multiple prediction models including:
- Weighted ensemble methods
- Dynamic weight optimization
- Performance-based model selection
- Confidence aggregation
- Model diversity tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EnsembleManager:
    """Comprehensive ensemble prediction manager"""
    
    def __init__(
        self,
        weight_method: str = 'weighted_average',  # 'mean', 'weighted_average', 'median', 'weighted_median'
        performance_based: bool = True,
        update_weights_online: bool = True
    ):
        """
        Initialize ensemble manager
        
        Args:
            weight_method: Method for combining predictions
            performance_based: Use performance-based weighting
            update_weights_online: Update weights based on recent performance
        """
        self.weight_method = weight_method
        self.performance_based = performance_based
        self.update_weights_online = update_weights_online
        
        self.model_weights = {}
        self.model_performance = {}
        self.prediction_history = []
        self.actual_history = []
        
        self.ensemble_stats = {
            'total_predictions': 0,
            'mean_squared_error': 0.0,
            'mean_absolute_error': 0.0,
            'accuracy': 0.0
        }
    
    def ensemble_predict(
        self,
        predictions: Dict[str, float],
        confidences: Optional[Dict[str, float]] = None,
        model_performances: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: Dictionary of model_name -> prediction
            confidences: Dictionary of model_name -> confidence
            model_performances: Dictionary of model_name -> performance_score
            
        Returns:
            Tuple of (ensemble_prediction, ensemble_confidence)
        """
        try:
            if not predictions:
                logger.error("No predictions provided for ensemble")
                return 0.0, 0.0
            
            pred_values = list(predictions.values())
            model_names = list(predictions.keys())
            
            # Calculate weights
            if self.performance_based and model_performances:
                weights = self._calculate_performance_weights(model_names, model_performances)
            elif confidences:
                weights = self._calculate_confidence_weights(model_names, confidences)
            else:
                weights = self._calculate_equal_weights(model_names)
            
            # Combine predictions based on method
            if self.weight_method == 'mean':
                ensemble_pred = np.mean(pred_values)
                
            elif self.weight_method == 'weighted_average':
                ensemble_pred = np.average(pred_values, weights=weights)
                
            elif self.weight_method == 'median':
                ensemble_pred = np.median(pred_values)
                
            elif self.weight_method == 'weighted_median':
                ensemble_pred = self._weighted_median(pred_values, weights)
                
            elif self.weight_method == 'trimmed_mean':
                ensemble_pred = self._trimmed_mean(pred_values, trim_fraction=0.2)
                
            else:
                logger.warning(f"Unknown method {self.weight_method}, using weighted average")
                ensemble_pred = np.average(pred_values, weights=weights)
            
            # Calculate ensemble confidence
            if confidences:
                confidence_values = [confidences.get(name, 0.5) for name in model_names]
                ensemble_confidence = np.average(confidence_values, weights=weights)
            else:
                ensemble_confidence = np.mean(weights)
            
            # Normalize confidence
            ensemble_confidence = max(0.0, min(1.0, ensemble_confidence))
            
            logger.debug(f"Ensemble prediction: {ensemble_pred:.4f}, confidence: {ensemble_confidence:.4f}")
            
            return ensemble_pred, ensemble_confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 0.0, 0.0
    
    def _calculate_performance_weights(
        self,
        model_names: List[str],
        model_performances: Dict[str, float]
    ) -> List[float]:
        """Calculate weights based on model performance"""
        try:
            weights = []
            
            for name in model_names:
                if name in model_performances:
                    # Use performance score as weight
                    performance = model_performances[name]
                    weights.append(max(0.0, performance))
                else:
                    weights.append(0.1)  # Low weight for unknown performance
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating performance weights: {e}")
            return [1.0 / len(model_names)] * len(model_names)
    
    def _calculate_confidence_weights(
        self,
        model_names: List[str],
        confidences: Dict[str, float]
    ) -> List[float]:
        """Calculate weights based on prediction confidence"""
        try:
            weights = []
            
            for name in model_names:
                confidence = confidences.get(name, 0.5)
                # Square confidence to emphasize higher confidence predictions
                weights.append(confidence ** 2)
            
            # Normalize
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating confidence weights: {e}")
            return [1.0 / len(model_names)] * len(model_names)
    
    def _calculate_equal_weights(self, model_names: List[str]) -> List[float]:
        """Calculate equal weights for all models"""
        return [1.0 / len(model_names)] * len(model_names)
    
    def _weighted_median(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted median"""
        try:
            # Create weighted list
            weighted_values = []
            for val, weight in zip(values, weights):
                weighted_values.extend([val] * int(weight * 100))
            
            return np.median(weighted_values) if weighted_values else np.median(values)
            
        except Exception as e:
            logger.error(f"Error calculating weighted median: {e}")
            return np.median(values)
    
    def _trimmed_mean(self, values: List[float], trim_fraction: float = 0.2) -> float:
        """Calculate trimmed mean (remove extreme values)"""
        try:
            sorted_values = sorted(values)
            n = len(sorted_values)
            trim_count = int(n * trim_fraction)
            
            if trim_count > 0:
                trimmed = sorted_values[trim_count:-trim_count]
                return np.mean(trimmed) if trimmed else np.mean(values)
            else:
                return np.mean(values)
                
        except Exception as e:
            logger.error(f"Error calculating trimmed mean: {e}")
            return np.mean(values)
    
    def update_weights_from_performance(
        self,
        model_name: str,
        actual_value: float,
        predicted_value: float
    ):
        """Update model weights based on prediction accuracy"""
        try:
            if self.update_weights_online:
                # Calculate error
                error = abs(predicted_value - actual_value)
                relative_error = error / actual_value if actual_value != 0 else 1.0
                
                # Update performance score (lower error = higher score)
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = 1.0
                
                # Exponential moving average of performance
                alpha = 0.1  # Learning rate
                self.model_performance[model_name] = (
                    alpha * (1 - relative_error) + 
                    (1 - alpha) * self.model_performance[model_name]
                )
                
                # Keep performance in [0, 1]
                self.model_performance[model_name] = max(0.0, min(1.0, self.model_performance[model_name]))
                
                logger.debug(f"Updated performance for {model_name}: {self.model_performance[model_name]:.4f}")
                
        except Exception as e:
            logger.error(f"Error updating weights from performance: {e}")
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        return self.ensemble_stats.copy()
    
    def add_prediction(self, prediction: float, actual: float = None):
        """Add prediction to history"""
        try:
            self.prediction_history.append(prediction)
            
            if actual is not None:
                self.actual_history.append(actual)
                self.ensemble_stats['total_predictions'] += 1
                
                # Update error metrics
                if len(self.prediction_history) > 1 and len(self.actual_history) > 1:
                    pred_hist = np.array(self.prediction_history[-100:])
                    actual_hist = np.array(self.actual_history[-100:])
                    
                    self.ensemble_stats['mean_squared_error'] = float(np.mean((pred_hist - actual_hist) ** 2))
                    self.ensemble_stats['mean_absolute_error'] = float(np.mean(np.abs(pred_hist - actual_hist)))
                    
                    # Calculate accuracy for direction predictions
                    correct = np.sum(np.sign(pred_hist) == np.sign(actual_hist))
                    self.ensemble_stats['accuracy'] = correct / len(pred_hist)
                
        except Exception as e:
            logger.error(f"Error adding prediction: {e}")
    
    def save_ensemble_config(self, filepath: str):
        """Save ensemble configuration"""
        try:
            config = {
                'weight_method': self.weight_method,
                'performance_based': self.performance_based,
                'model_weights': self.model_weights,
                'model_performance': self.model_performance,
                'ensemble_stats': self.ensemble_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Ensemble config saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble config: {e}")
    
    def load_ensemble_config(self, filepath: str) -> bool:
        """Load ensemble configuration"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.weight_method = config.get('weight_method', 'weighted_average')
            self.performance_based = config.get('performance_based', True)
            self.model_weights = config.get('model_weights', {})
            self.model_performance = config.get('model_performance', {})
            self.ensemble_stats = config.get('ensemble_stats', self.ensemble_stats)
            
            logger.info(f"Ensemble config loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble config: {e}")
            return False
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get model importance scores based on weights and performance"""
        try:
            importance = {}
            
            for model_name in self.model_performance.keys():
                weight = self.model_weights.get(model_name, 0.1)
                performance = self.model_performance[model_name]
                importance[model_name] = weight * performance
            
            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting model importance: {e}")
            return {}

