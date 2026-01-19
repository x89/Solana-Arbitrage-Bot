#!/usr/bin/env python3
"""
Prediction Validator Module
Comprehensive prediction validation system including:
- Range validation
- Statistical validation
- Anomaly detection
- Confidence thresholding
- Consistency checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    validation_score: float
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

class PredictionValidator:
    """Comprehensive prediction validation"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_price_change: float = 0.5,  # 50% max change per prediction
        check_consistency: bool = True,
        check_anomalies: bool = True
    ):
        """
        Initialize prediction validator
        
        Args:
            confidence_threshold: Minimum confidence for valid prediction
            max_price_change: Maximum allowed price change (percentage)
            check_consistency: Enable consistency checks
            check_anomalies: Enable anomaly detection
        """
        self.confidence_threshold = confidence_threshold
        self.max_price_change = max_price_change
        self.check_consistency = check_consistency
        self.check_anomalies = check_anomalies
        
        self.validation_history = []
        self.statistics = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0
        }
    
    def validate(
        self,
        prediction: float,
        confidence: float,
        historical_data: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a prediction
        
        Args:
            prediction: Predicted value
            confidence: Prediction confidence score
            historical_data: Historical data for context
            metadata: Additional metadata
            
        Returns:
            ValidationResult object
        """
        try:
            warnings = []
            errors = []
            validation_score = 1.0
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                errors.append(f"Low confidence: {confidence:.4f} < {self.confidence_threshold}")
                validation_score -= 0.3
            
            # Check if prediction is finite
            if not np.isfinite(prediction):
                errors.append(f"Prediction is not finite: {prediction}")
                validation_score = 0.0
            
            # Check range if historical data provided
            if historical_data is not None and len(historical_data) > 0:
                range_check = self._check_range(prediction, historical_data)
                if not range_check['is_valid']:
                    errors.append(f"Prediction out of reasonable range: {range_check['reason']}")
                    validation_score -= 0.2
                
                # Check consistency
                if self.check_consistency:
                    consistency_check = self._check_consistency(prediction, historical_data)
                    if not consistency_check['is_valid']:
                        warnings.append(f"Inconsistency detected: {consistency_check['reason']}")
                        validation_score -= 0.1
                
                # Check for anomalies
                if self.check_anomalies:
                    anomaly_check = self._check_anomaly(prediction, historical_data)
                    if anomaly_check['is_anomaly']:
                        warnings.append(f"Anomaly detected: {anomaly_check['reason']}")
                        validation_score -= 0.15
            
            # Check metadata if provided
            if metadata:
                metadata_check = self._check_metadata(metadata)
                if not metadata_check['is_valid']:
                    warnings.extend(metadata_check['warnings'])
                    errors.extend(metadata_check['errors'])
                    validation_score -= len(metadata_check['errors']) * 0.1
            
            # Determine final validity
            is_valid = len(errors) == 0 and validation_score > 0.5
            
            # Create result
            result = ValidationResult(
                is_valid=is_valid,
                validation_score=max(0.0, min(1.0, validation_score)),
                warnings=warnings,
                errors=errors,
                metadata={
                    'confidence': confidence,
                    'prediction': prediction,
                    'validation_timestamp': datetime.now().isoformat()
                }
            )
            
            # Update statistics
            self.statistics['total_validations'] += 1
            if is_valid:
                self.statistics['passed_validations'] += 1
            else:
                self.statistics['failed_validations'] += 1
            self.statistics['warnings_count'] += len(warnings)
            
            self.validation_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                warnings=[],
                errors=[f"Validation error: {str(e)}"],
                metadata={}
            )
    
    def _check_range(
        self,
        prediction: float,
        historical_data: List[float]
    ) -> Dict[str, Any]:
        """Check if prediction is within reasonable range"""
        try:
            historical_array = np.array(historical_data)
            
            # Calculate statistics
            mean = np.mean(historical_array)
            std = np.std(historical_array)
            min_val = np.min(historical_array)
            max_val = np.max(historical_array)
            
            # Define reasonable range
            range_low = max(min_val, mean - 4 * std)
            range_high = min(max_val, mean + 4 * std)
            
            # Allow for larger moves
            range_expanded = max(
                self.max_price_change * abs(mean),
                3 * std
            )
            
            range_low = min(min_val, mean - range_expanded)
            range_high = max(max_val, mean + range_expanded)
            
            if prediction < range_low or prediction > range_high:
                return {
                    'is_valid': False,
                    'reason': f"Prediction {prediction:.4f} outside range [{range_low:.4f}, {range_high:.4f}]"
                }
            
            return {'is_valid': True, 'reason': 'Within range'}
            
        except Exception as e:
            logger.error(f"Error checking range: {e}")
            return {'is_valid': True, 'reason': 'Range check failed'}
    
    def _check_consistency(
        self,
        prediction: float,
        historical_data: List[float]
    ) -> Dict[str, Any]:
        """Check prediction consistency with historical trends"""
        try:
            if len(historical_data) < 5:
                return {'is_valid': True, 'reason': 'Insufficient data'}
            
            recent_data = np.array(historical_data[-20:])  # Last 20 data points
            
            # Calculate trend
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            last_value = historical_data[-1]
            
            # Check if prediction follows trend
            if trend > 0:  # Uptrend
                if prediction < last_value:
                    return {
                        'is_valid': False,
                        'reason': 'Prediction contradicts uptrend'
                    }
            elif trend < 0:  # Downtrend
                if prediction > last_value:
                    return {
                        'is_valid': False,
                        'reason': 'Prediction contradicts downtrend'
                    }
            
            # Check for sudden reversals
            recent_volatility = np.std(recent_data)
            prediction_change = abs(prediction - last_value)
            
            if prediction_change > 3 * recent_volatility:
                return {
                    'is_valid': False,
                    'reason': 'Sudden large move detected'
                }
            
            return {'is_valid': True, 'reason': 'Consistent with trend'}
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return {'is_valid': True, 'reason': 'Consistency check failed'}
    
    def _check_anomaly(
        self,
        prediction: float,
        historical_data: List[float]
    ) -> Dict[str, Any]:
        """Detect anomalies in prediction"""
        try:
            if len(historical_data) < 10:
                return {'is_anomaly': False, 'reason': 'Insufficient data'}
            
            recent_data = np.array(historical_data[-50:])
            last_value = historical_data[-1]
            
            # Z-score test
            mean = np.mean(recent_data)
            std = np.std(recent_data)
            
            if std > 0:
                z_score = (prediction - mean) / std
                
                if abs(z_score) > 3:
                    return {
                        'is_anomaly': True,
                        'reason': f'Extreme z-score: {z_score:.2f}',
                        'z_score': z_score
                    }
            
            # IQR method
            q1 = np.percentile(recent_data, 25)
            q3 = np.percentile(recent_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 2.5 * iqr
            
            if prediction < lower_bound or prediction > upper_bound:
                return {
                    'is_anomaly': True,
                    'reason': f'Outlier: {prediction:.4f} outside IQR bounds',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            return {'is_anomaly': False, 'reason': 'No anomaly detected'}
            
        except Exception as e:
            logger.error(f"Error checking anomaly: {e}")
            return {'is_anomaly': False, 'reason': 'Anomaly check failed'}
    
    def _check_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check metadata for issues"""
        try:
            warnings = []
            errors = []
            
            # Check model name
            if 'model_name' not in metadata:
                warnings.append("Missing model_name in metadata")
            
            # Check timestamp
            if 'timestamp' not in metadata:
                warnings.append("Missing timestamp in metadata")
            
            # Check features
            if 'features_used' in metadata:
                if len(metadata['features_used']) == 0:
                    warnings.append("No features used for prediction")
            
            return {
                'is_valid': len(errors) == 0,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Error checking metadata: {e}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [f"Metadata check error: {str(e)}"]
            }
    
    def batch_validate(
        self,
        predictions: List[float],
        confidences: List[float],
        historical_data: Optional[List[List[float]]] = None
    ) -> List[ValidationResult]:
        """Validate a batch of predictions"""
        results = []
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            hist_data = historical_data[i] if historical_data else None
            result = self.validate(pred, conf, hist_data)
            results.append(result)
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.statistics.copy()
        
        if stats['total_validations'] > 0:
            stats['pass_rate'] = stats['passed_validations'] / stats['total_validations']
            stats['warning_rate'] = stats['warnings_count'] / stats['total_validations']
        else:
            stats['pass_rate'] = 0.0
            stats['warning_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.statistics = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0
        }
        self.validation_history = []
        logger.info("Validation statistics reset")

