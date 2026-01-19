#!/usr/bin/env python3
"""
Reverse Coefficient Analyzer
Analyze reverse momentum and mean reversion coefficients
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CoefficientData:
    """Coefficient analysis data"""
    timestamp: datetime
    symbol: str
    price: float
    coefficient: float
    reverse_probability: float
    momentum: float
    volatility: float

class ReverseCoefficientAnalyzer:
    """Analyze reverse coefficients for mean reversion strategies"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.coefficient_history = []
        self.analysis_results = []
        logger.info("ReverseCoefficientAnalyzer initialized")
    
    def calculate_reverse_coefficient(self, price_data: pd.Series) -> float:
        """Calculate reverse coefficient"""
        try:
            if len(price_data) < self.lookback_period:
                return 0.0
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate autocorrelation
            autocorr = returns.autocorr(lag=1)
            
            # Reverse coefficient: negative autocorrelation suggests mean reversion
            reverse_coef = -autocorr if not np.isnan(autocorr) else 0.0
            
            return reverse_coef
            
        except Exception as e:
            logger.error(f"Error calculating reverse coefficient: {e}")
            return 0.0
    
    def calculate_hurst_exponent(self, price_data: pd.Series) -> float:
        """Calculate Hurst exponent for mean reversion detection"""
        try:
            # Simplified Hurst calculation
            n = len(price_data)
            mean_price = price_data.mean()
            deviations = price_data - mean_price
            cumulative = deviations.cumsum()
            ranges = []
            
            for lag in range(1, min(n // 4, 50)):
                range_val = (cumulative[lag:].max() - cumulative[lag:].min()) / lag
                std_val = price_data[:lag+1].std()
                if std_val > 0:
                    ranges.append(range_val / std_val)
            
            if ranges:
                hurst = np.mean(np.log(ranges)) / np.log(2) if ranges else 0.5
            else:
                hurst = 0.5
            
            return hurst
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def analyze_mean_reversion(self, price_data: pd.Series) -> Dict[str, Any]:
        """Analyze mean reversion characteristics"""
        try:
            reverse_coef = self.calculate_reverse_coefficient(price_data)
            hurst = self.calculate_hurst_exponent(price_data)
            
            # Calculate mean
            mean_price = price_data.mean()
            current_price = price_data.iloc[-1]
            
            # Calculate deviation from mean
            deviation = (current_price - mean_price) / mean_price
            
            # Determine reversion signal strength
            if reverse_coef > 0.3 and abs(deviation) > 0.02:
                signal_strength = 'strong'
            elif reverse_coef > 0.1 and abs(deviation) > 0.01:
                signal_strength = 'medium'
            else:
                signal_strength = 'weak'
            
            # Hurst interpretation
            if hurst < 0.5:
                trend_type = 'mean_reverting'
            elif hurst > 0.5:
                trend_type = 'trending'
            else:
                trend_type = 'random_walk'
            
            analysis = {
                'reverse_coefficient': reverse_coef,
                'hurst_exponent': hurst,
                'deviation_from_mean': deviation,
                'signal_strength': signal_strength,
                'trend_type': trend_type,
                'mean_price': mean_price,
                'current_price': current_price,
                'expected_return': self._estimate_expected_return(reverse_coef, deviation)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in mean reversion analysis: {e}")
            return {}
    
    def _estimate_expected_return(self, reverse_coef: float, deviation: float) -> float:
        """Estimate expected return based on mean reversion"""
        # Expected return is negative of current deviation times reverse coefficient
        expected_return = -deviation * reverse_coef
        return expected_return
    
    def calculate_arima_coefficients(self, price_data: pd.Series, p: int = 1, d: int = 1, q: int = 1) -> Dict[str, Any]:
        """Calculate ARIMA coefficients"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Fit ARIMA model
            model = ARIMA(price_data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Extract coefficients
            ar_coefs = fitted_model.arparams if hasattr(fitted_model, 'arparams') else []
            ma_coefs = fitted_model.maparams if hasattr(fitted_model, 'maparams') else []
            
            return {
                'ar_coefficients': ar_coefs.tolist() if len(ar_coefs) > 0 else [],
                'ma_coefficients': ma_coefs.tolist() if len(ma_coefs) > 0 else [],
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except ImportError:
            logger.warning("statsmodels not available for ARIMA analysis")
            return {}
        except Exception as e:
            logger.error(f"Error calculating ARIMA coefficients: {e}")
            return {}
    
    def detect_regime_change(self, price_data: pd.Series, window: int = 50) -> Dict[str, Any]:
        """Detect regime changes in price series"""
        try:
            if len(price_data) < window * 2:
                return {}
            
            # Split into windows
            first_half = price_data[:window]
            second_half = price_data[-window:]
            
            # Calculate statistics for each half
            mean_first = first_half.mean()
            mean_second = second_half.mean()
            vol_first = first_half.std()
            vol_second = second_half.std()
            
            # Test for significant difference
            mean_change = abs(mean_second - mean_first) / mean_first
            vol_change = abs(vol_second - vol_first) / vol_first if vol_first > 0 else 0
            
            # Determine if regime change occurred
            has_regime_change = mean_change > 0.05 or vol_change > 0.3
            
            return {
                'has_regime_change': has_regime_change,
                'mean_change': mean_change,
                'volatility_change': vol_change,
                'first_half_mean': mean_first,
                'second_half_mean': mean_second
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime change: {e}")
            return {}

def main():
    """Example usage"""
    analyzer = ReverseCoefficientAnalyzer()
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    price_data = pd.Series(prices, index=dates)
    
    # Analyze
    analysis = analyzer.analyze_mean_reversion(price_data)
    
    print("Reverse Coefficient Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

