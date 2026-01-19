"""
Reverse Coefficient Analyzer Configuration
"""

REVERSE_COEFFICIENT_CONFIG = {
    'analyzer': {
        'lookback_period': 20,
        'mean_reversion_threshold': 0.5,
        'hurst_window': 50
    },
    'arima': {
        'p': 1,
        'd': 1,
        'q': 1
    }
}

