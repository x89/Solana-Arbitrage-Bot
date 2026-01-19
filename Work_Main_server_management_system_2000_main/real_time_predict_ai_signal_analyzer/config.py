"""
Real-time AI Signal Analyzer Configuration
"""

SIGNAL_ANALYZER_CONFIG = {
    'analyzer': {
        'confidence_threshold': 0.7,
        'update_interval': 1.0,  # seconds
        'signal_history_size': 1000
    },
    
    'filters': {
        'min_confidence': 0.5,
        'allowed_symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        'allowed_signal_types': ['buy', 'sell']
    },
    
    'aggregation': {
        'aggregation_window_minutes': 5,
        'consensus_algorithm': 'majority_vote'
    }
}

