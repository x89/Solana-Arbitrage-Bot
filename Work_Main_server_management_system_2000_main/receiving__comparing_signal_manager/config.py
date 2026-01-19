"""
Signal Receiver Configuration
"""

SIGNAL_RECEIVER_CONFIG = {
    'receiver': {
        'max_history': 1000,
        'sources': ['ai_prediction', 'technical_analysis', 'sentiment_analysis'],
        'min_confidence': 0.5
    }
}

