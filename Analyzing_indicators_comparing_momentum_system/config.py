"""
Indicators Configuration
Comprehensive configuration for technical indicators analysis and momentum comparison system
"""

CONFIG = {
    # Indicator settings
    'indicators': {
        'trend_indicators': ['sma', 'ema', 'wma', 'tema', 'dema'],
        'oscillators': ['rsi', 'stochastic', 'williams_r', 'cci', 'roc', 'mfi'],
        'momentum_indicators': ['macd', 'adx', 'aroon', 'momentum', 'rsi'],
        'volatility_indicators': ['bollinger_bands', 'atr', 'keltner_channels', 'donchian'],
        'volume_indicators': ['obv', 'ad_line', 'adosc', 'volume_ratio', 'pvt'],
        'custom_indicators': ['fisher_transform', 'supertrend', 'volume_profile', 'market_structure'],
        'advanced_indicators': ['dtfx_zones', 'zigzag_momentum', 'advanced_cci', 'advanced_bollinger', 'advanced_supertrend']
    },
    
    # Advanced indicator settings
    'advanced_indicators': {
        'cci_period': 20,
        'bollinger_period': 20,
        'bollinger_std': 2.0,
        'supertrend_period': 10,
        'supertrend_multiplier': 3.0,
        'dtfx_structure_len': 10,
        'dtfx_fib_levels': [0, 0.3, 0.5, 0.7, 1.0],
        'zigzag_momentum_type': 'macd',  # 'macd', 'ma', 'qqe'
        'show_advanced': True
    },
    
    # Period settings
    'periods': {
        'short': [5, 9, 12, 14],
        'medium': [20, 21, 26, 50],
        'long': [100, 200],
        'rsi': [14, 21],
        'stochastic': [14, 21],
        'bollinger': [20],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'atr': [14, 20]
    },
    
    # Analysis settings
    'analysis': {
        'momentum_score_range': (-1.0, 1.0),
        'signal_strength_thresholds': {
            'weak': 0.3,
            'moderate': 0.5,
            'strong': 0.7,
            'very_strong': 0.9
        },
        'correlation_threshold': 0.7,
        'min_data_points': 50,
        'trend_direction_signals': ['bullish', 'bearish', 'neutral']
    },
    
    # Timeframe settings
    'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
    'default_timeframe': '1h',
    
    # Symbols to analyze
    'symbols': ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT'],
    
    # Database settings
    'database': {
        'path': 'indicators_analysis.db',
        'backup_interval': 3600,  # seconds
        'retention_days': 30
    },
    
    # Visualization settings
    'visualization': {
        'figsize': (15, 10),
        'style': 'dark_background',
        'save_format': 'png',
        'dpi': 150,
        'colors': {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#ffa726',
            'signal': '#42a5f5'
        }
    },
    
    # Risk management
    'risk_management': {
        'high_momentum_threshold': 0.8,
        'position_size_adjustment': 0.5,
        'stop_loss_percentage': 0.02,
        'take_profit_percentage': 0.04
    },
    
    # Logging settings
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'indicators_analysis.log'
    },
    
    # Real-time settings
    'real_time': {
        'update_interval': 60,  # seconds
        'max_cached_data': 1000,
        'enable_alerts': True
    }
}

