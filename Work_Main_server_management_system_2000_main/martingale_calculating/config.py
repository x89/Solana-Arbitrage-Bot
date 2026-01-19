#!/usr/bin/env python3
"""
Martingale Calculator Configuration with ATR Dynamic Stop-Loss
"""

MARTINGALE_CONFIG = {
    'strategies': {
        'classic': {
            'initial_bet': 100.0,
            'multiplier': 2.0,
            'max_bet': 1000.0,
            'max_loss': 5000.0,
            'max_consecutive_losses': 5
        },
        'fibonacci': {
            'initial_bet': 100.0,
            'multiplier': 1.618,  # Golden ratio
            'max_consecutive_losses': 7
        },
        'dalembert': {
            'initial_bet': 100.0,
            'multiplier': 1.0,  # Linear increase
            'max_bet': 500.0
        }
    },
    
    'risk_management': {
        'max_drawdown': 0.3,  # 30%
        'stop_loss': 0.25,  # 25%
        'take_profit': 0.50,  # 50%
        'position_sizing': 'kelly',  # 'kelly', 'fixed', 'martingale'
    },
    
    'atr_dynamic_stop_loss': {
        # ATR Parameters
        'atr_period': 10,  # 5-20 range
        'atr_multiplier': 1.8,  # Stop-loss multiplier (1.5-2.5 range)
        'atr_based_stops': True,
        'use_trailing_stops': False,
        'trail_multiplier': 0.5,  # Trailing stop multiplier
        
        # Risk-Reward
        'risk_reward_ratio': 3.0,  # 2.0-5.0 range, default 3:1
        'position_size_pct': 15.0,  # 15% of account equity per trade
        'max_position_size': 1000.0,
        
        # Indicator Parameters
        'supertrend_factor': 2.8,  # 2.0-3.5 range
        'supertrend_period': 10,  # 5-20 range
        'ma_length': 20,  # 10-50 range
        'use_close_filter': True,  # Conservative crossover (False = aggressive)
        
        # Entry Confirmation
        'require_ma_cross': True,
        'require_supertrend': True,
        'require_multiple_confirm': True,
        'confirmation_periods': 1,  # Min periods to hold signal
        
        # Dynamic Adjustment
        'adjust_to_volatility': True,
        'volatility_lookback': 20,
        'volatility_threshold': 1.5,  # High/low volatility threshold
        
        # Session Filters
        'filter_by_session': False,
        'trading_sessions': ['london', 'new_york'],  # 'london', 'new_york', 'tokyo', 'sydney'
        'avoid_session': ['tokyo'],  # Sessions to avoid
        
        # AI Optimization
        'use_ai_optimization': True,
        'ai_optimization_metric': 'sharpe_ratio',  # 'sharpe_ratio', 'total_return', 'win_rate', 'calmar'
        'ai_optimization_frequency': 'weekly',  # 'daily', 'weekly', 'monthly'
        'ai_backtest_window': 252,  # Trading days
        'ai_min_trades': 50,  # Minimum trades for optimization
        
        # Adaptive Parameters
        'adaptive_atr_multiplier': False,  # Dynamically adjust ATR multiplier
        'adaptive_rr_ratio': False,  # Dynamically adjust risk-reward ratio
        'min_atr_multiplier': 1.2,
        'max_atr_multiplier': 2.5,
        'min_rr_ratio': 2.0,
        'max_rr_ratio': 5.0
    },
    
    'monitoring': {
        'update_interval': 1.0,  # seconds
        'lookback_period': 100,
        'alert_threshold_drawdown': 0.2,
        'alert_threshold_losses': 3,
        'track_atr_history': True,
        'track_stop_loss_history': True,
        'alert_on_volatility_spike': True,
        'volatility_spike_threshold': 2.0  # 2x normal volatility
    },
    
    'optimization': {
        'n_trials': 100,
        'optimization_metric': 'sharpe_ratio',
        'use_monte_carlo': True,
        'monte_carlo_simulations': 1000,
        'use_ai_search': True,  # Use AI for parameter search
        'ai_search_method': 'bayesian',  # 'grid', 'random', 'bayesian'
        'ai_precision': 'high',  # 'low', 'medium', 'high'
        'optimization_ranges': {
            'atr_multiplier': [1.2, 1.5, 1.8, 2.0, 2.5],
            'risk_reward_ratio': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            'supertrend_factor': [2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5],
            'ma_length': [10, 15, 20, 25, 30, 50]
        }
    },
    
    'ai_components': {
        'ml_prediction': False,  # Use ML for entry prediction
        'reinforcement_learning': False,  # Use RL for parameter optimization
        'neural_network_config': {
            'hidden_layers': [64, 32, 16],
            'activation': 'relu',
            'dropout': 0.2,
            'learning_rate': 0.001
        },
        'feature_engineering': {
            'use_technical_indicators': True,
            'use_market_sentiment': False,
            'use_time_features': True,
            'feature_normalization': True
        }
    }
}

def get_config(key: str = None):
    """Get configuration value(s)"""
    if key is None:
        return MARTINGALE_CONFIG
    
    # Handle nested keys
    keys = key.split('.')
    value = MARTINGALE_CONFIG
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    
    return value

def update_config(key: str, value):
    """Update configuration value"""
    try:
        keys = key.split('.')
        config_ref = MARTINGALE_CONFIG
        
        # Navigate to the nested key
        for k in keys[:-1]:
            if not isinstance(config_ref, dict) or k not in config_ref:
                return False
            config_ref = config_ref[k]
        
        # Update the value
        config_ref[keys[-1]] = value
        return True
        
    except Exception:
        return False

