#!/usr/bin/env python3
"""
Prediction System Configuration
Comprehensive configuration management for the AI predicting model generating system including:
- Model configurations (XGBoost, LSTM, Transformer, TFT, N-BEATS, GNN, DQN)
- Ensemble settings and weights
- Feature engineering parameters
- Inference settings
- Performance thresholds
- Resource allocation
"""

import os
from typing import Dict, List, Any, Optional
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configurations
MODEL_CONFIGS = {
    'xgboost': {
        'name': 'XGBoost',
        'type': 'tree_based',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        'requires_gpu': False,
        'inference_time_ms': 10,
        'max_features': 100
    },
    'lstm': {
        'name': 'LSTM',
        'type': 'deep_learning',
        'hyperparameters': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'sequence_length': 60
        },
        'requires_gpu': True,
        'inference_time_ms': 25,
        'max_features': 50
    },
    'transformer': {
        'name': 'Transformer',
        'type': 'deep_learning',
        'hyperparameters': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'sequence_length': 100
        },
        'requires_gpu': True,
        'inference_time_ms': 50,
        'max_features': 100
    },
    'tft': {
        'name': 'Temporal Fusion Transformer',
        'type': 'deep_learning',
        'hyperparameters': {
            'd_model': 256,
            'nhead': 4,
            'num_layers': 2,
            'quantiles': [0.1, 0.5, 0.9],
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'requires_gpu': True,
        'inference_time_ms': 75,
        'max_features': 50
    },
    'nbeats': {
        'name': 'N-BEATS',
        'type': 'deep_learning',
        'hyperparameters': {
            'forecast_length': 24,
            'backcast_length': 168,
            'stack_types': ['trend', 'seasonality'],
            'n_blocks': 3,
            'n_layers': 4,
            'layer_width': 512,
            'dropout': 0.1,
            'learning_rate': 0.001
        },
        'requires_gpu': True,
        'inference_time_ms': 100,
        'max_features': 30
    },
    'gnn': {
        'name': 'Graph Neural Network',
        'type': 'deep_learning',
        'hyperparameters': {
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'num_heads': 4,
            'learning_rate': 0.001,
            'batch_size': 16
        },
        'requires_gpu': True,
        'inference_time_ms': 150,
        'max_features': 100
    },
    'dqn': {
        'name': 'Deep Q-Network',
        'type': 'reinforcement_learning',
        'hyperparameters': {
            'state_dim': 100,
            'action_dim': 3,
            'hidden_dim': 128,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.1,
            'batch_size': 32
        },
        'requires_gpu': True,
        'inference_time_ms': 30,
        'max_features': 50
    },
    'lgmm': {
        'name': 'Latent Gaussian Mixture Model',
        'type': 'unsupervised_clustering',
        'hyperparameters': {
            'n_components': 3,
            'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical'
            'init_params': 'kmeans',  # 'kmeans', 'k-means++', 'random'
            'max_iter': 100,
            'tol': 1e-3,
            'reg_covar': 1e-6,
            'random_state': 42,
            'warm_start': False,
            'n_init': 1
        },
        'requires_gpu': False,
        'inference_time_ms': 20,
        'max_features': 100,
        'description': 'LGMM identifies hidden market regimes (stable, volatile, mixed) by modeling data as mixtures of Gaussian distributions',
        'use_case': 'Market regime detection, volatility clustering, trend state identification'
    },
    'chronos_bolt': {
        'name': 'Chronos Bolt',
        'type': 'time_series',
        'hyperparameters': {
            'context_length': 64,
            'prediction_length': 24,
            'num_samples': 100,
            'temperature': 1.0
        },
        'requires_gpu': True,
        'inference_time_ms': 200,
        'max_features': 1,
        'description': 'Amazon Chronos Bolt - Lightweight and fast time series forecasting model',
        'use_case': 'Real-time momentum prediction, fast multi-horizon forecasting'
    },
    'chronos_t5': {
        'name': 'Chronos T5',
        'type': 'time_series',
        'hyperparameters': {
            'context_length': 128,
            'prediction_length': 64,
            'num_samples': 100,
            'temperature': 1.0
        },
        'requires_gpu': True,
        'inference_time_ms': 400,
        'max_features': 1,
        'description': 'Amazon Chronos T5 - Large scale time series forecasting model',
        'use_case': 'High-accuracy momentum prediction, long-horizon forecasting'
    }
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'weight_method': 'dynamic',  # 'static', 'dynamic', 'performance_based'
    'static_weights': {
        'chronos_bolt': 0.20,  # Fast real-time predictions
        'chronos_t5': 0.15,    # Accurate long-horizon predictions
        'xgboost': 0.15,
        'lstm': 0.15,
        'transformer': 0.12,
        'lgmm': 0.08,          # Regime detection
        'tft': 0.05,
        'nbeats': 0.05,
        'gnn': 0.03,
        'dqn': 0.02
    },
    'performance_metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae'],
    'dynamic_weight_window': 100,  # Number of predictions to use for dynamic weighting
    'min_models_for_ensemble': 2,
    'max_models_in_ensemble': 5
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_indicators': {
        'rsi': {'period': 14, 'enabled': True},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9, 'enabled': True},
        'bollinger': {'period': 20, 'std_dev': 2, 'enabled': True},
        'ema': {'periods': [12, 26, 50, 200], 'enabled': True},
        'sma': {'periods': [5, 10, 20, 50], 'enabled': True},
        'stochastic': {'k_period': 14, 'd_period': 3, 'enabled': True},
        'williams_r': {'period': 14, 'enabled': True},
        'adx': {'period': 14, 'enabled': True},
        'cci': {'period': 20, 'enabled': True},
        'atr': {'period': 14, 'enabled': True}
    },
    'price_features': {
        'price_change': True,
        'price_change_5': True,
        'price_change_20': True,
        'price_change_50': True,
        'volatility': True,
        'momentum': True,
        'acceleration': True
    },
    'volume_features': {
        'volume_sma_20': True,
        'volume_ratio': True,
        'volume_price_trend': True,
        'obv': True,
        'vwap': True
    },
    'sequence_features': {
        'sequence_length': 60,
        'create_lags': True,
        'lag_periods': [1, 2, 3, 5, 10, 20],
        'rolling_windows': [5, 10, 20, 50]
    }
}

# Inference Configuration
INFERENCE_CONFIG = {
    'batch_size': 32,
    'cache_enabled': True,
    'cache_size': 1000,
    'parallel_inference': True,
    'max_parallel_models': 3,
    'timeout_seconds': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 1
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.6,
    'min_precision': 0.55,
    'min_recall': 0.55,
    'min_f1_score': 0.55,
    'max_mse': 1.0,
    'max_mae': 0.8,
    'confidence_threshold': 0.6,
    'prediction_stability_threshold': 0.8
}

# Resource Allocation
RESOURCE_CONFIG = {
    'gpu_allocation': {
        'enabled': True,
        'preferred_devices': None,  # None = auto-detect
        'memory_limit_mb': None,  # None = no limit
        'gpu_timeout_seconds': 60
    },
    'cpu_allocation': {
        'num_threads': None,  # None = auto-detect
        'memory_limit_gb': None  # None = no limit
    },
    'cloud_services': {
        'enabled': False,
        'service_provider': 'aws',  # 'aws', 'gcp', 'azure'
        'instance_type': 't3.medium',
        'max_cost_per_hour': 10.0
    }
}

# Data Processing Configuration
DATA_CONFIG = {
    'normalization': {
        'method': 'minmax',  # 'minmax', 'standard', 'robust'
        'feature_range': (0, 1),
        'fit_on_training_only': True
    },
    'handling_missing_values': {
        'strategy': 'forward_fill',  # 'forward_fill', 'backward_fill', 'interpolate', 'mean', 'median'
        'max_consecutive_nans': 5,
        'drop_if_high_missing': 0.5
    },
    'outlier_handling': {
        'enabled': True,
        'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
        'threshold': 3.0,
        'clip_extremes': True
    }
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'prediction_horizon': 24,  # Number of future periods
    'prediction_types': ['price', 'direction', 'volatility'],
    'confidence_calculation': {
        'method': 'calibration',  # 'calibration', 'monte_carlo', 'ensemble_variance'
        'num_samples': 100,
        'temperature': 1.0
    },
    'result_formatting': {
        'include_confidence_intervals': True,
        'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
        'include_timestamps': True,
        'include_metadata': True
    }
}

# Model Registry Configuration
MODEL_REGISTRY_CONFIG = {
    'storage_path': 'trained_models',
    'checkpoint_frequency': 10,  # Save checkpoint every N epochs
    'model_versioning': True,
    'backup_enabled': True,
    'compression': False,
    'model_format': 'pytorch',  # 'pytorch', 'onnx', 'tensorrt', 'torchscript'
    'export_formats': ['pytorch', 'onnx']
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'logging': {
        'level': 'INFO',
        'log_predictions': True,
        'log_performance_metrics': True,
        'log_errors': True,
        'log_file': 'prediction_system.log'
    },
    'metrics_tracking': {
        'enabled': True,
        'track_prediction_accuracy': True,
        'track_inference_time': True,
        'track_resource_usage': True,
        'dashboard_enabled': False
    },
    'alerts': {
        'enabled': True,
        'performance_degradation_threshold': 0.1,
        'send_email_alerts': False,
        'send_slack_alerts': False
    }
}

# Main Configuration Dictionary
CONFIG = {
    'models': MODEL_CONFIGS,
    'ensemble': ENSEMBLE_CONFIG,
    'features': FEATURE_CONFIG,
    'inference': INFERENCE_CONFIG,
    'performance': PERFORMANCE_THRESHOLDS,
    'resources': RESOURCE_CONFIG,
    'data_processing': DATA_CONFIG,
    'prediction': PREDICTION_CONFIG,
    'model_registry': MODEL_REGISTRY_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'environment': {
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
        'max_workers': int(os.getenv('MAX_WORKERS', '4')),
        'timezone': os.getenv('TIMEZONE', 'UTC'),
        'data_timezone': os.getenv('DATA_TIMEZONE', 'UTC')
    }
}

def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name)

def get_feature_list() -> List[str]:
    """Get list of enabled features"""
    features = []
    
    # Technical indicators
    for indicator, config in FEATURE_CONFIG['technical_indicators'].items():
        if config.get('enabled', False):
            features.append(indicator)
    
    # Price features
    for feature, enabled in FEATURE_CONFIG['price_features'].items():
        if enabled:
            features.append(feature)
    
    # Volume features
    for feature, enabled in FEATURE_CONFIG['volume_features'].items():
        if enabled:
            features.append(feature)
    
    return features

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check model configurations
        for model_name, model_config in MODEL_CONFIGS.items():
            assert 'name' in model_config, f"Model {model_name} missing 'name'"
            assert 'type' in model_config, f"Model {model_name} missing 'type'"
            assert 'hyperparameters' in model_config, f"Model {model_name} missing 'hyperparameters'"
        
        # Check ensemble weights sum to 1
        static_weights = ENSEMBLE_CONFIG['static_weights']
        total_weight = sum(static_weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Ensemble weights sum to {total_weight}, not 1.0"
        
        # Check performance thresholds
        assert PERFORMANCE_THRESHOLDS['min_accuracy'] > 0, "min_accuracy must be positive"
        
        logger.info("Configuration validated successfully")
        return True
        
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def update_config(section: str, key: str, value: Any) -> bool:
    """Update configuration value"""
    try:
        if section in CONFIG:
            if isinstance(CONFIG[section], dict):
                CONFIG[section][key] = value
                logger.info(f"Updated {section}.{key} = {value}")
                return True
            else:
                logger.error(f"Cannot update non-dict section: {section}")
                return False
        else:
            logger.error(f"Unknown configuration section: {section}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

if __name__ == "__main__":
    """Main function to demonstrate configuration"""
    logger.info("Prediction System Configuration")
    logger.info("=" * 60)
    
    # Validate configuration
    validate_config()
    
    # Display available models
    logger.info("\nAvailable Models:")
    for model_name, model_config in MODEL_CONFIGS.items():
        logger.info(f"  - {model_name}: {model_config['name']} ({model_config['type']})")
    
    # Display enabled features
    features = get_feature_list()
    logger.info(f"\nEnabled Features: {len(features)}")
    logger.info(f"  {features[:10]}..." if len(features) > 10 else f"  {features}")
    
    logger.info("\nConfiguration loaded successfully!")

