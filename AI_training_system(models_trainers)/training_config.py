#!/usr/bin/env python3
"""
Configuration for AI Training System
Comprehensive configuration management for all model types including LGMM
"""

TRAINING_CONFIG = {
    # General training settings
    'epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    'validation_split': 0.2,
    'test_split': 0.1,
    'early_stopping': True,
    'patience': 10,
    
    # LGMM-specific configuration
    'lgmm': {
        'n_components': 3,
        'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical'
        'init_params': 'kmeans',  # 'kmeans', 'k-means++', 'random'
        'max_iter': 100,
        'tol': 1e-3,
        'reg_covar': 1e-6,
        'random_state': 42,
        'warm_start': False,
        'n_init': 1,
        'normalize': True,
        'features': ['returns', 'volume_changes'],
        'description': 'Latent Gaussian Mixture Model for market regime detection',
        'use_case': 'Market regime identification, volatility clustering, trend state detection'
    },
    
    # XGBoost configuration
    'xgboost': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    
    # LSTM configuration
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 60,
        'bidirectional': False
    },
    
    # Transformer configuration
    'transformer': {
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'sequence_length': 100,
        'prediction_horizon': 24
    },
    
    # TFT configuration
    'tft': {
        'd_model': 256,
        'nhead': 4,
        'num_layers': 2,
        'quantiles': [0.1, 0.5, 0.9],
        'dropout': 0.1
    },
    
    # N-BEATS configuration
    'nbeats': {
        'forecast_length': 24,
        'backcast_length': 168,
        'stack_types': ['trend', 'seasonality'],
        'n_blocks': 3,
        'n_layers': 4,
        'layer_width': 512,
        'dropout': 0.1
    },
    
    # GNN configuration
    'gnn': {
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'num_heads': 4
    },
    
    # DQN configuration
    'dqn': {
        'state_dim': 100,
        'action_dim': 3,
        'hidden_dim': 128,
        'dropout': 0.2,
        'gamma': 0.99,
        'epsilon': 0.1,
        'batch_size': 32
    },
    
    # LLM+RL Hybrid configuration
    'llm_rl_hybrid': {
        'llm_model': 'gpt-4o-mini',
        'llm_temperature': 0.0,  # Deterministic for strategy generation
        'llm_max_tokens': 500,
        'llm_frequency_penalty': 1.0,
        'llm_presence_penalty': 0.25,
        'llm_context_window': 128000,
        'llm_seed': 49,
        'strategy_frequency': 'monthly',  # 'daily', 'weekly', 'monthly'
        'strategist_persona': 'expert_algorithmic_trader',
        'analyst_enabled': True,
        'use_in_context_memory': True,
        'use_instruction_decomposition': True,
        'use_news_sentiment': True,
        'rl_state_dim': 100,
        'rl_action_dim': 3,  # 0=hold, 1=buy, 2=sell
        'rl_hidden_dim': 128,
        'rl_learning_rate': 0.0001,
        'rl_gamma': 0.99,
        'rl_epsilon_start': 1.0,
        'rl_epsilon_end': 0.01,
        'rl_epsilon_decay': 0.995,
        'rl_batch_size': 32,
        'rl_replay_buffer_size': 10000,
        'rl_update_target_freq': 100,
        'interaction_term_type': 'direction_strength',  # How to combine LLM signal
        'confidence_scaling': 'entropy_adjusted',  # How to scale confidence
        'evaluation_metrics': ['sharpe_ratio', 'max_drawdown', 'returns'],
        'expert_review_enabled': False,
        'cost_tracking': True
    },
    
    # Device settings
    'device': 'auto',  # 'auto', 'cpu', 'cuda'
    'num_workers': 4,
    
    # MLOps settings
    'use_mlflow': True,
    'use_wandb': True,
    'use_tensorboard': True,
    'experiment_name': 'ai_training_system',
    
    # Data settings
    'data_path': 'data',
    'model_save_path': 'models',
    'log_path': 'logs',
    
    # Hyperparameter optimization
    'optuna': {
        'n_trials': 100,
        'timeout': 3600,  # seconds
        'direction': 'minimize',  # 'minimize' or 'maximize'
        'metric': 'val_loss'
    }
}

def get_config(model_type: str = None) -> dict:
    """
    Get configuration for a specific model type
    
    Args:
        model_type: Model type ('lgmm', 'lstm', 'transformer', etc.)
        
    Returns:
        Configuration dictionary
    """
    if model_type is None:
        return TRAINING_CONFIG
    
    if model_type in TRAINING_CONFIG:
        return TRAINING_CONFIG[model_type]
    
    return {}

def update_config(model_type: str, updates: dict) -> bool:
    """
    Update configuration for a model type
    
    Args:
        model_type: Model type
        updates: Dictionary of updates
        
    Returns:
        True if successful
    """
    try:
        if model_type in TRAINING_CONFIG:
            TRAINING_CONFIG[model_type].update(updates)
            return True
        return False
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

