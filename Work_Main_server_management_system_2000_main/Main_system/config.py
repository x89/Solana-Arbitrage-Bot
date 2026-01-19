"""
Main Server Configuration
Comprehensive configuration for the management system
"""

from typing import Dict, List

CONFIG = {
    # Monitoring settings
    'monitoring_interval': 60,  # seconds
    'health_check_interval': 300,  # seconds
    'retention_period': 86400,  # 24 hours
    
    # Alert thresholds
    'alert_threshold': 0.8,
    'cpu_warning_threshold': 70.0,
    'cpu_critical_threshold': 85.0,
    'memory_warning_threshold': 75.0,
    'memory_critical_threshold': 90.0,
    'disk_warning_threshold': 80.0,
    'disk_critical_threshold': 95.0,
    'network_timeout': 5.0,
    'response_time_warning': 1.0,
    'response_time_critical': 3.0,
    
    # Email alerts
    'email_enabled': False,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email_username': '',
    'email_password': '',
    'email_recipients': [],
    
    # Slack notifications
    'slack_enabled': False,
    'slack_webhook_url': '',
    
    # Discord notifications
    'discord_enabled': False,
    'discord_webhook_url': '',
    
    # Generic webhooks
    'webhook_enabled': False,
    'webhook_url': '',
    'webhook_headers': {'Content-Type': 'application/json'},
    
    # Dashboard settings
    'dashboard_port': 5000,
    'dashboard_host': '0.0.0.0',
    
    # Subsystem management
    'auto_start_subsystems': False,
    'auto_restart_failed': True,
    'max_restart_attempts': 3,
    'restart_delay': 10,  # seconds
    
    # Database settings
    'database_path': 'management.db',
    'log_level': 'INFO',
    
    # Performance tracking
    'performance_tracking_enabled': True,
    'performance_data_retention_days': 30,
    
    # Auto-scaling
    'auto_scaling_enabled': False,
    'scale_up_threshold': 80.0,  # CPU/Memory %
    'scale_down_threshold': 30.0,
    
    # Subsystems configuration
    'subsystems': {
        'data_collection': {
            'path': '../Data_collecting_system_bitget',
            'enabled': True,
            'start_script': 'advanced_data_collector.py',
            'required': True
        },
        'ai_training': {
            'path': '../AI_training_system',
            'enabled': True,
            'start_script': 'advanced_ai_trainer.py',
            'required': False
        },
        'ai_prediction': {
            'path': '../AI_predicting_model_generating_system',
            'enabled': True,
            'start_script': 'prediction_engine.py',
            'required': True
        },
        'pattern_detection': {
            'path': '../AI_pattern_detecting_system',
            'enabled': True,
            'start_script': 'pattern_detector.py',
            'required': False
        },
        'sentiment_analysis': {
            'path': '../AI_sentiment_training_analyzing_system',
            'enabled': True,
            'start_script': 'sentiment_analyzer.py',
            'required': False
        },
        'momentum_prediction': {
            'path': '../AI_momentum_real_time_predicting_system',
            'enabled': True,
            'start_script': 'momentum_predictor.py',
            'required': False
        },
        'indicator_analysis': {
            'path': '../Analyzing_indicators_comparing_momentum_system',
            'enabled': True,
            'start_script': 'indicator_analyzer.py',
            'required': False
        },
        'backtesting': {
            'path': '../Backtesting_checking_system',
            'enabled': True,
            'start_script': 'backtest_engine.py',
            'required': False
        },
        'signal_testing': {
            'path': '../Signal_testing_system',
            'enabled': True,
            'start_script': 'signal_tester.py',
            'required': False
        },
        'training_manager': {
            'path': '../Training_parallel_processers_manager',
            'enabled': True,
            'start_script': 'training_manager.py',
            'required': False
        }
    }
}

