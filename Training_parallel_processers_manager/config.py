#!/usr/bin/env python3
"""
Training Parallel Processors Configuration
Comprehensive configuration for parallel training management
"""

from typing import Dict, Any

CONFIG = {
    # Job Management
    'max_concurrent_jobs': 4,
    'max_jobs_per_node': 8,
    'job_timeout_seconds': 86400,  # 24 hours
    'job_checkpoint_frequency': 300,  # Save every 5 minutes
    'enable_job_cancellation': True,
    'max_job_retries': 3,
    'retry_delay_seconds': 60,
    
    # GPU Allocation
    'gpu_per_job': 1,
    'min_gpu_memory_gb': 4.0,
    'max_gpus_per_job': 8,
    'gpu_allocation_strategy': 'first_fit',  # 'first_fit', 'best_fit', 'round_robin'
    'prefer_cooler_gpus': True,
    'enable_gpu_isolation': True,
    
    # Resource Monitoring
    'resource_monitoring_interval': 5.0,
    'resource_history_size': 1000,
    'enable_cpu_monitoring': True,
    'enable_memory_monitoring': True,
    'enable_gpu_monitoring': True,
    'enable_network_monitoring': False,
    'enable_disk_monitoring': True,
    'monitoring_alert_thresholds': {
        'cpu_usage': 90,
        'memory_usage': 90,
        'gpu_usage': 95,
        'disk_usage': 90,
        'temperature': 85
    },
    
    # Distributed Training
    'enable_distributed_training': True,
    'distributed_backend': 'nccl',  # 'nccl' for GPU, 'gloo' for CPU
    'master_addr': 'localhost',
    'master_port': '12355',
    'init_method': 'tcp://localhost:12355',
    'sync_bn': False,
    'ddp_find_unused_parameters': False,
    'ddp_bucket_cap_mb': 25,
    'ddp_broadcast_buffers': True,
    
    # Training Settings
    'training_batch_size': 32,
    'training_eval_frequency': 1000,  # Steps
    'training_save_frequency': 10,  # Epochs
    'training_log_frequency': 100,  # Steps
    'enable_mixed_precision': True,
    'enable_gradient_accumulation': False,
    'gradient_accumulation_steps': 1,
    'gradient_clipping': 1.0,
    
    # Checkpoint Management
    'checkpoint_save_path': 'checkpoints',
    'max_checkpoints_to_keep': 5,
    'checkpoint_format': 'pth',  # 'pth', 'pt', 'h5'
    'enable_snapshot': True,
    'snapshot_frequency': 50,  # Epochs
    
    # Dataset Configuration
    'dataset_cache_size': 1000,
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,
    'dataloader_prefetch_factor': 2,
    'dataloader_persistent_workers': True,
    
    # Database
    'database_path': 'training_jobs.db',
    'database_backup_frequency': 3600,  # 1 hour
    'database_backup_path': 'backups',
    'enable_job_history': True,
    'max_job_history_days': 90,
    
    # Scheduling
    'scheduler_strategy': 'priority',  # 'priority', 'fifo', 'fair'
    'enable_job_prioritization': True,
    'priority_weights': {
        'critical': 4,
        'high': 3,
        'medium': 2,
        'low': 1
    },
    'enable_job_preemption': False,
    'max_queue_size': 100,
    
    # Performance Optimization
    'enable_auto_optimization': True,
    'optimization_check_interval': 60,  # seconds
    'enable_batch_size_auto_tune': True,
    'enable_learning_rate_auto_tune': False,
    'memory_optimization': 'auto',  # 'auto', 'aggressive', 'conservative'
    
    # Logging
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_directory': 'logs',
    'enable_tensorboard': True,
    'enable_wandb': False,
    'enable_mlflow': False,
    'tensorboard_dir': 'runs',
    'log_every_n_steps': 100,
    'save_summary_steps': 1000,
    
    # Error Handling
    'enable_error_recovery': True,
    'max_consecutive_errors': 3,
    'error_backoff_delay': 60,  # seconds
    'enable_auto_restart': False,
    'restart_max_attempts': 2,
    
    # Network Settings
    'enable_nccl_timeout': True,
    'nccl_timeout_seconds': 1800,  # 30 minutes
    'enable_nccl_debug': False,
    'nccl_debug_level': 'INFO',
    
    # Security
    'enable_resource_limits': True,
    'max_memory_per_job_gb': 32,
    'max_cpu_per_job': 8,
    'enable_job_isolation': True,
    'job_timeout_action': 'kill',  # 'kill', 'suspend', 'terminate'
    
    # Cost Management
    'enable_cost_tracking': False,
    'cost_per_gpu_hour': 0.50,  # USD
    'cost_per_cpu_hour': 0.01,  # USD
    'cost_per_memory_gb_hour': 0.001,  # USD
    'budget_limit': None,
    
    # Health Checks
    'enable_health_checks': True,
    'health_check_interval': 300,  # 5 minutes
    'max_consecutive_health_failures': 3,
    'auto_recovery_enabled': True,
    
    # Data Persistence
    'save_training_metrics': True,
    'save_resource_usage': True,
    'save_job_traces': True,
    'metrics_save_path': 'metrics',
    'traces_save_path': 'traces',
    
    # Alerting
    'enable_alerts': True,
    'alert_on_job_failure': True,
    'alert_on_resource_exhaustion': True,
    'alert_on_health_check_failure': True,
    'alert_on_budget_exceeded': True,
    'alert_channels': ['log'],  # 'log', 'email', 'slack', 'webhook'
    
    # Advanced Features
    'enable_job_chaining': False,
    'enable_conditional_execution': False,
    'enable_data_pipeline_caching': True,
    'cache_directory': '.cache',
    'enable_model_versioning': True,
    'model_versioning_format': 'v{timestamp}'
}

def get_config(key: str = None):
    """
    Get configuration value(s)
    
    Args:
        key: Configuration key (returns entire config if None)
        
    Returns:
        Configuration value or entire config dict
    """
    if key is None:
        return CONFIG
    
    # Handle nested keys
    keys = key.split('.')
    value = CONFIG
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    
    return value

def update_config(key: str, value):
    """
    Update configuration value
    
    Args:
        key: Configuration key (supports nested keys with '.')
        value: New value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        keys = key.split('.')
        config_ref = CONFIG
        
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

def validate_config() -> Dict[str, Any]:
    """
    Validate configuration
    
    Returns:
        Dictionary with validation results
    """
    warnings = []
    errors = []
    
    # Check max_concurrent_jobs
    if CONFIG['max_concurrent_jobs'] > CONFIG['max_jobs_per_node']:
        warnings.append(f"max_concurrent_jobs ({CONFIG['max_concurrent_jobs']}) exceeds max_jobs_per_node ({CONFIG['max_jobs_per_node']})")
    
    # Check GPU allocation
    if CONFIG['max_gpus_per_job'] < CONFIG['gpu_per_job']:
        errors.append(f"max_gpus_per_job ({CONFIG['max_gpus_per_job']}) is less than gpu_per_job ({CONFIG['gpu_per_job']})")
    
    # Check monitoring intervals
    if CONFIG['resource_monitoring_interval'] < 1.0:
        warnings.append(f"resource_monitoring_interval ({CONFIG['resource_monitoring_interval']}) is very low, may impact performance")
    
    # Check timeout
    if CONFIG['job_timeout_seconds'] < 300:
        warnings.append(f"job_timeout_seconds ({CONFIG['job_timeout_seconds']}) is very short")
    
    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors
}

