#!/usr/bin/env python3
"""
GPU Manager Configuration
Comprehensive configuration for GPU resource management
"""

CONFIG = {
    # Memory Management
    'max_memory_usage': 0.9,  # 90% max memory usage per GPU
    'min_free_memory_mb': 1000,  # Minimum 1GB free memory required
    'memory_allocation_strategy': 'first_fit',  # 'first_fit', 'best_fit', 'worst_fit'
    'enable_memory_pooling': True,
    'memory_pool_size_mb': 5000,  # 5GB pool per GPU
    
    # Task Queue
    'task_queue_size': 100,
    'max_concurrent_tasks': 4,  # Per GPU
    'task_timeout_seconds': 3600,  # 1 hour timeout
    
    # GPU Monitoring
    'monitoring_interval_seconds': 5,
    'performance_history_size': 1000,  # Keep last 1000 performance records
    'enable_real_time_monitoring': True,
    'track_temperature': True,
    'track_power_usage': True,
    'temperature_threshold_celsius': 80,
    'power_usage_threshold_watts': 250,
    
    # Task Scheduling
    'task_scheduling_strategy': 'priority',  # 'priority', 'fifo', 'round_robin'
    'enable_task_preemption': False,  # Allow high priority tasks to interrupt low priority
    'max_task_retries': 3,
    'task_retry_delay_seconds': 5,
    
    # GPU Selection
    'gpu_selection_strategy': 'best_fit',  # 'best_fit', 'first_available', 'load_balance'
    'prefer_cooler_gpus': True,
    'prefer_lower_power_usage': False,
    
    # Performance Optimization
    'enable_auto_optimization': True,
    'optimization_check_interval_seconds': 10,
    'enable_memory_compression': False,
    'enable_model_caching': True,
    'cache_size_mb': 10000,  # 10GB cache
    
    # Task Priority Weights
    'priority_weights': {
        'CRITICAL': 4.0,
        'HIGH': 3.0,
        'MEDIUM': 2.0,
        'LOW': 1.0
    },
    
    # Task Type Default Settings
    'task_defaults': {
        'training': {
            'expected_memory_mb': 2000,
            'expected_duration_seconds': 300,
            'preferred_gpu': None,  # None = auto-select
            'allow_preemption': False
        },
        'inference': {
            'expected_memory_mb': 500,
            'expected_duration_seconds': 10,
            'preferred_gpu': None,
            'allow_preemption': True
        },
        'pattern_detection': {
            'expected_memory_mb': 800,
            'expected_duration_seconds': 30,
            'preferred_gpu': None,
            'allow_preemption': False
        },
        'sentiment_analysis': {
            'expected_memory_mb': 300,
            'expected_duration_seconds': 5,
            'preferred_gpu': None,
            'allow_preemption': True
        }
    },
    
    # System Limits
    'max_gpu_utilization': 95,  # Maximum GPU utilization percentage
    'min_gpu_utilization': 10,  # Minimum for optimization checks
    'enable_gpu_throttling': True,
    'enable_thermal_throttling': True,
    
    # Logging and Reporting
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'enable_performance_logging': True,
    'performance_log_interval_seconds': 60,
    'enable_task_logging': True,
    'enable_system_logging': True,
    
    # Resource Limits
    'max_tasks_per_gpu': 5,
    'max_total_memory_usage_gb': None,  # None = no limit
    'memory_cleanup_threshold': 0.85,  # Clean up when usage exceeds 85%
    
    # CUDA Settings
    'cuda_device_order': 'PCI_BUS_ID',  # 'PCI_BUS_ID', 'FASTEST_FIRST'
    'enable_cuda_peer_access': True,  # Enable peer-to-peer access between GPUs
    'enable_unified_memory': False,  # Enable CUDA unified memory
    'max_split_size_mb': 256,  # Maximum memory split size
    
    # Auto-scaling and Load Balancing
    'enable_auto_scaling': True,
    'load_balance_strategy': 'round_robin',  # 'round_robin', 'least_utilized', 'most_free_memory'
    'minimum_free_gpus': 1,  # Always keep at least 1 GPU free if possible
    
    # Health Checks
    'enable_health_checks': True,
    'health_check_interval_seconds': 60,
    'max_consecutive_failures': 3,
    'auto_recovery_enabled': True,
    
    # Data Persistence
    'save_task_history': True,
    'task_history_file': 'gpu_task_history.json',
    'save_performance_metrics': True,
    'performance_metrics_file': 'gpu_performance_metrics.json',
    
    # Alerts and Notifications
    'enable_alerts': True,
    'alert_on_high_temperature': True,
    'alert_on_high_memory_usage': True,
    'alert_on_task_failure': True,
    'alert_on_gpu_unavailable': True
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
    return CONFIG.get(key)

def update_config(key: str, value):
    """
    Update configuration value
    
    Args:
        key: Configuration key
        value: New value
        
    Returns:
        True if successful
    """
    if key in CONFIG:
        CONFIG[key] = value
        return True
    return False

