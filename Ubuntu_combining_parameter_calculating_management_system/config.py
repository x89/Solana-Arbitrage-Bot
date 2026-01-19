"""
Cloud Configuration Module
Comprehensive configuration management for cloud operations including:
- AWS region and credentials configuration
- Instance type specifications
- Cost thresholds and budgets
- SSH connection settings
- Task dispatching parameters
- Optimization settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_CONFIG = {
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
    'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'session_token': os.getenv('AWS_SESSION_TOKEN'),
    'default_vpc': True,
    'security_groups': ['default']
}

# Instance Configuration
INSTANCE_CONFIG = {
    'default_instance_type': 't3.medium',
    'instance_types': {
        't3.micro': {'vcpus': 2, 'memory': 1, 'hourly_cost': 0.0104},
        't3.small': {'vcpus': 2, 'memory': 2, 'hourly_cost': 0.0208},
        't3.medium': {'vcpus': 2, 'memory': 4, 'hourly_cost': 0.0416},
        't3.large': {'vcpus': 2, 'memory': 8, 'hourly_cost': 0.0832},
        'm5.large': {'vcpus': 2, 'memory': 8, 'hourly_cost': 0.096},
        'c5.large': {'vcpus': 2, 'memory': 4, 'hourly_cost': 0.085},
    },
    'preferred_instance_types': ['t3.medium', 't3.large', 'm5.large'],
    'min_instances': 1,
    'max_instances': 10,
    'auto_scaling_enabled': True
}

# SSH Configuration
SSH_CONFIG = {
    'default_username': 'root',
    'default_port': 22,
    'connection_timeout': 30,
    'connection_pool_size': 5,
    'private_key_path': os.getenv('SSH_PRIVATE_KEY_PATH', '~/.ssh/id_rsa'),
    'password': 'N$##O#@PKUE#',
    'known_hosts_path': '~/.ssh/known_hosts'
}

# Ubuntu Server Configuration for Parallel CPU Calculating
UBUNTU_SERVER_CONFIG = {
    'ip_address': '96.8.113.136',
    'username': 'root',
    'password': 'N$##O#@PKUE#',
    'port': 22,
    'hostname': 'ubuntu-cpu-server',
    'description': 'Ubuntu server for parallel CPU real-time calculating',
    'max_connections': 5,
    'enabled': True,
    'region': 'production',
    'purpose': 'real-time-cpu-calculating'
}

# Task Dispatching Configuration
TASK_CONFIG = {
    'max_concurrent_tasks': 5,
    'task_timeout': 3600,  # 1 hour
    'max_retries': 3,
    'retry_delay': 60,  # 1 minute
    'queue_check_interval': 5,  # seconds
    'priority_levels': ['low', 'normal', 'high', 'urgent'],
    'priority_weights': {'low': 1, 'normal': 5, 'high': 10, 'urgent': 20}
}

# Cost Optimization Configuration
COST_CONFIG = {
    'budget_threshold': float(os.getenv('BUDGET_THRESHOLD', '1000.0')),
    'alert_threshold': 0.8,  # Alert at 80% of budget
    'optimization_enabled': os.getenv('OPTIMIZATION_ENABLED', 'true').lower() == 'true',
    'auto_stop_enabled': os.getenv('AUTO_STOP_ENABLED', 'false').lower() == 'true',
    'cost_tracking_enabled': True,
    'max_hourly_cost': 1.0,
    'cost_reports_interval': 24 * 3600,  # 24 hours
    'low_utilization_threshold': 10  # hours
}

# Cloud Handler Configuration
CLOUD_CONFIG = {
    'cloud_handler_enabled': True,
    'instance_lifetime_hours': 24,
    'spare_instance_pool': 2,
    'health_check_interval': 300,  # 5 minutes
    'auto_cleanup': True,
    'resource_tags': {
        'Environment': 'development',
        'Project': 'ai-trading',
        'ManagedBy': 'cloud_handler'
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'cloud_handler.log',
    'max_bytes': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# Complete Configuration
CONFIG = {
    'aws': AWS_CONFIG,
    'instance': INSTANCE_CONFIG,
    'ssh': SSH_CONFIG,
    'ubuntu_server': UBUNTU_SERVER_CONFIG,
    'task': TASK_CONFIG,
    'cost': COST_CONFIG,
    'cloud': CLOUD_CONFIG,
    'logging': LOGGING_CONFIG
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default config
            for key, value in loaded_config.items():
                CONFIG[key].update(value)
            
            logger.info(f"Configuration loaded from {config_path}")
        
        return CONFIG
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return CONFIG

def save_config(config_path: str, config: Dict[str, Any] = None):
    """
    Save configuration to file
    
    Args:
        config_path: Path to save configuration
        config: Configuration to save (uses current CONFIG if None)
    """
    try:
        config = config or CONFIG
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def get_config(key: str) -> Any:
    """
    Get configuration value by key
    
    Args:
        key: Configuration key (supports dot notation: 'aws.region')
        
    Returns:
        Configuration value or None
    """
    try:
        keys = key.split('.')
        value = CONFIG
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return None

def set_config(key: str, value: Any):
    """
    Set configuration value by key
    
    Args:
        key: Configuration key (supports dot notation)
        value: Configuration value to set
    """
    try:
        keys = key.split('.')
        target = CONFIG
        
        # Navigate to target dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set value
        target[keys[-1]] = value
        
        logger.info(f"Configuration {key} set to {value}")
        
    except Exception as e:
        logger.error(f"Error setting configuration: {e}")

def validate_config() -> bool:
    """
    Validate configuration
    
    Returns:
        True if configuration is valid
    """
    try:
        # Check required environment variables
        required_vars = []
        
        if not os.getenv('AWS_REGION'):
            required_vars.append('AWS_REGION')
        
        # Validate instance configuration
        if not INSTANCE_CONFIG.get('default_instance_type'):
            logger.error("No default instance type specified")
            return False
        
        # Validate cost configuration
        if COST_CONFIG.get('budget_threshold', 0) <= 0:
            logger.error("Invalid budget threshold")
            return False
        
        logger.info("Configuration validated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def print_config():
    """Print current configuration"""
    try:
        logger.info("=" * 60)
        logger.info("Current Configuration")
        logger.info("=" * 60)
        
        for section, settings in CONFIG.items():
            logger.info(f"\n{section.upper()}:")
            for key, value in settings.items():
                # Mask sensitive values
                if 'password' in key.lower() or 'key' in key.lower():
                    value = '***' if value else 'Not set'
                logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Error printing configuration: {e}")

def main():
    """Main function to demonstrate configuration management"""
    try:
        logger.info("=" * 60)
        logger.info("Configuration Management Demo")
        logger.info("=" * 60)
        
        # Print current configuration
        print_config()
        
        # Validate configuration
        is_valid = validate_config()
        logger.info(f"\nConfiguration is valid: {is_valid}")
        
        # Get specific config value
        region = get_config('aws.region')
        logger.info(f"\nAWS Region: {region}")
        
        # Set config value
        set_config('aws.region', 'us-west-2')
        logger.info(f"Updated region to: {get_config('aws.region')}")
        
        logger.info("=" * 60)
        logger.info("Configuration Management Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

