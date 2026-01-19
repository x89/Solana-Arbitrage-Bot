#!/usr/bin/env python3
"""
Ubuntu Server Connection Helper
Helper module for connecting to your Ubuntu server at 96.8.113.136
for parallel CPU real-time calculating
"""

import logging
from typing import Optional, Dict, Any
from ssh_manager import SSHManager
from config import UBUNTU_SERVER_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ubuntu_server_connection() -> Optional[SSHManager]:
    """
    Create SSH connection to Ubuntu server for CPU calculating
    
    Returns:
        SSHManager instance or None
    """
    try:
        config = UBUNTU_SERVER_CONFIG
        
        logger.info(f"Connecting to Ubuntu server: {config['ip_address']}")
        
        ssh_manager = SSHManager(
            host=config['ip_address'],
            username=config['username'],
            password=config['password'],
            port=config['port'],
            timeout=30,
            connection_pool_size=config['max_connections']
        )
        
        # Test connection
        if ssh_manager.check_connection():
            logger.info("Successfully connected to Ubuntu server")
            return ssh_manager
        else:
            logger.error("Failed to establish connection to Ubuntu server")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Ubuntu server connection: {e}")
        return None

def execute_cpu_calculation_task(
    ssh_manager: SSHManager,
    calculation_script: str,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute CPU calculation task on Ubuntu server
    
    Args:
        ssh_manager: SSH connection manager
        calculation_script: Script or command to execute
        parameters: Optional parameters for calculation
        
    Returns:
        Task execution results
    """
    try:
        logger.info("Executing CPU calculation task on Ubuntu server...")
        
        # Build command
        command = calculation_script
        
        if parameters:
            param_str = " ".join([f"{k}={v}" for k, v in parameters.items()])
            command = f"{command} {param_str}"
        
        # Execute command
        success, stdout, stderr = ssh_manager.execute_command(command)
        
        result = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'server': 'ubuntu-cpu-server',
            'ip': '96.8.113.136'
        }
        
        if success:
            logger.info("CPU calculation task completed successfully")
        else:
            logger.warning(f"CPU calculation task failed: {stderr}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing CPU calculation task: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_server_status(ssh_manager: SSHManager) -> Dict[str, Any]:
    """
    Get Ubuntu server status
    
    Args:
        ssh_manager: SSH connection manager
        
    Returns:
        Server status information
    """
    try:
        logger.info("Checking Ubuntu server status...")
        
        # Get CPU info
        success_cpu, cpu_info, _ = ssh_manager.execute_command(
            "lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs"
        )
        
        # Get memory info
        success_mem, mem_info, _ = ssh_manager.execute_command(
            "free -h | grep Mem | awk '{print $2}'"
        )
        
        # Get disk info
        success_disk, disk_info, _ = ssh_manager.execute_command(
            "df -h / | tail -1 | awk '{print $2}'"
        )
        
        # Get uptime
        success_uptime, uptime_info, _ = ssh_manager.execute_command("uptime")
        
        status = {
            'ip_address': '96.8.113.136',
            'hostname': 'ubuntu-cpu-server',
            'status': 'online' if ssh_manager.check_connection() else 'offline',
            'cpu': cpu_info.strip() if success_cpu else 'Unknown',
            'memory': mem_info.strip() if success_mem else 'Unknown',
            'disk': disk_info.strip() if success_disk else 'Unknown',
            'uptime': uptime_info.strip() if success_uptime else 'Unknown'
        }
        
        logger.info(f"Server status: {status['status']}")
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        return {
            'ip_address': '96.8.113.136',
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main function to demonstrate Ubuntu server connection"""
    try:
        logger.info("=" * 60)
        logger.info("Ubuntu Server Connection Demo")
        logger.info("=" * 60)
        
        # Create connection
        ssh = create_ubuntu_server_connection()
        
        if not ssh:
            logger.error("Failed to connect to Ubuntu server")
            return
        
        # Get server status
        logger.info("\nGetting server status...")
        status = get_server_status(ssh)
        logger.info(f"Server IP: {status['ip_address']}")
        logger.info(f"Status: {status['status']}")
        if 'cpu' in status:
            logger.info(f"CPU: {status['cpu']}")
        if 'memory' in status:
            logger.info(f"Memory: {status['memory']}")
        
        # Example: Execute a simple calculation
        logger.info("\nExecuting sample calculation...")
        result = execute_cpu_calculation_task(
            ssh,
            "echo 'Sample calculation complete'"
        )
        logger.info(f"Result: {result['success']}")
        
        # Close connection
        ssh.close_all_connections()
        
        logger.info("=" * 60)
        logger.info("Ubuntu Server Connection Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
