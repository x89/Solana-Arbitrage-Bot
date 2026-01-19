#!/usr/bin/env python3
"""
SSH Manager Module
Comprehensive SSH connection management system including:
- Secure connection establishment
- Command execution and file transfer
- Connection pooling and reuse
- Error handling and retry logic
- Connection health monitoring
- Session management
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Optional import
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logging.warning("paramiko not available. SSH features will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSHManager:
    """Comprehensive SSH connection manager"""
    
    def __init__(
        self,
        host: str,
        username: str,
        password: Optional[str] = None,
        private_key_path: Optional[str] = None,
        port: int = 22,
        timeout: int = 30,
        connection_pool_size: int = 5
    ):
        """
        Initialize SSH manager
        
        Args:
            host: Hostname or IP address
            username: SSH username
            password: SSH password (optional if using key)
            private_key_path: Path to private key file
            port: SSH port
            timeout: Connection timeout in seconds
            connection_pool_size: Maximum number of connections to pool
        """
        self.host = host
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.port = port
        self.timeout = timeout
        self.connection_pool_size = connection_pool_size
        
        # Connection pool
        self.connections = []
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'reused_connections': 0
        }
        
        logger.info(f"SSH manager initialized for {host}:{port}")
    
    def _get_connection(self):
        """
        Get SSH connection from pool or create new
        
        Returns:
            SSH connection client
        """
        try:
            if not PARAMIKO_AVAILABLE:
                logger.error("paramiko not available. Cannot create SSH connection.")
                return None
            
            # Try to get existing connection
            if self.connections:
                conn = self.connections.pop()
                self.connection_stats['reused_connections'] += 1
                logger.debug("Reusing existing SSH connection")
                return conn
            
            # Create new connection
            self.connection_stats['total_connections'] += 1
            logger.info(f"Creating new SSH connection to {self.host}:{self.port}")
            
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Load private key if provided
            key = None
            if self.private_key_path and os.path.exists(self.private_key_path):
                key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            
            # Connect
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                pkey=key,
                timeout=self.timeout,
                allow_agent=False,
                look_for_keys=False
            )
            
            self.connection_stats['active_connections'] += 1
            
            return client
            
        except Exception as e:
            self.connection_stats['failed_connections'] += 1
            logger.error(f"Error creating SSH connection: {e}")
            return None
    
    def _return_connection(self, client):
        """Return connection to pool"""
        try:
            if client is None:
                return
                
            if len(self.connections) < self.connection_pool_size:
                self.connections.append(client)
                logger.debug("Connection returned to pool")
            else:
                if hasattr(client, 'close'):
                    client.close()
                self.connection_stats['active_connections'] -= 1
                logger.debug("Connection pool full, closing connection")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
    
    def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        silent: bool = False
    ) -> Tuple[bool, str, str]:
        """
        Execute command on remote host
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            silent: Whether to suppress output
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            if not silent:
                logger.info(f"Executing command: {command}")
            
            client = self._get_connection()
            
            if not client:
                return False, "", "Failed to establish SSH connection"
            
            # Execute command
            timeout = timeout or self.timeout
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Read output
            exit_status = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode()
            stderr_text = stderr.read().decode()
            
            # Return connection to pool
            self._return_connection(client)
            
            success = exit_status == 0
            
            if not silent:
                if success:
                    logger.info(f"Command executed successfully")
                else:
                    logger.warning(f"Command failed with exit status {exit_status}")
            
            return success, stdout_text, stderr_text
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False, "", str(e)
    
    def upload_file(
        self,
        local_path: str,
        remote_path: str
    ) -> bool:
        """
        Upload file to remote host
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Uploading {local_path} to {remote_path}")
            
            client = self._get_connection()
            
            if not client:
                logger.error("Failed to establish SSH connection")
                return False
            
            # Use SFTP
            sftp = client.open_sftp()
            
            # Upload file
            sftp.put(local_path, remote_path)
            
            sftp.close()
            self._return_connection(client)
            
            logger.info("File uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> bool:
        """
        Download file from remote host
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading {remote_path} to {local_path}")
            
            client = self._get_connection()
            
            if not client:
                logger.error("Failed to establish SSH connection")
                return False
            
            # Use SFTP
            sftp = client.open_sftp()
            
            # Download file
            sftp.get(remote_path, local_path)
            
            sftp.close()
            self._return_connection(client)
            
            logger.info("File downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def check_connection(self) -> bool:
        """
        Check if SSH connection is healthy
        
        Returns:
            True if connection is healthy
        """
        try:
            success, _, _ = self.execute_command("echo test", silent=True)
            return success
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, int]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            'pooled_connections': len(self.connections)
        }
    
    def close_all_connections(self):
        """Close all connections in pool"""
        try:
            logger.info("Closing all SSH connections")
            
            for client in self.connections:
                try:
                    if hasattr(client, 'close'):
                        client.close()
                except:
                    pass
            
            self.connections.clear()
            self.connection_stats['active_connections'] = 0
            
            logger.info("All connections closed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    @contextmanager
    def connection_context(self):
        """Context manager for SSH connection"""
        client = self._get_connection()
        try:
            yield client
        finally:
            if client:
                self._return_connection(client)

def main():
    """Main function to demonstrate SSH manager"""
    try:
        logger.info("=" * 60)
        logger.info("SSH Manager Demo")
        logger.info("=" * 60)
        
        # Initialize SSH manager (example)
        ssh = SSHManager(
            host="example.com",
            username="ubuntu",
            port=22
        )
        
        # Check connection
        logger.info("Checking connection...")
        is_connected = ssh.check_connection()
        logger.info(f"Connection status: {is_connected}")
        
        # Execute command
        logger.info("\nExecuting command...")
        success, stdout, stderr = ssh.execute_command("uptime")
        if success:
            logger.info(f"Output: {stdout}")
        
        # Get stats
        stats = ssh.get_connection_stats()
        logger.info(f"\nConnection stats: {stats}")
        
        # Close connections
        ssh.close_all_connections()
        
        logger.info("=" * 60)
        logger.info("SSH Manager Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

