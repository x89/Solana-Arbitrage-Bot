#!/usr/bin/env python3
"""
Ubuntu Cloud Service Handler
Advanced cloud service integration system including:
- Cloud instance management
- Remote task execution
- Cloud resource monitoring
- Distributed computing coordination
- Cloud storage management
- Cost optimization
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import subprocess
import os
import time
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logging.warning("paramiko not available. SSH features will be limited.")

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. AWS features will be limited.")

try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    logging.warning("asyncio/aiohttp not available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CloudInstance:
    """Cloud instance structure"""
    instance_id: str
    instance_type: str
    region: str
    state: str
    public_ip: str
    private_ip: str
    launch_time: datetime
    tags: Dict[str, str]
    cost_per_hour: float
    metadata: Dict[str, Any]

@dataclass
class CloudTask:
    """Cloud task structure"""
    task_id: str
    instance_id: str
    command: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class CloudResource:
    """Cloud resource structure"""
    resource_type: str
    resource_id: str
    region: str
    status: str
    cost_per_hour: float
    utilization: float
    metadata: Dict[str, Any]

class AWSManager:
    """AWS cloud service manager"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.ec2_client = None
        self.s3_client = None
        self.cloudwatch_client = None
        self.instances = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients"""
        try:
            if BOTO3_AVAILABLE:
                self.ec2_client = boto3.client('ec2', region_name=self.region)
                self.s3_client = boto3.client('s3', region_name=self.region)
                self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
                
                logger.info(f"AWS clients initialized for region {self.region}")
            else:
                logger.warning("AWS clients not initialized (boto3 not available)")
            
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {e}")
    
    def launch_instance(self, instance_type: str = 't3.micro', 
                       ami_id: str = 'ami-0c02fb55956c7d316',  # Ubuntu 20.04 LTS
                       key_name: str = None, security_group: str = None) -> Optional[str]:
        """Launch a new EC2 instance"""
        try:
            if not BOTO3_AVAILABLE or not self.ec2_client:
                logger.error("AWS EC2 client not available")
                return None
            
            # Create security group if not provided
            if not security_group:
                security_group = self._create_security_group()
            
            # Launch instance
            response = self.ec2_client.run_instances(
                ImageId=ami_id,
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_type,
                KeyName=key_name,
                SecurityGroupIds=[security_group],
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'trading-bot-{int(time.time())}'},
                            {'Key': 'Purpose', 'Value': 'AI-Trading'},
                            {'Key': 'CreatedBy', 'Value': 'TradingBotSystem'}
                        ]
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get instance information
            instance_info = self.get_instance_info(instance_id)
            if instance_info:
                self.instances[instance_id] = instance_info
            
            logger.info(f"Launched instance {instance_id} of type {instance_type}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Error launching instance: {e}")
            return None
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an EC2 instance"""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            
            # Wait for instance to be terminated
            waiter = self.ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])
            
            if instance_id in self.instances:
                del self.instances[instance_id]
            
            logger.info(f"Terminated instance {instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error terminating instance {instance_id}: {e}")
            return False
    
    def get_instance_info(self, instance_id: str) -> Optional[CloudInstance]:
        """Get instance information"""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            
            if not response['Reservations']:
                return None
            
            instance = response['Reservations'][0]['Instances'][0]
            
            # Get public IP
            public_ip = instance.get('PublicIpAddress', '')
            private_ip = instance.get('PrivateIpAddress', '')
            
            # Get tags
            tags = {}
            for tag in instance.get('Tags', []):
                tags[tag['Key']] = tag['Value']
            
            # Calculate cost (simplified)
            cost_per_hour = self._get_instance_cost(instance['InstanceType'])
            
            return CloudInstance(
                instance_id=instance_id,
                instance_type=instance['InstanceType'],
                region=self.region,
                state=instance['State']['Name'],
                public_ip=public_ip,
                private_ip=private_ip,
                launch_time=instance['LaunchTime'],
                tags=tags,
                cost_per_hour=cost_per_hour,
                metadata={'response': instance}
            )
            
        except Exception as e:
            logger.error(f"Error getting instance info for {instance_id}: {e}")
            return None
    
    def list_instances(self) -> List[CloudInstance]:
        """List all instances"""
        try:
            response = self.ec2_client.describe_instances()
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    instance_info = self.get_instance_info(instance_id)
                    if instance_info:
                        instances.append(instance_info)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error listing instances: {e}")
            return []
    
    def _create_security_group(self) -> str:
        """Create security group for trading bot"""
        try:
            group_name = f'trading-bot-sg-{int(time.time())}'
            
            response = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description='Security group for AI Trading Bot'
            )
            
            group_id = response['GroupId']
            
            # Add SSH access
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            # Add HTTP access
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 80,
                        'ToPort': 80,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            # Add HTTPS access
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            logger.info(f"Created security group {group_id}")
            
            return group_id
            
        except Exception as e:
            logger.error(f"Error creating security group: {e}")
            return ""
    
    def _get_instance_cost(self, instance_type: str) -> float:
        """Get instance cost per hour (simplified)"""
        try:
            # Simplified cost mapping (in USD per hour)
            cost_map = {
                't3.micro': 0.0104,
                't3.small': 0.0208,
                't3.medium': 0.0416,
                't3.large': 0.0832,
                't3.xlarge': 0.1664,
                't3.2xlarge': 0.3328,
                'm5.large': 0.096,
                'm5.xlarge': 0.192,
                'm5.2xlarge': 0.384,
                'm5.4xlarge': 0.768,
                'c5.large': 0.085,
                'c5.xlarge': 0.17,
                'c5.2xlarge': 0.34,
                'c5.4xlarge': 0.68,
                'p3.2xlarge': 3.06,
                'p3.8xlarge': 12.24,
                'p3.16xlarge': 24.48
            }
            
            return cost_map.get(instance_type, 0.1)  # Default cost
            
        except Exception as e:
            logger.error(f"Error getting instance cost: {e}")
            return 0.1
    
    def upload_to_s3(self, local_file: str, bucket: str, s3_key: str) -> bool:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_file, bucket, s3_key)
            logger.info(f"Uploaded {local_file} to s3://{bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False
    
    def download_from_s3(self, bucket: str, s3_key: str, local_file: str) -> bool:
        """Download file from S3"""
        try:
            self.s3_client.download_file(bucket, s3_key, local_file)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            return False

class SSHManager:
    """SSH connection manager for remote execution"""
    
    def __init__(self):
        self.connections = {}
        self.ssh_key_path = None
    
    def set_ssh_key(self, key_path: str):
        """Set SSH key path"""
        self.ssh_key_path = key_path
    
    def connect(self, host: str, username: str = 'ubuntu', port: int = 22) -> bool:
        """Connect to remote host via SSH"""
        try:
            if not PARAMIKO_AVAILABLE:
                logger.error("paramiko not available. Cannot establish SSH connection.")
                return False
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if self.ssh_key_path:
                ssh.connect(host, username=username, key_filename=self.ssh_key_path, port=port)
            else:
                ssh.connect(host, username=username, port=port)
            
            self.connections[host] = ssh
            
            logger.info(f"Connected to {host} via SSH")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {host}: {e}")
            return False
    
    def disconnect(self, host: str):
        """Disconnect from remote host"""
        try:
            if host in self.connections:
                self.connections[host].close()
                del self.connections[host]
                logger.info(f"Disconnected from {host}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {host}: {e}")
    
    def execute_command(self, host: str, command: str) -> Tuple[int, str, str]:
        """Execute command on remote host"""
        try:
            if host not in self.connections:
                logger.error(f"No connection to {host}")
                return -1, "", "No connection"
            
            ssh = self.connections[host]
            stdin, stdout, stderr = ssh.exec_command(command)
            
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            return exit_code, output, error
            
        except Exception as e:
            logger.error(f"Error executing command on {host}: {e}")
            return -1, "", str(e)
    
    def upload_file(self, host: str, local_path: str, remote_path: str) -> bool:
        """Upload file to remote host"""
        try:
            if host not in self.connections:
                logger.error(f"No connection to {host}")
                return False
            
            ssh = self.connections[host]
            sftp = ssh.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            
            logger.info(f"Uploaded {local_path} to {host}:{remote_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file to {host}: {e}")
            return False
    
    def download_file(self, host: str, remote_path: str, local_path: str) -> bool:
        """Download file from remote host"""
        try:
            if host not in self.connections:
                logger.error(f"No connection to {host}")
                return False
            
            ssh = self.connections[host]
            sftp = ssh.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            
            logger.info(f"Downloaded {host}:{remote_path} to {local_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from {host}: {e}")
            return False

class CloudTaskManager:
    """Manage cloud tasks and execution"""
    
    def __init__(self):
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.task_counter = 0
        self.ssh_manager = SSHManager()
        self.running = False
        self.task_thread = None
    
    def start(self):
        """Start task manager"""
        try:
            self.running = True
            self.task_thread = threading.Thread(target=self._process_tasks)
            self.task_thread.daemon = True
            self.task_thread.start()
            
            logger.info("Cloud task manager started")
            
        except Exception as e:
            logger.error(f"Error starting task manager: {e}")
    
    def stop(self):
        """Stop task manager"""
        try:
            self.running = False
            if self.task_thread:
                self.task_thread.join(timeout=5)
            
            logger.info("Cloud task manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping task manager: {e}")
    
    def submit_task(self, instance_id: str, command: str, metadata: Dict[str, Any] = None) -> str:
        """Submit a task for execution"""
        try:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time())}"
            
            task = CloudTask(
                task_id=task_id,
                instance_id=instance_id,
                command=command,
                status='pending',
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                result=None,
                error_message=None,
                metadata=metadata or {}
            )
            
            self.tasks[task_id] = task
            self.task_queue.put(task)
            
            logger.info(f"Submitted task {task_id} for instance {instance_id}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return ""
    
    def _process_tasks(self):
        """Process tasks in a separate thread"""
        try:
            while self.running:
                try:
                    task = self.task_queue.get(timeout=1)
                    self._execute_task(task)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                
        except Exception as e:
            logger.error(f"Error in task processing thread: {e}")
    
    def _execute_task(self, task: CloudTask):
        """Execute a task on remote instance"""
        try:
            # Get instance information
            aws_manager = AWSManager()
            instance_info = aws_manager.get_instance_info(task.instance_id)
            
            if not instance_info or not instance_info.public_ip:
                task.status = 'failed'
                task.error_message = "Instance not found or no public IP"
                task.completed_at = datetime.now()
                return
            
            # Connect to instance
            host = instance_info.public_ip
            if not self.ssh_manager.connect(host):
                task.status = 'failed'
                task.error_message = "Failed to connect to instance"
                task.completed_at = datetime.now()
                return
            
            # Update task status
            task.status = 'running'
            task.started_at = datetime.now()
            
            # Execute command
            exit_code, output, error = self.ssh_manager.execute_command(host, task.command)
            
            # Update task result
            task.completed_at = datetime.now()
            
            if exit_code == 0:
                task.status = 'completed'
                task.result = output
            else:
                task.status = 'failed'
                task.error_message = error
            
            # Disconnect
            self.ssh_manager.disconnect(host)
            
            logger.info(f"Task {task.task_id} completed with status {task.status}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[CloudTask]:
        """Get task status"""
        try:
            return self.tasks.get(task_id)
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    def list_tasks(self) -> List[CloudTask]:
        """List all tasks"""
        try:
            return list(self.tasks.values())
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []

class CloudResourceMonitor:
    """Monitor cloud resources and costs"""
    
    def __init__(self):
        self.resource_history = []
        self.cost_history = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 300):  # 5 minutes
        """Start resource monitoring"""
        try:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Cloud resource monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting resource monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        try:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            logger.info("Cloud resource monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping resource monitoring: {e}")
    
    def _monitor_resources(self, interval: float):
        """Monitor resources in a separate thread"""
        try:
            while self.monitoring:
                # Monitor AWS resources
                aws_manager = AWSManager()
                instances = aws_manager.list_instances()
                
                # Calculate total cost
                total_cost = sum(instance.cost_per_hour for instance in instances)
                
                # Record resource usage
                resource_data = {
                    'timestamp': datetime.now(),
                    'instances': len(instances),
                    'total_cost_per_hour': total_cost,
                    'running_instances': len([i for i in instances if i.state == 'running'])
                }
                
                self.resource_history.append(resource_data)
                
                # Keep only last 1000 records
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")
    
    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_resources = [r for r in self.resource_history if r['timestamp'] >= cutoff_time]
            
            if not recent_resources:
                return {'total_cost': 0, 'avg_cost_per_hour': 0, 'instances': 0}
            
            total_cost = sum(r['total_cost_per_hour'] for r in recent_resources) * (hours / len(recent_resources))
            avg_cost_per_hour = sum(r['total_cost_per_hour'] for r in recent_resources) / len(recent_resources)
            avg_instances = sum(r['instances'] for r in recent_resources) / len(recent_resources)
            
            return {
                'total_cost': total_cost,
                'avg_cost_per_hour': avg_cost_per_hour,
                'avg_instances': avg_instances,
                'hours_analyzed': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {}

class CloudDatabase:
    """Database management for cloud services"""
    
    def __init__(self, db_path: str = "cloud_services.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize cloud database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cloud_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT UNIQUE NOT NULL,
                instance_type TEXT NOT NULL,
                region TEXT NOT NULL,
                state TEXT NOT NULL,
                public_ip TEXT,
                private_ip TEXT,
                launch_time DATETIME NOT NULL,
                cost_per_hour REAL NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cloud_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                instance_id TEXT NOT NULL,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                completed_at DATETIME,
                result TEXT,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (instance_id) REFERENCES cloud_instances (instance_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cloud_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                region TEXT NOT NULL,
                status TEXT NOT NULL,
                cost_per_hour REAL NOT NULL,
                utilization REAL NOT NULL,
                metadata TEXT,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_instance(self, instance: CloudInstance) -> bool:
        """Save cloud instance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cloud_instances 
                (instance_id, instance_type, region, state, public_ip, private_ip,
                 launch_time, cost_per_hour, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                instance.instance_id,
                instance.instance_type,
                instance.region,
                instance.state,
                instance.public_ip,
                instance.private_ip,
                instance.launch_time,
                instance.cost_per_hour,
                json.dumps(instance.tags),
                json.dumps(instance.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving instance: {e}")
            return False
    
    def save_task(self, task: CloudTask) -> bool:
        """Save cloud task to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cloud_tasks 
                (task_id, instance_id, command, status, created_at, started_at,
                 completed_at, result, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.instance_id,
                task.command,
                task.status,
                task.created_at,
                task.started_at,
                task.completed_at,
                task.result,
                task.error_message,
                json.dumps(task.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving task: {e}")
            return False

class CloudManager:
    """Main cloud service handler"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.aws_manager = AWSManager(self.config.get('aws_region', 'us-east-1'))
        self.ssh_manager = SSHManager()
        self.task_manager = CloudTaskManager()
        self.resource_monitor = CloudResourceMonitor()
        self.database = CloudDatabase()
        
        # Settings
        self.ssh_key_path = self.config.get('ssh_key_path')
        self.max_instances = self.config.get('max_instances', 5)
        self.auto_terminate_idle_minutes = self.config.get('auto_terminate_idle_minutes', 60)
        
        # Running state
        self.running = False
    
    def start(self):
        """Start cloud manager"""
        try:
            self.running = True
            
            # Set SSH key if provided
            if self.ssh_key_path:
                self.ssh_manager.set_ssh_key(self.ssh_key_path)
            
            # Start task manager
            self.task_manager.start()
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            logger.info("Cloud manager started")
            
        except Exception as e:
            logger.error(f"Error starting cloud manager: {e}")
    
    def stop(self):
        """Stop cloud manager"""
        try:
            self.running = False
            
            # Stop task manager
            self.task_manager.stop()
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            logger.info("Cloud manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping cloud manager: {e}")
    
    def launch_trading_instance(self, instance_type: str = 't3.medium') -> Optional[str]:
        """Launch a trading bot instance"""
        try:
            # Check instance limit
            instances = self.aws_manager.list_instances()
            running_instances = [i for i in instances if i.state == 'running']
            
            if len(running_instances) >= self.max_instances:
                logger.warning(f"Instance limit reached ({self.max_instances})")
                return None
            
            # Launch instance
            instance_id = self.aws_manager.launch_instance(instance_type)
            
            if instance_id:
                # Save to database
                instance_info = self.aws_manager.get_instance_info(instance_id)
                if instance_info:
                    self.database.save_instance(instance_info)
                
                logger.info(f"Launched trading instance {instance_id}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Error launching trading instance: {e}")
            return None
    
    def setup_trading_environment(self, instance_id: str) -> bool:
        """Setup trading environment on instance"""
        try:
            # Get instance information
            instance_info = self.aws_manager.get_instance_info(instance_id)
            
            if not instance_info or not instance_info.public_ip:
                logger.error(f"Instance {instance_id} not found or no public IP")
                return False
            
            # Commands to setup trading environment
            setup_commands = [
                "sudo apt-get update",
                "sudo apt-get install -y python3 python3-pip git",
                "pip3 install torch torchvision torchaudio",
                "pip3 install pandas numpy scikit-learn",
                "pip3 install boto3 paramiko",
                "git clone https://github.com/your-repo/trading-bot.git /home/ubuntu/trading-bot",
                "cd /home/ubuntu/trading-bot && pip3 install -r requirements.txt"
            ]
            
            # Execute setup commands
            for command in setup_commands:
                task_id = self.task_manager.submit_task(instance_id, command)
                
                # Wait for task completion
                while True:
                    task = self.task_manager.get_task_status(task_id)
                    if task and task.status in ['completed', 'failed']:
                        if task.status == 'failed':
                            logger.error(f"Setup command failed: {task.error_message}")
                            return False
                        break
                    time.sleep(5)
            
            logger.info(f"Trading environment setup completed on {instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up trading environment: {e}")
            return False
    
    def run_trading_task(self, instance_id: str, task_type: str, parameters: Dict[str, Any]) -> str:
        """Run a trading task on instance"""
        try:
            # Construct command based on task type
            if task_type == 'data_collection':
                command = f"cd /home/ubuntu/trading-bot && python3 data_collector.py --symbol {parameters.get('symbol', 'SOLUSDT')}"
            elif task_type == 'model_training':
                command = f"cd /home/ubuntu/trading-bot && python3 train_model.py --model {parameters.get('model', 'lstm')} --epochs {parameters.get('epochs', 10)}"
            elif task_type == 'backtesting':
                command = f"cd /home/ubuntu/trading-bot && python3 backtest.py --strategy {parameters.get('strategy', 'ma_crossover')}"
            else:
                command = f"cd /home/ubuntu/trading-bot && python3 {task_type}.py"
            
            # Submit task
            task_id = self.task_manager.submit_task(instance_id, command, parameters)
            
            logger.info(f"Submitted {task_type} task {task_id} to instance {instance_id}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error running trading task: {e}")
            return ""
    
    def get_instance_status(self, instance_id: str) -> Optional[CloudInstance]:
        """Get instance status"""
        try:
            return self.aws_manager.get_instance_info(instance_id)
            
        except Exception as e:
            logger.error(f"Error getting instance status: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[CloudTask]:
        """Get task status"""
        try:
            return self.task_manager.get_task_status(task_id)
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate instance"""
        try:
            success = self.aws_manager.terminate_instance(instance_id)
            
            if success:
                logger.info(f"Terminated instance {instance_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error terminating instance: {e}")
            return False
    
    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary"""
        try:
            return self.resource_monitor.get_cost_summary(hours)
            
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {}
    
    def optimize_costs(self) -> Dict[str, Any]:
        """Optimize cloud costs"""
        try:
            instances = self.aws_manager.list_instances()
            running_instances = [i for i in instances if i.state == 'running']
            
            optimization_suggestions = {
                'total_instances': len(running_instances),
                'total_cost_per_hour': sum(i.cost_per_hour for i in running_instances),
                'suggestions': []
            }
            
            # Check for idle instances
            for instance in running_instances:
                # This would check if instance is actually being used
                # For now, just suggest based on instance type
                if instance.instance_type.startswith('p3'):  # GPU instances
                    optimization_suggestions['suggestions'].append(
                        f"Consider terminating GPU instance {instance.instance_id} if not actively training"
                    )
                
                if instance.cost_per_hour > 1.0:  # Expensive instances
                    optimization_suggestions['suggestions'].append(
                        f"Consider downsizing expensive instance {instance.instance_id} ({instance.instance_type})"
                    )
            
            return optimization_suggestions
            
        except Exception as e:
            logger.error(f"Error optimizing costs: {e}")
            return {}

def main():
    """Main function to demonstrate cloud service handler"""
    try:
        # Initialize cloud manager
        config = {
            'aws_region': 'us-east-1',
            'max_instances': 3,
            'auto_terminate_idle_minutes': 60
        }
        
        manager = CloudManager(config)
        
        # Start the manager
        manager.start()
        
        # Launch a trading instance
        logger.info("Launching trading instance...")
        instance_id = manager.launch_trading_instance('t3.medium')
        
        if instance_id:
            logger.info(f"Launched instance {instance_id}")
            
            # Wait for instance to be ready
            time.sleep(30)
            
            # Setup trading environment
            logger.info("Setting up trading environment...")
            setup_success = manager.setup_trading_environment(instance_id)
            
            if setup_success:
                # Run a trading task
                logger.info("Running trading task...")
                task_id = manager.run_trading_task(
                    instance_id, 
                    'data_collection', 
                    {'symbol': 'SOLUSDT'}
                )
                
                if task_id:
                    # Monitor task
                    logger.info(f"Monitoring task {task_id}...")
                    for i in range(30):  # Monitor for 30 seconds
                        time.sleep(1)
                        task = manager.get_task_status(task_id)
                        if task:
                            logger.info(f"Task status: {task.status}")
                            if task.status in ['completed', 'failed']:
                                break
                
                # Get cost summary
                cost_summary = manager.get_cost_summary()
                logger.info(f"Cost summary: {cost_summary}")
                
                # Get optimization suggestions
                optimization = manager.optimize_costs()
                logger.info(f"Optimization suggestions: {optimization['suggestions']}")
            
            # Terminate instance
            logger.info("Terminating instance...")
            manager.terminate_instance(instance_id)
        
        # Stop the manager
        manager.stop()
        
        logger.info("Cloud service handler test completed!")
        
    except Exception as e:
        logger.error(f"Error in main cloud handler function: {e}")

if __name__ == "__main__":
    main()
