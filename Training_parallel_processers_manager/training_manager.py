#!/usr/bin/env python3
"""
Training Parallel Processors Manager
Advanced parallel processing system for AI model training including:
- Multi-GPU training coordination
- Distributed training across multiple nodes
- Training job scheduling and management
- Resource allocation and optimization
- Training progress monitoring
- Model checkpointing and recovery
"""

import queue
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import sqlite3
import os
import signal
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("Numpy not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available")

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingJob:
    """Training job structure"""
    job_id: str
    model_type: str
    dataset_path: str
    config: Dict[str, Any]
    priority: int
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float
    gpu_ids: List[int]
    node_id: str
    metadata: Dict[str, Any]

@dataclass
class ResourceInfo:
    """Resource information structure"""
    node_id: str
    cpu_count: int
    cpu_usage: float
    memory_total: float
    memory_used: float
    memory_available: float
    gpu_count: int
    gpu_info: List[Dict[str, Any]]
    disk_usage: float
    network_usage: float
    timestamp: datetime

@dataclass
class TrainingResult:
    """Training result structure"""
    job_id: str
    model_path: str
    metrics: Dict[str, float]
    training_time: float
    gpu_utilization: List[float]
    memory_usage: List[float]
    checkpoint_paths: List[str]
    logs: List[str]
    status: str
    error_message: Optional[str]

class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.resource_history = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        try:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Resource monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting resource monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        try:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            logger.info("Resource monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping resource monitoring: {e}")
    
    def _monitor_resources(self, interval: float):
        """Monitor resources in a separate thread"""
        try:
            while self.monitoring:
                resource_info = self.get_current_resources()
                self.resource_history.append(resource_info)
                
                # Keep only last 1000 records
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")
    
    def get_current_resources(self) -> ResourceInfo:
        """Get current resource information"""
        try:
            # CPU information
            if PSUTIL_AVAILABLE:
                cpu_count = psutil.cpu_count()
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory information
                memory = psutil.virtual_memory()
                memory_total = memory.total / (1024**3)  # GB
                memory_used = memory.used / (1024**3)  # GB
                memory_available = memory.available / (1024**3)  # GB
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
            else:
                cpu_count = 0
                cpu_usage = 0.0
                memory_total = 0.0
                memory_used = 0.0
                memory_available = 0.0
                disk_usage = 0.0
            
            # GPU information
            gpu_info = []
            gpu_count = 0
            
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    gpu_count = len(gpus)
                    
                    for gpu in gpus:
                        gpu_info.append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total': gpu.memoryTotal,
                            'memory_used': gpu.memoryUsed,
                            'memory_free': gpu.memoryFree,
                            'utilization': gpu.load * 100,
                            'temperature': gpu.temperature
                        })
                except Exception as e:
                    logger.warning(f"Error getting GPU info: {e}")
            
            # Network usage (simplified)
            network_usage = 0.0  # Would need to track network stats
            
            # Get node ID (cross-platform)
            try:
                if hasattr(os, 'uname'):
                    node_id = os.uname().nodename
                else:
                    # Windows
                    node_id = os.environ.get('COMPUTERNAME', 'unknown')
            except Exception:
                node_id = 'unknown'
            
            return ResourceInfo(
                node_id=node_id,
                cpu_count=cpu_count,
                cpu_usage=cpu_usage,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_available=memory_available,
                gpu_count=gpu_count,
                gpu_info=gpu_info,
                disk_usage=disk_usage,
                network_usage=network_usage,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting current resources: {e}")
            return ResourceInfo(
                node_id='unknown',
                cpu_count=0,
                cpu_usage=0,
                memory_total=0,
                memory_used=0,
                memory_available=0,
                gpu_count=0,
                gpu_info=[],
                disk_usage=0,
                network_usage=0,
                timestamp=datetime.now()
            )
    
    def get_available_gpus(self, min_memory: float = 4.0) -> List[int]:
        """Get available GPUs with sufficient memory"""
        try:
            available_gpus = []
            current_resources = self.get_current_resources()
            
            for gpu in current_resources.gpu_info:
                if gpu['memory_free'] >= min_memory * 1024:  # Convert GB to MB
                    available_gpus.append(gpu['id'])
            
            return available_gpus
            
        except Exception as e:
            logger.error(f"Error getting available GPUs: {e}")
            return []
    
    def get_resource_utilization(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get resource utilization over time"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_resources = [r for r in self.resource_history if r.timestamp >= cutoff_time]
            
            if not recent_resources:
                return {}
            
            utilization = {
                'cpu_usage': [r.cpu_usage for r in recent_resources],
                'memory_usage': [r.memory_used for r in recent_resources],
                'gpu_utilization': [],
                'timestamps': [r.timestamp for r in recent_resources]
            }
            
            # GPU utilization (average across all GPUs)
            for r in recent_resources:
                if r.gpu_info:
                    if NUMPY_AVAILABLE:
                        avg_gpu_util = np.mean([gpu['utilization'] for gpu in r.gpu_info])
                    else:
                        avg_gpu_util = sum([gpu['utilization'] for gpu in r.gpu_info]) / len(r.gpu_info) if r.gpu_info else 0.0
                    utilization['gpu_utilization'].append(avg_gpu_util)
                else:
                    utilization['gpu_utilization'].append(0)
            
            return utilization
            
        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            return {}

class TrainingScheduler:
    """Schedule and manage training jobs"""
    
    def __init__(self, max_concurrent_jobs: int = 4):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = queue.PriorityQueue()
        self.running_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        self.job_counter = 0
        self.scheduler_thread = None
        self.scheduling = False
    
    def start_scheduler(self):
        """Start the training scheduler"""
        try:
            self.scheduling = True
            self.scheduler_thread = threading.Thread(target=self._schedule_jobs)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            logger.info("Training scheduler started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    def stop_scheduler(self):
        """Stop the training scheduler"""
        try:
            self.scheduling = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            
            logger.info("Training scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def submit_job(self, model_type: str, dataset_path: str, config: Dict[str, Any], 
                   priority: int = 1) -> str:
        """Submit a training job"""
        try:
            self.job_counter += 1
            job_id = f"job_{self.job_counter}_{int(time.time())}"
            
            job = TrainingJob(
                job_id=job_id,
                model_type=model_type,
                dataset_path=dataset_path,
                config=config,
                priority=priority,
                status='pending',
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                progress=0.0,
                gpu_ids=[],
                node_id='',
                metadata={}
            )
            
            # Add to queue (lower priority number = higher priority)
            self.job_queue.put((priority, job))
            
            logger.info(f"Job {job_id} submitted with priority {priority}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return ""
    
    def _schedule_jobs(self):
        """Schedule jobs in a separate thread"""
        try:
            while self.scheduling:
                # Check if we can start more jobs
                if len(self.running_jobs) < self.max_concurrent_jobs and not self.job_queue.empty():
                    try:
                        priority, job = self.job_queue.get_nowait()
                        
                        # Start the job
                        self._start_job(job)
                        
                    except queue.Empty:
                        pass
                
                # Check running jobs
                self._check_running_jobs()
                
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in scheduler thread: {e}")
    
    def _start_job(self, job: TrainingJob):
        """Start a training job"""
        try:
            # Allocate resources
            resource_monitor = ResourceMonitor()
            available_gpus = resource_monitor.get_available_gpus()
            
            if not available_gpus:
                logger.warning(f"No available GPUs for job {job.job_id}")
                # Put job back in queue
                self.job_queue.put((job.priority, job))
                return
            
            # Assign GPUs
            job.gpu_ids = available_gpus[:job.config.get('gpu_count', 1)]
            job.node_id = resource_monitor.get_current_resources().node_id
            job.status = 'running'
            job.started_at = datetime.now()
            
            # Start training process
            training_process = self._create_training_process(job)
            
            self.running_jobs[job.job_id] = {
                'job': job,
                'process': training_process
            }
            
            logger.info(f"Started job {job.job_id} on GPUs {job.gpu_ids}")
            
        except Exception as e:
            logger.error(f"Error starting job {job.job_id}: {e}")
            job.status = 'failed'
            self.failed_jobs[job.job_id] = job
    
    def _create_training_process(self, job: TrainingJob) -> subprocess.Popen:
        """Create a training process"""
        try:
            # Create training script
            script_path = self._create_training_script(job)
            
            # Set environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, job.gpu_ids))
            env['JOB_ID'] = job.job_id
            
            # Start process
            process = subprocess.Popen(
                ['python', script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return process
            
        except Exception as e:
            logger.error(f"Error creating training process: {e}")
            return None
    
    def _create_training_script(self, job: TrainingJob) -> str:
        """Create a training script for the job"""
        try:
            script_content = f"""
import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        job_id = os.environ.get('JOB_ID', 'unknown')
        logger.info(f"Starting training for job {job_id}")
        
        # Load job configuration
        config = {json.dumps(job.config)}
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {{device}}")
        
        # Create dummy model for demonstration
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        ).to(device)
        
        # Create dummy data
        X = torch.randn(1000, 100).to(device)
        y = torch.randint(0, 10, (1000,)).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = config.get('epochs', 10)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {{epoch}}, Loss: {{loss.item():.4f}}")
        
        # Save model
        model_path = f"models/model_{{job_id}}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Training completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in training: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()
"""
            
            script_path = f"training_scripts/train_{job.job_id}.py"
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            return script_path
            
        except Exception as e:
            logger.error(f"Error creating training script: {e}")
            return ""
    
    def _check_running_jobs(self):
        """Check status of running jobs"""
        try:
            completed_jobs = []
            
            for job_id, job_info in self.running_jobs.items():
                job = job_info['job']
                process = job_info['process']
                
                if process and process.poll() is not None:
                    # Process completed
                    if process.returncode == 0:
                        job.status = 'completed'
                        job.completed_at = datetime.now()
                        job.progress = 100.0
                        self.completed_jobs[job_id] = job
                        logger.info(f"Job {job_id} completed successfully")
                    else:
                        job.status = 'failed'
                        job.completed_at = datetime.now()
                        self.failed_jobs[job_id] = job
                        logger.error(f"Job {job_id} failed with return code {process.returncode}")
                    
                    completed_jobs.append(job_id)
            
            # Remove completed jobs from running jobs
            for job_id in completed_jobs:
                del self.running_jobs[job_id]
            
        except Exception as e:
            logger.error(f"Error checking running jobs: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status"""
        try:
            # Check running jobs
            if job_id in self.running_jobs:
                return self.running_jobs[job_id]['job']
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            
            # Check failed jobs
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        try:
            if job_id in self.running_jobs:
                job_info = self.running_jobs[job_id]
                process = job_info['process']
                
                if process:
                    process.terminate()
                    process.wait(timeout=5)
                
                job = job_info['job']
                job.status = 'cancelled'
                job.completed_at = datetime.now()
                
                del self.running_jobs[job_id]
                
                logger.info(f"Job {job_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        try:
            return {
                'queue_size': self.job_queue.qsize(),
                'running_jobs': len(self.running_jobs),
                'completed_jobs': len(self.completed_jobs),
                'failed_jobs': len(self.failed_jobs),
                'max_concurrent_jobs': self.max_concurrent_jobs
            }
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {}

class DistributedTrainingManager:
    """Manage distributed training across multiple nodes"""
    
    def __init__(self, master_addr: str = 'localhost', master_port: str = '12355'):
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = 1
        self.rank = 0
        self.distributed = False
    
    def setup_distributed_training(self, world_size: int, rank: int):
        """Setup distributed training"""
        try:
            self.world_size = world_size
            self.rank = rank
            
            if world_size > 1:
                os.environ['MASTER_ADDR'] = self.master_addr
                os.environ['MASTER_PORT'] = self.master_port
                os.environ['WORLD_SIZE'] = str(world_size)
                os.environ['RANK'] = str(rank)
                
                dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
                self.distributed = True
                
                logger.info(f"Distributed training setup: rank {rank}/{world_size}")
            
        except Exception as e:
            logger.error(f"Error setting up distributed training: {e}")
    
    def cleanup_distributed_training(self):
        """Cleanup distributed training"""
        try:
            if self.distributed:
                dist.destroy_process_group()
                self.distributed = False
                logger.info("Distributed training cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up distributed training: {e}")
    
    def create_distributed_model(self, model: nn.Module) -> nn.Module:
        """Create distributed model"""
        try:
            if self.distributed:
                model = model.to(f'cuda:{self.rank}')
                model = DDP(model, device_ids=[self.rank])
                logger.info(f"Created distributed model on rank {self.rank}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating distributed model: {e}")
            return model
    
    def create_distributed_dataloader(self, dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create distributed dataloader"""
        try:
            if self.distributed:
                sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
                logger.info(f"Created distributed dataloader on rank {self.rank}")
            else:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Error creating distributed dataloader: {e}")
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class TrainingDatabase:
    """Database management for training jobs"""
    
    def __init__(self, db_path: str = "training_jobs.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize training database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                config TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                completed_at DATETIME,
                progress REAL NOT NULL,
                gpu_ids TEXT,
                node_id TEXT,
                metadata TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                model_path TEXT NOT NULL,
                metrics TEXT NOT NULL,
                training_time REAL NOT NULL,
                gpu_utilization TEXT,
                memory_usage TEXT,
                checkpoint_paths TEXT,
                logs TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES training_jobs (job_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                cpu_usage REAL NOT NULL,
                memory_usage REAL NOT NULL,
                gpu_utilization TEXT,
                timestamp DATETIME NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_job(self, job: TrainingJob) -> bool:
        """Save training job to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO training_jobs 
                (job_id, model_type, dataset_path, config, priority, status,
                 created_at, started_at, completed_at, progress, gpu_ids, node_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id,
                job.model_type,
                job.dataset_path,
                json.dumps(job.config),
                job.priority,
                job.status,
                job.created_at,
                job.started_at,
                job.completed_at,
                job.progress,
                json.dumps(job.gpu_ids),
                job.node_id,
                json.dumps(job.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving job: {e}")
            return False
    
    def save_result(self, result: TrainingResult) -> bool:
        """Save training result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_results 
                (job_id, model_path, metrics, training_time, gpu_utilization,
                 memory_usage, checkpoint_paths, logs, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.job_id,
                result.model_path,
                json.dumps(result.metrics),
                result.training_time,
                json.dumps(result.gpu_utilization),
                json.dumps(result.memory_usage),
                json.dumps(result.checkpoint_paths),
                json.dumps(result.logs),
                result.status,
                result.error_message
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False
    
    def save_resource_usage(self, resource_info: ResourceInfo) -> bool:
        """Save resource usage to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resource_usage 
                (node_id, cpu_usage, memory_usage, gpu_utilization, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                resource_info.node_id,
                resource_info.cpu_usage,
                resource_info.memory_used,
                json.dumps([gpu['utilization'] for gpu in resource_info.gpu_info]),
                resource_info.timestamp
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving resource usage: {e}")
            return False

class TrainingManager:
    """Main training parallel processors manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.resource_monitor = ResourceMonitor()
        self.training_scheduler = TrainingScheduler(
            max_concurrent_jobs=self.config.get('max_concurrent_jobs', 4)
        )
        self.distributed_manager = DistributedTrainingManager()
        self.database = TrainingDatabase()
        
        # Settings
        self.max_concurrent_jobs = self.config.get('max_concurrent_jobs', 4)
        self.resource_monitoring_interval = self.config.get('resource_monitoring_interval', 5.0)
        
        # Running state
        self.running = False
    
    def start(self):
        """Start the training manager"""
        try:
            self.running = True
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring(self.resource_monitoring_interval)
            
            # Start training scheduler
            self.training_scheduler.start_scheduler()
            
            logger.info("Training manager started")
            
        except Exception as e:
            logger.error(f"Error starting training manager: {e}")
    
    def stop(self):
        """Stop the training manager"""
        try:
            self.running = False
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Stop training scheduler
            self.training_scheduler.stop_scheduler()
            
            # Cleanup distributed training
            self.distributed_manager.cleanup_distributed_training()
            
            logger.info("Training manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping training manager: {e}")
    
    def submit_training_job(self, model_type: str, dataset_path: str, 
                           config: Dict[str, Any], priority: int = 1) -> str:
        """Submit a training job"""
        try:
            job_id = self.training_scheduler.submit_job(
                model_type, dataset_path, config, priority
            )
            
            if job_id:
                logger.info(f"Training job {job_id} submitted")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting training job: {e}")
            return ""
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status"""
        try:
            return self.training_scheduler.get_job_status(job_id)
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            return self.training_scheduler.cancel_job(job_id)
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            # Get resource information
            current_resources = self.resource_monitor.get_current_resources()
            
            # Get queue status
            queue_status = self.training_scheduler.get_queue_status()
            
            # Get resource utilization
            resource_utilization = self.resource_monitor.get_resource_utilization()
            
            return {
                'resources': asdict(current_resources),
                'queue_status': queue_status,
                'resource_utilization': resource_utilization,
                'running': self.running
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation"""
        try:
            current_resources = self.resource_monitor.get_current_resources()
            available_gpus = self.resource_monitor.get_available_gpus()
            
            optimization_suggestions = {
                'available_gpus': available_gpus,
                'cpu_utilization': current_resources.cpu_usage,
                'memory_utilization': current_resources.memory_used / current_resources.memory_total,
                'recommendations': []
            }
            
            # CPU recommendations
            if current_resources.cpu_usage > 80:
                optimization_suggestions['recommendations'].append(
                    "High CPU usage detected. Consider reducing concurrent jobs."
                )
            
            # Memory recommendations
            if current_resources.memory_used / current_resources.memory_total > 0.9:
                optimization_suggestions['recommendations'].append(
                    "High memory usage detected. Consider reducing batch sizes."
                )
            
            # GPU recommendations
            if not available_gpus:
                optimization_suggestions['recommendations'].append(
                    "No available GPUs. Consider waiting for current jobs to complete."
                )
            
            return optimization_suggestions
            
        except Exception as e:
            logger.error(f"Error optimizing resource allocation: {e}")
            return {}

def main():
    """Main function to demonstrate training parallel processors manager"""
    try:
        # Initialize training manager
        config = {
            'max_concurrent_jobs': 2,
            'resource_monitoring_interval': 5.0
        }
        
        manager = TrainingManager(config)
        
        # Start the manager
        manager.start()
        
        # Submit some training jobs
        logger.info("Submitting training jobs...")
        
        job_configs = [
            {
                'model_type': 'lstm',
                'dataset_path': 'data/lstm_dataset.csv',
                'config': {'epochs': 10, 'batch_size': 32, 'gpu_count': 1},
                'priority': 1
            },
            {
                'model_type': 'transformer',
                'dataset_path': 'data/transformer_dataset.csv',
                'config': {'epochs': 20, 'batch_size': 16, 'gpu_count': 2},
                'priority': 2
            },
            {
                'model_type': 'cnn',
                'dataset_path': 'data/cnn_dataset.csv',
                'config': {'epochs': 15, 'batch_size': 64, 'gpu_count': 1},
                'priority': 1
            }
        ]
        
        job_ids = []
        for job_config in job_configs:
            job_id = manager.submit_training_job(**job_config)
            if job_id:
                job_ids.append(job_id)
        
        # Monitor jobs
        logger.info("Monitoring training jobs...")
        for i in range(30):  # Monitor for 30 seconds
            time.sleep(1)
            
            # Check job status
            for job_id in job_ids:
                status = manager.get_job_status(job_id)
                if status:
                    logger.info(f"Job {job_id}: {status.status} ({status.progress:.1f}%)")
            
            # Get system status
            if i % 10 == 0:  # Every 10 seconds
                system_status = manager.get_system_status()
                logger.info(f"System Status: {system_status['queue_status']}")
        
        # Get optimization suggestions
        optimization = manager.optimize_resource_allocation()
        logger.info(f"Optimization suggestions: {optimization['recommendations']}")
        
        # Stop the manager
        manager.stop()
        
        logger.info("Training parallel processors manager test completed!")
        
    except Exception as e:
        logger.error(f"Error in main training manager function: {e}")

if __name__ == "__main__":
    main()
