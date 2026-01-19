#!/usr/bin/env python3
"""
GPU Usage Processors
Advanced GPU resource management system for AI trading bot including:
- GPU monitoring and utilization tracking
- Dynamic GPU allocation
- Memory management
- Performance optimization
- Multi-GPU support
- GPU task scheduling
"""

import time
import logging
import threading
import queue
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import subprocess
import os
import sys
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU features will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available. GPU monitoring will be limited.")

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    PATTERN_DETECTION = "pattern_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

@dataclass
class GPUTask:
    """GPU task structure"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    model_name: str
    input_data: Any
    expected_memory: int  # MB
    expected_duration: float  # seconds
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    gpu_id: Optional[int] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None

@dataclass
class GPUInfo:
    """GPU information structure"""
    gpu_id: int
    name: str
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    utilization: float  # percentage
    temperature: float  # celsius
    power_usage: float  # watts
    driver_version: str
    cuda_version: str
    is_available: bool

@dataclass
class GPUPerformance:
    """GPU performance metrics"""
    gpu_id: int
    timestamp: datetime
    utilization: float
    memory_usage: float
    temperature: float
    power_usage: float
    throughput: float  # tasks per second
    latency: float  # average task latency

class GPUMonitor:
    """GPU monitoring and information gathering"""
    
    def __init__(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_count = 0
        self.gpu_info = {}
        self.performance_history = []
        
        if self.gpu_count > 0:
            self._initialize_gpu_info()
        else:
            logger.info("No GPUs detected. Running in CPU-only mode.")
    
    def _get_fallback_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get GPU info when torch/GUtil not available"""
        return GPUInfo(
            gpu_id=gpu_id,
            name="Unknown",
            memory_total=0,
            memory_used=0,
            memory_free=0,
            utilization=0.0,
            temperature=0.0,
            power_usage=0.0,
            driver_version="Unknown",
            cuda_version="Unknown",
            is_available=False
        )
    
    def _initialize_gpu_info(self):
        """Initialize GPU information"""
        try:
            for i in range(self.gpu_count):
                gpu_info = self._get_gpu_info(i)
                self.gpu_info[i] = gpu_info
                
            logger.info(f"Initialized {self.gpu_count} GPU(s)")
            
        except Exception as e:
            logger.error(f"Error initializing GPU info: {e}")
    
    def _get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get detailed GPU information"""
        try:
            # Get GPU properties
            if TORCH_AVAILABLE:
                props = torch.cuda.get_device_properties(gpu_id)
            else:
                return self._get_fallback_gpu_info(gpu_id)
            
            # Get current GPU stats
            gpu_stats = None
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    gpu_stats = gpus[gpu_id] if gpu_id < len(gpus) else None
                except Exception:
                    pass
            
            if gpu_stats:
                memory_used = int(gpu_stats.memoryUsed)
                memory_total = int(gpu_stats.memoryTotal)
                memory_free = memory_total - memory_used
                utilization = gpu_stats.load * 100
                temperature = gpu_stats.temperature
                power_usage = getattr(gpu_stats, 'powerDraw', 0)
            else:
                # Fallback to torch
                if TORCH_AVAILABLE:
                    memory_used = torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)
                    memory_total = props.total_memory // (1024 * 1024)
                else:
                    memory_used = 0
                    memory_total = 0
                memory_free = memory_total - memory_used
                utilization = 0.0
                temperature = 0.0
                power_usage = 0.0
            
            # Get driver and CUDA versions
            if TORCH_AVAILABLE:
                try:
                    # Check if method exists
                    if hasattr(torch.cuda, 'get_driver_version'):
                        driver_version = torch.cuda.get_driver_version()
                    else:
                        driver_version = "Unknown"
                    cuda_version = getattr(torch.version, 'cuda', "Unknown")
                except Exception:
                    driver_version = "Unknown"
                    cuda_version = "Unknown"
            else:
                driver_version = "Unknown"
                cuda_version = "Unknown"
            
            return GPUInfo(
                gpu_id=gpu_id,
                name=props.name,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                driver_version=f"{driver_version[0]}.{driver_version[1]}",
                cuda_version=cuda_version,
                is_available=True
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU info for GPU {gpu_id}: {e}")
            return GPUInfo(
                gpu_id=gpu_id,
                name="Unknown",
                memory_total=0,
                memory_used=0,
                memory_free=0,
                utilization=0.0,
                temperature=0.0,
                power_usage=0.0,
                driver_version="Unknown",
                cuda_version="Unknown",
                is_available=False
            )
    
    def get_all_gpu_info(self) -> Dict[int, GPUInfo]:
        """Get information for all GPUs"""
        try:
            for gpu_id in range(self.gpu_count):
                self.gpu_info[gpu_id] = self._get_gpu_info(gpu_id)
            
            return self.gpu_info.copy()
            
        except Exception as e:
            logger.error(f"Error getting all GPU info: {e}")
            return {}
    
    def get_gpu_performance(self, gpu_id: int) -> GPUPerformance:
        """Get current GPU performance metrics"""
        try:
            gpu_info = self.gpu_info.get(gpu_id)
            if not gpu_info:
                gpu_info = self._get_gpu_info(gpu_id)
            
            # Calculate throughput and latency from history
            throughput = self._calculate_throughput(gpu_id)
            latency = self._calculate_latency(gpu_id)
            
            performance = GPUPerformance(
                gpu_id=gpu_id,
                timestamp=datetime.now(),
                utilization=gpu_info.utilization,
                memory_usage=gpu_info.memory_used / gpu_info.memory_total * 100,
                temperature=gpu_info.temperature,
                power_usage=gpu_info.power_usage,
                throughput=throughput,
                latency=latency
            )
            
            # Store performance history
            self.performance_history.append(performance)
            
            # Keep only recent history (last 1000 entries)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting GPU performance: {e}")
            return GPUPerformance(
                gpu_id=gpu_id,
                timestamp=datetime.now(),
                utilization=0.0,
                memory_usage=0.0,
                temperature=0.0,
                power_usage=0.0,
                throughput=0.0,
                latency=0.0
            )
    
    def _calculate_throughput(self, gpu_id: int) -> float:
        """Calculate GPU throughput (tasks per second)"""
        try:
            # This would be implemented based on task completion history
            # For now, return a placeholder value
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
            return 0.0
    
    def _calculate_latency(self, gpu_id: int) -> float:
        """Calculate average task latency"""
        try:
            # This would be implemented based on task completion history
            # For now, return a placeholder value
            return 0.1
            
        except Exception as e:
            logger.error(f"Error calculating latency: {e}")
            return 0.0
    
    def is_gpu_available(self, gpu_id: int, required_memory: int = 0) -> bool:
        """Check if GPU is available for tasks"""
        try:
            gpu_info = self.gpu_info.get(gpu_id)
            if not gpu_info or not gpu_info.is_available:
                return False
            
            # Check memory availability
            if required_memory > 0:
                return gpu_info.memory_free >= required_memory
            
            # Check utilization
            return gpu_info.utilization < 90.0  # Less than 90% utilization
            
        except Exception as e:
            logger.error(f"Error checking GPU availability: {e}")
            return False
    
    def get_best_gpu(self, required_memory: int = 0) -> Optional[int]:
        """Get the best available GPU for a task"""
        try:
            best_gpu = None
            best_score = -1
            
            for gpu_id in range(self.gpu_count):
                if self.is_gpu_available(gpu_id, required_memory):
                    gpu_info = self.gpu_info[gpu_id]
                    
                    # Score based on free memory and low utilization
                    memory_score = gpu_info.memory_free / gpu_info.memory_total
                    utilization_score = 1.0 - (gpu_info.utilization / 100.0)
                    
                    score = memory_score * 0.6 + utilization_score * 0.4
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu_id
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error finding best GPU: {e}")
            return None

class GPUTaskScheduler:
    """GPU task scheduling and management"""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Task execution threads
        self.executor = ThreadPoolExecutor(max_workers=self.gpu_monitor.gpu_count)
        self.running = False
        
        # Task statistics
        self.task_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_completion_time': 0.0,
            'throughput': 0.0
        }
    
    def start_scheduler(self):
        """Start the GPU task scheduler"""
        self.running = True
        
        # Start task processing threads
        for gpu_id in range(self.gpu_monitor.gpu_count):
            thread = threading.Thread(target=self._process_tasks, args=(gpu_id,))
            thread.daemon = True
            thread.start()
        
        logger.info("GPU task scheduler started")
    
    def stop_scheduler(self):
        """Stop the GPU task scheduler"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("GPU task scheduler stopped")
    
    def submit_task(self, task: GPUTask) -> bool:
        """Submit a task for execution"""
        try:
            # Calculate priority score (higher number = higher priority)
            priority_score = task.priority.value * 1000 - self.task_stats['total_tasks']
            
            # Add to queue
            self.task_queue.put((priority_score, task))
            
            self.task_stats['total_tasks'] += 1
            
            logger.info(f"Task {task.task_id} submitted with priority {task.priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return False
    
    def _process_tasks(self, gpu_id: int):
        """Process tasks for a specific GPU"""
        while self.running:
            try:
                # Get next task
                if not self.task_queue.empty():
                    priority_score, task = self.task_queue.get()
                    
                    # Check if GPU is available
                    if self.gpu_monitor.is_gpu_available(gpu_id, task.expected_memory):
                        # Execute task
                        self._execute_task(task, gpu_id)
                    else:
                        # Put task back in queue
                        self.task_queue.put((priority_score, task))
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing tasks for GPU {gpu_id}: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: GPUTask, gpu_id: int):
        """Execute a task on a specific GPU"""
        try:
            # Update task status
            task.status = "running"
            task.started_at = datetime.now()
            task.gpu_id = gpu_id
            
            self.running_tasks[task.task_id] = task
            
            # Set CUDA device
            if TORCH_AVAILABLE:
                torch.cuda.set_device(gpu_id)
            
            # Execute task based on type
            if task.task_type == TaskType.TRAINING:
                result = self._execute_training_task(task)
            elif task.task_type == TaskType.INFERENCE:
                result = self._execute_inference_task(task)
            elif task.task_type == TaskType.PATTERN_DETECTION:
                result = self._execute_pattern_detection_task(task)
            elif task.task_type == TaskType.SENTIMENT_ANALYSIS:
                result = self._execute_sentiment_analysis_task(task)
            else:
                result = self._execute_generic_task(task)
            
            # Update task completion
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.running_tasks[task.task_id]
            
            # Update statistics
            self._update_task_stats(task)
            
            logger.info(f"Task {task.task_id} completed on GPU {gpu_id}")
            
        except Exception as e:
            # Handle task failure
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error = str(e)
            
            self.failed_tasks[task.task_id] = task
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            logger.error(f"Task {task.task_id} failed on GPU {gpu_id}: {e}")
    
    def _execute_training_task(self, task: GPUTask) -> Any:
        """Execute training task"""
        try:
            # This would implement actual model training
            # For now, simulate training
            time.sleep(task.expected_duration)
            
            return {
                'model_path': f'models/{task.model_name}_trained.pth',
                'accuracy': 0.95,
                'loss': 0.05,
                'epochs': 100
            }
            
        except Exception as e:
            logger.error(f"Error executing training task: {e}")
            raise
    
    def _execute_inference_task(self, task: GPUTask) -> Any:
        """Execute inference task"""
        try:
            # This would implement actual model inference
            # For now, simulate inference
            time.sleep(task.expected_duration)
            
            return {
                'predictions': [0.8, 0.2, 0.9],
                'confidence': 0.85,
                'processing_time': task.expected_duration
            }
            
        except Exception as e:
            logger.error(f"Error executing inference task: {e}")
            raise
    
    def _execute_pattern_detection_task(self, task: GPUTask) -> Any:
        """Execute pattern detection task"""
        try:
            # This would implement actual pattern detection
            # For now, simulate pattern detection
            time.sleep(task.expected_duration)
            
            return {
                'patterns_detected': ['head_and_shoulders', 'double_top'],
                'confidence_scores': [0.8, 0.7],
                'processing_time': task.expected_duration
            }
            
        except Exception as e:
            logger.error(f"Error executing pattern detection task: {e}")
            raise
    
    def _execute_sentiment_analysis_task(self, task: GPUTask) -> Any:
        """Execute sentiment analysis task"""
        try:
            # This would implement actual sentiment analysis
            # For now, simulate sentiment analysis
            time.sleep(task.expected_duration)
            
            return {
                'sentiment_score': 0.6,
                'confidence': 0.8,
                'keywords': ['bullish', 'growth', 'positive'],
                'processing_time': task.expected_duration
            }
            
        except Exception as e:
            logger.error(f"Error executing sentiment analysis task: {e}")
            raise
    
    def _execute_generic_task(self, task: GPUTask) -> Any:
        """Execute generic task"""
        try:
            # Generic task execution
            time.sleep(task.expected_duration)
            
            return {
                'result': 'generic_task_completed',
                'processing_time': task.expected_duration
            }
            
        except Exception as e:
            logger.error(f"Error executing generic task: {e}")
            raise
    
    def _update_task_stats(self, task: GPUTask):
        """Update task statistics"""
        try:
            if task.status == "completed":
                self.task_stats['completed_tasks'] += 1
                
                # Update average completion time
                if task.started_at and task.completed_at:
                    completion_time = (task.completed_at - task.started_at).total_seconds()
                    total_completed = self.task_stats['completed_tasks']
                    
                    # Calculate running average
                    current_avg = self.task_stats['avg_completion_time']
                    self.task_stats['avg_completion_time'] = (
                        (current_avg * (total_completed - 1) + completion_time) / total_completed
                    )
                
            elif task.status == "failed":
                self.task_stats['failed_tasks'] += 1
            
            # Calculate throughput (tasks per minute)
            total_tasks = self.task_stats['total_tasks']
            if total_tasks > 0:
                # This is a simplified calculation
                self.task_stats['throughput'] = self.task_stats['completed_tasks'] / total_tasks
            
        except Exception as e:
            logger.error(f"Error updating task stats: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[GPUTask]:
        """Get task status by ID"""
        # Check all task dictionaries
        for task_dict in [self.running_tasks, self.completed_tasks, self.failed_tasks]:
            if task_id in task_dict:
                return task_dict[task_id]
        
        return None
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """Get current GPU utilization"""
        try:
            utilization = {}
            
            for gpu_id in range(self.gpu_monitor.gpu_count):
                gpu_info = self.gpu_monitor.gpu_info.get(gpu_id)
                if gpu_info:
                    utilization[gpu_id] = gpu_info.utilization
            
            return utilization
            
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
            return {}
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        return self.task_stats.copy()

class GPUMemoryManager:
    """GPU memory management"""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.memory_usage = {}
        self.memory_history = []
    
    def allocate_memory(self, gpu_id: int, size_mb: int) -> bool:
        """Allocate GPU memory"""
        try:
            gpu_info = self.gpu_monitor.gpu_info.get(gpu_id)
            if not gpu_info:
                return False
            
            # Check if enough memory is available
            if gpu_info.memory_free < size_mb:
                logger.warning(f"Not enough memory on GPU {gpu_id}: {gpu_info.memory_free}MB < {size_mb}MB")
                return False
            
            # Update memory usage tracking
            if gpu_id not in self.memory_usage:
                self.memory_usage[gpu_id] = 0
            
            self.memory_usage[gpu_id] += size_mb
            
            logger.info(f"Allocated {size_mb}MB on GPU {gpu_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error allocating memory: {e}")
            return False
    
    def deallocate_memory(self, gpu_id: int, size_mb: int):
        """Deallocate GPU memory"""
        try:
            if gpu_id in self.memory_usage:
                self.memory_usage[gpu_id] = max(0, self.memory_usage[gpu_id] - size_mb)
                logger.info(f"Deallocated {size_mb}MB from GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error deallocating memory: {e}")
    
    def get_memory_usage(self, gpu_id: int) -> Dict[str, int]:
        """Get memory usage for a GPU"""
        try:
            gpu_info = self.gpu_monitor.gpu_info.get(gpu_id)
            if not gpu_info:
                return {}
            
            allocated = self.memory_usage.get(gpu_id, 0)
            
            return {
                'total': gpu_info.memory_total,
                'used': gpu_info.memory_used,
                'free': gpu_info.memory_free,
                'allocated': allocated,
                'available': gpu_info.memory_free - allocated
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def clear_gpu_memory(self, gpu_id: int):
        """Clear GPU memory cache"""
        try:
            if TORCH_AVAILABLE:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            # Reset memory usage tracking
            if gpu_id in self.memory_usage:
                self.memory_usage[gpu_id] = 0
            
            logger.info(f"Cleared memory cache for GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")

class GPUPerformanceOptimizer:
    """GPU performance optimization"""
    
    def __init__(self, gpu_monitor: GPUMonitor, task_scheduler: GPUTaskScheduler):
        self.gpu_monitor = gpu_monitor
        self.task_scheduler = task_scheduler
        self.optimization_enabled = True
        self.optimization_thread = None
    
    def start_optimization(self):
        """Start performance optimization"""
        if self.optimization_enabled:
            self.optimization_thread = threading.Thread(target=self._optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            logger.info("GPU performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization"""
        self.optimization_enabled = False
        if self.optimization_thread:
            self.optimization_thread.join()
        logger.info("GPU performance optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_enabled:
            try:
                # Monitor GPU performance
                for gpu_id in range(self.gpu_monitor.gpu_count):
                    self._optimize_gpu(gpu_id)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(30)
    
    def _optimize_gpu(self, gpu_id: int):
        """Optimize specific GPU performance"""
        try:
            gpu_info = self.gpu_monitor.gpu_info.get(gpu_id)
            if not gpu_info:
                return
            
            # Check for high temperature
            if gpu_info.temperature > 80:
                logger.warning(f"High temperature on GPU {gpu_id}: {gpu_info.temperature}°C")
                self._reduce_gpu_load(gpu_id)
            
            # Check for high memory usage
            memory_usage_percent = gpu_info.memory_used / gpu_info.memory_total * 100
            if memory_usage_percent > 90:
                logger.warning(f"High memory usage on GPU {gpu_id}: {memory_usage_percent:.1f}%")
                self._optimize_memory_usage(gpu_id)
            
            # Check for low utilization
            if gpu_info.utilization < 10 and len(self.task_scheduler.running_tasks) > 0:
                logger.info(f"Low utilization on GPU {gpu_id}: {gpu_info.utilization:.1f}%")
                self._increase_gpu_load(gpu_id)
            
        except Exception as e:
            logger.error(f"Error optimizing GPU {gpu_id}: {e}")
    
    def _reduce_gpu_load(self, gpu_id: int):
        """Reduce GPU load to prevent overheating"""
        try:
            # This would implement load reduction strategies
            # For now, just log the action
            logger.info(f"Reducing load on GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error reducing GPU load: {e}")
    
    def _optimize_memory_usage(self, gpu_id: int):
        """Optimize memory usage on GPU"""
        try:
            # Clear memory cache
            if TORCH_AVAILABLE:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            logger.info(f"Optimized memory usage on GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
    
    def _increase_gpu_load(self, gpu_id: int):
        """Increase GPU load for better utilization"""
        try:
            # This would implement load balancing strategies
            # For now, just log the action
            logger.info(f"Increasing load on GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error increasing GPU load: {e}")

class GPUManager:
    """Main GPU management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.task_scheduler = GPUTaskScheduler(self.gpu_monitor)
        self.memory_manager = GPUMemoryManager(self.gpu_monitor)
        self.performance_optimizer = GPUPerformanceOptimizer(self.gpu_monitor, self.task_scheduler)
        
        # System status
        self.running = False
        
        # Statistics
        self.system_stats = {
            'total_gpus': self.gpu_monitor.gpu_count,
            'available_gpus': 0,
            'total_memory': 0,
            'free_memory': 0,
            'avg_utilization': 0.0,
            'avg_temperature': 0.0
        }
    
    def start_system(self):
        """Start the GPU management system"""
        try:
            if self.gpu_monitor.gpu_count == 0:
                logger.warning("No GPUs available")
                return False
            
            self.running = True
            
            # Start components
            self.task_scheduler.start_scheduler()
            self.performance_optimizer.start_optimization()
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitoring_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            logger.info("GPU management system started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting GPU system: {e}")
            return False
    
    def stop_system(self):
        """Stop the GPU management system"""
        try:
            self.running = False
            
            # Stop components
            self.task_scheduler.stop_scheduler()
            self.performance_optimizer.stop_optimization()
            
            logger.info("GPU management system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping GPU system: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update GPU information
                self.gpu_monitor.get_all_gpu_info()
                
                # Update system statistics
                self._update_system_stats()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _update_system_stats(self):
        """Update system statistics"""
        try:
            total_memory = 0
            free_memory = 0
            total_utilization = 0
            total_temperature = 0
            available_gpus = 0
            
            for gpu_id, gpu_info in self.gpu_monitor.gpu_info.items():
                if gpu_info.is_available:
                    available_gpus += 1
                    total_memory += gpu_info.memory_total
                    free_memory += gpu_info.memory_free
                    total_utilization += gpu_info.utilization
                    total_temperature += gpu_info.temperature
            
            self.system_stats.update({
                'available_gpus': available_gpus,
                'total_memory': total_memory,
                'free_memory': free_memory,
                'avg_utilization': total_utilization / max(available_gpus, 1),
                'avg_temperature': total_temperature / max(available_gpus, 1)
            })
            
        except Exception as e:
            logger.error(f"Error updating system stats: {e}")
    
    def submit_task(self, task_type: TaskType, model_name: str, input_data: Any,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   expected_memory: int = 100,
                   expected_duration: float = 10.0) -> str:
        """Submit a task for GPU execution"""
        try:
            task_id = f"{task_type.value}_{model_name}_{int(time.time())}"
            
            task = GPUTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                model_name=model_name,
                input_data=input_data,
                expected_memory=expected_memory,
                expected_duration=expected_duration,
                created_at=datetime.now()
            )
            
            success = self.task_scheduler.submit_task(task)
            
            if success:
                logger.info(f"Task {task_id} submitted successfully")
                return task_id
            else:
                logger.error(f"Failed to submit task {task_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return ""
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result by ID"""
        try:
            task = self.task_scheduler.get_task_status(task_id)
            if task and task.status == "completed":
                return task.result
            return None
            
        except Exception as e:
            logger.error(f"Error getting task result: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            status = {
                'system_running': self.running,
                'gpu_count': self.gpu_monitor.gpu_count,
                'system_stats': self.system_stats.copy(),
                'task_stats': self.task_scheduler.get_task_statistics(),
                'gpu_info': {gpu_id: asdict(gpu_info) for gpu_id, gpu_info in self.gpu_monitor.gpu_info.items()},
                'gpu_utilization': self.task_scheduler.get_gpu_utilization()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}

def main():
    """Main function to demonstrate GPU management"""
    try:
        # Initialize GPU manager
        config = {
            'monitoring_interval': 5,
            'optimization_enabled': True,
            'max_memory_usage': 0.9
        }
        
        gpu_manager = GPUManager(config)
        
        # Start system
        if gpu_manager.start_system():
            logger.info("GPU management system started successfully")
            
            # Submit some test tasks
            task_ids = []
            
            # Submit training task
            task_id = gpu_manager.submit_task(
                TaskType.TRAINING,
                'lstm_model',
                {'data': 'sample_data'},
                TaskPriority.HIGH,
                expected_memory=500,
                expected_duration=5.0
            )
            if task_id:
                task_ids.append(task_id)
            
            # Submit inference task
            task_id = gpu_manager.submit_task(
                TaskType.INFERENCE,
                'xgboost_model',
                {'features': [1, 2, 3, 4, 5]},
                TaskPriority.MEDIUM,
                expected_memory=100,
                expected_duration=2.0
            )
            if task_id:
                task_ids.append(task_id)
            
            # Monitor tasks
            for i in range(30):  # Monitor for 30 seconds
                time.sleep(1)
                
                # Check task results
                for task_id in task_ids:
                    result = gpu_manager.get_task_result(task_id)
                    if result:
                        logger.info(f"Task {task_id} completed with result: {result}")
                        task_ids.remove(task_id)
                
                # Print system status every 10 seconds
                if i % 10 == 0:
                    status = gpu_manager.get_system_status()
                    logger.info(f"System status: {status['system_stats']}")
            
            # Stop system
            gpu_manager.stop_system()
            
        else:
            logger.error("Failed to start GPU management system")
        
    except Exception as e:
        logger.error(f"Error in main GPU management function: {e}")

if __name__ == "__main__":
    main()
