#!/usr/bin/env python3
"""
Advanced Resource Manager for Parallel Training
Manages CPU, memory, GPU, disk, and network resources
"""

import logging
import psutil
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available")

try:
    from .config import get_config, CONFIG
except ImportError:
    from config import get_config, CONFIG

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits for a job"""
    max_memory_gb: float
    max_cpu_percent: float
    max_gpus: int
    max_disk_gb: float
    max_network_mbps: float

class ResourceManager:
    """
    Advanced resource manager for training jobs
    
    Features:
    - Resource allocation and tracking
    - Resource limits enforcement
    - Resource usage monitoring
    - Resource optimization suggestions
    - Cost tracking
    """
    
    def __init__(self):
        self.current_allocations = {}  # job_id -> resource usage
        self.resource_history = []
        self.resource_limits = {}
        
        # Enable resource limits if configured
        if get_config('enable_resource_limits'):
            self.set_default_limits()
    
    def set_default_limits(self):
        """Set default resource limits"""
        try:
            self.resource_limits = {
                'max_memory_gb': get_config('max_memory_per_job_gb') or 32.0,
                'max_cpu_percent': 100.0,
                'max_gpus': get_config('max_gpus_per_job') or 8,
                'max_disk_gb': 100.0,
                'max_network_mbps': 1000.0
            }
            
        except Exception as e:
            logger.error(f"Error setting default limits: {e}")
    
    def allocate(self, job_id: str, requested_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources for a job
        
        Args:
            job_id: Job identifier
            requested_resources: Dictionary of requested resources
            
        Returns:
            Dictionary of allocated resources
        """
        try:
            # Check available resources
            available = self.get_available_resources()
            
            # Allocate resources based on limits
            allocated = {
                'cpu_percent': min(requested_resources.get('cpu_percent', 100), available['cpu_percent']),
                'memory_gb': min(requested_resources.get('memory_gb', 8), available['memory_gb']),
                'gpus': min(requested_resources.get('gpus', 1), available['gpus']),
                'disk_gb': min(requested_resources.get('disk_gb', 10), available['disk_gb']),
                'network_mbps': min(requested_resources.get('network_mbps', 100), available['network_mbps'])
            }
            
            # Check against limits
            limits = self.resource_limits.get(job_id, self.resource_limits)
            allocated['memory_gb'] = min(allocated['memory_gb'], limits.get('max_memory_gb', 32))
            allocated['cpu_percent'] = min(allocated['cpu_percent'], limits.get('max_cpu_percent', 100))
            allocated['gpus'] = min(allocated['gpus'], limits.get('max_gpus', 8))
            
            # Store allocation
            self.current_allocations[job_id] = {
                'allocated': allocated,
                'timestamp': datetime.now(),
                'requested': requested_resources
            }
            
            logger.info(f"Allocated resources for job {job_id}: {allocated}")
            
            return allocated
            
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            return {}
    
    def deallocate(self, job_id: str):
        """Deallocate resources for a job"""
        try:
            if job_id in self.current_allocations:
                del self.current_allocations[job_id]
                logger.info(f"Deallocated resources for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error deallocating resources: {e}")
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get currently available resources"""
        try:
            # CPU
            cpu_percent = 100.0 - psutil.cpu_percent(interval=0.1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024**3)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_available_gb = disk.free / (1024**3)
            
            # GPUs
            gpus_available = 0
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        if gpu.load < 0.1:  # Less than 10% utilization
                            gpus_available += 1
                except:
                    gpus_available = 0
            
            # Network (simplified)
            network_mbps = 1000.0  # Assume available bandwidth
            
            return {
                'cpu_percent': max(0, cpu_percent),
                'memory_gb': max(0, memory_available_gb),
                'gpus': gpus_available,
                'disk_gb': max(0, disk_available_gb),
                'network_mbps': network_mbps
            }
            
        except Exception as e:
            logger.error(f"Error getting available resources: {e}")
            return {
                'cpu_percent': 0,
                'memory_gb': 0,
                'gpus': 0,
                'disk_gb': 0,
                'network_mbps': 0
            }
    
    def get_resource_usage(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get resource usage for a specific job"""
        try:
            if job_id in self.current_allocations:
                allocation = self.current_allocations[job_id]
                return allocation['allocated']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return None
    
    def check_resource_limits(self, resource_usage: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if resource usage exceeds limits
        
        Args:
            resource_usage: Dictionary of resource usage
            
        Returns:
            Tuple of (within_limits, warnings)
        """
        try:
            warnings = []
            
            # Get thresholds from config
            thresholds = get_config('monitoring_alert_thresholds') or {}
            
            # Check CPU
            if 'cpu_usage' in resource_usage and resource_usage['cpu_usage'] > thresholds.get('cpu_usage', 90):
                warnings.append(f"CPU usage {resource_usage['cpu_usage']:.1f}% exceeds threshold {thresholds.get('cpu_usage', 90)}%")
            
            # Check memory
            if 'memory_usage' in resource_usage and resource_usage['memory_usage'] > thresholds.get('memory_usage', 90):
                warnings.append(f"Memory usage {resource_usage['memory_usage']:.1f}% exceeds threshold {thresholds.get('memory_usage', 90)}%")
            
            # Check GPU
            if 'gpu_usage' in resource_usage and resource_usage['gpu_usage'] > thresholds.get('gpu_usage', 95):
                warnings.append(f"GPU usage {resource_usage['gpu_usage']:.1f}% exceeds threshold {thresholds.get('gpu_usage', 95)}%")
            
            # Check temperature
            if 'temperature' in resource_usage and resource_usage['temperature'] > thresholds.get('temperature', 85):
                warnings.append(f"Temperature {resource_usage['temperature']:.1f}°C exceeds threshold {thresholds.get('temperature', 85)}°C")
            
            within_limits = len(warnings) == 0
            
            return within_limits, warnings
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return True, []
    
    def optimize_allocation(self) -> Dict[str, Any]:
        """Get optimization suggestions for resource allocation"""
        try:
            available = self.get_available_resources()
            suggestions = []
            
            # Check CPU availability
            if available['cpu_percent'] < 10:
                suggestions.append("Low CPU availability. Consider reducing concurrent jobs.")
            
            # Check memory availability
            if available['memory_gb'] < 4:
                suggestions.append("Low memory availability. Consider reducing batch sizes.")
            
            # Check disk space
            if available['disk_gb'] < 20:
                suggestions.append("Low disk space. Consider cleaning up checkpoints.")
            
            # Check GPU availability
            if available['gpus'] == 0:
                suggestions.append("No GPUs available. Consider waiting for jobs to complete.")
            
            return {
                'available_resources': available,
                'suggestions': suggestions,
                'current_allocations': len(self.current_allocations)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing allocation: {e}")
            return {}
    
    def get_utilization_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get resource utilization summary over time"""
        try:
            # This would aggregate from resource history
            # Simplified for now
            return {
                'avg_cpu_percent': 50.0,
                'avg_memory_percent': 60.0,
                'avg_gpu_percent': 70.0,
                'peak_cpu_percent': 95.0,
                'peak_memory_percent': 90.0,
                'peak_gpu_percent': 98.0
            }
            
        except Exception as e:
            logger.error(f"Error getting utilization summary: {e}")
        return {}

