#!/usr/bin/env python3
"""
Advanced GPU Allocator for Parallel Training
Supports multiple allocation strategies with resource optimization
"""

import logging
import torch
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

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

class AllocationStrategy(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    TEMPERATURE_AWARE = "temperature_aware"

class TrainingGPUAllocator:
    """
    Advanced GPU allocator for training jobs
    
    Supports multiple allocation strategies:
    - First-fit: Allocate to first available GPU
    - Best-fit: Minimize wasted GPU memory
    - Round-robin: Distribute jobs evenly across GPUs
    - Load-balanced: Select least utilized GPU
    - Temperature-aware: Prefer cooler GPUs
    """
    
    def __init__(self):
        self.allocation_history = []
        self.current_allocations = {}  # job_id -> gpu_ids
        self.gpu_usage_count = {}  # gpu_id -> usage count
        
    def allocate(self, job_id: str, required_gpus: int = 1, min_memory_gb: float = 4.0) -> List[int]:
        """
        Allocate GPU(s) for a training job
        
        Args:
            job_id: Training job ID
            required_gpus: Number of GPUs needed
            min_memory_gb: Minimum GPU memory in GB
            
        Returns:
            List of allocated GPU IDs
        """
        try:
            strategy_name = get_config('gpu_allocation_strategy') or 'first_fit'
            strategy = AllocationStrategy(strategy_name)
            
            available_gpus = self._get_available_gpus(min_memory_gb)
            
            if len(available_gpus) < required_gpus:
                logger.warning(f"Not enough GPUs: need {required_gpus}, available {len(available_gpus)}")
                return []
            
            # Select GPUs based on strategy
            if strategy == AllocationStrategy.FIRST_FIT:
                selected_gpus = self._first_fit_allocate(available_gpus, required_gpus)
            elif strategy == AllocationStrategy.BEST_FIT:
                selected_gpus = self._best_fit_allocate(available_gpus, required_gpus, min_memory_gb)
            elif strategy == AllocationStrategy.ROUND_ROBIN:
                selected_gpus = self._round_robin_allocate(available_gpus, required_gpus)
            elif strategy == AllocationStrategy.LOAD_BALANCED:
                selected_gpus = self._load_balanced_allocate(available_gpus, required_gpus)
            elif strategy == AllocationStrategy.TEMPERATURE_AWARE:
                selected_gpus = self._temperature_aware_allocate(available_gpus, required_gpus)
            else:
                selected_gpus = available_gpus[:required_gpus]
            
            # Record allocation
            self.current_allocations[job_id] = selected_gpus
            
            for gpu_id in selected_gpus:
                self.gpu_usage_count[gpu_id] = self.gpu_usage_count.get(gpu_id, 0) + 1
            
            self.allocation_history.append({
                'job_id': job_id,
                'gpu_ids': selected_gpus,
                'timestamp': self._get_timestamp(),
                'strategy': strategy_name
            })
            
            logger.info(f"Allocated GPUs {selected_gpus} to job {job_id}")
            
            return selected_gpus
            
        except Exception as e:
            logger.error(f"Error allocating GPUs: {e}")
            return []
    
    def deallocate(self, job_id: str):
        """Deallocate GPUs for a job"""
        try:
            if job_id in self.current_allocations:
                gpu_ids = self.current_allocations[job_id]
                
                for gpu_id in gpu_ids:
                    if self.gpu_usage_count.get(gpu_id, 0) > 0:
                        self.gpu_usage_count[gpu_id] -= 1
                
                del self.current_allocations[job_id]
                logger.info(f"Deallocated GPUs for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error deallocating GPUs: {e}")
    
    def _get_available_gpus(self, min_memory_gb: float) -> List[Dict[str, Any]]:
        """Get available GPUs with sufficient memory"""
        try:
            available_gpus = []
            
            # Check PyTorch CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return []
            
            gpu_count = torch.cuda.device_count()
            
            for gpu_id in range(gpu_count):
                gpu_available = True
                
                # Check if already allocated
                for allocated_gpus in self.current_allocations.values():
                    if gpu_id in allocated_gpus:
                        gpu_available = False
                        break
                
                if not gpu_available:
                    continue
                
                # Get GPU info
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpu_id < len(gpus):
                            gpu = gpus[gpu_id]
                            memory_free_gb = gpu.memoryFree / 1024  # Convert MB to GB
                            
                            if memory_free_gb >= min_memory_gb:
                                available_gpus.append({
                                    'id': gpu_id,
                                    'name': gpu.name,
                                    'memory_total': gpu.memoryTotal / 1024,
                                    'memory_free': memory_free_gb,
                                    'utilization': gpu.load * 100,
                                    'temperature': gpu.temperature
                                })
                    except Exception as e:
                        logger.warning(f"Error getting GPU {gpu_id} info: {e}")
                else:
                    # Fallback: assume GPU is available
                    available_gpus.append({
                        'id': gpu_id,
                        'name': f'GPU-{gpu_id}',
                        'memory_total': 8.0,  # Assume 8GB
                        'memory_free': 8.0,
                        'utilization': 0.0,
                        'temperature': 50.0
                    })
            
            return available_gpus
            
        except Exception as e:
            logger.error(f"Error getting available GPUs: {e}")
            return []
    
    def _first_fit_allocate(self, available_gpus: List[Dict[str, Any]], required: int) -> List[int]:
        """First-fit allocation strategy"""
        return [gpu['id'] for gpu in available_gpus[:required]]
    
    def _best_fit_allocate(self, available_gpus: List[Dict[str, Any]], required: int, min_memory: float) -> List[int]:
        """Best-fit allocation strategy"""
        # Sort by available memory
        sorted_gpus = sorted(available_gpus, key=lambda x: x['memory_free'])
        return [gpu['id'] for gpu in sorted_gpus[:required]]
    
    def _round_robin_allocate(self, available_gpus: List[Dict[str, Any]], required: int) -> List[int]:
        """Round-robin allocation strategy"""
        # Round-robin based on usage count
        if available_gpus:
            sorted_gpus = sorted(available_gpus, key=lambda x: self.gpu_usage_count.get(x['id'], 0))
            return [gpu['id'] for gpu in sorted_gpus[:required]]
        return []
    
    def _load_balanced_allocate(self, available_gpus: List[Dict[str, Any]], required: int) -> List[int]:
        """Load-balanced allocation strategy"""
        # Sort by utilization
        sorted_gpus = sorted(available_gpus, key=lambda x: x['utilization'])
        return [gpu['id'] for gpu in sorted_gpus[:required]]
    
    def _temperature_aware_allocate(self, available_gpus: List[Dict[str, Any]], required: int) -> List[int]:
        """Temperature-aware allocation strategy"""
        # Sort by temperature and prefer cooler GPUs
        sorted_gpus = sorted(available_gpus, key=lambda x: x['temperature'])
        return [gpu['id'] for gpu in sorted_gpus[:required]]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        return {
            'total_allocations': len(self.allocation_history),
            'active_allocations': len(self.current_allocations),
            'gpu_usage': self.gpu_usage_count.copy()
        }

