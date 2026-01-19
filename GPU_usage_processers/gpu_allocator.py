#!/usr/bin/env python3
"""
GPU Allocator Module
Advanced GPU resource allocation with multiple strategies
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import torch

try:
    from .gpu_manager import GPUManager, GPUMonitor, GPUTask, TaskPriority, GPUInfo
    from .config import CONFIG
except ImportError:
    from gpu_manager import GPUManager, GPUMonitor, GPUTask, TaskPriority, GPUInfo
    from config import CONFIG

logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"

class GPUAllocator:
    """Advanced GPU resource allocator with multiple strategies"""
    
    def __init__(self, gpu_manager: GPUManager = None):
        """
        Initialize GPU allocator
        
        Args:
            gpu_manager: GPUManager instance (creates new one if None)
        """
        self.gpu_manager = gpu_manager
        if gpu_manager is None:
            self.gpu_manager = GPUManager()
            self.gpu_manager.start_system()
        
        self.allocation_history = []
        self.current_allocations = {}  # task_id -> gpu_id
        
        logger.info("GPUAllocator initialized")
    
    def allocate(
        self,
        task: GPUTask,
        strategy: Optional[AllocationStrategy] = None
    ) -> Optional[int]:
        """
        Allocate GPU for a task
        
        Args:
            task: GPU task to allocate
            strategy: Allocation strategy (uses config default if None)
            
        Returns:
            GPU ID if successful, None otherwise
        """
        try:
            if strategy is None:
                strategy_name = CONFIG.get('gpu_selection_strategy', 'best_fit')
                strategy = AllocationStrategy(strategy_name)
            
            gpu_id = None
            
            if strategy == AllocationStrategy.FIRST_FIT:
                gpu_id = self._first_fit(task)
            elif strategy == AllocationStrategy.BEST_FIT:
                gpu_id = self._best_fit(task)
            elif strategy == AllocationStrategy.WORST_FIT:
                gpu_id = self._worst_fit(task)
            elif strategy == AllocationStrategy.ROUND_ROBIN:
                gpu_id = self._round_robin(task)
            elif strategy == AllocationStrategy.LOAD_BALANCED:
                gpu_id = self._load_balanced(task)
            
            if gpu_id is not None:
                self.current_allocations[task.task_id] = gpu_id
                self.allocation_history.append({
                    'task_id': task.task_id,
                    'gpu_id': gpu_id,
                    'timestamp': task.created_at,
                    'strategy': strategy.value
                })
                logger.info(f"Allocated GPU {gpu_id} for task {task.task_id}")
            
            return gpu_id
            
        except Exception as e:
            logger.error(f"Error allocating GPU: {e}")
            return None
    
    def _first_fit(self, task: GPUTask) -> Optional[int]:
        """First-fit allocation strategy"""
        try:
            gpu_monitor = self.gpu_manager.gpu_monitor
            required_memory = task.expected_memory
            
            for gpu_id in range(gpu_monitor.gpu_count):
                if gpu_monitor.is_gpu_available(gpu_id, required_memory):
                    return gpu_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error in first-fit allocation: {e}")
            return None
    
    def _best_fit(self, task: GPUTask) -> Optional[int]:
        """Best-fit allocation strategy (minimize wasted memory)"""
        try:
            gpu_monitor = self.gpu_manager.gpu_monitor
            required_memory = task.expected_memory
            
            best_gpu = None
            best_waste = float('inf')
            
            for gpu_id in range(gpu_monitor.gpu_count):
                if gpu_monitor.is_gpu_available(gpu_id, required_memory):
                    gpu_info = gpu_monitor.gpu_info.get(gpu_id)
                    if gpu_info:
                        waste = gpu_info.memory_free - required_memory
                        if waste >= 0 and waste < best_waste:
                            best_waste = waste
                            best_gpu = gpu_id
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error in best-fit allocation: {e}")
            return None
    
    def _worst_fit(self, task: GPUTask) -> Optional[int]:
        """Worst-fit allocation strategy (maximize remaining free memory)"""
        try:
            gpu_monitor = self.gpu_manager.gpu_monitor
            required_memory = task.expected_memory
            
            best_gpu = None
            best_remaining = -1
            
            for gpu_id in range(gpu_monitor.gpu_count):
                if gpu_monitor.is_gpu_available(gpu_id, required_memory):
                    gpu_info = gpu_monitor.gpu_info.get(gpu_id)
                    if gpu_info:
                        remaining = gpu_info.memory_free - required_memory
                        if remaining > best_remaining:
                            best_remaining = remaining
                            best_gpu = gpu_id
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error in worst-fit allocation: {e}")
            return None
    
    def _round_robin(self, task: GPUTask) -> Optional[int]:
        """Round-robin allocation strategy"""
        try:
            gpu_monitor = self.gpu_manager.gpu_monitor
            required_memory = task.expected_memory
            
            # Get last allocated GPU
            if self.allocation_history:
                last_gpu = self.allocation_history[-1]['gpu_id']
                start_gpu = (last_gpu + 1) % gpu_monitor.gpu_count
            else:
                start_gpu = 0
            
            # Try to allocate starting from start_gpu
            for offset in range(gpu_monitor.gpu_count):
                gpu_id = (start_gpu + offset) % gpu_monitor.gpu_count
                if gpu_monitor.is_gpu_available(gpu_id, required_memory):
                    return gpu_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error in round-robin allocation: {e}")
            return None
    
    def _load_balanced(self, task: GPUTask) -> Optional[int]:
        """Load-balanced allocation strategy (least utilized GPU)"""
        try:
            gpu_monitor = self.gpu_manager.gpu_monitor
            required_memory = task.expected_memory
            
            best_gpu = None
            best_score = -1
            
            for gpu_id in range(gpu_monitor.gpu_count):
                if gpu_monitor.is_gpu_available(gpu_id, required_memory):
                    gpu_info = gpu_monitor.gpu_info.get(gpu_id)
                    if gpu_info:
                        # Score based on low utilization and high free memory
                        memory_score = gpu_info.memory_free / gpu_info.memory_total
                        utilization_score = 1.0 - (gpu_info.utilization / 100.0)
                        
                        # Combined score
                        score = memory_score * 0.5 + utilization_score * 0.5
                        
                        # Penalize high temperature if configured
                        if CONFIG.get('prefer_cooler_gpus', True):
                            temp_penalty = max(0, (gpu_info.temperature - 70) / 100.0)
                            score -= temp_penalty * 0.2
                        
                        if score > best_score:
                            best_score = score
                            best_gpu = gpu_id
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error in load-balanced allocation: {e}")
            return None
    
    def deallocate(self, task_id: str) -> bool:
        """
        Deallocate GPU for a task
        
        Args:
            task_id: Task ID to deallocate
            
        Returns:
            True if successful
        """
        try:
            if task_id in self.current_allocations:
                gpu_id = self.current_allocations[task_id]
                del self.current_allocations[task_id]
                logger.info(f"Deallocated GPU {gpu_id} for task {task_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deallocating GPU: {e}")
            return False
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        try:
            stats = {
                'total_allocations': len(self.allocation_history),
                'current_allocations': len(self.current_allocations),
                'gpu_usage': {}
            }
            
            # Calculate GPU usage
            gpu_counts = {}
            for task_id, gpu_id in self.current_allocations.items():
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
            
            stats['gpu_usage'] = gpu_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting allocation stats: {e}")
            return {}
    
    def get_best_gpu_for_task(self, task: GPUTask) -> Optional[int]:
        """Get best GPU for a task using configured strategy"""
        try:
            return self.allocate(task)
        except Exception as e:
            logger.error(f"Error getting best GPU: {e}")
            return None

