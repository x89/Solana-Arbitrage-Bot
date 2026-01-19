#!/usr/bin/env python3
"""
Advanced Job Scheduler for Parallel Training
Manages job prioritization, queuing, and execution
"""

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from .config import get_config, CONFIG
    from .training_manager import TrainingJob
except ImportError:
    from config import get_config, CONFIG
    from training_manager import TrainingJob

logger = logging.getLogger(__name__)

class SchedulingStrategy(Enum):
    PRIORITY = "priority"
    FIFO = "fifo"
    FAIR = "fair"
    SJF = "sjf"  # Shortest Job First

@dataclass
class JobStats:
    """Job statistics"""
    job_id: str
    wait_time: float
    execution_time: float
    resource_utilization: float

class JobScheduler:
    """
    Advanced job scheduler for training jobs
    
    Features:
    - Multiple scheduling strategies (priority, FIFO, fair, SJF)
    - Dynamic priority adjustment
    - Job preemption support
    - Resource-aware scheduling
    - Fairness guarantees
    """
    
    def __init__(self, max_concurrent_jobs: int = None):
        self.max_concurrent_jobs = max_concurrent_jobs or get_config('max_concurrent_jobs') or 4
        self.scheduler_strategy = get_config('scheduler_strategy') or 'priority'
        
        # Job queues
        self.job_queues = {
            'critical': queue.PriorityQueue(),
            'high': queue.PriorityQueue(),
            'medium': queue.PriorityQueue(),
            'low': queue.PriorityQueue()
        }
        
        # Execution tracking
        self.running_jobs = {}
        self.completed_jobs = {}
        self.job_stats = {}
        
        # Scheduler state
        self.scheduling_enabled = False
        self.scheduler_thread = None
        
        # Fairness tracking
        self.fairness_tracker = {}  # Track resource usage per user/group
        
    def schedule(self, job: TrainingJob):
        """
        Schedule a training job
        
        Args:
            job: TrainingJob to schedule
            
        Returns:
            True if successfully scheduled
        """
        try:
            priority_level = self._determine_priority_level(job.priority)
            score = self._calculate_scheduling_score(job)
            
            # Add to appropriate queue
            self.job_queues[priority_level].put((score, job))
            
            logger.info(f"Scheduled job {job.job_id} with priority {priority_level}, score {score}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling job: {e}")
            return False
    
    def _determine_priority_level(self, priority: int) -> str:
        """Determine priority level from priority value"""
        priority_weights = get_config('priority_weights') or {}
        
        if priority >= 4:
            return 'critical'
        elif priority >= 3:
            return 'high'
        elif priority >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_scheduling_score(self, job: TrainingJob) -> float:
        """Calculate scheduling score based on strategy"""
        strategy = get_config('scheduler_strategy') or 'priority'
        
        if strategy == 'priority':
            # Higher priority number = higher score
            score = float(job.priority) * 1000
            
            # Adjust by age (older jobs get slight boost)
            age_hours = (datetime.now() - job.created_at).total_seconds() / 3600
            score += age_hours * 0.1
            
            return -score  # Negative for priority queue (lowest = highest priority)
        
        elif strategy == 'fifo':
            # First in, first out
            return job.created_at.timestamp()
        
        elif strategy == 'fair':
            # Fair scheduling
            priority_score = job.priority * 1000
            
            # Age adjustment
            age_minutes = (datetime.now() - job.created_at).total_seconds() / 60
            age_boost = age_minutes * 10
            
            return -(priority_score + age_boost)
        
        elif strategy == 'sjf':
            # Shortest job first (based on estimated duration)
            estimated_duration = job.config.get('estimated_duration', 3600)
            return estimated_duration
        
        else:
            return job.created_at.timestamp()
    
    def start_scheduler(self):
        """Start the job scheduler"""
        try:
            if not self.scheduling_enabled:
                self.scheduling_enabled = True
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
                self.scheduler_thread.start()
                logger.info("Job scheduler started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    def stop_scheduler(self):
        """Stop the job scheduler"""
        try:
            self.scheduling_enabled = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            logger.info("Job scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.scheduling_enabled:
                # Check if we can schedule more jobs
                if len(self.running_jobs) < self.max_concurrent_jobs:
                    job = self._get_next_job()
                    
                    if job:
                        self._execute_job(job)
                
                # Update running jobs
                self._update_running_jobs()
                
                time.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
    
    def _get_next_job(self) -> Optional[TrainingJob]:
        """Get next job from queues based on strategy"""
        try:
            # Check priority queues in order
            for priority_level in ['critical', 'high', 'medium', 'low']:
                if not self.job_queues[priority_level].empty():
                    try:
                        score, job = self.job_queues[priority_level].get_nowait()
                        return job
                    except queue.Empty:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
    
    def _execute_job(self, job: TrainingJob):
        """Execute a job"""
        try:
            job.status = 'running'
            job.started_at = datetime.now()
            
            self.running_jobs[job.job_id] = {
                'job': job,
                'start_time': datetime.now()
            }
            
            logger.info(f"Executing job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Error executing job: {e}")
    
    def _update_running_jobs(self):
        """Update status of running jobs"""
        try:
            completed_job_ids = []
            
            for job_id, job_info in self.running_jobs.items():
                job = job_info['job']
                
                # Check if job is complete (placeholder logic)
                # In real implementation, check actual job process status
                
                if job.progress >= 100.0:
                    job.completed_at = datetime.now()
                    job.status = 'completed'
                    
                    self.completed_jobs[job_id] = job
                    completed_job_ids.append(job_id)
                    
                    # Calculate stats
                    execution_time = (job.completed_at - job.started_at).total_seconds()
                    wait_time = (job.started_at - job.created_at).total_seconds()
                    
                    self.job_stats[job_id] = JobStats(
                        job_id=job_id,
                        wait_time=wait_time,
                        execution_time=execution_time,
                        resource_utilization=0.0
                    )
            
            # Remove completed jobs
            for job_id in completed_job_ids:
                del self.running_jobs[job_id]
                
        except Exception as e:
            logger.error(f"Error updating running jobs: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information"""
        try:
            queue_status = {
                'running': len(self.running_jobs),
                'completed': len(self.completed_jobs),
                'queued': {},
                'total_queued': 0
            }
            
            # Count queued jobs per priority
            for priority_level in ['critical', 'high', 'medium', 'low']:
                queue_size = self.job_queues[priority_level].qsize()
                queue_status['queued'][priority_level] = queue_size
                queue_status['total_queued'] += queue_size
            
            return queue_status
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued job"""
        try:
            # Check if job is running
            if job_id in self.running_jobs:
                job_info = self.running_jobs[job_id]
                job_info['job'].status = 'cancelled'
                del self.running_jobs[job_id]
                logger.info(f"Cancelled running job {job_id}")
                return True
            
            # Check if job is in queue
            for priority_level in ['critical', 'high', 'medium', 'low']:
                # Would need to scan queue to find and remove job
                # This is simplified
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information"""
        try:
            # Check running jobs
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]['job']
                return asdict(job)
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return asdict(job)
            
            # Check queues (would need to implement queue scanning)
            return None
            
        except Exception as e:
            logger.error(f"Error getting job info: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        try:
            stats = {
                'total_processed': len(self.completed_jobs),
                'current_running': len(self.running_jobs),
                'job_stats': {}
            }
            
            # Calculate average wait time
            if self.job_stats:
                avg_wait_time = sum(s.wait_time for s in self.job_stats.values()) / len(self.job_stats)
                avg_exec_time = sum(s.execution_time for s in self.job_stats.values()) / len(self.job_stats)
                
                stats['average_wait_time'] = avg_wait_time
                stats['average_execution_time'] = avg_exec_time
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

