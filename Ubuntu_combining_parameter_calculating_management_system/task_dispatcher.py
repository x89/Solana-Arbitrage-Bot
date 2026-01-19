#!/usr/bin/env python3
"""
Task Dispatcher Module
Comprehensive cloud task dispatching system including:
- Task queuing and prioritization
- Remote execution via SSH
- Task monitoring and status tracking
- Retry logic and error handling
- Resource scheduling
- Result collection
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Task data class"""
    id: str
    command: str
    priority: int = 0
    max_retries: int = 3
    timeout: int = 3600
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    callback: Optional[Callable] = None

class TaskDispatcher:
    """Comprehensive task dispatcher for cloud computing"""
    
    def __init__(
        self,
        ssh_manager: Any,
        max_concurrent_tasks: int = 5,
        task_timeout: int = 3600
    ):
        """
        Initialize task dispatcher
        
        Args:
            ssh_manager: SSH manager instance for remote execution
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Default task timeout in seconds
        """
        self.ssh_manager = ssh_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.running_tasks: Dict[str, Task] = {}
        
        # Statistics
        self.stats = {
            'total_dispatched': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retries': 0
        }
        
        # Threading
        self.dispatch_thread = None
        self.running = False
        
        logger.info("Task dispatcher initialized")
    
    def add_task(
        self,
        task_id: str,
        command: str,
        priority: int = 0,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Add task to dispatch queue
        
        Args:
            task_id: Unique task identifier
            command: Command to execute
            priority: Task priority (higher = more important)
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            callback: Callback function for task completion
            
        Returns:
            True if task added successfully
        """
        try:
            if task_id in self.tasks:
                logger.warning(f"Task {task_id} already exists")
                return False
            
            task = Task(
                id=task_id,
                command=command,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout or self.task_timeout,
                callback=callback
            )
            
            self.tasks[task_id] = task
            
            # Add to priority queue
            self._insert_task_by_priority(task_id)
            
            logger.info(f"Task {task_id} added to queue with priority {priority}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return False
    
    def _insert_task_by_priority(self, task_id: str):
        """Insert task in queue by priority"""
        try:
            task = self.tasks[task_id]
            
            # Find insertion point
            insert_index = 0
            for i, queued_id in enumerate(self.task_queue):
                if self.tasks[queued_id].priority < task.priority:
                    insert_index = i
                    break
                insert_index = i + 1
            
            self.task_queue.insert(insert_index, task_id)
            
        except Exception as e:
            logger.error(f"Error inserting task by priority: {e}")
    
    def dispatch_task(self, task_id: str) -> bool:
        """
        Dispatch a single task
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if dispatched successfully
        """
        try:
            if task_id not in self.tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self.tasks[task_id]
            
            if task.status not in [TaskStatus.PENDING, TaskStatus.FAILED]:
                logger.warning(f"Task {task_id} is not in dispatchable state")
                return False
            
            # Update status
            task.status = TaskStatus.RUNNING
            self.running_tasks[task_id] = task
            
            logger.info(f"Dispatching task {task_id}: {task.command}")
            
            # Execute command via SSH
            success, stdout, stderr = self.ssh_manager.execute_command(
                task.command,
                timeout=task.timeout
            )
            
            # Update task result
            if success:
                task.status = TaskStatus.COMPLETED
                task.result = {
                    'stdout': stdout,
                    'stderr': stderr,
                    'timestamp': datetime.now().isoformat()
                }
                self.stats['total_completed'] += 1
                
                logger.info(f"Task {task_id} completed successfully")
            else:
                task.status = TaskStatus.FAILED
                task.error = stderr
                task.retry_count += 1
                self.stats['total_retries'] += 1
                
                # Check if should retry
                if task.retry_count < task.max_retries:
                    logger.warning(f"Task {task_id} failed, will retry ({task.retry_count}/{task.max_retries})")
                    task.status = TaskStatus.PENDING
                else:
                    self.stats['total_failed'] += 1
                    logger.error(f"Task {task_id} failed permanently")
            
            # Remove from running
            self.running_tasks.pop(task_id, None)
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error dispatching task: {e}")
            return False
    
    def dispatch_all_pending(self):
        """Dispatch all pending tasks"""
        try:
            pending_tasks = [
                t_id for t_id, task in self.tasks.items()
                if task.status == TaskStatus.PENDING
            ]
            
            logger.info(f"Dispatching {len(pending_tasks)} pending tasks")
            
            # Dispatch up to max concurrent
            while len(self.running_tasks) < self.max_concurrent_tasks and self.task_queue:
                task_id = self.task_queue.pop(0)
                self.stats['total_dispatched'] += 1
                
                # Dispatch in background thread
                thread = threading.Thread(target=self.dispatch_task, args=(task_id,))
                thread.daemon = True
                thread.start()
            
            # Continue dispatching remaining tasks
            while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
                if not self.running:
                    break
                
                # Wait for a slot
                time.sleep(1)
                
                if self.task_queue:
                    task_id = self.task_queue.pop(0)
                    self.stats['total_dispatched'] += 1
                    
                    thread = threading.Thread(target=self.dispatch_task, args=(task_id,))
                    thread.daemon = True
                    thread.start()
            
        except Exception as e:
            logger.error(f"Error dispatching all tasks: {e}")
    
    def start_dispatch_loop(self):
        """Start continuous dispatch loop"""
        try:
            if self.running:
                logger.warning("Dispatch loop already running")
                return
            
            self.running = True
            self.dispatch_thread = threading.Thread(target=self._dispatch_loop)
            self.dispatch_thread.daemon = True
            self.dispatch_thread.start()
            
            logger.info("Dispatch loop started")
            
        except Exception as e:
            logger.error(f"Error starting dispatch loop: {e}")
    
    def stop_dispatch_loop(self):
        """Stop continuous dispatch loop"""
        try:
            self.running = False
            
            if self.dispatch_thread:
                self.dispatch_thread.join(timeout=5)
            
            logger.info("Dispatch loop stopped")
            
        except Exception as e:
            logger.error(f"Error stopping dispatch loop: {e}")
    
    def _dispatch_loop(self):
        """Internal dispatch loop"""
        while self.running:
            try:
                if self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
                    self.dispatch_all_pending()
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
                time.sleep(1)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary
        """
        try:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            return {
                'id': task.id,
                'status': task.status.value,
                'created_at': task.created_at,
                'retry_count': task.retry_count,
                'result': task.result,
                'error': task.error
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatch statistics"""
        return {
            **self.stats,
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'running_tasks': len(self.running_tasks),
            'total_tasks': len(self.tasks)
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancelled successfully
        """
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.status == TaskStatus.RUNNING:
                logger.warning(f"Cannot cancel running task {task_id}")
                return False
            
            task.status = TaskStatus.CANCELLED
            
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            logger.info(f"Task {task_id} cancelled")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            return False

def main():
    """Main function to demonstrate task dispatcher"""
    try:
        logger.info("=" * 60)
        logger.info("Task Dispatcher Demo")
        logger.info("=" * 60)
        
        # Note: Requires SSH manager instance
        # ssh_manager = SSHManager(...)
        # dispatcher = TaskDispatcher(ssh_manager)
        
        logger.info("Task dispatcher ready")
        logger.info("=" * 60)
        logger.info("Task Dispatcher Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

