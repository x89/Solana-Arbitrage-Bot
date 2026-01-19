#!/usr/bin/env python3
"""
Subsystem Registry
Track and manage all subsystems in the project
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class SubsystemStatus(Enum):
    """Subsystem status levels"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    DEGRADED = "degraded"

@dataclass
class SubsystemInfo:
    """Subsystem information"""
    name: str
    path: str
    start_script: str
    enabled: bool
    required: bool
    status: SubsystemStatus
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    restart_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class SubsystemRegistry:
    """Registry for all subsystems"""
    
    def __init__(self, subsystems_config: Dict[str, Any]):
        self.subsystems: Dict[str, SubsystemInfo] = {}
        self.initialize_subsystems(subsystems_config)
        logger.info("SubsystemRegistry initialized")
    
    def initialize_subsystems(self, subsystems_config: Dict[str, Any]):
        """Initialize subsystems from configuration"""
        for name, config in subsystems_config.items():
            if config.get('enabled', True):
                subsystem = SubsystemInfo(
                    name=name,
                    path=config['path'],
                    start_script=config['start_script'],
                    enabled=True,
                    required=config.get('required', False),
                    status=SubsystemStatus.STOPPED,
                    metadata={}
                )
                self.subsystems[name] = subsystem
                logger.info(f"Registered subsystem: {name}")
    
    def get_subsystem(self, name: str) -> Optional[SubsystemInfo]:
        """Get subsystem by name"""
        return self.subsystems.get(name)
    
    def update_subsystem_status(self, name: str, status: SubsystemStatus, **kwargs):
        """Update subsystem status"""
        if name in self.subsystems:
            subsystem = self.subsystems[name]
            subsystem.status = status
            
            if 'pid' in kwargs:
                subsystem.pid = kwargs['pid']
            if 'error_message' in kwargs:
                subsystem.error_message = kwargs['error_message']
            if status == SubsystemStatus.RUNNING and not subsystem.start_time:
                subsystem.start_time = datetime.now()
            
            subsystem.last_heartbeat = datetime.now()
            
            logger.info(f"Subsystem {name} status updated to {status.value}")
    
    def get_all_subsystems(self) -> List[SubsystemInfo]:
        """Get all subsystems"""
        return list(self.subsystems.values())
    
    def get_running_subsystems(self) -> List[SubsystemInfo]:
        """Get all running subsystems"""
        return [s for s in self.subsystems.values() if s.status == SubsystemStatus.RUNNING]
    
    def get_stopped_subsystems(self) -> List[SubsystemInfo]:
        """Get all stopped subsystems"""
        return [s for s in self.subsystems.values() if s.status == SubsystemStatus.STOPPED]
    
    def get_error_subsystems(self) -> List[SubsystemInfo]:
        """Get all subsystems in error state"""
        return [s for s in self.subsystems.values() if s.status in [SubsystemStatus.ERROR, SubsystemStatus.DEGRADED]]
    
    def get_required_subsystems(self) -> List[SubsystemInfo]:
        """Get all required subsystems"""
        return [s for s in self.subsystems.values() if s.required]
    
    def increment_restart_count(self, name: str):
        """Increment restart count for a subsystem"""
        if name in self.subsystems:
            self.subsystems[name].restart_count += 1
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry summary"""
        all_subsystems = list(self.subsystems.values())
        
        return {
            'total_subsystems': len(all_subsystems),
            'running': sum(1 for s in all_subsystems if s.status == SubsystemStatus.RUNNING),
            'stopped': sum(1 for s in all_subsystems if s.status == SubsystemStatus.STOPPED),
            'error': sum(1 for s in all_subsystems if s.status in [SubsystemStatus.ERROR, SubsystemStatus.DEGRADED]),
            'enabled': sum(1 for s in all_subsystems if s.enabled),
            'required': sum(1 for s in all_subsystems if s.required),
            'subsystems': [asdict(s) for s in all_subsystems]
        }
    
    def verify_subsystem_paths(self) -> Dict[str, bool]:
        """Verify that subsystem paths exist"""
        results = {}
        
        for name, subsystem in self.subsystems.items():
            full_path = os.path.join(subsystem.path, subsystem.start_script)
            exists = os.path.exists(full_path)
            results[name] = exists
            
            if not exists:
                logger.warning(f"Subsystem {name} path not found: {full_path}")
        
        return results
    
    def get_subsystem_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a subsystem"""
        if name in self.subsystems:
            subsystem = self.subsystems[name]
            
            # Check if process is actually running
            is_process_running = False
            if subsystem.pid:
                import psutil
                try:
                    if psutil.Process(subsystem.pid).is_running():
                        is_process_running = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return {
                'name': subsystem.name,
                'path': subsystem.path,
                'start_script': subsystem.start_script,
                'enabled': subsystem.enabled,
                'required': subsystem.required,
                'status': subsystem.status.value,
                'pid': subsystem.pid,
                'is_process_running': is_process_running,
                'start_time': subsystem.start_time.isoformat() if subsystem.start_time else None,
                'last_heartbeat': subsystem.last_heartbeat.isoformat() if subsystem.last_heartbeat else None,
                'restart_count': subsystem.restart_count,
                'error_message': subsystem.error_message,
                'uptime_seconds': (datetime.now() - subsystem.start_time).total_seconds() 
                    if subsystem.start_time else 0
            }
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'subsystems': {name: asdict(s) for name, s in self.subsystems.items()}
        }

