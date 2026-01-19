#!/usr/bin/env python3
"""
Subsystem Orchestrator
Orchestrate and manage all project subsystems
"""

import logging
import os
import subprocess
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from subsystem_registry import SubsystemRegistry, SubsystemStatus
from alert_manager import AlertManager, AlertLevel
from health_checker import HealthChecker
from server_monitor import ServerMonitor

logger = logging.getLogger(__name__)

class SubsystemOrchestrator:
    """Orchestrator for managing all subsystems"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = SubsystemRegistry(config['subsystems'])
        self.alert_manager = AlertManager(config)
        self.health_checker = HealthChecker(config)
        self.server_monitor = ServerMonitor(config)
        
        # Auto-restart settings
        self.auto_restart_enabled = config.get('auto_restart_failed', True)
        self.max_restart_attempts = config.get('max_restart_attempts', 3)
        self.restart_delay = config.get('restart_delay', 10)
        
        # Management thread
        self.management_thread = None
        self.is_running = False
        
        logger.info("SubsystemOrchestrator initialized")
    
    def start_all(self):
        """Start all enabled subsystems"""
        logger.info("Starting all subsystems...")
        
        for name, subsystem in self.registry.subsystems.items():
            if subsystem.enabled and subsystem.status == SubsystemStatus.STOPPED:
                self.start_subsystem(name)
        
        logger.info("All subsystems started")
    
    def stop_all(self):
        """Stop all running subsystems"""
        logger.info("Stopping all subsystems...")
        
        for name, subsystem in self.registry.subsystems.items():
            if subsystem.status == SubsystemStatus.RUNNING:
                self.stop_subsystem(name)
        
        logger.info("All subsystems stopped")
    
    def start_subsystem(self, name: str) -> bool:
        """Start a specific subsystem"""
        subsystem = self.registry.get_subsystem(name)
        
        if not subsystem:
            logger.error(f"Subsystem {name} not found")
            return False
        
        if not subsystem.enabled:
            logger.warning(f"Subsystem {name} is disabled")
            return False
        
        if subsystem.status == SubsystemStatus.RUNNING:
            logger.warning(f"Subsystem {name} is already running")
            return True
        
        try:
            logger.info(f"Starting subsystem: {name}")
            self.registry.update_subsystem_status(name, SubsystemStatus.STARTING)
            
            # Construct command
            script_path = os.path.join(subsystem.path, subsystem.start_script)
            
            if not os.path.exists(script_path):
                error_msg = f"Script not found: {script_path}"
                logger.error(error_msg)
                self.registry.update_subsystem_status(
                    name, 
                    SubsystemStatus.ERROR,
                    error_message=error_msg
                )
                return False
            
            # Start process
            process = subprocess.Popen(
                ['python', script_path],
                cwd=subsystem.path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Update registry
            self.registry.update_subsystem_status(
                name,
                SubsystemStatus.RUNNING,
                pid=process.pid
            )
            
            logger.info(f"Subsystem {name} started with PID {process.pid}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error starting subsystem {name}: {e}"
            logger.error(error_msg)
            self.registry.update_subsystem_status(
                name,
                SubsystemStatus.ERROR,
                error_message=error_msg
            )
            return False
    
    def stop_subsystem(self, name: str) -> bool:
        """Stop a specific subsystem"""
        subsystem = self.registry.get_subsystem(name)
        
        if not subsystem:
            logger.error(f"Subsystem {name} not found")
            return False
        
        if subsystem.status != SubsystemStatus.RUNNING:
            logger.warning(f"Subsystem {name} is not running")
            return True
        
        try:
            logger.info(f"Stopping subsystem: {name}")
            self.registry.update_subsystem_status(name, SubsystemStatus.STOPPING)
            
            # Try to terminate gracefully
            if subsystem.pid:
                import psutil
                try:
                    process = psutil.Process(subsystem.pid)
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info(f"Subsystem {name} terminated")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    # If graceful termination fails, try force kill
                    try:
                        process.kill()
                        logger.warning(f"Subsystem {name} force killed")
                    except:
                        pass
            
            self.registry.update_subsystem_status(name, SubsystemStatus.STOPPED)
            logger.info(f"Subsystem {name} stopped")
            
            return True
            
        except Exception as e:
            error_msg = f"Error stopping subsystem {name}: {e}"
            logger.error(error_msg)
            self.registry.update_subsystem_status(
                name,
                SubsystemStatus.ERROR,
                error_message=error_msg
            )
            return False
    
    def restart_subsystem(self, name: str) -> bool:
        """Restart a specific subsystem"""
        logger.info(f"Restarting subsystem: {name}")
        
        self.registry.increment_restart_count(name)
        
        # Stop first
        self.stop_subsystem(name)
        
        # Wait before restart
        time.sleep(self.restart_delay)
        
        # Start again
        return self.start_subsystem(name)
    
    def start_auto_management(self):
        """Start automatic management"""
        if self.is_running:
            logger.warning("Auto management already running")
            return
        
        self.is_running = True
        self.management_thread = threading.Thread(target=self._auto_management_loop, daemon=True)
        self.management_thread.start()
        logger.info("Auto management started")
    
    def stop_auto_management(self):
        """Stop automatic management"""
        self.is_running = False
        if self.management_thread:
            self.management_thread.join(timeout=5)
        logger.info("Auto management stopped")
    
    def _auto_management_loop(self):
        """Automatic management loop"""
        while self.is_running:
            try:
                # Check subsystem health
                self._check_subsystem_health()
                
                # Auto-restart failed subsystems
                if self.auto_restart_enabled:
                    self._auto_restart_failed_subsystems()
                
                # Update subsystem heartbeats
                self._update_heartbeats()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in auto management loop: {e}")
                time.sleep(30)
    
    def _check_subsystem_health(self):
        """Check health of all subsystems"""
        for name, subsystem in self.registry.subsystems.items():
            if subsystem.status == SubsystemStatus.RUNNING:
                # Check if process is still running
                if subsystem.pid:
                    import psutil
                    try:
                        process = psutil.Process(subsystem.pid)
                        if not process.is_running():
                            logger.warning(f"Subsystem {name} process died")
                            self.registry.update_subsystem_status(
                                name,
                                SubsystemStatus.ERROR,
                                error_message="Process died unexpectedly"
                            )
                            self.alert_manager.create_alert(
                                'system',
                                'Subsystem Process Died',
                                f'Subsystem {name} process unexpectedly terminated',
                                'orchestrator',
                                level=AlertLevel.WARNING
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        logger.warning(f"Subsystem {name} process not found")
                        self.registry.update_subsystem_status(
                            name,
                            SubsystemStatus.ERROR,
                            error_message="Process not found"
                        )
    
    def _auto_restart_failed_subsystems(self):
        """Auto-restart subsystems that have failed"""
        error_subsystems = self.registry.get_error_subsystems()
        
        for subsystem in error_subsystems:
            if subsystem.restart_count < self.max_restart_attempts:
                logger.info(f"Auto-restarting subsystem {subsystem.name} (attempt {subsystem.restart_count + 1})")
                
                # Alert on auto-restart
                self.alert_manager.create_alert(
                    'system',
                    'Auto-restarting Subsystem',
                    f'Subsystem {subsystem.name} failed, attempting restart',
                    'orchestrator',
                    level=AlertLevel.INFO
                )
                
                self.restart_subsystem(subsystem.name)
            else:
                logger.error(f"Subsystem {subsystem.name} exceeded max restart attempts")
                
                self.alert_manager.create_alert(
                    'system',
                    'Subsystem Restart Limit Exceeded',
                    f'Subsystem {subsystem.name} failed {subsystem.restart_count} times and will not be restarted',
                    'orchestrator',
                    level=AlertLevel.CRITICAL
                )
    
    def _update_heartbeats(self):
        """Update subsystem heartbeats"""
        for subsystem in self.registry.get_running_subsystems():
            self.registry.update_subsystem_status(subsystem.name, SubsystemStatus.RUNNING)
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        registry_summary = self.registry.get_registry_summary()
        health_summary = self.health_checker.get_health_summary()
        server_status = self.server_monitor.get_current_status()
        
        return {
            'orchestrator_status': 'running' if self.is_running else 'stopped',
            'auto_restart_enabled': self.auto_restart_enabled,
            'registry': registry_summary,
            'system_health': health_summary,
            'server_status': server_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_subsystem_logs(self, name: str, lines: int = 100) -> List[str]:
        """Get logs for a subsystem"""
        # This is a placeholder - in production this would read actual log files
        subsystem = self.registry.get_subsystem(name)
        
        if not subsystem:
            return []
        
        logs = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Subsystem {name} initialized",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Status: {subsystem.status.value}",
        ]
        
        return logs
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        return {
            'orchestrator': self.get_orchestrator_status(),
            'subsystems': {
                'total': len(self.registry.get_all_subsystems()),
                'running': len(self.registry.get_running_subsystems()),
                'stopped': len(self.registry.get_stopped_subsystems()),
                'error': len(self.registry.get_error_subsystems())
            },
            'alerts': self.alert_manager.get_alert_summary(),
            'health': self.health_checker.get_health_summary(),
            'server': self.server_monitor.get_current_status(),
            'timestamp': datetime.now().isoformat()
        }

