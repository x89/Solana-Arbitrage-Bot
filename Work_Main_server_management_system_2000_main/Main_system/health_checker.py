#!/usr/bin/env python3
"""
Health Checker Module
Comprehensive health checking for all system components
"""

import logging
import psutil
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import sqlite3

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class ComponentHealth:
    """Component health status"""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class HealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.thresholds = {
            'cpu_warning': self.config.get('cpu_warning_threshold', 70.0),
            'cpu_critical': self.config.get('cpu_critical_threshold', 85.0),
            'memory_warning': self.config.get('memory_warning_threshold', 75.0),
            'memory_critical': self.config.get('memory_critical_threshold', 90.0),
            'disk_warning': self.config.get('disk_warning_threshold', 80.0),
            'disk_critical': self.config.get('disk_critical_threshold', 95.0),
            'network_timeout': self.config.get('network_timeout', 5.0),
            'response_time_warning': self.config.get('response_time_warning', 1.0),
            'response_time_critical': self.config.get('response_time_critical', 3.0)
        }
        
        self.component_health: Dict[str, ComponentHealth] = {}
        self.check_history: List[Dict[str, Any]] = []
        
        # Subsystems to check
        self.subsystems = [
            'data_collection',
            'ai_training',
            'ai_prediction',
            'pattern_detection',
            'sentiment_analysis',
            'momentum_prediction',
            'indicator_analysis',
            'signal_testing',
            'backtesting',
            'training_manager'
        ]
        
        logger.info("HealthChecker initialized")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            start_time = time.time()
            
            # Collect health data
            cpu_usage = self._check_cpu()
            memory = self._check_memory()
            disk = self._check_disk()
            network = self._check_network()
            processes = self._check_processes()
            
            overall_status = self._determine_overall_status(
                cpu_usage, memory, disk, network
            )
            
            check_duration = time.time() - start_time
            
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'status': overall_status.value,
                'cpu': cpu_usage,
                'memory': memory,
                'disk': disk,
                'network': network,
                'processes': processes,
                'check_duration': check_duration,
                'uptime': self._get_system_uptime()
            }
            
            # Keep last 100 checks
            self.check_history.append(health_data)
            if len(self.check_history) > 100:
                self.check_history.pop(0)
            
            logger.debug(f"System health check completed: {overall_status.value}")
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error in system health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': HealthStatus.CRITICAL.value,
                'error': str(e)
            }
    
    def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component"""
        try:
            start_time = time.time()
            
            # Determine if component is running
            status = self._check_component_status(component_name)
            response_time = time.time() - start_time
            
            # Determine health status based on response time
            if response_time > self.thresholds['response_time_critical']:
                health_status = HealthStatus.CRITICAL
            elif response_time > self.thresholds['response_time_warning']:
                health_status = HealthStatus.DEGRADED
            elif status:
                health_status = HealthStatus.HEALTHY
            else:
                health_status = HealthStatus.DOWN
            
            component_health = ComponentHealth(
                name=component_name,
                status=health_status,
                last_check=datetime.now(),
                response_time=response_time,
                details={'running': status, 'check_method': self._get_check_method(component_name)}
            )
            
            self.component_health[component_name] = component_health
            
            return component_health
            
        except Exception as e:
            logger.error(f"Error checking component {component_name}: {e}")
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0,
                details={},
                error_message=str(e)
            )
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            status = HealthStatus.HEALTHY
            if cpu_percent > self.thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL
            elif cpu_percent > self.thresholds['cpu_warning']:
                status = HealthStatus.DEGRADED
            
            return {
                'usage_percent': cpu_percent,
                'core_count': cpu_count,
                'per_core': cpu_per_core,
                'status': status.value
            }
        except Exception as e:
            logger.error(f"Error checking CPU: {e}")
            return {'error': str(e), 'status': HealthStatus.CRITICAL.value}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = HealthStatus.HEALTHY
            if mem.percent > self.thresholds['memory_critical']:
                status = HealthStatus.CRITICAL
            elif mem.percent > self.thresholds['memory_warning']:
                status = HealthStatus.DEGRADED
            
            return {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_gb': mem.used / (1024**3),
                'usage_percent': mem.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
                'status': status.value
            }
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return {'error': str(e), 'status': HealthStatus.CRITICAL.value}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            if disk.percent > self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
            elif disk.percent > self.thresholds['disk_warning']:
                status = HealthStatus.DEGRADED
            
            return {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': disk.percent,
                'status': status.value
            }
        except Exception as e:
            logger.error(f"Error checking disk: {e}")
            return {'error': str(e), 'status': HealthStatus.CRITICAL.value}
    
    def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Check basic network interface
            net_io = psutil.net_io_counters()
            
            # Check if we can resolve DNS
            import socket
            try:
                socket.gethostbyname('google.com')
                internet_status = True
            except:
                internet_status = False
            
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'internet_connected': internet_status,
                'status': HealthStatus.HEALTHY.value if internet_status else HealthStatus.DEGRADED.value
            }
        except Exception as e:
            logger.error(f"Error checking network: {e}")
            return {'error': str(e), 'status': HealthStatus.DEGRADED.value}
    
    def _check_processes(self) -> Dict[str, Any]:
        """Check running processes"""
        try:
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return {
                'total_processes': len(psutil.pids()),
                'python_processes': len(python_processes),
                'python_details': python_processes
            }
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
            return {'error': str(e)}
    
    def _check_component_status(self, component_name: str) -> bool:
        """Check if a component is running"""
        # This is a simplified check - in production, this would check actual endpoints
        # For now, check if there's a process or file that indicates the component is active
        
        try:
            # Check for common indicators that a component might be running
            # This is a placeholder implementation
            
            component_map = {
                'data_collection': ['advanced_data_collector.py'],
                'ai_training': ['advanced_ai_trainer.py', 'integrated_trainer.py'],
                'ai_prediction': ['prediction_engine.py', 'model_loader.py'],
                'pattern_detection': ['pattern_detector.py'],
                'sentiment_analysis': ['sentiment_analyzer.py'],
                'momentum_prediction': ['momentum_predictor.py'],
                'indicator_analysis': ['indicator_analyzer.py'],
                'signal_testing': ['signal_tester.py'],
                'backtesting': ['backtest_engine.py'],
                'training_manager': ['training_manager.py']
            }
            
            files_to_check = component_map.get(component_name, [])
            
            # Check if files exist (simplified check)
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    for file_to_check in files_to_check:
                        if file_to_check in cmdline and 'python' in proc.info['name'].lower():
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking component status: {e}")
            return False
    
    def _get_check_method(self, component_name: str) -> str:
        """Get the check method for a component"""
        return "process_check"  # Simplified for now
    
    def _determine_overall_status(self, cpu, memory, disk, network) -> HealthStatus:
        """Determine overall system status"""
        # Check for any critical issues
        if cpu.get('status') == HealthStatus.CRITICAL.value or \
           memory.get('status') == HealthStatus.CRITICAL.value or \
           disk.get('status') == HealthStatus.CRITICAL.value:
            return HealthStatus.CRITICAL
        
        # Check for any degraded status
        if cpu.get('status') == HealthStatus.DEGRADED.value or \
           memory.get('status') == HealthStatus.DEGRADED.value or \
           disk.get('status') == HealthStatus.DEGRADED.value:
            return HealthStatus.DEGRADED
        
        # Check for network issues
        if network.get('status') != HealthStatus.HEALTHY.value:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds"""
        try:
            return time.time() - psutil.boot_time()
        except:
            return 0
    
    def check_all_components(self) -> List[ComponentHealth]:
        """Check health of all components"""
        results = []
        for subsystem in self.subsystems:
            result = self.check_component_health(subsystem)
            results.append(result)
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        system_health = self.check_system_health()
        components = self.check_all_components()
        
        healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        critical_count = sum(1 for c in components if c.status == HealthStatus.CRITICAL)
        down_count = sum(1 for c in components if c.status == HealthStatus.DOWN)
        
        return {
            'system_health': system_health,
            'components': [{
                'name': c.name,
                'status': c.status.value,
                'last_check': c.last_check.isoformat(),
                'response_time': c.response_time
            } for c in components],
            'summary': {
                'total_components': len(components),
                'healthy': healthy_count,
                'degraded': degraded_count,
                'critical': critical_count,
                'down': down_count
            }
        }

