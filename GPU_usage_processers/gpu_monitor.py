#!/usr/bin/env python3
"""
GPU Monitor Module
Enhanced GPU monitoring with real-time tracking and alerts
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    from .gpu_manager import GPUMonitor as BaseGPUMonitor, GPUInfo, GPUPerformance
    from .config import CONFIG, get_config
except ImportError:
    from gpu_manager import GPUMonitor as BaseGPUMonitor, GPUInfo, GPUPerformance
    from config import CONFIG, get_config

logger = logging.getLogger(__name__)

@dataclass
class GPUAlert:
    """GPU alert structure"""
    gpu_id: int
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    value: float
    threshold: float

class EnhancedGPUMonitor(BaseGPUMonitor):
    """Enhanced GPU monitor with alerting and historical tracking"""
    
    def __init__(self):
        """Initialize enhanced monitor"""
        super().__init__()
        self.alerts = []
        self.alert_callbacks = []
        self.monitoring_enabled = True
        self.alert_history = []
        self.performance_trends = {}
        
        # Start monitoring thread if enabled
        if CONFIG.get('enable_real_time_monitoring', True):
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Check all GPUs for alerts
                for gpu_id in range(self.gpu_count):
                    self._check_gpu_alerts(gpu_id)
                
                # Update performance trends
                self._update_performance_trends()
                
                # Sleep for monitoring interval
                time.sleep(CONFIG.get('monitoring_interval_seconds', 5))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _check_gpu_alerts(self, gpu_id: int):
        """Check for GPU alerts and trigger if necessary"""
        try:
            gpu_info = self.gpu_info.get(gpu_id)
            if not gpu_info:
                return
            
            # Check temperature alert
            if CONFIG.get('alert_on_high_temperature', True):
                temp_threshold = CONFIG.get('temperature_threshold_celsius', 80)
                if gpu_info.temperature > temp_threshold:
                    self._trigger_alert(gpu_id, 'temperature', 'warning',
                                      f"High temperature on GPU {gpu_id}",
                                      gpu_info.temperature, temp_threshold)
            
            # Check memory usage alert
            if CONFIG.get('alert_on_high_memory_usage', True) and gpu_info.memory_total > 0:
                memory_usage_pct = (gpu_info.memory_used / gpu_info.memory_total) * 100
                if memory_usage_pct > 90:
                    self._trigger_alert(gpu_id, 'memory', 'warning',
                                      f"High memory usage on GPU {gpu_id}",
                                      memory_usage_pct, 90)
            
            # Check GPU unavailable
            if CONFIG.get('alert_on_gpu_unavailable', True):
                if not gpu_info.is_available:
                    self._trigger_alert(gpu_id, 'availability', 'error',
                                      f"GPU {gpu_id} is unavailable",
                                      0, 1)
            
        except Exception as e:
            logger.error(f"Error checking GPU alerts: {e}")
    
    def _trigger_alert(self, gpu_id: int, alert_type: str, severity: str,
                      message: str, value: float, threshold: float):
        """Trigger an alert"""
        try:
            alert = GPUAlert(
                gpu_id=gpu_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                value=value,
                threshold=threshold
            )
            
            self.alerts.append(alert)
            self.alert_history.append(alert)
            
            # Keep only recent alerts (last 100)
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Log alert
            logger.warning(f"GPU {gpu_id} Alert [{severity.upper()}]: {message} (value: {value:.2f}, threshold: {threshold:.2f})")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def register_alert_callback(self, callback):
        """Register a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, minutes: int = 60) -> List[GPUAlert]:
        """Get recent alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent = [a for a in self.alert_history if a.timestamp >= cutoff_time]
            return recent
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def get_alerts_by_type(self, alert_type: str) -> List[GPUAlert]:
        """Get alerts by type"""
        try:
            return [a for a in self.alert_history if a.alert_type == alert_type]
        except Exception as e:
            logger.error(f"Error getting alerts by type: {e}")
            return []
    
    def _update_performance_trends(self):
        """Update performance trends for analysis"""
        try:
            for gpu_id in range(self.gpu_count):
                if gpu_id not in self.performance_trends:
                    self.performance_trends[gpu_id] = {
                        'utilization': [],
                        'memory': [],
                        'temperature': [],
                        'timestamps': []
                    }
                
                gpu_info = self.gpu_info.get(gpu_id)
                if gpu_info:
                    now = datetime.now()
                    trend = self.performance_trends[gpu_id]
                    
                    trend['utilization'].append(gpu_info.utilization)
                    if gpu_info.memory_total > 0:
                        trend['memory'].append(gpu_info.memory_used / gpu_info.memory_total * 100)
                    else:
                        trend['memory'].append(0.0)
                    trend['temperature'].append(gpu_info.temperature)
                    trend['timestamps'].append(now)
                    
                    # Keep only last 60 data points (5 minutes at 5s intervals)
                    if len(trend['utilization']) > 60:
                        for key in trend:
                            trend[key] = trend[key][-60:]
        except Exception as e:
            logger.error(f"Error updating performance trends: {e}")
    
    def get_average_utilization(self, gpu_id: int, minutes: int = 5) -> float:
        """Get average GPU utilization over time period"""
        try:
            if gpu_id not in self.performance_trends:
                return 0.0
            
            trend = self.performance_trends[gpu_id]
            if not trend['utilization']:
                return 0.0
            
            # Get data for specified minutes
            cutoff = datetime.now() - timedelta(minutes=minutes)
            indices = [i for i, ts in enumerate(trend['timestamps']) if ts >= cutoff]
            
            if indices:
                values = [trend['utilization'][i] for i in indices]
                return sum(values) / len(values)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting average utilization: {e}")
            return 0.0
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        try:
            summary = {
                'total_gpus': self.gpu_count,
                'available_gpus': sum(1 for info in self.gpu_info.values() if info.is_available),
                'gpu_details': {}
            }
            
            for gpu_id in range(self.gpu_count):
                gpu_info = self.gpu_info.get(gpu_id)
                if gpu_info:
                    memory_pct = (gpu_info.memory_used / gpu_info.memory_total * 100) if gpu_info.memory_total > 0 else 0.0
                    summary['gpu_details'][gpu_id] = {
                        'utilization': gpu_info.utilization,
                        'memory_used_pct': memory_pct,
                        'memory_free_mb': gpu_info.memory_free,
                        'temperature': gpu_info.temperature,
                        'power_usage': gpu_info.power_usage,
                        'is_available': gpu_info.is_available,
                        'avg_utilization_5m': self.get_average_utilization(gpu_id, 5)
                    }
            
            # Add recent alerts
            summary['recent_alerts'] = len(self.get_recent_alerts(60))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            return {}
    
    def clear_alerts(self):
        """Clear all current alerts"""
        self.alerts = []
        logger.info("Alerts cleared")
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        try:
            data = {
                'gpu_info': {gpu_id: {
                    'name': info.name,
                    'memory_total': info.memory_total,
                    'memory_used': info.memory_used,
                    'utilization': info.utilization,
                    'temperature': info.temperature
                } for gpu_id, info in self.gpu_info.items()},
                'alerts': [{
                    'gpu_id': alert.gpu_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'value': alert.value
                } for alert in self.alert_history[-100:]],
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Monitoring data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")

# Backward compatibility
def get_usage():
    """Get simple GPU usage (legacy function)"""
    try:
        monitor = EnhancedGPUMonitor()
        summary = monitor.get_usage_summary()
        
        usage = {}
        for gpu_id, details in summary.get('gpu_details', {}).items():
            usage[f'gpu{gpu_id}'] = int(details.get('utilization', 0))
        
        return usage
        
    except Exception as e:
        logger.error(f"Error getting GPU usage: {e}")
        return {'gpu0': 0}

