#!/usr/bin/env python3
"""
Main Server Management System
Entry point for the management system
"""

import logging
import time
import sys
from datetime import datetime
from typing import Dict, Any

from config import CONFIG
from subsystem_orchestrator import SubsystemOrchestrator
from health_checker import HealthChecker
from server_monitor import ServerMonitor
from alert_manager import AlertManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('management.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ManagementSystem:
    """Main management system"""
    
    def __init__(self):
        logger.info("Initializing Management System...")
        
        # Initialize components
        self.alert_manager = AlertManager(CONFIG)
        self.health_checker = HealthChecker(CONFIG)
        self.server_monitor = ServerMonitor(CONFIG)
        self.orchestrator = SubsystemOrchestrator(CONFIG)
        
        # System state
        self.is_running = False
        
        logger.info("Management System initialized")
    
    def start(self):
        """Start the management system"""
        if self.is_running:
            logger.warning("Management system already running")
            return
        
        logger.info("Starting Management System...")
        
        try:
            # Start server monitoring
            self.server_monitor.start_monitoring()
            
            # Start health checking
            self.health_checker.check_system_health()
            
            # Start auto management
            self.orchestrator.start_auto_management()
            
            # Auto-start subsystems if configured
            if CONFIG.get('auto_start_subsystems', False):
                self.orchestrator.start_all()
            
            self.is_running = True
            
            self.alert_manager.create_alert(
                'system',
                'Management System Started',
                'The main server management system has started successfully',
                'main',
                level=AlertLevel.INFO
            )
            
            logger.info("Management System started successfully")
            logger.info(f"Auto-start subsystems: {CONFIG.get('auto_start_subsystems', False)}")
            logger.info(f"Auto-restart failed: {CONFIG.get('auto_restart_failed', True)}")
            
        except Exception as e:
            logger.error(f"Error starting management system: {e}")
            raise
    
    def stop(self):
        """Stop the management system"""
        logger.info("Stopping Management System...")
        
        try:
            # Stop auto management
            self.orchestrator.stop_auto_management()
            
            # Stop all subsystems
            self.orchestrator.stop_all()
            
            # Stop server monitoring
            self.server_monitor.stop_monitoring()
            
            self.is_running = False
            
            self.alert_manager.create_alert(
                'system',
                'Management System Stopped',
                'The main server management system has stopped',
                'main',
                level=AlertLevel.INFO
            )
            
            logger.info("Management System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping management system: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.orchestrator.get_system_overview()
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health"""
        return self.orchestrator.get_orchestrator_status()
    
    def start_subsystem(self, name: str) -> bool:
        """Start a specific subsystem"""
        return self.orchestrator.start_subsystem(name)
    
    def stop_subsystem(self, name: str) -> bool:
        """Stop a specific subsystem"""
        return self.orchestrator.stop_subsystem(name)
    
    def restart_subsystem(self, name: str) -> bool:
        """Restart a specific subsystem"""
        return self.orchestrator.restart_subsystem(name)

def main():
    """Main function"""
    try:
        logger.info("="*80)
        logger.info("AI Trading Bot - Main Server Management System")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Initialize management system
        management_system = ManagementSystem()
        
        # Start the system
        management_system.start()
        
        # Print system status
        logger.info("System Status:")
        logger.info("-------------")
        status = management_system.get_status()
        
        logger.info(f"Orchestrator: {status['orchestrator']['orchestrator_status']}")
        logger.info(f"Auto-restart: {status['orchestrator']['auto_restart_enabled']}")
        logger.info("")
        logger.info("Subsystems:")
        for key, value in status['subsystems'].items():
            logger.info(f"  {key}: {value}")
        logger.info("")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                
                # Check for keyboard interrupt
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if line.strip().lower() == 'quit' or line.strip().lower() == 'exit':
                        break
                        
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal")
        
        # Stop the system
        management_system.stop()
        
        logger.info("="*80)
        logger.info("Management System shutdown complete")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

