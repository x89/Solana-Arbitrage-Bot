#!/usr/bin/env python3
"""
Master Server
Main server to run ALL bots, models, signal processors, and subsystems
"""

import logging
import threading
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from flask import Flask, jsonify
from flask_cors import CORS
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'test_signal'))
sys.path.insert(0, os.path.join(current_dir, 'martingale_calculating'))
sys.path.insert(0, os.path.join(current_dir, 'real_time_predict_ai_signal_analyzer'))
sys.path.insert(0, os.path.join(current_dir, 'receiving__comparing_signal_manager'))
sys.path.insert(0, os.path.join(current_dir, 'reverse_cofficient_analyzer'))
sys.path.insert(0, os.path.join(current_dir, 'Training_resistance_support'))

from alert_manager import AlertManager, AlertLevel
from health_checker import HealthChecker
from server_monitor import ServerMonitor
from config import CONFIG

@dataclass
class BotProcess:
    """Information about a running bot process"""
    name: str
    process: subprocess.Popen
    script_path: str
    enabled: bool = True
    status: str = "running"
    start_time: datetime = None

class MasterServer:
    """Master server that runs ALL bots, models, and subsystems"""
    
    def __init__(self):
        logger.info("Initializing Master Server...")
        
        # Core management
        self.alert_manager = AlertManager(CONFIG)
        self.health_checker = HealthChecker(CONFIG)
        self.server_monitor = ServerMonitor(CONFIG)
        
        # Bot processes
        self.bot_processes: Dict[str, BotProcess] = {}
        self.api_server = None
        self.running = False
        
        # Bot configurations
        self.bot_configs = {
            'signal_generator': {
                'script': 'test_signal/test_version_signal_generator.py',
                'enabled': True,
                'required': True
            },
            'signal_backend': {
                'script': 'test_signal/signal_backend_fastapi.py',
                'enabled': True,
                'required': True
            },
            'ai_bot': {
                'script': 'test_signal/json_update_ai_bot.py',
                'enabled': True,
                'required': False
            },
            'telegram_monitor': {
                'script': 'test_signal/telegram_monitoring_bot/telegrambot_enhanced.py',
                'enabled': False,  # Disabled by default
                'required': False
            }
        }
        
        logger.info("Master Server initialized")
    
    def start(self):
        """Start all bots and subsystems"""
        if self.running:
            logger.warning("Server already running")
            return
        
        try:
            logger.info("="*80)
            logger.info("Starting Master Server")
            logger.info("="*80)
            
            # Start server monitoring
            self.server_monitor.start_monitoring()
            
            # Start health checking
            threading.Thread(target=self._health_check_loop, daemon=True).start()
            
            # Start all bots
            self._start_all_bots()
            
            # Create API server
            self.api_server = self._create_api_server()
            
            self.running = True
            
            self.alert_manager.create_alert(
                'system',
                'Master Server Started',
                'All bots and subsystems are now running',
                'master_server',
                level=AlertLevel.INFO
            )
            
            logger.info("Master Server started successfully")
            logger.info(f"API server running on http://localhost:5000")
            logger.info(f"FastAPI backend running on http://localhost:8000")
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise
    
    def stop(self):
        """Stop all bots and subsystems"""
        try:
            logger.info("Stopping Master Server...")
            
            # Stop all bots
            self._stop_all_bots()
            
            # Stop server monitoring
            self.server_monitor.stop_monitoring()
            
            self.running = False
            
            self.alert_manager.create_alert(
                'system',
                'Master Server Stopped',
                'All bots and subsystems have been stopped',
                'master_server',
                level=AlertLevel.INFO
            )
            
            logger.info("Master Server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    def _start_all_bots(self):
        """Start all configured bots"""
        logger.info("Starting all bots...")
        
        for bot_name, config in self.bot_configs.items():
            if config['enabled']:
                try:
                    self._start_bot(bot_name)
                except Exception as e:
                    logger.error(f"Error starting bot {bot_name}: {e}")
                    if config['required']:
                        self.alert_manager.create_alert(
                            'system',
                            f'Failed to Start Required Bot: {bot_name}',
                            f'Bot {bot_name} failed to start: {e}',
                            'master_server',
                            level=AlertLevel.CRITICAL
                        )
        
        logger.info(f"Started {len(self.bot_processes)} bots")
    
    def _start_bot(self, bot_name: str):
        """Start a specific bot"""
        config = self.bot_configs[bot_name]
        script_path = os.path.join(current_dir, config['script'])
        
        if not os.path.exists(script_path):
            logger.warning(f"Bot script not found: {script_path}")
            return
        
        try:
            # Start bot process
            process = subprocess.Popen(
                ['python', script_path],
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            bot_process = BotProcess(
                name=bot_name,
                process=process,
                script_path=script_path,
                status='running',
                start_time=datetime.now()
            )
            
            self.bot_processes[bot_name] = bot_process
            
            logger.info(f"Started bot: {bot_name} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Error starting bot {bot_name}: {e}")
            raise
    
    def _stop_all_bots(self):
        """Stop all running bots"""
        logger.info("Stopping all bots...")
        
        for bot_name, bot_process in self.bot_processes.items():
            try:
                logger.info(f"Stopping bot: {bot_name}")
                bot_process.process.terminate()
                bot_process.process.wait(timeout=5)
                bot_process.status = 'stopped'
                logger.info(f"Bot stopped: {bot_name}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing bot: {bot_name}")
                bot_process.process.kill()
            except Exception as e:
                logger.error(f"Error stopping bot {bot_name}: {e}")
        
        self.bot_processes.clear()
    
    def _health_check_loop(self):
        """Background health checking"""
        while self.running:
            try:
                time.sleep(30)
                
                # Check bot processes
                for bot_name, bot_process in list(self.bot_processes.items()):
                    if bot_process.process.poll() is not None:
                        logger.warning(f"Bot {bot_name} has stopped")
                        bot_process.status = 'stopped'
                        
                        # Try to restart if required
                        config = self.bot_configs[bot_name]
                        if config['required']:
                            logger.info(f"Restarting bot: {bot_name}")
                            self._start_bot(bot_name)
                
                # Check system health
                health = self.health_checker.check_system_health()
                
                if health.get('status') in ['warning', 'critical']:
                    self.alert_manager.create_alert(
                        'system',
                        f'System Health: {health["status"].upper()}',
                        'System health check detected issues',
                        'health_checker',
                        level=AlertLevel.WARNING
                    )
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(30)
    
    def _create_api_server(self) -> Flask:
        """Create Flask API server"""
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/api/master/status', methods=['GET'])
        def get_master_status():
            """Get master server status"""
            bots_status = {}
            for name, bot_process in self.bot_processes.items():
                bots_status[name] = {
                    'status': bot_process.status,
                    'running': bot_process.process.poll() is None,
                    'pid': bot_process.process.pid,
                    'start_time': bot_process.start_time.isoformat() if bot_process.start_time else None
                }
            
            return jsonify({
                'running': self.running,
                'timestamp': datetime.now().isoformat(),
                'bots': bots_status,
                'server_health': self.server_monitor.get_current_status(),
                'active_alerts': len(self.alert_manager.get_active_alerts())
            })
        
        @app.route('/api/master/bots', methods=['GET'])
        def get_bots():
            """Get all bots information"""
            bots = []
            for name, bot_process in self.bot_processes.items():
                bots.append({
                    'name': name,
                    'status': bot_process.status,
                    'enabled': self.bot_configs[name]['enabled'],
                    'required': self.bot_configs[name]['required']
                })
            return jsonify(bots)
        
        @app.route('/api/master/bots/<bot_name>/restart', methods=['POST'])
        def restart_bot(bot_name: str):
            """Restart a specific bot"""
            try:
                if bot_name in self.bot_processes:
                    bot_process = self.bot_processes[bot_name]
                    bot_process.process.terminate()
                    bot_process.process.wait(timeout=5)
                    time.sleep(2)
                    self._start_bot(bot_name)
                    return jsonify({'success': True, 'message': f'Bot {bot_name} restarted'})
                else:
                    return jsonify({'success': False, 'error': 'Bot not found'}), 404
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @app.route('/api/master/health', methods=['GET'])
        def get_health():
            """Get health summary"""
            try:
                health = self.health_checker.get_health_summary()
                return jsonify(health)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/master/alerts', methods=['GET'])
        def get_alerts():
            """Get active alerts"""
            try:
                alerts = self.alert_manager.get_active_alerts()
                alerts_data = [alert.to_dict() for alert in alerts]
                return jsonify(alerts_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/master/control/start', methods=['POST'])
        def start_service():
            """Start master server"""
            if not self.running:
                self.start()
                return jsonify({'success': True, 'message': 'Master server started'})
            return jsonify({'success': False, 'message': 'Already running'})
        
        @app.route('/api/master/control/stop', methods=['POST'])
        def stop_service():
            """Stop master server"""
            if self.running:
                self.stop()
                return jsonify({'success': True, 'message': 'Master server stopped'})
            return jsonify({'success': False, 'message': 'Not running'})
        
        return app
    
    def run_api_server(self, host='0.0.0.0', port=5000):
        """Run the API server"""
        if not self.api_server:
            self.api_server = self._create_api_server()
        
        logger.info(f"Starting API server on {host}:{port}")
        
        try:
            self.api_server.run(host=host, port=port, debug=False)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

def main():
    """Main function"""
    try:
        logger.info("="*80)
        logger.info("AI Trading Bot - Master Server")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Create master server
        server = MasterServer()
        
        # Start server
        server.start()
        
        # Print status
        logger.info("Master Server Status:")
        logger.info("-------------")
        logger.info("Running bots:")
        for name, bot_process in server.bot_processes.items():
            logger.info(f"  - {name}: {bot_process.status} (PID: {bot_process.process.pid})")
        logger.info("")
        logger.info("API Endpoints:")
        logger.info("  - http://localhost:5000/api/master/status")
        logger.info("  - http://localhost:5000/api/master/bots")
        logger.info("  - http://localhost:5000/api/master/health")
        logger.info("  - http://localhost:5000/api/master/alerts")
        logger.info("")
        
        # Run API server
        try:
            server.run_api_server(host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal")
        finally:
            server.stop()
        
        logger.info("="*80)
        logger.info("Master Server shutdown complete")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

