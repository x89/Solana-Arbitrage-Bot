#!/usr/bin/env python3
"""
Integrating Server
Unified server integrating all subsystems and management components
"""

import logging
import asyncio
import threading
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pandas as pd

# Import core subsystems
from alert_manager import AlertManager, AlertLevel
from health_checker import HealthChecker
from server_monitor import ServerMonitor
from subsystem_orchestrator import SubsystemOrchestrator
from subsystem_registry import SubsystemRegistry, SubsystemStatus
from config import CONFIG

# Add paths for importing new subsystems
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'martingale_calculating'))
sys.path.insert(0, os.path.join(current_dir, 'real_time_predict_ai_signal_analyzer'))
sys.path.insert(0, os.path.join(current_dir, 'receiving__comparing_signal_manager'))
sys.path.insert(0, os.path.join(current_dir, 'reverse_cofficient_analyzer'))
sys.path.insert(0, os.path.join(current_dir, 'Training_resistance_support'))

logger = logging.getLogger(__name__)

class IntegratingServer:
    """Main integrating server for all subsystems"""
    
    def __init__(self):
        logger.info("Initializing Integrating Server...")
        
        # Core management components
        self.alert_manager = AlertManager(CONFIG)
        self.health_checker = HealthChecker(CONFIG)
        self.server_monitor = ServerMonitor(CONFIG)
        self.orchestrator = SubsystemOrchestrator(CONFIG)
        
        # Subsystem instances
        self.subsystems = {}
        self.api_server = None
        self.running = False
        
        logger.info("Integrating Server initialized")
    
    def initialize_subsystems(self):
        """Initialize all subsystem instances"""
        try:
            logger.info("Initializing subsystem instances...")
            
            # Martingale Calculator
            try:
                from martingale_calculator import MartingaleCalculator, MartingaleConfig, MartingaleType
                martingale_config = MartingaleConfig(
                    strategy_type=MartingaleType.CLASSIC,
                    initial_bet=100.0,
                    multiplier=2.0
                )
                self.subsystems['martingale_calculator'] = MartingaleCalculator(martingale_config)
                logger.info("Martingale calculator initialized")
            except Exception as e:
                logger.error(f"Error initializing martingale calculator: {e}")
            
            # Signal Analyzer
            try:
                from predictive_signal_analyzer import PredictiveSignalAnalyzer
                self.subsystems['signal_analyzer'] = PredictiveSignalAnalyzer()
                logger.info("Signal analyzer initialized")
            except Exception as e:
                logger.error(f"Error initializing signal analyzer: {e}")
            
            # Signal Receiver
            try:
                from signal_receiver import SignalReceiver
                self.subsystems['signal_receiver'] = SignalReceiver()
                logger.info("Signal receiver initialized")
            except Exception as e:
                logger.error(f"Error initializing signal receiver: {e}")
            
            # Reverse Coefficient Analyzer
            try:
                from reverse_coefficient_analyzer import ReverseCoefficientAnalyzer
                self.subsystems['reverse_analyzer'] = ReverseCoefficientAnalyzer()
                logger.info("Reverse coefficient analyzer initialized")
            except Exception as e:
                logger.error(f"Error initializing reverse analyzer: {e}")
            
            # Support/Resistance Trainer
            try:
                from resistance_support_trainer import SupportResistanceTrainer
                self.subsystems['resistance_trainer'] = SupportResistanceTrainer()
                logger.info("Resistance trainer initialized")
            except Exception as e:
                logger.error(f"Error initializing resistance trainer: {e}")
            
            logger.info(f"Initialized {len(self.subsystems)} subsystems")
            
        except Exception as e:
            logger.error(f"Error initializing subsystems: {e}")
    
    def start(self):
        """Start the integrating server"""
        if self.running:
            logger.warning("Server already running")
            return
        
        try:
            logger.info("Starting Integrating Server...")
            
            # Initialize subsystems
            self.initialize_subsystems()
            
            # Start server monitoring
            self.server_monitor.start_monitoring()
            
            # Start health checking in background
            threading.Thread(target=self._health_check_loop, daemon=True).start()
            
            # Start orchestrator auto-management
            self.orchestrator.start_auto_management()
            
            # Create Flask API server
            self.api_server = self._create_api_server()
            
            self.running = True
            
            self.alert_manager.create_alert(
                'system',
                'Integrating Server Started',
                'All subsystems initialized and server is running',
                'integrating_server',
                level=AlertLevel.INFO
            )
            
            logger.info("Integrating Server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise
    
    def stop(self):
        """Stop the integrating server"""
        try:
            logger.info("Stopping Integrating Server...")
            
            # Stop API server
            if self.api_server:
                # Flask doesn't have built-in stop, would need to use a WSGI server for production
                pass
            
            # Stop orchestrator
            self.orchestrator.stop_auto_management()
            
            # Stop server monitoring
            self.server_monitor.stop_monitoring()
            
            self.running = False
            
            self.alert_manager.create_alert(
                'system',
                'Integrating Server Stopped',
                'All subsystems stopped and server is shutting down',
                'integrating_server',
                level=AlertLevel.INFO
            )
            
            logger.info("Integrating Server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    def _health_check_loop(self):
        """Background health checking loop"""
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check system health
                health = self.health_checker.check_system_health()
                
                if health['status'] in ['warning', 'critical']:
                    self.alert_manager.create_alert(
                        'system',
                        f'System Health: {health["status"].upper()}',
                        f'System health check detected issues',
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
        
        @app.route('/api/status', methods=['GET'])
        def get_status():
            """Get overall system status"""
            try:
                status = {
                    'running': self.running,
                    'timestamp': datetime.now().isoformat(),
                    'system_health': self.health_checker.check_system_health(),
                    'server_status': self.server_monitor.get_current_status(),
                    'subsystems': {name: 'initialized' for name in self.subsystems.keys()},
                    'orchestrator': self.orchestrator.get_orchestrator_status()
                }
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/health', methods=['GET'])
        def get_health():
            """Get health summary"""
            try:
                health = self.health_checker.get_health_summary()
                return jsonify(health)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/subsystems', methods=['GET'])
        def get_subsystems():
            """Get all subsystems"""
            try:
                subsystems_info = {}
                for name, subsystem in self.subsystems.items():
                    subsystems_info[name] = {
                        'name': name,
                        'type': type(subsystem).__name__,
                        'status': 'initialized'
                    }
                return jsonify(subsystems_info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/alerts', methods=['GET'])
        def get_alerts():
            """Get active alerts"""
            try:
                alerts = self.alert_manager.get_active_alerts()
                alerts_data = [alert.to_dict() for alert in alerts]
                return jsonify(alerts_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/alerts/summary', methods=['GET'])
        def get_alert_summary():
            """Get alert summary"""
            try:
                summary = self.alert_manager.get_alert_summary()
                return jsonify(summary)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/martingale/calculate', methods=['POST'])
        def calculate_martingale():
            """Calculate martingale parameters"""
            try:
                data = request.json
                last_result = data.get('last_result')
                
                if 'martingale_calculator' in self.subsystems:
                    calculator = self.subsystems['martingale_calculator']
                    next_bet = calculator.calculate_next_bet(last_result)
                    stats = calculator.get_statistics()
                    
                    return jsonify({
                        'next_bet': next_bet,
                        'statistics': stats
                    })
                else:
                    return jsonify({'error': 'Martingale calculator not initialized'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/signal/analyze', methods=['POST'])
        def analyze_signal():
            """Analyze a prediction signal"""
            try:
                data = request.json
                from predictive_signal_analyzer import PredictionSignal
                
                signal = PredictionSignal(
                    signal_id=data.get('signal_id'),
                    timestamp=datetime.now(),
                    symbol=data.get('symbol'),
                    signal_type=data.get('signal_type'),
                    confidence=data.get('confidence'),
                    price_prediction=data.get('price_prediction'),
                    predicted_change=data.get('predicted_change'),
                    features=data.get('features', {}),
                    model_version=data.get('model_version')
                )
                
                if 'signal_analyzer' in self.subsystems:
                    analyzer = self.subsystems['signal_analyzer']
                    analysis = analyzer.analyze_signal(signal)
                    
                    return jsonify({
                        'analysis_score': analysis.analysis_score,
                        'recommendation': analysis.recommendation,
                        'risk_level': analysis.risk_level,
                        'expected_return': analysis.expected_return,
                        'reasoning': analysis.reasoning
                    })
                else:
                    return jsonify({'error': 'Signal analyzer not initialized'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/resistance/identify', methods=['POST'])
        def identify_resistance_support():
            """Identify support and resistance levels"""
            try:
                data = request.json
                price_data_json = data.get('price_data')
                
                # Convert to DataFrame
                price_data = pd.DataFrame(price_data_json)
                
                if 'resistance_trainer' in self.subsystems:
                    trainer = self.subsystems['resistance_trainer']
                    levels = trainer.identify_levels(price_data)
                    
                    levels_data = [{
                        'level_type': level.level_type,
                        'price': level.price,
                        'strength': level.strength,
                        'touches': level.touches
                    } for level in levels]
                    
                    return jsonify({'levels': levels_data})
                else:
                    return jsonify({'error': 'Resistance trainer not initialized'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/system/overview', methods=['GET'])
        def get_system_overview():
            """Get comprehensive system overview"""
            try:
                overview = {
                    'timestamp': datetime.now().isoformat(),
                    'server_running': self.running,
                    'system_health': self.health_checker.get_health_summary(),
                    'server_status': self.server_monitor.get_current_status(),
                    'active_alerts': len(self.alert_manager.get_active_alerts()),
                    'subsystems_count': len(self.subsystems),
                    'subsystems': list(self.subsystems.keys()),
                    'orchestrator_status': self.orchestrator.get_orchestrator_status()
                }
                return jsonify(overview)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return app
    
    def run_api_server(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        if not self.api_server:
            self.api_server = self._create_api_server()
        
        logger.info(f"Starting API server on {host}:{port}")
        self.api_server.run(host=host, port=port, debug=debug)

def main():
    """Main function"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('integrating_server.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("="*80)
        logger.info("AI Trading Bot - Integrating Server")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Create and start server
        server = IntegratingServer()
        server.start()
        
        # Run API server
        try:
            server.run_api_server(host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal")
        finally:
            server.stop()
        
        logger.info("="*80)
        logger.info("Integrating Server shutdown complete")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

