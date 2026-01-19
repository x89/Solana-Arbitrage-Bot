#!/usr/bin/env python3
"""
Master Orchestrator
Runs ALL systems in the Deploying folder
Discovers and starts every subsystem
"""

import logging
import subprocess
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Installing fastapi and uvicorn...")
    os.system("pip install fastapi uvicorn")
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_orchestrator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Get the Deploying folder (parent of Main_server_management_system)
current_file = Path(__file__).resolve()
main_dir = current_file.parent  # Main_server_management_system folder
deploying_dir = main_dir.parent  # Deploying folder

logger.info(f"Current file: {current_file}")
logger.info(f"Main dir: {main_dir}")
logger.info(f"Deploying directory: {deploying_dir}")

# Make sure we're in the Deploying folder
if deploying_dir.name != "Deploying":
    # Try to find the correct path
    possible_paths = [
        main_dir.parent,
        Path(__file__).resolve().parents[2],
        Path.cwd().parent,
        Path.cwd()
    ]
    
    for path in possible_paths:
        if (path / "Main_server_management_system").exists():
            deploying_dir = path
            break
    else:
        # Use current working directory as fallback
        deploying_dir = Path.cwd()
        logger.warning(f"Could not find Deploying folder, using: {deploying_dir}")

logger.info(f"Using Deploying directory: {deploying_dir}")

# Create FastAPI app
app = FastAPI(
    title="Master Orchestrator - Complete System Controller",
    description="Discovers and runs ALL systems in the Deploying folder",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class SystemProcess:
    """Information about a running system"""
    name: str
    path: str
    process: subprocess.Popen
    start_time: datetime
    status: str = "running"

class MasterOrchestrator:
    """Orchestrates ALL systems"""
    
    def __init__(self):
        self.system_processes: Dict[str, SystemProcess] = {}
        self.running = False
        self.systems_discovered = []
        self.startup_time = datetime.now()
        
        logger.info("Initializing Master Orchestrator...")
        self._discover_all_systems()
        logger.info(f"Discovered {len(self.systems_discovered)} systems")
    
    def _discover_all_systems(self):
        """Discover all systems in the Deploying folder"""
        try:
            # List of system directories
            system_dirs = [
                # AI Systems
                ('AI_forecasting_fine-turning_system', 'forecasting.py'),
                ('AI_training_system', 'integrated_trainer.py'),
                ('AI_predicting_model_generating_system', 'prediction_engine.py'),
                ('AI_pattern_detecting_system', 'pattern_detector.py'),
                ('AI_sentiment_training_analyzing_system', 'sentiment_analyzer.py'),
                ('AI_momentum_real_time_predicting_system', 'momentum_predictor.py'),
                ('AI_news_training_system', 'analyzer.py'),
                
                # Trading Systems
                ('AI_trading_prediction_signal_steps', 'signal_generator.py'),
                ('Analyzing_indicators_comparing_momentum_system', 'indicator_analyzer.py'),
                ('Backtesting_checking_system', 'backtest_engine.py'),
                ('Signal_testing_system', 'signal_tester.py'),
                
                # Infrastructure
                ('Data_collecting_system_bitget', 'advanced_data_collector.py'),
                ('Training_parallel_processers_manager', 'training_manager.py'),
                ('GPU_usage_processers', 'gpu_monitor.py'),
                ('News_collecting_system', 'news_processor.py'),
                
                # Main Server (this one)
                ('Main_server_management_system', 'main.py'),
            ]
            
            for system_name, main_file in system_dirs:
                system_path = deploying_dir / system_name / main_file
                
                # Also check for alternative files
                if not system_path.exists():
                    # Check for common alternative names
                    for alt_name in ['main.py', '__main__.py', f'{system_name}.py', 'run.py']:
                        alt_path = deploying_dir / system_name / alt_name
                        if alt_path.exists():
                            system_path = alt_path
                            break
                
                if system_path.exists():
                    self.systems_discovered.append({
                        'name': system_name,
                        'path': str(system_path),
                        'main_file': main_file,
                        'exists': True
                    })
                    logger.info(f"✓ Discovered: {system_name} at {system_path}")
                else:
                    self.systems_discovered.append({
                        'name': system_name,
                        'path': str(system_path),
                        'main_file': main_file,
                        'exists': False
                    })
                    logger.warning(f"✗ Not found: {system_name}")
        
        except Exception as e:
            logger.error(f"Error discovering systems: {e}")
    
    def start_all_systems(self):
        """Start all discovered systems"""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("="*80)
        logger.info("Starting ALL Systems")
        logger.info("="*80)
        
        started_count = 0
        
        for system in self.systems_discovered:
            if system['exists']:
                try:
                    self._start_system(system)
                    started_count += 1
                    time.sleep(1)  # Small delay between starts
                except Exception as e:
                    logger.error(f"Error starting {system['name']}: {e}")
        
        self.running = True
        
        logger.info(f"Started {started_count} out of {len(self.systems_discovered)} systems")
        logger.info("")
    
    def _start_system(self, system: Dict[str, Any]):
        """Start a specific system"""
        system_name = system['name']
        system_path = Path(system['path'])
        
        # Skip if main file doesn't exist
        if not system_path.exists():
            logger.warning(f"Main file not found: {system_path}")
            return
        
        try:
            # Change to system directory
            system_dir = system_path.parent
            
            # Start the process
            process = subprocess.Popen(
                ['python', system_path.name],
                cwd=str(system_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            system_process = SystemProcess(
                name=system_name,
                path=str(system_path),
                process=process,
                start_time=datetime.now(),
                status='running'
            )
            
            self.system_processes[system_name] = system_process
            logger.info(f"✓ Started: {system_name} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Error starting system {system_name}: {e}")
    
    def stop_all_systems(self):
        """Stop all running systems"""
        logger.info("Stopping all systems...")
        
        for name, system_process in self.system_processes.items():
            try:
                logger.info(f"Stopping: {name}")
                system_process.process.terminate()
                system_process.process.wait(timeout=5)
                system_process.status = 'stopped'
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing: {name}")
                system_process.process.kill()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.system_processes.clear()
        self.running = False
    
    def check_system_health(self):
        """Check health of all running systems"""
        health_status = {
            'total_systems': len(self.systems_discovered),
            'running': 0,
            'stopped': 0,
            'systems': []
        }
        
        for system in self.systems_discovered:
            if not system['exists']:
                continue
            
            name = system['name']
            if name in self.system_processes:
                process = self.system_processes[name].process
                is_running = process.poll() is None
                
                health_status['systems'].append({
                    'name': name,
                    'running': is_running,
                    'pid': process.pid if is_running else None
                })
                
                if is_running:
                    health_status['running'] += 1
                else:
                    health_status['stopped'] += 1
            else:
                health_status['systems'].append({
                    'name': name,
                    'running': False,
                    'pid': None
                })
                health_status['stopped'] += 1
        
        return health_status

# Create global orchestrator
orchestrator = MasterOrchestrator()

# ==================== FastAPI Routes ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Master Orchestrator - Complete System Controller",
        "version": "1.0.0",
        "docs": "/docs",
        "uptime": str(datetime.now() - orchestrator.startup_time),
        "systems": len(orchestrator.systems_discovered)
    }

@app.get("/api/systems")
async def get_all_systems():
    """Get all discovered systems"""
    return {
        'total': len(orchestrator.systems_discovered),
        'systems': orchestrator.systems_discovered
    }

@app.get("/api/status")
async def get_status():
    """Get status of all systems"""
    health = orchestrator.check_system_health()
    
    return {
        'running': orchestrator.running,
        'uptime': str(datetime.now() - orchestrator.startup_time),
        'health': health,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/api/systems/{system_name}/status")
async def get_system_status(system_name: str):
    """Get status of a specific system"""
    if system_name in orchestrator.system_processes:
        process = orchestrator.system_processes[system_name]
        return {
            'name': system_name,
            'running': process.process.poll() is None,
            'pid': process.process.pid,
            'start_time': process.start_time.isoformat(),
            'status': process.status
        }
    else:
        raise HTTPException(status_code=404, detail="System not found or not started")

@app.post("/api/systems/{system_name}/restart")
async def restart_system(system_name: str):
    """Restart a specific system"""
    try:
        # Stop if running
        if system_name in orchestrator.system_processes:
            process = orchestrator.system_processes[system_name]
            process.process.terminate()
            time.sleep(2)
        
        # Find and start
        for system in orchestrator.systems_discovered:
            if system['name'] == system_name and system['exists']:
                orchestrator._start_system(system)
                return {"success": True, "message": f"System {system_name} restarted"}
        
        raise HTTPException(status_code=404, detail="System not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/control/start-all")
async def start_all():
    """Start all systems"""
    orchestrator.start_all_systems()
    return {"success": True, "message": "All systems started"}

@app.post("/api/control/stop-all")
async def stop_all():
    """Stop all systems"""
    orchestrator.stop_all_systems()
    return {"success": True, "message": "All systems stopped"}

@app.get("/api/stats")
async def get_stats():
    """Get statistics"""
    return {
        'total_systems': len(orchestrator.systems_discovered),
        'running_systems': len([s for s in orchestrator.system_processes.values() 
                                if s.process.poll() is None]),
        'uptime': str(datetime.now() - orchestrator.startup_time),
        'start_time': orchestrator.startup_time.isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("Master Orchestrator starting...")
    logger.info(f"Discovered {len(orchestrator.systems_discovered)} systems")
    
    # Auto-start all systems
    orchestrator.start_all_systems()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Shutting down Master Orchestrator...")
    orchestrator.stop_all_systems()

def main():
    """Main function"""
    try:
        logger.info("="*80)
        logger.info("Master Orchestrator - Complete System Controller")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        logger.info(f"Deploying directory: {deploying_dir}")
        logger.info(f"Discovered {len(orchestrator.systems_discovered)} systems")
        logger.info("")
        logger.info("Starting uvicorn server...")
        logger.info("API docs: http://localhost:8000/docs")
        logger.info("")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

