#!/usr/bin/env python3
"""
AI Trading Prediction Signal Bot - Setup Script
Automated setup and configuration for the trading bot
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'charts', 
        'models',
        'logs',
        'backtests',
        'reports'
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}/")

def create_env_file():
    """Create .env file template"""
    env_content = """# AI Trading Prediction Signal Bot - Environment Variables

# API Keys (get these from respective services)
TIMEGPT_API_KEY=your_timegpt_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here

# Broker Configuration - IBKR REMOVED
# IBKR integration has been disabled. Use Bitget for data collection.
# IBKR_HOST=127.0.0.1
# IBKR_PAPER_PORT=7497
# IBKR_LIVE_PORT=7496
# IBKR_CLIENT_ID=1

# Trading Parameters
INITIAL_CAPITAL=10000
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.15
TRANSACTION_COST_PCT=0.001

# Data Sources
BITGET_API_URL=https://api.bitget.com
YAHOO_FINANCE_ENABLED=true

# Monitoring
LOG_LEVEL=INFO
ENABLE_EMAIL_ALERTS=false
ENABLE_SLACK_ALERTS=false
ENABLE_DISCORD_ALERTS=false

# Development
DEBUG_MODE=false
SIMULATION_MODE=true
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
    else:
        print("⚠️  .env file already exists")

def create_config_file():
    """Create custom configuration file"""
    config_content = {
        "data": {
            "DEFAULT_SYMBOL": "SOLUSDT",
            "DEFAULT_TIMEFRAME": "15min",
            "HISTORICAL_DAYS": 30,
            "MAX_CANDLES": 200
        },
        "trading": {
            "MODE": "SIMULATION",
            "DEFAULT_QUANTITY": 100,
            "MAX_POSITION_SIZE": 0.1,
            "MIN_POSITION_SIZE": 0.01,
            "STOP_LOSS_PCT": 0.048,
            "TAKE_PROFIT_PCT": 0.125,
            "MAX_DAILY_LOSS": 0.05,
            "MAX_DRAWDOWN": 0.15
        },
        "ai": {
            "TIMEGPT_ENABLED": True,
            "CHRONOS_T5_ENABLED": True,
            "CHRONOS_BOLT_ENABLED": True,
            "TIMESFM1_ENABLED": True,
            "TIMESFM2_ENABLED": True,
            "PATTERN_DETECTION_ENABLED": True,
            "SENTIMENT_ENABLED": True,
            "ML_MODEL_ENABLED": True
        },
        "signal": {
            "WEIGHTS": {
                "forecast": 0.25,
                "pattern": 0.20,
                "sentiment": 0.15,
                "technical": 0.25,
                "ml": 0.15
            },
            "MIN_SIGNAL_CONFIDENCE": 0.6,
            "MIN_SIGNAL_STRENGTH": "MODERATE"
        },
        "monitoring": {
            "LOG_LEVEL": "INFO",
            "PERFORMANCE_TRACKING_ENABLED": True,
            "DAILY_REPORT_ENABLED": True
        },
        "bot": {
            "DEBUG_MODE": False,
            "SIMULATION_MODE": True,
            "DATA_UPDATE_INTERVAL": 15,
            "SIGNAL_CHECK_INTERVAL": 60
        }
    }
    
    config_file = 'custom_config.json'
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(config_content, f, indent=2)
        print(f"✅ Created {config_file}")
    else:
        print(f"⚠️  {config_file} already exists")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please install manually: pip install -r requirements.txt")

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    else:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def check_gpu_support():
    """Check for GPU support"""
    print("🎮 Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU support available: {gpu_count} device(s)")
            print(f"   Primary GPU: {gpu_name}")
        else:
            print("⚠️  GPU not available - will use CPU")
    except ImportError:
        print("⚠️  PyTorch not installed - GPU check skipped")

def create_startup_script():
    """Create startup script"""
    startup_content = """#!/bin/bash
# AI Trading Prediction Signal Bot - Startup Script

echo "🤖 Starting AI Trading Prediction Signal Bot..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one from .env template."
fi

# Run the trading bot
echo "Starting trading bot..."
python trading_bot.py

echo "Trading bot stopped."
"""
    
    startup_file = 'start_bot.sh'
    if not os.path.exists(startup_file):
        with open(startup_file, 'w') as f:
            f.write(startup_content)
        os.chmod(startup_file, 0o755)  # Make executable
        print(f"✅ Created {startup_file}")
    else:
        print(f"⚠️  {startup_file} already exists")

def create_windows_startup():
    """Create Windows startup script"""
    startup_content = """@echo off
REM AI Trading Prediction Signal Bot - Windows Startup Script

echo 🤖 Starting AI Trading Prediction Signal Bot...

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found. Please create one from .env template.
)

REM Run the trading bot
echo Starting trading bot...
python trading_bot.py

echo Trading bot stopped.
pause
"""
    
    startup_file = 'start_bot.bat'
    if not os.path.exists(startup_file):
        with open(startup_file, 'w') as f:
            f.write(startup_content)
        print(f"✅ Created {startup_file}")
    else:
        print(f"⚠️  {startup_file} already exists")

def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import sklearn
        print("✅ Core dependencies imported successfully")
        
        # Test configuration
        from config import Config
        config = Config()
        print("✅ Configuration loaded successfully")
        
        # Test basic functionality
        from signal_generator import SignalGenerator
        signal_gen = SignalGenerator(config)
        print("✅ Signal generator initialized successfully")
        
        print("✅ All basic tests passed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies")
    except Exception as e:
        print(f"❌ Test error: {e}")

def main():
    """Main setup function"""
    print("🚀 AI Trading Prediction Signal Bot - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_config_file()
    
    # Create startup scripts
    create_startup_script()
    create_windows_startup()
    
    # Check GPU support
    check_gpu_support()
    
    # Install dependencies
    install_dependencies()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Review custom_config.json for your preferences")
    print("3. Run: python example_usage.py (to test)")
    print("4. Run: python trading_bot.py (to start trading)")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
