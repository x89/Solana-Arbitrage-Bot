#!/usr/bin/env python3
"""
Main Entry Point - Real-Time Indicator Analysis
This is the MAIN file to run for continuous real-time analysis
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_indicators import main

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       Real-Time Indicator Analysis - MAIN                       ║
    ║       Press Ctrl+C to stop                                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    main()

