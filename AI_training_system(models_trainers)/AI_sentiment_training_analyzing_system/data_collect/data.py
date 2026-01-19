#!/usr/bin/env python3
"""
Data Collection Module - IBKR Integration DISABLED

⚠️ NOTE: Interactive Brokers (IBKR) functionality has been DISABLED.
The system now uses Bitget for data collection instead.

This file is kept for reference only. All IBKR code is disabled.
Use the Bitget data collector instead.
"""

import logging
from typing import Optional
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBDataCollector:
    """
    IBKR Data Collector - DISABLED
    
    ⚠️ This class is disabled. Use Bitget data collection instead.
    
    The system no longer uses Interactive Brokers.
    All data collection now uses the Bitget API.
    
    See: Data_collecting_system_bitget/advanced_data_collector.py
    """
    
    def __init__(self):
        logger.warning("⚠️ IBDataCollector is DISABLED. Use Bitget data collection instead.")
        logger.info("See: Data_collecting_system_bitget/advanced_data_collector.py")
    
    def connect(self, *args, **kwargs):
        """Disabled: IBKR connection is no longer supported"""
        logger.error("❌ IBKR is disabled. Use Bitget for data collection.")
        return False
    
    def fetch_historical_data(self, *args, **kwargs):
        """Disabled: IBKR data fetching is no longer supported"""
        logger.error("❌ IBKR is disabled. Use Bitget for data collection.")
        return pd.DataFrame()  # Empty DataFrame
    
    def disconnect(self):
        """Disabled"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Legacy functions - all disabled
def connect_ib(*args, **kwargs):
    """Disabled: Use Bitget instead"""
    logger.error("❌ IBKR connect_ib() is disabled. Use Bitget.")
    return None

def fetch_historical_data(*args, **kwargs):
    """Disabled: Use Bitget instead"""
    logger.error("❌ IBKR fetch_historical_data() is disabled. Use Bitget.")
    return pd.DataFrame()

def main():
    """Disabled: IBKR functionality removed"""
    print("\n" + "="*80)
    print("⚠️  IBKR INTEGRATION DISABLED")
    print("="*80)
    print("\nInteractive Brokers integration has been removed from the system.")
    print("\n✅ Use Bitget for data collection instead:")
    print("   File: Data_collecting_system_bitget/advanced_data_collector.py")
    print("\n📝 Bitget Data Collection:")
    print("   - No TWS/Gateway required")
    print("   - Direct API access")
    print("   - Better for crypto/data collection")
    print("   - Free data collection")
    print("\n❌ IBKR was removed because:")
    print("   - Requires TWS/Gateway installation")
    print("   - More complex setup")
    print("   - Not needed for crypto trading")
    print("   - Bitget works better for our use case")

if __name__ == "__main__":
    main()
