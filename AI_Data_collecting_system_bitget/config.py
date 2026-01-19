#!/usr/bin/env python3
"""
Configuration Module
System-wide configuration management
"""

from dataclasses import dataclass

@dataclass
class CollectorConfig:
    """Data collection configuration"""
    symbols: list = None
    timeframes: list = None
    exchanges: list = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h']
        if self.exchanges is None:
            self.exchanges = ['bitget', 'binance']

