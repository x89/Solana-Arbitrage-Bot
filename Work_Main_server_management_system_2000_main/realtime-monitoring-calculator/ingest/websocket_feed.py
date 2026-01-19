"""
Real-time WebSocket market data feed for Bitget
Optimized for low-latency, incremental updates
"""

import asyncio
import json
import logging
from typing import Dict, Callable, Optional
from datetime import datetime
from collections import deque

import websockets
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Market tick data structure"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: list  # [(price, size), ...]
    asks: list  # [(price, size), ...]


class BitgetWebSocketFeed:
    """
    Real-time WebSocket feed from Bitget exchange
    Provides incremental ticks and order book updates
    """
    
    def __init__(
        self,
        symbol: str = "SOLUSDT",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        callback: Optional[Callable] = None
    ):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.callback = callback
        self.ws = None
        self.running = False
        self.tick_buffer = deque(maxlen=1000)
        
        # Bitget WebSocket URLs
        self.base_url = "wss://ws.bitget.com"
        self.public_channel = f"swap/tick@{self.symbol}"
        self.orderbook_channel = f"swap/depth@{self.symbol}"
    
    async def connect(self):
        """Connect to Bitget WebSocket"""
        try:
            url = f"{self.base_url}/v2/ws/public"
            self.ws = await websockets.connect(url)
            logger.info(f"Connected to Bitget WebSocket: {url}")
            
            # Subscribe to tick data
            await self.subscribe_ticks()
            
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe_ticks(self):
        """Subscribe to real-time tick data"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {
                    "instType": "SPOT",
                    "channel": "ticker",
                    "instId": self.symbol
                }
            ]
        }
        
        await self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to tick data for {self.symbol}")
    
    async def start_stream(self):
        """Start streaming market data"""
        if not await self.connect():
            return
        
        self.running = True
        
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                # Parse and process tick
                tick = self._parse_tick(data)
                if tick:
                    self.tick_buffer.append(tick)
                    
                    # Call callback if provided
                    if self.callback:
                        await self.callback(tick)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            self.running = False
    
    def _parse_tick(self, data: dict) -> Optional[MarketTick]:
        """Parse raw WebSocket data into MarketTick"""
        try:
            if 'data' in data and len(data['data']) > 0:
                tick_data = data['data'][0]
                
                return MarketTick(
                    timestamp=datetime.now(),
                    symbol=self.symbol,
                    price=float(tick_data.get('lastPr', 0)),
                    volume=float(tick_data.get('volume24h', 0)),
                    side='buy' if float(tick_data.get('lastPr', 0)) > float(tick_data.get('open24h', 0)) else 'sell',
                    trade_id=str(tick_data.get('ts', 0))
                )
            return None
        except Exception as e:
            logger.error(f"Error parsing tick: {e}")
            return None
    
    async def stop(self):
        """Stop the stream"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("WebSocket stream stopped")
    
    def get_latest_ticks(self, n: int = 100) -> list:
        """Get latest N ticks from buffer"""
        return list(self.tick_buffer)[-n:]


class CandleGenerator:
    """
    Generate OHLCV candles from ticks in real-time
    Supports 1s, 1m, 5m, 15m, 1h timeframes
    """
    
    def __init__(self, timeframe: str = "1m"):
        self.timeframe = timeframe
        self.candles = {}
        self.current_candle = None
        self.tick_count = 0
        self.volume = 0.0
        
    def _get_timeframe_seconds(self) -> int:
        """Get timeframe in seconds"""
        mapping = {
            "1s": 1,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        return mapping.get(self.timeframe, 60)
    
    def process_tick(self, tick: MarketTick) -> Optional[dict]:
        """
        Process tick and return new candle if completed
        Returns None if candle is still forming
        """
        # Initialize or update current candle
        if not self.current_candle:
            self._start_new_candle(tick)
            return None
        
        # Check if candle should be closed
        elapsed = (tick.timestamp - self.current_candle['timestamp']).total_seconds()
        
        if elapsed >= self._get_timeframe_seconds():
            # Close current candle
            completed_candle = self.current_candle.copy()
            self._start_new_candle(tick)
            
            return completed_candle
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], tick.price)
            self.current_candle['low'] = min(self.current_candle['low'], tick.price)
            self.current_candle['close'] = tick.price
            self.current_candle['volume'] += tick.volume
            
            return None
    
    def _start_new_candle(self, tick: MarketTick):
        """Start a new candle"""
        self.current_candle = {
            'timestamp': tick.timestamp,
            'symbol': tick.symbol,
            'timeframe': self.timeframe,
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price,
            'volume': tick.volume
        }


async def example_usage():
    """Example usage of WebSocket feed"""
    
    async def on_tick(tick: MarketTick):
        """Callback for new ticks"""
        print(f"Tick: {tick.symbol} @ {tick.price} ({tick.side})")
    
    # Create feed
    feed = BitgetWebSocketFeed(
        symbol="SOLUSDT",
        callback=on_tick
    )
    
    # Start streaming
    await feed.start_stream()


if __name__ == "__main__":
    asyncio.run(example_usage())

