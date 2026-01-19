#!/usr/bin/env python3
"""
Signal Receiver
Receive and manage incoming trading signals
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SignalData:
    """Trading signal data structure"""
    signal_id: str
    timestamp: datetime
    source: str
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    metadata: Dict[str, Any]

class SignalReceiver:
    """Receive trading signals from multiple sources"""
    
    def __init__(self):
        self.received_signals = deque(maxlen=1000)
        self.signal_sources = {}
        self.receiving_stats = {
            'total_received': 0,
            'by_source': {},
            'errors': 0
        }
        logger.info("SignalReceiver initialized")
    
    async def receive_signal(self, signal_data: Dict[str, Any]) -> Optional[SignalData]:
        """Receive a signal"""
        try:
            # Validate signal
            if not self._validate_signal(signal_data):
                self.receiving_stats['errors'] += 1
                return None
            
            # Parse signal
            signal = self._parse_signal(signal_data)
            
            if signal:
                self.received_signals.append(signal)
                self.receiving_stats['total_received'] += 1
                
                # Track by source
                if signal.source not in self.receiving_stats['by_source']:
                    self.receiving_stats['by_source'][signal.source] = 0
                self.receiving_stats['by_source'][signal.source] += 1
                
                logger.info(f"Signal received: {signal.signal_id} from {signal.source}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error receiving signal: {e}")
            self.receiving_stats['errors'] += 1
            return None
    
    def _validate_signal(self, data: Dict[str, Any]) -> bool:
        """Validate signal data"""
        required_fields = ['signal_id', 'timestamp', 'source', 'symbol', 'signal_type']
        return all(field in data for field in required_fields)
    
    def _parse_signal(self, data: Dict[str, Any]) -> Optional[SignalData]:
        """Parse signal data"""
        try:
            signal = SignalData(
                signal_id=data['signal_id'],
                timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
                source=data['source'],
                symbol=data['symbol'],
                signal_type=data['signal_type'],
                entry_price=float(data.get('entry_price', 0.0)),
                target_price=float(data.get('target_price', 0.0)),
                stop_loss=float(data.get('stop_loss', 0.0)),
                confidence=float(data.get('confidence', 0.5)),
                metadata=data.get('metadata', {})
            )
            return signal
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            return None
    
    def get_signals_by_source(self, source: str) -> List[SignalData]:
        """Get signals from specific source"""
        return [s for s in self.received_signals if s.source == source]
    
    def get_recent_signals(self, n: int = 10) -> List[SignalData]:
        """Get recent signals"""
        return list(self.received_signals)[-n:]
    
    def get_receiving_stats(self) -> Dict[str, Any]:
        """Get receiving statistics"""
        return self.receiving_stats.copy()

class SignalComparator:
    """Compare signals from multiple sources"""
    
    def __init__(self):
        self.comparison_results = []
        logger.info("SignalComparator initialized")
    
    def compare_signals(self, signals: List[SignalData]) -> Dict[str, Any]:
        """Compare multiple signals"""
        if not signals:
            return {}
        
        # Count by type
        buy_count = sum(1 for s in signals if s.signal_type == 'buy')
        sell_count = sum(1 for s in signals if s.signal_type == 'sell')
        hold_count = sum(1 for s in signals if s.signal_type == 'hold')
        
        # Calculate consensus
        consensus = self._calculate_consensus(signals, buy_count, sell_count, hold_count)
        
        # Analyze confidence levels
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Price range
        entry_prices = [s.entry_price for s in signals]
        min_price = min(entry_prices)
        max_price = max(entry_prices)
        
        comparison = {
            'consensus': consensus,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'avg_confidence': avg_confidence,
            'price_range': {
                'min': min_price,
                'max': max_price,
                'spread': max_price - min_price
            },
            'total_signals': len(signals)
        }
        
        self.comparison_results.append(comparison)
        return comparison
    
    def _calculate_consensus(self, signals: List[SignalData], buy_count: int, sell_count: int, hold_count: int) -> str:
        """Calculate consensus signal"""
        if buy_count > sell_count and buy_count > hold_count:
            return 'buy'
        elif sell_count > buy_count and sell_count > hold_count:
            return 'sell'
        else:
            return 'hold'
    
    def get_consistency_score(self, signals: List[SignalData]) -> float:
        """Calculate consistency score"""
        if len(signals) < 2:
            return 1.0
        
        types = [s.signal_type for s in signals]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        max_count = max(type_counts.values())
        consistency = max_count / len(signals)
        
        return consistency

def main():
    """Example usage"""
    receiver = SignalReceiver()
    
    signal_data = {
        'signal_id': 'test_001',
        'timestamp': datetime.now().isoformat(),
        'source': 'ai_model_v1',
        'symbol': 'BTCUSDT',
        'signal_type': 'buy',
        'entry_price': 50000.0,
        'target_price': 52000.0,
        'stop_loss': 48000.0,
        'confidence': 0.85
    }
    
    import asyncio
    signal = asyncio.run(receiver.receive_signal(signal_data))
    
    if signal:
        print(f"Received signal: {signal.signal_id}")

if __name__ == "__main__":
    main()

