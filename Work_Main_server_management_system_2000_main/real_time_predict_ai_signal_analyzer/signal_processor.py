#!/usr/bin/env python3
"""
Signal Processor
Process and validate real-time AI prediction signals
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

from predictive_signal_analyzer import PredictionSignal

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Process incoming prediction signals"""
    
    def __init__(self):
        self.signal_queue = asyncio.Queue()
        self.processed_signals = deque(maxlen=1000)
        self.processing_stats = {
            'total_processed': 0,
            'valid_signals': 0,
            'invalid_signals': 0
        }
        logger.info("SignalProcessor initialized")
    
    async def process_signal(self, signal_data: Dict[str, Any]) -> Optional[PredictionSignal]:
        """Process incoming signal data"""
        try:
            # Validate and parse signal
            signal = self._parse_signal(signal_data)
            
            if signal is None:
                self.processing_stats['invalid_signals'] += 1
                return None
            
            # Add to queue
            await self.signal_queue.put(signal)
            
            # Store processed signal
            self.processed_signals.append(signal)
            self.processing_stats['total_processed'] += 1
            self.processing_stats['valid_signals'] += 1
            
            logger.info(f"Signal processed: {signal.signal_id}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    def _parse_signal(self, data: Dict[str, Any]) -> Optional[PredictionSignal]:
        """Parse signal data into PredictionSignal object"""
        try:
            signal = PredictionSignal(
                signal_id=data.get('signal_id', 'unknown'),
                timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
                symbol=data['symbol'],
                signal_type=data['signal_type'],
                confidence=float(data['confidence']),
                price_prediction=float(data['price_prediction']),
                predicted_change=float(data.get('predicted_change', 0.0)),
                features=data.get('features', {}),
                model_version=data.get('model_version', 'unknown')
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            return None
    
    async def get_next_signal(self) -> Optional[PredictionSignal]:
        """Get next signal from queue"""
        try:
            return await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    def get_recent_signals(self, n: int = 10) -> List[PredictionSignal]:
        """Get recent processed signals"""
        return list(self.processed_signals)[-n:]

class SignalFilter:
    """Filter signals based on criteria"""
    
    def __init__(self):
        self.filters = []
        logger.info("SignalFilter initialized")
    
    def add_filter(self, filter_name: str, filter_func: callable):
        """Add custom filter"""
        self.filters.append((filter_name, filter_func))
    
    def apply_filters(self, signal: PredictionSignal) -> bool:
        """Apply all filters to signal"""
        for filter_name, filter_func in self.filters:
            try:
                if not filter_func(signal):
                    logger.debug(f"Signal {signal.signal_id} rejected by filter: {filter_name}")
                    return False
            except Exception as e:
                logger.error(f"Error applying filter {filter_name}: {e}")
                return False
        
        return True
    
    def confidence_filter(self, min_confidence: float):
        """Filter by minimum confidence"""
        def filter_func(signal: PredictionSignal) -> bool:
            return signal.confidence >= min_confidence
        self.add_filter('confidence', filter_func)
    
    def symbol_filter(self, allowed_symbols: List[str]):
        """Filter by allowed symbols"""
        def filter_func(signal: PredictionSignal) -> bool:
            return signal.symbol in allowed_symbols
        self.add_filter('symbol', filter_func)
    
    def signal_type_filter(self, allowed_types: List[str]):
        """Filter by signal types"""
        def filter_func(signal: PredictionSignal) -> bool:
            return signal.signal_type in allowed_types
        self.add_filter('signal_type', filter_func)

class SignalAggregator:
    """Aggregate multiple signals into single decision"""
    
    def __init__(self):
        self.aggregation_window = timedelta(minutes=5)
        logger.info("SignalAggregator initialized")
    
    def aggregate_signals(self, signals: List[PredictionSignal]) -> Dict[str, Any]:
        """Aggregate multiple signals"""
        if not signals:
            return {}
        
        # Count by signal type
        buy_count = sum(1 for s in signals if s.signal_type == 'buy')
        sell_count = sum(1 for s in signals if s.signal_type == 'sell')
        hold_count = sum(1 for s in signals if s.signal_type == 'hold')
        
        # Calculate weighted confidence
        total_confidence = sum(s.confidence for s in signals)
        avg_confidence = total_confidence / len(signals) if signals else 0
        
        # Determine consensus
        if buy_count > sell_count and buy_count > hold_count:
            consensus = 'buy'
        elif sell_count > buy_count and sell_count > hold_count:
            consensus = 'sell'
        else:
            consensus = 'hold'
        
        # Calculate expected price change
        predicted_changes = [s.predicted_change for s in signals]
        avg_change = np.mean(predicted_changes)
        
        return {
            'consensus': consensus,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_signals': len(signals),
            'avg_confidence': avg_confidence,
            'avg_predicted_change': avg_change,
            'confidence': abs(buy_count - sell_count) / len(signals) if signals else 0
        }

def main():
    """Example usage"""
    processor = SignalProcessor()
    
    # Sample signal data
    signal_data = {
        'signal_id': 'test_001',
        'timestamp': datetime.now().isoformat(),
        'symbol': 'BTCUSDT',
        'signal_type': 'buy',
        'confidence': 0.85,
        'price_prediction': 50000.0,
        'predicted_change': 0.03,
        'features': {
            'volume': 1.2,
            'rsi': 45.0
        },
        'model_version': 'model_v1'
    }
    
    # Process signal
    import asyncio
    signal = asyncio.run(processor.process_signal(signal_data))
    
    if signal:
        print(f"Processed signal: {signal.signal_id}")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Type: {signal.signal_type}")
        print(f"  Confidence: {signal.confidence:.2%}")

if __name__ == "__main__":
    main()

