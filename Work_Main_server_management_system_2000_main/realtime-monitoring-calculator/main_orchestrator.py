"""
Main Orchestrator
Ties together all components: ingest → features → models → signals → execution
Real-time pipeline with low-latency processing
"""

import asyncio
import logging
import time
from typing import Dict, List
import numpy as np

from ingest.websocket_feed import BitgetWebSocketFeed
from ingest.candle_generator import CandleGenerator
from features.feature_engine import IncrementalFeatureEngine
from serving.predictor import ModelPredictor, EnsemblePredictor
from strategy.signal_generator import SignalGenerator, SignalType
from strategy.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """
    Main orchestrator for real-time trading system
    Coordinates all components for end-to-end pipeline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        # Initialize components
        self.feed = None
        self.candle_gen = None
        self.feature_engine = None
        self.predictor = None
        self.signal_gen = None
        self.risk_manager = None
        
        # State
        self.sequence_count = 0
        self.last_prediction_time = 0
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing components...")
        
        # Configuration
        symbol = self.config.get('symbol', 'SOLUSDT')
        timeframe = self.config.get('timeframe', '1m')
        
        # 1. Candle Generator
        self.candle_gen = CandleGenerator(symbol, timeframe)
        logger.info("✓ Candle generator initialized")
        
        # 2. Feature Engine
        self.feature_engine = IncrementalFeatureEngine(
            window_size=self.config.get('feature_window', 120)
        )
        logger.info("✓ Feature engine initialized")
        
        # 3. Model Predictor
        model_configs = self.config.get('models', [])
        if len(model_configs) > 1:
            self.predictor = EnsemblePredictor(model_configs)
        else:
            model_config = model_configs[0] if model_configs else {}
            self.predictor = ModelPredictor(
                model_path=model_config.get('path', 'models/chronos_t5.pth'),
                model_type=model_config.get('type', 'chronos_t5')
            )
        logger.info("✓ Model predictor initialized")
        
        # 4. Signal Generator
        self.signal_gen = SignalGenerator(
            min_confidence=self.config.get('min_confidence', 0.6),
            min_prediction_return=self.config.get('min_prediction_return', 0.002)
        )
        logger.info("✓ Signal generator initialized")
        
        # 5. Risk Manager
        self.risk_manager = RiskManager(
            initial_capital=self.config.get('initial_capital', 10000.0),
            risk_per_trade=self.config.get('risk_per_trade', 0.02)
        )
        logger.info("✓ Risk manager initialized")
        
        # 6. WebSocket Feed (will connect on start)
        api_key = self.config.get('bitget_api_key')
        api_secret = self.config.get('bitget_api_secret')
        
        self.feed = BitgetWebSocketFeed(
            symbol=symbol,
            api_key=api_key,
            api_secret=api_secret,
            callback=self._on_tick
        )
        logger.info("✓ WebSocket feed initialized")
        
        logger.info("All components initialized ✓")
    
    async def _on_tick(self, tick: Dict):
        """Process incoming tick"""
        try:
            # Generate candle if ready
            completed_candle = self.candle_gen.process_tick(tick)
            
            if completed_candle:
                await self._process_candle(completed_candle)
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    async def _process_candle(self, candle: Dict):
        """Process completed candle through pipeline"""
        self.sequence_count += 1
        
        try:
            # 1. Extract features
            features_dict = self.feature_engine.update(candle)
            feature_vector = self.feature_engine.get_feature_vector(features_dict)
            
            # 2. Get model prediction
            prediction = self.predictor.predict(
                feature_vector,
                horizon=self.config.get('prediction_horizon', 6)
            )
            
            # Log latency
            latency = prediction.get('latency_ms', 0)
            self.last_prediction_time = latency
            
            if self.sequence_count % 10 == 0:
                logger.info(
                    f"Sequence {self.sequence_count} - "
                    f"Latency: {latency:.2f}ms, "
                    f"Current price: {candle['close']:.2f}"
                )
            
            # 3. Generate signal
            signal = self.signal_gen.generate_signal(
                prediction,
                current_price=candle['close'],
                metadata={'sequence': self.sequence_count}
            )
            
            if signal:
                logger.info(
                    f"Generated {signal.signal_type.name} signal - "
                    f"Confidence: {signal.confidence:.2f}, "
                    f"Return: {signal.prediction_return:.4f}"
                )
                
                # 4. Execute signal (or just log for now)
                await self._execute_signal(signal, candle['close'])
            
        except Exception as e:
            logger.error(f"Error in candle processing pipeline: {e}")
    
    async def _execute_signal(self, signal, current_price: float):
        """Execute trading signal through risk manager"""
        try:
            # Convert signal to dict format for risk manager
            signal_dict = {
                'signal_type': signal.signal_type,
                'prediction_return': signal.prediction_return,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp
            }
            
            # Create position
            position = self.risk_manager.open_position(
                symbol=self.config.get('symbol', 'SOLUSDT'),
                signal=signal_dict,
                current_price=current_price
            )
            
            if position:
                logger.info(f"Opened position: {position.side} {position.quantity:.4f}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting trading system...")
        self.running = True
        
        try:
            # Start WebSocket feed
            await self.feed.start_stream()
            
        except KeyboardInterrupt:
            logger.info("Stopping system...")
            await self.stop()
        except Exception as e:
            logger.error(f"System error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading system"""
        self.running = False
        
        if self.feed:
            await self.feed.stop()
        
        logger.info("Trading system stopped")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'sequence_count': self.sequence_count,
            'last_prediction_latency_ms': self.last_prediction_time,
            'risk_metrics': self.risk_manager.get_risk_metrics().__dict__ if self.risk_manager else {}
        }
        
        if self.predictor:
            latency_stats = self.predictor.get_latency_stats()
            stats['model_latency'] = latency_stats
        
        return stats


def load_config() -> Dict:
    """Load configuration from file or defaults"""
    # In production, load from config.yaml
    # For now, return defaults
    
    return {
        'symbol': 'SOLUSDT',
        'timeframe': '1m',
        'initial_capital': 10000.0,
        'risk_per_trade': 0.02,
        'min_confidence': 0.6,
        'min_prediction_return': 0.002,
        'feature_window': 120,
        'prediction_horizon': 6,
        
        # Model configurations
        'models': [
            {
                'name': 'chronos_t5',
                'type': 'chronos_t5',
                'path': 'models/chronos_t5.pth',
                'weight': 0.4
            },
            {
                'name': 'chronos_bolt',
                'type': 'chronos_bolt',
                'path': 'models/chronos_bolt.pth',
                'weight': 0.3
            },
            {
                'name': 'xgboost',
                'type': 'torchscript',
                'path': 'models/xgboost.pth',
                'weight': 0.3
            }
        ],
        
        # API credentials (should be in .env)
        'bitget_api_key': None,  # Load from .env
        'bitget_api_secret': None  # Load from .env
    }


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("AI Crypto Trading System - Real-time Predictor")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create orchestrator
    orchestrator = TradingSystemOrchestrator(config)
    
    # Start system
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())

