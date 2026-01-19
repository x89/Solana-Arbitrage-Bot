#!/usr/bin/env python3
"""
Real-Time Momentum Prediction with Live Data
Continuously fetches real-time data and predicts momentum until stopped
"""

import numpy as np
import pandas as pd
import logging
import time
import signal
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info("\n" + "="*70)
    logger.info("Shutting down gracefully...")
    logger.info("="*70)
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RealtimeMomentumPredictor:
    """Real-time momentum prediction system"""
    
    def __init__(self, symbol='SOLUSDT', timeframe='15m', interval=15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval = interval  # seconds
        
        self.predictor = None
        self.ensemble_mgr = None
        self.momentum_buffer = []
        self.data_buffer = pd.DataFrame()
        
    def initialize(self):
        """Initialize prediction components"""
        logger.info("Initializing prediction system...")
        
        from chronos_momentum_predictor import ChronosMomentumPredictor
        from momentum_calculator import MomentumCalculator
        from ensemble_manager import EnsembleManager
        
        # Initialize Chronos predictor
        self.predictor = ChronosMomentumPredictor(
            context_length=64,
            prediction_horizon=24,
            model_type='bolt',
            device='auto'
        )
        
        # Initialize momentum calculator
        self.calculator = MomentumCalculator()
        
        # Initialize ensemble manager
        self.ensemble_mgr = EnsembleManager()
        
        logger.info("✓ Prediction system initialized")
    
    def fetch_live_data(self):
        """Fetch live market data (implement based on your data source)"""
        # This is a placeholder - implement based on your data feed
        # Options:
        # 1. yfinance for stocks
        # 2. ccxt for crypto exchanges
        # 3. WebSocket for real-time data
        # 4. Database queries
        
        # Example: Simulate data fetch
        try:
            # TODO: Replace with actual data fetching
            # For demo purposes, generating simulated data
            current_price = np.random.normal(100, 5)
            
            new_data = {
                'timestamp': datetime.now(),
                'open': current_price * (1 + np.random.normal(0, 0.001)),
                'high': current_price * (1 + abs(np.random.normal(0, 0.005))),
                'low': current_price * (1 - abs(np.random.normal(0, 0.005))),
                'close': current_price,
                'volume': np.random.randint(1000, 10000)
            }
            
            return pd.DataFrame([new_data])
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def update_momentum(self, new_data):
        """Update momentum calculation from new data"""
        try:
            # Add to buffer
            if self.data_buffer.empty:
                self.data_buffer = new_data.copy()
            else:
                self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
            
            # Keep only last 200 rows for performance
            if len(self.data_buffer) > 200:
                self.data_buffer = self.data_buffer.tail(200).reset_index(drop=True)
            
            # Calculate momentum if we have enough data
            if len(self.data_buffer) >= 50:
                df_with_momentum = self.calculator.calculate_momentum_indicators(self.data_buffer)
                
                if 'composite_momentum' in df_with_momentum.columns:
                    latest_momentum = df_with_momentum['composite_momentum'].iloc[-1]
                    
                    # Update buffer
                    self.momentum_buffer.append(latest_momentum)
                    
                    # Keep only last 64 values
                    if len(self.momentum_buffer) > 64:
                        self.momentum_buffer = self.momentum_buffer[-64:]
                    
                    return latest_momentum
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating momentum: {e}")
            return None
    
    def predict(self):
        """Generate momentum prediction"""
        try:
            if len(self.momentum_buffer) < 64:
                return None
            
            # Predict using Chronos Bolt
            prediction = self.predictor.predict_next_momentum(
                self.momentum_buffer,
                model_name='bolt'
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
    
    def run(self):
        """Run real-time prediction loop"""
        global running
        
        self.initialize()
        
        logger.info("\n" + "="*70)
        logger.info(f"Real-Time Momentum Predictor - {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Update Interval: {self.interval}s")
        logger.info("="*70)
        logger.info("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        while running:
            start_time = time.time()
            iteration += 1
            
            try:
                # Fetch new data
                new_data = self.fetch_live_data()
                
                if new_data is not None and not new_data.empty:
                    # Update momentum
                    current_momentum = self.update_momentum(new_data)
                    
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] #{iteration}")
                    
                    if current_momentum is not None:
                        print(f"  Current Momentum: {current_momentum:.6f}")
                        
                        # Predict
                        prediction = self.predict()
                        
                        if prediction is not None:
                            print(f"  → Predicted Next Momentum: {prediction:.6f}")
                            
                            # Calculate direction
                            if prediction > current_momentum:
                                direction = "↑ BULLISH"
                                color = "\033[92m"  # Green
                            elif prediction < current_momentum:
                                direction = "↓ BEARISH"
                                color = "\033[91m"  # Red
                            else:
                                direction = "- NEUTRAL"
                                color = "\033[93m"  # Yellow
                            
                            print(f"  {color}{direction}\033[0m")
                        else:
                            print("  ⏳ Building prediction buffer...")
                    else:
                        print("  ⏳ Calculating momentum indicators...")
                else:
                    print(f"  ⚠ No data received")
                
                # Sleep until next iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                running = False
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(1)
        
        logger.info(f"\n✓ Stopped after {iteration} iterations")


def main():
    """Main entry point"""
    try:
        # Configuration
        symbol = 'SOLUSDT'
        timeframe = '15m'
        interval = 15  # seconds between updates
        
        logger.info("Starting real-time momentum prediction system...")
        logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}, Interval: {interval}s")
        
        # Initialize and run
        predictor = RealtimeMomentumPredictor(
            symbol=symbol,
            timeframe=timeframe,
            interval=interval
        )
        
        predictor.run()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

