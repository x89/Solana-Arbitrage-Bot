#!/usr/bin/env python3
"""
Real-Time Momentum Prediction System
Runs continuous momentum predictions until stopped
"""

import numpy as np
import pandas as pd
import logging
import time
import signal
import sys
from datetime import datetime, timedelta
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

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def simulate_realtime_data(buffer, new_value=None):
    """Simulate new real-time momentum data"""
    if new_value is None:
        # Generate new random momentum value
        new_value = np.sin(time.time() * 0.1) + np.random.normal(0, 0.05)
    
    buffer.append(new_value)
    
    # Keep buffer at maximum size (64 for Chronos context)
    if len(buffer) > 64:
        buffer = buffer[-64:]
    
    return buffer


def main():
    """Main real-time prediction loop"""
    global running
    
    try:
        print("=" * 70)
        print("Real-Time Momentum Prediction System")
        print("=" * 70)
        print()
        
        # Import required modules
        logger.info("Loading modules...")
        from chronos_momentum_predictor import ChronosMomentumPredictor
        from momentum_calculator import MomentumCalculator
        from ensemble_manager import EnsembleManager
        
        logger.info("✓ Modules loaded")
        
        # Initialize predictor
        logger.info("\nInitializing Chronos predictor...")
        predictor = ChronosMomentumPredictor(
            context_length=64,
            prediction_horizon=24,
            model_type='bolt',  # Use Bolt for faster predictions
            device='auto'
        )
        logger.info("✓ Chronos predictor initialized")
        
        # Initialize ensemble manager
        ensemble_mgr = EnsembleManager()
        
        # Initialize momentum buffer
        momentum_buffer = []
        
        # Configuration
        prediction_interval = 15  # seconds between predictions
        min_buffer_size = 64  # Minimum data points needed
        
        logger.info(f"\nPrediction interval: {prediction_interval} seconds")
        logger.info(f"Minimum buffer size: {min_buffer_size} timesteps")
        logger.info("\n" + "=" * 70)
        logger.info("Starting real-time predictions...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70 + "\n")
        
        iteration = 0
        
        while running:
            start_time = time.time()
            iteration += 1
            
            try:
                # Simulate receiving new momentum data
                new_momentum = np.sin(time.time() * 0.1) + np.random.normal(0, 0.05)
                momentum_buffer = simulate_realtime_data(momentum_buffer, new_momentum)
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration #{iteration}")
                print(f"  Current momentum: {new_momentum:.6f}")
                print(f"  Buffer size: {len(momentum_buffer)}")
                
                # Check if we have enough data
                if len(momentum_buffer) >= min_buffer_size:
                    # Predict using Chronos Bolt (fast)
                    bolt_prediction = predictor.predict_next_momentum(
                        momentum_buffer,
                        model_name='bolt'
                    )
                    
                    print(f"  Chronos Bolt prediction: {bolt_prediction:.6f}")
                    
                    # If we have enough data, try ensemble predictions
                    if len(momentum_buffer) >= 96:
                        # Create DataFrame for ensemble prediction
                        df = pd.DataFrame({
                            'momentum': momentum_buffer[-min_buffer_size:]
                        })
                        
                        # Try to get T5 prediction as well
                        try:
                            t5_prediction = predictor.predict_next_momentum(
                                momentum_buffer,
                                model_name='t5'
                            )
                            print(f"  Chronos T5 prediction: {t5_prediction:.6f}")
                            
                            # Create ensemble prediction
                            all_predictions = {
                                'chronos_bolt': bolt_prediction,
                                'chronos_t5': t5_prediction
                            }
                            confidences = {
                                'chronos_bolt': 0.8,
                                'chronos_t5': 0.85
                            }
                            
                            ensemble_pred, ensemble_conf = ensemble_mgr.ensemble_predict(
                                all_predictions,
                                confidences
                            )
                            
                            print(f"  ✓ Ensemble prediction: {ensemble_pred:.6f}")
                            print(f"  ✓ Confidence: {ensemble_conf:.2f}")
                        except Exception as e:
                            logger.debug(f"T5 prediction skipped: {e}")
                            print(f"  ✓ Using Bolt only")
                    else:
                        print(f"  ✓ Using Bolt prediction (building buffer for ensemble...)")
                    
                else:
                    remaining = min_buffer_size - len(momentum_buffer)
                    print(f"  ⏳ Buffer building... need {remaining} more points")
                
                # Calculate time taken
                elapsed = time.time() - start_time
                print(f"  Time taken: {elapsed:.2f}s")
                
                # Sleep until next prediction
                sleep_time = max(0, prediction_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                running = False
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(1)  # Wait before retrying
        
        logger.info("\n" + "=" * 70)
        logger.info(f"Real-time prediction stopped after {iteration} iterations")
        logger.info("=" * 70)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install Chronos: pip install chronos-forecasting")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        sys.exit(0)

