#!/usr/bin/env python3
"""
Chronos Momentum Predictor Module
Integrates Chronos time series models for real-time momentum prediction
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from chronos import BaseChronosPipeline, ChronosBoltPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    logger.warning("Chronos not installed. Install with: pip install chronos-forecasting")


class ChronosMomentumPredictor:
    """Chronos-based momentum predictor for real-time forecasting"""
    
    def __init__(
        self,
        context_length: int = 128,
        prediction_horizon: int = 24,
        model_type: str = 'bolt',  # 'bolt' or 't5'
        device: str = 'auto'
    ):
        """
        Initialize Chronos Momentum Predictor
        
        Args:
            context_length: Number of past timesteps to use as input
            prediction_horizon: Number of future timesteps to predict
            model_type: 'bolt' (fast) or 't5' (accurate)
            device: 'cuda', 'cpu', or 'auto'
        """
        if not CHRONOS_AVAILABLE:
            logger.error("Chronos not available. Install with: pip install chronos-forecasting")
            raise ImportError("Chronos library not installed")
        
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.device = self._determine_device(device)
        
        # Load Chronos models
        self.chronos_t5 = None
        self.chronos_bolt = None
        self._load_models()
        
        logger.info(f"Chronos Momentum Predictor initialized (model={model_type}, device={self.device})")
    
    def _determine_device(self, device_str: str) -> str:
        """Determine computation device"""
        if device_str == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_str
    
    def _load_models(self):
        """Load Chronos models"""
        try:
            # Try to load from local checkpoints first
            chronos_t5_dir = os.getenv('CHRONOS_T5_DIR', './AI_fine-turning_system_forecasting_system/chronos_t5_ft')
            chronos_bolt_dir = os.getenv('CHRONOS_BOLT_DIR', './AI_fine-turning_system_forecasting_system/chronos_bolt_ft')
            
            # Load Chronos T5
            try:
                if os.path.isdir(chronos_t5_dir):
                    import glob
                    ckpts = sorted(glob.glob(os.path.join(chronos_t5_dir, "checkpoint-*/")))
                    load_dir = ckpts[-1] if ckpts else chronos_t5_dir
                    
                    self.chronos_t5 = BaseChronosPipeline.from_pretrained(
                        load_dir,
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                    )
                    logger.info(f"✓ Loaded Chronos T5 from local checkpoint: {load_dir}")
                else:
                    self.chronos_t5 = BaseChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-tiny" if self.device == 'cpu' else "amazon/chronos-t5-base",
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                    )
                    logger.info("✓ Loaded Chronos T5 from Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to load Chronos T5: {e}")
                self.chronos_t5 = None
            
            # Load Chronos Bolt
            try:
                if os.path.isdir(chronos_bolt_dir):
                    import glob
                    ckpts = sorted(glob.glob(os.path.join(chronos_bolt_dir, "checkpoint-*/")))
                    load_dir = ckpts[-1] if ckpts else chronos_bolt_dir
                    
                    self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
                        load_dir,
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                    )
                    logger.info(f"✓ Loaded Chronos Bolt from local checkpoint: {load_dir}")
                else:
                    self.chronos_bolt = ChronosBoltPipeline.from_pretrained(
                        "amazon/chronos-bolt-mini",
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                    )
                    logger.info("✓ Loaded Chronos Bolt from Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to load Chronos Bolt: {e}")
                self.chronos_bolt = None
            
        except Exception as e:
            logger.error(f"Error loading Chronos models: {e}")
    
    def prepare_momentum_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'composite_momentum'
    ) -> pd.DataFrame:
        """
        Prepare momentum data for Chronos
        
        Args:
            df: DataFrame with momentum features
            target_column: Column name for target momentum
            
        Returns:
            DataFrame in Chronos format with unique_id, ds, y columns
        """
        try:
            if target_column not in df.columns:
                logger.error(f"Target column {target_column} not found in DataFrame")
                return pd.DataFrame()
            
            # Create Chronos-format DataFrame
            chronos_df = pd.DataFrame({
                'unique_id': 'momentum',
                'ds': df.index if df.index.name else range(len(df)),
                'y': df[target_column].values
            })
            
            # Ensure ds is datetime
            chronos_df['ds'] = pd.to_datetime(chronos_df['ds'])
            
            # Remove any NaN values
            chronos_df = chronos_df.dropna()
            
            logger.debug(f"Prepared {len(chronos_df)} timesteps for Chronos")
            
            return chronos_df
            
        except Exception as e:
            logger.error(f"Error preparing momentum data: {e}")
            return pd.DataFrame()
    
    def predict_momentum(
        self,
        df: pd.DataFrame,
        target_column: str = 'composite_momentum',
        prediction_type: str = 'both'
    ) -> Dict[str, Any]:
        """
        Predict momentum using Chronos models
        
        Args:
            df: Input DataFrame with momentum data
            target_column: Column name for target momentum
            prediction_type: 'bolt', 't5', or 'both'
            
        Returns:
            Dictionary with predictions from selected models
        """
        try:
            # Prepare data for Chronos
            chronos_df = self.prepare_momentum_data(df, target_column)
            
            if len(chronos_df) < self.context_length:
                logger.error(f"Insufficient data: need at least {self.context_length} timesteps, got {len(chronos_df)}")
                return {}
            
            results = {}
            
            # Chronos Bolt prediction
            if prediction_type in ['bolt', 'both'] and self.chronos_bolt:
                try:
                    # Get context as torch tensor
                    context_values = chronos_df['y'].tail(self.context_length).values
                    context = torch.tensor(context_values, dtype=torch.float32)
                    
                    # Predict using correct API
                    bolt_pred = self.chronos_bolt.predict(context, self.prediction_horizon)
                    
                    # Extract median prediction
                    low, median, high = np.quantile(bolt_pred[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                    
                    results['bolt'] = {
                        'mean': float(median[-1]),
                        'median': median.tolist(),
                        'lower': low.tolist(),
                        'upper': high.tolist(),
                        'confidence': 0.8
                    }
                    logger.info("✓ Chronos Bolt prediction completed")
                except Exception as e:
                    logger.error(f"Error in Chronos Bolt prediction: {e}")
            
            # Chronos T5 prediction
            if prediction_type in ['t5', 'both'] and self.chronos_t5:
                try:
                    # Get context as torch tensor
                    context_values = chronos_df['y'].tail(self.context_length).values
                    context = torch.tensor(context_values, dtype=torch.float32)
                    
                    # Predict using correct API
                    t5_pred = self.chronos_t5.predict(context, self.prediction_horizon)
                    
                    # Extract median prediction
                    low, median, high = np.quantile(t5_pred[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                    
                    results['t5'] = {
                        'mean': float(median[-1]),
                        'median': median.tolist(),
                        'lower': low.tolist(),
                        'upper': high.tolist(),
                        'confidence': 0.85
                    }
                    logger.info("✓ Chronos T5 prediction completed")
                except Exception as e:
                    logger.error(f"Error in Chronos T5 prediction: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting momentum with Chronos: {e}")
            return {}
    
    def predict_next_momentum(
        self,
        momentum_series: List[float],
        model_name: str = 'bolt'
    ) -> float:
        """
        Quick prediction for next momentum value
        
        Args:
            momentum_series: List of recent momentum values
            model_name: 'bolt' or 't5'
            
        Returns:
            Predicted next momentum value
        """
        try:
            if len(momentum_series) < self.context_length:
                logger.error(f"Need at least {self.context_length} values, got {len(momentum_series)}")
                return 0.0
            
            # Get context as torch tensor
            context_values = momentum_series[-self.context_length:]
            context = torch.tensor(context_values, dtype=torch.float32)
            
            # Use appropriate model
            if model_name == 't5' and self.chronos_t5:
                model = self.chronos_t5
            elif model_name == 'bolt' and self.chronos_bolt:
                model = self.chronos_bolt
            else:
                logger.error(f"Model {model_name} not available")
                return 0.0
            
            # Predict using correct API
            prediction = model.predict(context, 1)  # Predict 1 step ahead
            
            # Extract median value
            low, median, high = np.quantile(prediction[0].numpy(), [0.1, 0.5, 0.9], axis=0)
            return float(median[0])  # Return first predicted value
                
        except Exception as e:
            logger.error(f"Error in quick momentum prediction: {e}")
            return 0.0
    
    def ensemble_predict(
        self,
        df: pd.DataFrame,
        target_column: str = 'composite_momentum',
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Ensemble prediction using both Bolt and T5
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            weights: Ensemble weights (default: 0.3 bolt, 0.7 t5)
            
        Returns:
            Ensemble prediction result
        """
        try:
            if weights is None:
                weights = {'bolt': 0.3, 't5': 0.7}
            
            # Get predictions from both models
            predictions = self.predict_momentum(df, target_column, 'both')
            
            if not predictions:
                return {'ensemble_prediction': 0.0, 'confidence': 0.0}
            
            # Calculate weighted ensemble
            ensemble_value = 0.0
            total_weight = 0.0
            
            for model_name, weight in weights.items():
                if model_name in predictions:
                    mean_val = predictions[model_name]['mean']
                    ensemble_value += mean_val * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_value /= total_weight
            else:
                ensemble_value = 0.0
            
            # Calculate ensemble confidence
            confidences = [pred['confidence'] for pred in predictions.values()]
            ensemble_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'ensemble_prediction': ensemble_value,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'ensemble_prediction': 0.0, 'confidence': 0.0}


def main():
    """Demonstrate Chronos momentum prediction"""
    try:
        print("=" * 70)
        print("Chronos Momentum Predictor Demo")
        print("=" * 70)
        
        if not CHRONOS_AVAILABLE:
            print("\n❌ Chronos library not installed")
            print("Install with: pip install chronos-forecasting")
            return
        
        # Initialize predictor
        predictor = ChronosMomentumPredictor(
            context_length=64,
            prediction_horizon=24,
            model_type='bolt'
        )
        
        # Create sample momentum data
        print("\nCreating sample momentum data...")
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')
        np.random.seed(42)
        
        # Generate momentum-like data (oscillating around 0)
        momentum_values = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
        
        df = pd.DataFrame({
            'composite_momentum': momentum_values
        }, index=dates)
        
        print(f"✓ Created {len(df)} timesteps of momentum data")
        
        # Test predictions
        print("\n" + "-" * 70)
        print("Testing Chronos Bolt prediction...")
        
        if predictor.chronos_bolt:
            result = predictor.predict_momentum(df, 'composite_momentum', 'bolt')
            if 'bolt' in result:
                print(f"✓ Bolt prediction: {result['bolt']['mean']:.4f}")
                print(f"  Confidence: {result['bolt']['confidence']:.2f}")
        
        # Test ensemble
        print("\n" + "-" * 70)
        print("Testing ensemble prediction...")
        ensemble_result = predictor.ensemble_predict(df, 'composite_momentum')
        
        if 'ensemble_prediction' in ensemble_result:
            print(f"✓ Ensemble prediction: {ensemble_result['ensemble_prediction']:.4f}")
            print(f"  Confidence: {ensemble_result['confidence']:.2f}")
        
        print("\n" + "=" * 70)
        print("✓ Chronos Momentum Predictor test completed!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in Chronos momentum demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

