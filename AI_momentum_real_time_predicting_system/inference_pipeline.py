#!/usr/bin/env python3
"""
Inference Pipeline Module
Comprehensive inference pipeline for prediction models including:
- Data preprocessing and transformation
- Model loading and management
- Batch inference processing
- Result post-processing
- Performance monitoring
- Caching and optimization
"""

import numpy as np
import pandas as pd
import torch
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferencePipeline:
    """Comprehensive inference pipeline for prediction models"""
    
    def __init__(self, cache_enabled: bool = True, cache_size: int = 1000):
        """
        Initialize inference pipeline
        
        Args:
            cache_enabled: Enable result caching
            cache_size: Maximum cache size
        """
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.cache_size = cache_size
        self.inference_stats = {
            'total_inferences': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
        
        logger.info("Inference pipeline initialized")
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        scaler: Any = None
    ) -> np.ndarray:
        """
        Preprocess data for inference
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            scaler: Optional pre-fitted scaler
            
        Returns:
            Preprocessed numpy array
        """
        try:
            # Select features
            feature_data = data[feature_columns].values
            
            # Scale if scaler provided
            if scaler is not None:
                feature_data = scaler.transform(feature_data)
            
            # Handle missing values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.debug(f"Preprocessed data shape: {feature_data.shape}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return np.array([])
    
    def load_model(self, model_path: str, model_type: str = 'sklearn'):
        """
        Load model from file
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('sklearn', 'pytorch', 'keras')
            
        Returns:
            Loaded model
        """
        try:
            if model_type == 'sklearn':
                import joblib
                model = joblib.load(model_path)
            elif model_type == 'pytorch':
                model = torch.load(model_path)
            elif model_type == 'keras':
                from tensorflow import keras
                model = keras.models.load_model(model_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        feature_columns: List[str],
        scaler: Any = None,
        model_type: str = 'sklearn',
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            model: Trained model
            data: Input DataFrame
            feature_columns: List of feature columns
            scaler: Optional scaler
            model_type: Type of model
            batch_size: Batch size for processing
            
        Returns:
            Predictions array
        """
        try:
            start_time = time.time()
            
            # Check cache
            cache_key = self._get_cache_key(data)
            if self.cache_enabled and cache_key in self.cache:
                self.inference_stats['cache_hits'] += 1
                logger.debug("Cache hit")
                return self.cache[cache_key]
            
            self.inference_stats['cache_misses'] += 1
            
            # Preprocess data
            processed_data = self.preprocess_data(data, feature_columns, scaler)
            
            if processed_data.size == 0:
                logger.error("Empty processed data")
                return np.array([])
            
            # Generate predictions
            if model_type == 'pytorch':
                predictions = self._predict_pytorch(model, processed_data, batch_size)
            elif model_type == 'keras':
                predictions = self._predict_keras(model, processed_data, batch_size)
            else:  # sklearn
                predictions = self._predict_sklearn(model, processed_data, batch_size)
            
            # Cache result
            if self.cache_enabled and len(self.cache) < self.cache_size:
                self.cache[cache_key] = predictions
            
            # Update stats
            inference_time = time.time() - start_time
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['total_time'] += inference_time
            
            logger.info(f"Inference completed in {inference_time:.4f}s")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return np.array([])
    
    def _predict_pytorch(self, model: torch.nn.Module, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Predict using PyTorch model"""
        try:
            model.eval()
            device = next(model.parameters()).device
            
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(data), batch_size):
                    batch = torch.FloatTensor(data[i:i+batch_size]).to(device)
                    batch_predictions = model(batch)
                    predictions.append(batch_predictions.cpu().numpy())
            
            return np.concatenate(predictions) if predictions else np.array([])
            
        except Exception as e:
            logger.error(f"Error in PyTorch prediction: {e}")
            return np.array([])
    
    def _predict_keras(self, model, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Predict using Keras model"""
        try:
            return model.predict(data, batch_size=batch_size, verbose=0)
            
        except Exception as e:
            logger.error(f"Error in Keras prediction: {e}")
            return np.array([])
    
    def _predict_sklearn(self, model, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Predict using sklearn model"""
        try:
            return model.predict(data)
            
        except Exception as e:
            logger.error(f"Error in sklearn prediction: {e}")
            return np.array([])
    
    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key from data"""
        try:
            # Use hash of data values
            return str(hash(data.values.tobytes()))
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(time.time())
    
    def postprocess_results(
        self,
        predictions: np.ndarray,
        inverse_transform: bool = True,
        scaler: Any = None
    ) -> np.ndarray:
        """
        Postprocess prediction results
        
        Args:
            predictions: Raw predictions
            inverse_transform: Whether to inverse transform
            scaler: Optional scaler for inverse transform
            
        Returns:
            Postprocessed predictions
        """
        try:
            # Inverse transform if needed
            if inverse_transform and scaler is not None:
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Clip predictions if needed
            predictions = np.clip(predictions, -1e6, 1e6)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error postprocessing results: {e}")
            return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        stats = self.inference_stats.copy()
        
        if stats['total_inferences'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_inferences']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_inferences']
        else:
            stats['avg_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear inference cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Inference cache cleared")
    
    def save_stats(self, filepath: str):
        """Save inference statistics"""
        try:
            stats = self.get_stats()
            stats['timestamp'] = datetime.now().isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Stats saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

def main():
    """Main function to demonstrate inference pipeline"""
    try:
        logger.info("=" * 60)
        logger.info("Inference Pipeline Demo")
        logger.info("=" * 60)
        
        # Initialize pipeline
        pipeline = InferencePipeline(cache_enabled=True)
        
        # Create sample data
        logger.info("Creating sample data...")
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        
        # Create dummy model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10)
        
        # Train model
        X = data.values
        y = np.random.rand(100)
        model.fit(X, y)
        
        # Run inference
        logger.info("Running inference...")
        predictions = pipeline.predict(
            model=model,
            data=data,
            feature_columns=['feature1', 'feature2', 'feature3'],
            model_type='sklearn'
        )
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Sample predictions: {predictions[:5]}")
        
        # Get stats
        stats = pipeline.get_stats()
        logger.info(f"Inference stats: {stats}")
        
        logger.info("=" * 60)
        logger.info("Inference Pipeline Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

