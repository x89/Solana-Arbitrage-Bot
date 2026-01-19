"""
Low-latency model predictor
Supports Chronos, TimesFM, Transformers
Optimized for real-time inference (<100ms target)
"""

import time
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Main prediction engine with multiple model support
    Optimized for low-latency inference
    """
    
    def __init__(self, model_path: str, model_type: str = "chronos_t5"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.device = self._get_device()
        self.inference_times = []
        
        self._load_model()
    
    def _get_device(self):
        """Get available device (GPU if available)"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _load_model(self):
        """Load model checkpoint"""
        try:
            if self.model_type == "chronos_t5":
                from chronos_forecasting import Chronos
                self.model = Chronos.from_pretrained("amazon/chronos-t5-base")
            
            elif self.model_type == "chronos_bolt":
                from chronos_forecasting import Chronos
                self.model = Chronos.from_pretrained("amazon/chronos-bolt-mini")
            
            elif self.model_type == "torchscript":
                # Load TorchScript model for fastest inference
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
            
            else:
                # Load regular PyTorch model
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = checkpoint['model']
                self.model.eval()
            
            self.model.to(self.device)
            logger.info(f"Loaded {self.model_type} model: {self.model_path}")
            
            # Warm up model
            self._warm_up()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _warm_up(self):
        """Warm up model with dummy input"""
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        # Run warm-up inferences
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        logger.info("Model warmed up")
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input for warm-up"""
        # Typical shape: (batch_size=1, seq_len, features)
        return torch.randn(1, 120, 5, device=self.device)  # OHLCV
    
    def predict(
        self,
        features: np.ndarray,
        horizon: int = 6,
        temperature: float = 1.0
    ) -> Dict:
        """
        Predict next horizon steps
        
        Args:
            features: Input features array (timesteps, features)
            horizon: Number of steps ahead to predict
            temperature: Sampling temperature for stochastic models
        
        Returns:
            Dict with predictions, confidence, metadata
        """
        start_time = time.time()
        
        # Convert to tensor
        if isinstance(features, np.ndarray):
            tensor = torch.from_numpy(features).float()
        else:
            tensor = features
        
        # Add batch dimension if needed
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(tensor)
        
        # Post-process prediction
        pred_np = prediction.cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Log latency
        if len(self.inference_times) % 100 == 0:
            logger.info(
                f"Latency - Median: {np.median(self.inference_times):.2f}ms, "
                f"P99: {np.percentile(self.inference_times, 99):.2f}ms"
            )
        
        return {
            'prediction': pred_np,
            'horizon': horizon,
            'latency_ms': inference_time,
            'timestamp': time.time()
        }
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            'mean': float(np.mean(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
            'max': float(np.max(times)),
            'min': float(np.min(times)),
            'count': len(times)
        }


class EnsemblePredictor:
    """
    Multi-model ensemble for consensus predictions
    Combines Chronos, TimesFM, and other models
    """
    
    def __init__(self, model_configs: List[Dict]):
        self.predictors = []
        
        for config in model_configs:
            predictor = ModelPredictor(
                model_path=config['path'],
                model_type=config['type']
            )
            self.predictors.append({
                'predictor': predictor,
                'weight': config.get('weight', 1.0),
                'name': config.get('name', 'unknown')
            })
        
        logger.info(f"Created ensemble with {len(self.predictors)} models")
    
    def predict(self, features: np.ndarray, horizon: int = 6) -> Dict:
        """
        Get ensemble prediction
        
        Args:
            features: Input features
            horizon: Prediction horizon
        
        Returns:
            Ensemble prediction with weights
        """
        predictions = []
        total_weight = 0.0
        
        # Get predictions from all models
        for model_info in self.predictors:
            pred_dict = model_info['predictor'].predict(features, horizon)
            weight = model_info['weight']
            
            predictions.append(pred_dict['prediction'] * weight)
            total_weight += weight
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        
        # Get latency stats
        latencies = [m['predictor'].get_latency_stats() for m in self.predictors]
        
        return {
            'prediction': ensemble_pred,
            'horizon': horizon,
            'individual_predictions': [p.tolist() for p in predictions],
            'model_latencies': latencies,
            'timestamp': time.time()
        }


class QuantizedPredictor:
    """
    CPU-optimized quantized model predictor
    Uses INT8 quantization for faster inference on CPU
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_quantized_model()
    
    def _load_quantized_model(self):
        """Load quantized model"""
        # Convert PyTorch model to TorchScript and quantize
        # This should be done offline during training
        # For now, load regular model and quantize at runtime
        
        try:
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.model.eval()
            logger.info("Loaded quantized model")
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
    
    def predict(self, features: np.ndarray, horizon: int = 6) -> Dict:
        """Quantized prediction"""
        # Same interface as regular predictor
        tensor = torch.from_numpy(features).float().unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            prediction = self.model(tensor)
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'prediction': prediction.numpy(),
            'horizon': horizon,
            'latency_ms': inference_time,
            'quantized': True
        }

