#!/usr/bin/env python3
"""
Model Loader Module
Comprehensive model loading and management system including:
- Multiple model format support (PyTorch, scikit-learn, XGBoost, TensorFlow)
- Model caching and optimization
- Metadata extraction
- Version management
- Model validation
"""

import os
import json
import joblib
import torch
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelLoader:
    """Comprehensive model loading and management"""
    
    def __init__(self, cache_size: int = 10, device: str = 'auto'):
        """
        Initialize model loader
        
        Args:
            cache_size: Maximum number of models to cache
            device: Device for models ('auto', 'cpu', 'cuda')
        """
        self.cache_size = cache_size
        self.device = self._determine_device(device)
        self.loaded_models = {}
        self.model_metadata = {}
        self.model_paths = {}
        
        logger.info(f"Model loader initialized with device: {self.device}")
    
    def _determine_device(self, device_str: str) -> str:
        """Determine computation device"""
        if device_str == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_str
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        force_reload: bool = False
    ) -> Optional[Any]:
        """
        Load a trained model
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to model file
            force_reload: Force reload even if already cached
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Check if already loaded
            if model_name in self.loaded_models and not force_reload:
                logger.info(f"Model {model_name} already loaded (use force_reload=True to reload)")
                return self.loaded_models[model_name]
            
            # Validate file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load based on file extension
            model_data = None
            
            if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
                model_data = self._load_sklearn_model(model_path)
                
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                model_data = self._load_pytorch_model(model_path)
                
            elif model_path.endswith('.h5') or model_path.endswith('.keras'):
                model_data = self._load_keras_model(model_path)
                
            elif model_path.endswith('.onnx'):
                model_data = self._load_onnx_model(model_path)
                
            elif model_path.endswith('.pb'):
                model_data = self._load_tensorflow_model(model_path)
                
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return None
            
            if model_data is None:
                return None
            
            # Store in cache
            if len(self.loaded_models) >= self.cache_size:
                # Remove oldest model
                oldest_key = next(iter(self.loaded_models))
                logger.info(f"Removing oldest model {oldest_key} from cache")
                del self.loaded_models[oldest_key]
            
            self.loaded_models[model_name] = model_data
            self.model_paths[model_name] = model_path
            
            # Load metadata
            self._load_metadata(model_name, model_path)
            
            logger.info(f"Loaded model {model_name} from {model_path}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_sklearn_model(self, path: str) -> Optional[Any]:
        """Load scikit-learn/XGBoost model"""
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading sklearn model: {e}")
            return None
    
    def _load_pytorch_model(self, path: str) -> Optional[Any]:
        """Load PyTorch model"""
        try:
            model_data = torch.load(path, map_location=self.device)
            
            # If model_data is a dictionary with 'model_state_dict', reconstruct model
            if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                logger.debug("Found PyTorch model with state dict")
                return model_data
            else:
                # Model is already a module
                if hasattr(model_data, 'eval'):
                    model_data.eval()
                return model_data
                
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None
    
    def _load_keras_model(self, path: str) -> Optional[Any]:
        """Load Keras/TensorFlow model"""
        try:
            try:
                from tensorflow import keras
                return keras.models.load_model(path)
            except ImportError:
                logger.error("TensorFlow not installed")
                return None
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            return None
    
    def _load_onnx_model(self, path: str) -> Optional[Any]:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(path)
        except ImportError:
            logger.error("ONNX Runtime not installed")
            return None
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return None
    
    def _load_tensorflow_model(self, path: str) -> Optional[Any]:
        """Load TensorFlow SavedModel"""
        try:
            try:
                import tensorflow as tf
                return tf.saved_model.load(path)
            except ImportError:
                logger.error("TensorFlow not installed")
                return None
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return None
    
    def _load_metadata(self, model_name: str, model_path: str):
        """Load model metadata"""
        try:
            # Try to load metadata file
            metadata_path = model_path.replace('.pkl', '_metadata.json') \
                                     .replace('.pt', '_metadata.json') \
                                     .replace('.pth', '_metadata.json') \
                                     .replace('.h5', '_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
            else:
                # Create basic metadata
                self.model_metadata[model_name] = {
                    'name': model_name,
                    'path': model_path,
                    'format': os.path.splitext(model_path)[1][1:],
                    'loaded_at': json.dumps(torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else None)
                }
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.model_metadata[model_name] = {}
    
    def load_models_from_directory(self, directory: str, pattern: str = '*.pkl') -> Dict[str, bool]:
        """
        Load all models from directory
        
        Args:
            directory: Directory containing models
            pattern: File pattern to match
            
        Returns:
            Dictionary of model_name -> success status
        """
        try:
            results = {}
            model_files = list(Path(directory).glob(pattern))
            
            for model_file in model_files:
                model_name = model_file.stem
                success = self.load_model(model_name, str(model_file))
                results[model_name] = success is not None
            
            logger.info(f"Loaded {sum(results.values())} models from {directory}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading models from directory: {e}")
            return {}
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model by name"""
        return self.loaded_models.get(model_name)
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        return self.model_metadata.get(model_name, {})
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.info(f"Unloaded model {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def unload_all_models(self):
        """Unload all models from memory"""
        try:
            count = len(self.loaded_models)
            self.loaded_models.clear()
            logger.info(f"Unloaded {count} models")
        except Exception as e:
            logger.error(f"Error unloading all models: {e}")
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded model names"""
        return list(self.loaded_models.keys())
    
    def model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            info = {
                'name': model_name,
                'loaded': model_name in self.loaded_models,
                'metadata': self.get_model_metadata(model_name),
                'path': self.model_paths.get(model_name, 'N/A')
            }
            
            if model_name in self.loaded_models:
                model_data = self.loaded_models[model_name]
                info['type'] = type(model_data).__name__
                info['has_predict'] = hasattr(model_data, 'predict')
                info['has_forward'] = hasattr(model_data, 'forward')
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def validate_model(self, model_name: str) -> bool:
        """Validate that a model is properly loaded and functional"""
        try:
            if model_name not in self.loaded_models:
                logger.error(f"Model {model_name} not loaded")
                return False
            
            model = self.loaded_models[model_name]
            
            # Check if model has predict/forward method
            has_predict = hasattr(model, 'predict')
            has_forward = hasattr(model, 'forward')
            
            if not (has_predict or has_forward):
                logger.warning(f"Model {model_name} has no predict or forward method")
                return False
            
            logger.info(f"Model {model_name} validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False

