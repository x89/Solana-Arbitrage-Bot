#!/usr/bin/env python3
"""
Model Registry Module
Comprehensive model registry system including:
- Model registration and versioning
- Metadata management
- Model discovery and querying
- Performance tracking
- Model lifecycle management
- Support for LGMM and all modern models
"""

from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    name: str
    model_type: str
    version: str
    created_at: str
    performance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    features: List[str]
    description: str = ""
    use_cases: List[str] = None

class ModelRegistry:
    """Central registry for AI models"""
    
    def __init__(self, storage_path: str = "model_registry.json"):
        """
        Initialize model registry
        
        Args:
            storage_path: Path to store registry data
        """
        self.storage_path = storage_path
        self.models = {}
        self.model_metadata = {}
        self.model_versions = {}
        
        # Load existing registry if available
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from file"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.model_metadata = data.get('metadata', {})
                    self.model_versions = data.get('versions', {})
                logger.info(f"Loaded registry with {len(self.model_metadata)} models")
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            data = {
                'metadata': self.model_metadata,
                'versions': self.model_versions,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register(
        self,
        name: str,
        model_class: Any,
        metadata: Dict[str, Any] = None,
        version: str = "1.0.0"
    ):
        """
        Register a model
        
        Args:
            name: Model name
            model_class: Model class or instance
            metadata: Model metadata
            version: Model version
        """
        try:
            self.models[name] = model_class
            
            # Create metadata entry
            if metadata is None:
                metadata = {}
            
            metadata_entry = ModelMetadata(
                name=name,
                model_type=metadata.get('model_type', 'unknown'),
                version=version,
                created_at=datetime.now().isoformat(),
                performance=metadata.get('performance', {}),
                hyperparameters=metadata.get('hyperparameters', {}),
                features=metadata.get('features', []),
                description=metadata.get('description', ''),
                use_cases=metadata.get('use_cases', [])
            )
            
            self.model_metadata[name] = asdict(metadata_entry)
            
            # Track versions
            if name not in self.model_versions:
                self.model_versions[name] = []
            self.model_versions[name].append({
                'version': version,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata
            })
            
            self._save_registry()
            
            logger.info(f"Registered model: {name} v{version}")
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
    
    def register_lgmm(self):
        """Register LGMM model for market regime detection"""
        try:
            from lgmm_trainer import LGMMTrainer
            
            metadata = {
                'model_type': 'unsupervised_clustering',
                'description': 'Latent Gaussian Mixture Model for market regime detection',
                'use_cases': [
                    'Market regime identification',
                    'Volatility clustering',
                    'Trend state detection',
                    'SPY data analysis'
                ],
                'hyperparameters': {
                    'n_components': 3,
                    'covariance_type': 'full',
                    'init_params': 'kmeans',
                    'max_iter': 100,
                    'tol': 1e-3
                },
                'features': ['returns', 'volume_changes'],
                'performance': {}
            }
            
            self.register('lgmm', LGMMTrainer, metadata, version='1.0.0')
            logger.info("LGMM model registered")
            
        except ImportError:
            logger.warning("LGMM trainer not available")
        except Exception as e:
            logger.error(f"Error registering LGMM: {e}")
    
    def get(self, name: str) -> Any:
        """
        Get model by name
        
        Args:
            name: Model name
            
        Returns:
            Model class or instance
        """
        return self.models.get(name)
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get model metadata
        
        Args:
            name: Model name
            
        Returns:
            Model metadata dictionary
        """
        return self.model_metadata.get(name, {})
    
    def list_models(self) -> List[str]:
        """
        List all registered models
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def search_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search for models by criteria
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags/use cases
            
        Returns:
            List of matching model names
        """
        try:
            matching_models = []
            
            for name, metadata in self.model_metadata.items():
                if model_type and metadata.get('model_type') != model_type:
                    continue
                
                if tags:
                    use_cases = metadata.get('use_cases', [])
                    if not any(tag in str(use_cases) for tag in tags):
                        continue
                
                matching_models.append(name)
            
            return matching_models
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    def update_performance(self, name: str, performance: Dict[str, float]):
        """
        Update model performance metrics
        
        Args:
            name: Model name
            performance: Performance metrics
        """
        try:
            if name in self.model_metadata:
                self.model_metadata[name]['performance'].update(performance)
                self._save_registry()
                logger.info(f"Updated performance for {name}")
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all registered models
        
        Returns:
            DataFrame with model summary
        """
        try:
            summary_data = []
            
            for name, metadata in self.model_metadata.items():
                summary_data.append({
                    'name': name,
                    'model_type': metadata.get('model_type'),
                    'version': metadata.get('version'),
                    'created_at': metadata.get('created_at'),
                    'description': metadata.get('description', '')[:100]  # Truncate
                })
            
            df = pd.DataFrame(summary_data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return pd.DataFrame()
    
    def export_registry(self, filepath: str):
        """
        Export registry to file
        
        Args:
            filepath: Path to export file
        """
        try:
            export_data = {
                'models': list(self.models.keys()),
                'metadata': self.model_metadata,
                'versions': self.model_versions,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Registry exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a model
        
        Args:
            name: Model name
            
        Returns:
            True if successful
        """
        try:
            if name in self.models:
                del self.models[name]
                del self.model_metadata[name]
                self._save_registry()
                logger.info(f"Unregistered model: {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering model: {e}")
            return False

