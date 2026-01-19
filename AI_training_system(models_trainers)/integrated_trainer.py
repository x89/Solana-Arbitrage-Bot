#!/usr/bin/env python3
"""
AI Training System Integration Module
Integrates modern AI models with the existing training infrastructure
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from modern_ai_models import (
        TransformerTimeSeriesModel,
        TemporalFusionTransformer,
        NBeatsModel,
        MarketGraphNetwork,
        DQNAgent,
        ModernTrainer
    )
    MODERN_MODELS_AVAILABLE = True
except ImportError:
    MODERN_MODELS_AVAILABLE = False
    logger.warning("Modern AI models not available. Some features disabled.")

try:
    from advanced_ai_trainer import (
        LSTMModel,
        TimeSeriesDataset,
        TrainingConfig,
        XGBoostTrainer,
        LSTMTrainer,
        TransformerTrainer,
        EnsembleTrainer
    )
    ADVANCED_TRAINER_AVAILABLE = True
except ImportError:
    ADVANCED_TRAINER_AVAILABLE = False
    logger.warning("Advanced trainer not available. Some features disabled.")

try:
    from llm_rl_hybrid import LLMRLHybridTrainer, LLMConfig, RLConfig
    LLM_RL_AVAILABLE = True
except ImportError:
    LLM_RL_AVAILABLE = False
    logger.warning("LLM+RL Hybrid not available. Install dependencies.")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

class IntegratedModelTrainer:
    """Integrated training system combining traditional and modern models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        if ADVANCED_TRAINER_AVAILABLE:
            self.xgb_trainer = XGBoostTrainer
            self.lstm_trainer = LSTMTrainer
            self.transformer_trainer = TransformerTrainer
            self.ensemble_trainer = EnsembleTrainer
        else:
            self.xgb_trainer = None
            self.lstm_trainer = None
            self.transformer_trainer = None
            self.ensemble_trainer = None
        
        if MODERN_MODELS_AVAILABLE:
            self.modern_trainer = ModernTrainer(
                use_mlflow=self.config.get('use_mlflow', False),
                use_wandb=self.config.get('use_wandb', False),
                use_tensorboard=self.config.get('use_tensorboard', False)
            )
        else:
            self.modern_trainer = None
        
        # Model registry
        self.model_registry = {}
        
        if ADVANCED_TRAINER_AVAILABLE:
            self.model_registry['lstm'] = LSTMModel
            self.model_registry['xgb'] = None  # Will use advanced_ai_trainer
            self.model_registry['random_forest'] = None
        
        if MODERN_MODELS_AVAILABLE:
            self.model_registry['transformer'] = TransformerTimeSeriesModel
            self.model_registry['tft'] = TemporalFusionTransformer
            self.model_registry['nbeats'] = NBeatsModel
            self.model_registry['gnn'] = MarketGraphNetwork
            self.model_registry['dqn'] = DQNAgent
        
        if LLM_RL_AVAILABLE:
            self.model_registry['llm_rl_hybrid'] = LLMRLHybridTrainer
        
        # Initialize LLM+RL if available
        if LLM_RL_AVAILABLE:
            self.llm_rl_trainer = None
    
    def create_model(self, model_type: str, input_dim: int, **kwargs) -> nn.Module:
        """Create a model instance"""
        try:
            if model_type in ['xgb', 'random_forest']:
                # These are handled by advanced_ai_trainer
                return None
            
            if model_type not in self.model_registry:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model_class = self.model_registry[model_type]
            
            if model_type == 'lstm':
                return model_class(
                    input_size=input_dim,
                    hidden_size=kwargs.get('hidden_size', 128),
                    num_layers=kwargs.get('num_layers', 2),
                    dropout=kwargs.get('dropout', 0.2),
                    output_size=kwargs.get('output_size', 1)
                )
            
            elif model_type == 'transformer':
                return model_class(
                    input_dim=input_dim,
                    d_model=kwargs.get('d_model', 512),
                    nhead=kwargs.get('nhead', 8),
                    num_layers=kwargs.get('num_layers', 6),
                    dim_feedforward=kwargs.get('dim_feedforward', 2048),
                    dropout=kwargs.get('dropout', 0.1),
                    sequence_length=kwargs.get('sequence_length', 100),
                    prediction_horizon=kwargs.get('prediction_horizon', 24)
                )
            
            elif model_type == 'tft':
                return model_class(
                    input_dim=input_dim,
                    d_model=kwargs.get('d_model', 256),
                    nhead=kwargs.get('nhead', 4),
                    num_layers=kwargs.get('num_layers', 2),
                    quantiles=kwargs.get('quantiles', [0.1, 0.5, 0.9]),
                    dropout=kwargs.get('dropout', 0.1)
                )
            
            elif model_type == 'nbeats':
                return model_class(
                    input_dim=input_dim,
                    forecast_length=kwargs.get('forecast_length', 24),
                    backcast_length=kwargs.get('backcast_length', 168),
                    stack_types=kwargs.get('stack_types', ['trend', 'seasonality']),
                    n_blocks=kwargs.get('n_blocks', 3),
                    n_layers=kwargs.get('n_layers', 4),
                    layer_width=kwargs.get('layer_width', 512),
                    dropout=kwargs.get('dropout', 0.1)
                )
            
            elif model_type == 'gnn':
                return model_class(
                    input_dim=input_dim,
                    hidden_dim=kwargs.get('hidden_dim', 128),
                    output_dim=kwargs.get('output_dim', 64),
                    num_layers=kwargs.get('num_layers', 3),
                    dropout=kwargs.get('dropout', 0.1),
                    num_heads=kwargs.get('num_heads', 4)
                )
            
            elif model_type == 'dqn':
                return model_class(
                    state_dim=input_dim,
                    action_dim=kwargs.get('action_dim', 3),
                    hidden_dim=kwargs.get('hidden_dim', 128),
                    dropout=kwargs.get('dropout', 0.2)
                )
            
            logger.info(f"Created {model_type} model with input dimension {input_dim}")
            return model_class(input_dim, **kwargs)
            
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {e}")
            return None
    
    def train_model(self, model_type: str, data: pd.DataFrame, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
        """Train a model on data"""
        try:
            logger.info(f"Training {model_type} model...")
            
            # Prepare data
            if model_type in ['xgb', 'random_forest']:
                # Use advanced trainer for sklearn models
                return self.advanced_trainer.train_all_models(data, kwargs.get('symbol', 'UNKNOWN'))
            
            # For neural network models
            # Convert data to tensors
            X = data.drop(columns=['target']).values if 'target' in data.columns else data.values
            y = data['target'].values if 'target' in data.columns else data.iloc[:, -1].values
            
            # Create model
            model = self.create_model(model_type, X.shape[1], **kwargs)
            
            if model is None:
                return None, {}
            
            # Create dataset
            sequence_length = kwargs.get('sequence_length', 60)
            dataset = TimeSeriesDataset(X, y, sequence_length)
            
            # Create data loaders
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train with tracking
            trained_model = self.modern_trainer.train_with_tracking(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=kwargs.get('epochs', 100),
                learning_rate=kwargs.get('learning_rate', 0.001),
                device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            # Return metrics (simplified)
            metrics = {
                'model_type': model_type,
                'train_size': train_size,
                'val_size': val_size,
                'parameters': sum(p.numel() for p in trained_model.parameters())
            }
            
            logger.info(f"Successfully trained {model_type} model")
            
            return trained_model, metrics
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return None, {}
    
    def evaluate_model(self, model: nn.Module, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a trained model"""
        try:
            # Prepare data
            X = data.drop(columns=['target']).values if 'target' in data.columns else data.values
            y = data['target'].values if 'target' in data.columns else data.iloc[:, -1].values
            
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for i in range(len(X) - 60):
                    sequence = torch.FloatTensor(X[i:i+60]).unsqueeze(0)
                    actual = y[i+60]
                    
                    pred = model(sequence)
                    predictions.append(pred.item())
                    actuals.append(actual)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def compare_models(self, data: pd.DataFrame, model_types: List[str]) -> pd.DataFrame:
        """Compare multiple models"""
        try:
            results = []
            
            for model_type in model_types:
                logger.info(f"Evaluating {model_type}...")
                
                model, train_metrics = self.train_model(model_type, data)
                
                if model is not None:
                    eval_metrics = self.evaluate_model(model, data)
                    results.append({
                        'model': model_type,
                        **train_metrics,
                        **eval_metrics
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()

def main():
    """Main function to demonstrate integrated training"""
    try:
        # Initialize trainer
        config = {
            'use_mlflow': True,
            'use_wandb': True,
            'use_tensorboard': True
        }
        
        trainer = IntegratedModelTrainer(config)
        
        logger.info("Integrated training system initialized")
        logger.info("Modern models available:")
        logger.info("- Transformer: State-of-the-art transformer architecture")
        logger.info("- TFT: Temporal Fusion Transformer for interpretable forecasting")
        logger.info("- N-BEATS: Neural basis expansion")
        logger.info("- GNN: Graph neural network for market structure")
        logger.info("- DQN: Deep Q-Network for reinforcement learning")
        
    except Exception as e:
        logger.error(f"Error in integrated training system: {e}")

if __name__ == "__main__":
    main()

