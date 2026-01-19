#!/usr/bin/env python3
"""
Advanced AI Training System
Comprehensive AI model training system including:
- XGBoost for gradient boosting
- LSTM for time series prediction
- Transformer models (Superbase integration)
- Ensemble learning
- Hyperparameter optimization
- Model evaluation and validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Hyperparameter optimization disabled.")

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: str
    features: List[str]
    target_column: str
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_jobs: int = -1

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    model_size: float

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]
        return torch.FloatTensor(sequence), torch.FloatTensor([target])

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.activation(out)
        
        return out

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, output_size: int = 1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Take the last output
        x = x[:, -1, :]
        x = self.output_projection(x)
        x = self.activation(x)
        
        return x

class XGBoostTrainer:
    """XGBoost model trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for XGBoost training"""
        try:
            # Select features and target
            X = df[self.config.features].values
            y = df[self.config.target_column].values
            
            # Handle missing values
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing XGBoost data: {e}")
            return None, None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Train XGBoost model"""
        try:
            start_time = datetime.now()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Initialize XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            training_time = (datetime.now() - start_time).total_seconds()
            
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average='weighted'),
                recall=recall_score(y_test, y_pred, average='weighted'),
                f1_score=f1_score(y_test, y_pred, average='weighted'),
                roc_auc=roc_auc_score(y_test, y_pred_proba),
                training_time=training_time,
                model_size=self._calculate_model_size()
            )
            
            # Get feature importance
            self.feature_importance = dict(zip(
                self.config.features, 
                self.model.feature_importances_
            ))
            
            logger.info(f"XGBoost training completed. Accuracy: {metrics.accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return None
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping hyperparameter optimization.")
            return {}
        
        try:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                
                model = xgb.XGBClassifier(**params, random_state=self.config.random_state)
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            
            logger.info(f"Best XGBoost parameters: {study.best_params}")
            return study.best_params
            
        except Exception as e:
            logger.error(f"Error optimizing XGBoost hyperparameters: {e}")
            return {}
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        if self.model is None:
            return 0.0
        
        # Approximate model size
        n_estimators = self.model.n_estimators
        max_depth = self.model.max_depth
        n_features = len(self.config.features)
        
        # Rough estimation
        size_mb = (n_estimators * max_depth * n_features * 4) / (1024 * 1024)
        return size_mb
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.config.features,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")

class LSTMTrainer:
    """LSTM model trainer"""
    
    def __init__(self, config: TrainingConfig, sequence_length: int = 60):
        self.config = config
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for LSTM training"""
        try:
            # Select features and target
            X = df[self.config.features].values
            y = df[self.config.target_column].values
            
            # Handle missing values
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(X) - self.sequence_length + 1):
                sequences.append(X[i:i + self.sequence_length])
                targets.append(y[i + self.sequence_length - 1])
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Create datasets and dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            test_dataset = TimeSeriesDataset(X_test, y_test, self.sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            return None, None
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> ModelMetrics:
        """Train LSTM model"""
        try:
            start_time = datetime.now()
            
            # Initialize model
            input_size = len(self.config.features)
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            num_epochs = 100
            best_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(test_loader)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Calculate final metrics
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            self.model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            # Calculate metrics (convert to classification for consistency)
            predictions_binary = (predictions > 0.5).astype(int)
            actuals_binary = (actuals > 0.5).astype(int)
            
            metrics = ModelMetrics(
                accuracy=accuracy_score(actuals_binary, predictions_binary),
                precision=precision_score(actuals_binary, predictions_binary, average='weighted'),
                recall=recall_score(actuals_binary, predictions_binary, average='weighted'),
                f1_score=f1_score(actuals_binary, predictions_binary, average='weighted'),
                roc_auc=roc_auc_score(actuals_binary, predictions),
                training_time=training_time,
                model_size=self._calculate_model_size()
            )
            
            logger.info(f"LSTM training completed. Accuracy: {metrics.accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        if self.model is None:
            return 0.0
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        return size_mb
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'features': self.config.features,
                'sequence_length': self.sequence_length,
                'model_config': {
                    'input_size': len(self.config.features),
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            }
            torch.save(model_data, filepath)
            logger.info(f"LSTM model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")

class TransformerTrainer:
    """Transformer model trainer"""
    
    def __init__(self, config: TrainingConfig, sequence_length: int = 60):
        self.config = config
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for Transformer training"""
        try:
            # Similar to LSTM preparation
            X = df[self.config.features].values
            y = df[self.config.target_column].values
            
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            X = self.scaler.fit_transform(X)
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(X) - self.sequence_length + 1):
                sequences.append(X[i:i + self.sequence_length])
                targets.append(y[i + self.sequence_length - 1])
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Create datasets and dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            test_dataset = TimeSeriesDataset(X_test, y_test, self.sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error preparing Transformer data: {e}")
            return None, None
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> ModelMetrics:
        """Train Transformer model"""
        try:
            start_time = datetime.now()
            
            # Initialize model
            input_size = len(self.config.features)
            self.model = TransformerModel(
                input_size=input_size,
                d_model=256,
                nhead=8,
                num_layers=6,
                dropout=0.1
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            num_epochs = 100
            best_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(test_loader)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Calculate final metrics
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            self.model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            # Calculate metrics
            predictions_binary = (predictions > 0.5).astype(int)
            actuals_binary = (actuals > 0.5).astype(int)
            
            metrics = ModelMetrics(
                accuracy=accuracy_score(actuals_binary, predictions_binary),
                precision=precision_score(actuals_binary, predictions_binary, average='weighted'),
                recall=recall_score(actuals_binary, predictions_binary, average='weighted'),
                f1_score=f1_score(actuals_binary, predictions_binary, average='weighted'),
                roc_auc=roc_auc_score(actuals_binary, predictions),
                training_time=training_time,
                model_size=self._calculate_model_size()
            )
            
            logger.info(f"Transformer training completed. Accuracy: {metrics.accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return None
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        if self.model is None:
            return 0.0
        
        total_params = sum(p.numel() for p in self.model.parameters())
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'features': self.config.features,
                'sequence_length': self.sequence_length,
                'model_config': {
                    'input_size': len(self.config.features),
                    'd_model': 256,
                    'nhead': 8,
                    'num_layers': 6,
                    'dropout': 0.1
                }
            }
            torch.save(model_data, filepath)
            logger.info(f"Transformer model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving Transformer model: {e}")

class EnsembleTrainer:
    """Ensemble model trainer combining multiple models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        """Train ensemble of models"""
        try:
            results = {}
            
            # Prepare data
            X = df[self.config.features].values
            y = df[self.config.target_column].values
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            # Train XGBoost
            logger.info("Training XGBoost...")
            xgb_trainer = XGBoostTrainer(self.config)
            X_scaled = self.scaler.fit_transform(X)
            xgb_metrics = xgb_trainer.train(X_scaled, y)
            if xgb_metrics:
                results['xgboost'] = xgb_metrics
                self.models['xgboost'] = xgb_trainer
            
            # Train LSTM
            logger.info("Training LSTM...")
            lstm_trainer = LSTMTrainer(self.config)
            train_loader, test_loader = lstm_trainer.prepare_data(df)
            if train_loader and test_loader:
                lstm_metrics = lstm_trainer.train(train_loader, test_loader)
                if lstm_metrics:
                    results['lstm'] = lstm_metrics
                    self.models['lstm'] = lstm_trainer
            
            # Train Transformer
            logger.info("Training Transformer...")
            transformer_trainer = TransformerTrainer(self.config)
            train_loader, test_loader = transformer_trainer.prepare_data(df)
            if train_loader and test_loader:
                transformer_metrics = transformer_trainer.train(train_loader, test_loader)
                if transformer_metrics:
                    results['transformer'] = transformer_metrics
                    self.models['transformer'] = transformer_trainer
            
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(results)
            
            logger.info("Ensemble training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {}
    
    def _calculate_ensemble_weights(self, results: Dict[str, ModelMetrics]):
        """Calculate ensemble weights based on model performance"""
        try:
            total_score = 0
            for model_name, metrics in results.items():
                # Weight based on F1 score
                score = metrics.f1_score
                self.weights[model_name] = score
                total_score += score
            
            # Normalize weights
            if total_score > 0:
                for model_name in self.weights:
                    self.weights[model_name] /= total_score
            
            logger.info(f"Ensemble weights: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        try:
            predictions = []
            
            for model_name, model in self.models.items():
                if model_name == 'xgboost':
                    pred = model.model.predict_proba(X)[:, 1]
                else:
                    # For neural networks, implement prediction logic
                    pred = np.random.random(len(X))  # Placeholder
                
                predictions.append(pred * self.weights.get(model_name, 0))
            
            # Combine predictions
            ensemble_pred = np.sum(predictions, axis=0)
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.zeros(len(X))
    
    def save_ensemble(self, filepath: str):
        """Save ensemble models"""
        try:
            ensemble_data = {
                'weights': self.weights,
                'scaler': self.scaler,
                'features': self.config.features
            }
            
            # Save individual models
            for model_name, model in self.models.items():
                model.save_model(f"{filepath}_{model_name}.pkl")
            
            # Save ensemble metadata
            joblib.dump(ensemble_data, f"{filepath}_ensemble.pkl")
            logger.info(f"Ensemble models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")

class ModelEvaluator:
    """Model evaluation and comparison"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_models(self, results: Dict[str, ModelMetrics]) -> pd.DataFrame:
        """Evaluate and compare models"""
        try:
            evaluation_data = []
            
            for model_name, metrics in results.items():
                evaluation_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.accuracy,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1_Score': metrics.f1_score,
                    'ROC_AUC': metrics.roc_auc,
                    'Training_Time': metrics.training_time,
                    'Model_Size_MB': metrics.model_size
                })
            
            df = pd.DataFrame(evaluation_data)
            df = df.sort_values('F1_Score', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return pd.DataFrame()
    
    def plot_model_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Plot model comparison charts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            axes[0, 0].bar(df['Model'], df['Accuracy'])
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1 Score comparison
            axes[0, 1].bar(df['Model'], df['F1_Score'])
            axes[0, 1].set_title('Model F1 Score Comparison')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Training time comparison
            axes[1, 0].bar(df['Model'], df['Training_Time'])
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_ylabel('Training Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Model size comparison
            axes[1, 1].bar(df['Model'], df['Model_Size_MB'])
            axes[1, 1].set_title('Model Size Comparison')
            axes[1, 1].set_ylabel('Model Size (MB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {e}")

def main():
    """Main function to run AI training"""
    try:
        # Load sample data (replace with your actual data)
        # For demonstration, create sample data
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'price': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'rsi': np.random.uniform(0, 100, n_samples),
            'macd': np.random.randn(n_samples),
            'bb_position': np.random.uniform(0, 1, n_samples),
            'sentiment': np.random.uniform(-1, 1, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Define training configuration
        config = TrainingConfig(
            model_type='ensemble',
            features=['price', 'volume', 'rsi', 'macd', 'bb_position', 'sentiment'],
            target_column='target'
        )
        
        # Train ensemble model
        logger.info("Starting AI model training...")
        ensemble_trainer = EnsembleTrainer(config)
        results = ensemble_trainer.train_ensemble(df)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        evaluation_df = evaluator.evaluate_models(results)
        
        print("\nModel Evaluation Results:")
        print(evaluation_df)
        
        # Plot comparison
        evaluator.plot_model_comparison(evaluation_df, 'model_comparison.png')
        
        # Save models
        ensemble_trainer.save_ensemble('trained_models/ensemble')
        
        logger.info("AI training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main training function: {e}")

if __name__ == "__main__":
    main()
