#!/usr/bin/env python3
"""
AI Predicting Model Generating System
Advanced model generation system for real-time predictions including:
- Model ensemble generation
- Real-time prediction pipeline
- Model performance monitoring
- Adaptive model selection
- Prediction confidence scoring
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result structure"""
    symbol: str
    timestamp: datetime
    prediction: float
    confidence: float
    model_name: str
    features_used: List[str]
    prediction_type: str  # 'price', 'direction', 'volatility'
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: datetime
    prediction_count: int
    correct_predictions: int

class ModelLoader:
    """Model loading and management"""
    
    def __init__(self, models_directory: str = "trained_models"):
        self.models_directory = models_directory
        self.loaded_models = {}
        self.model_metadata = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_name: str, model_path: str) -> bool:
        """Load a trained model"""
        try:
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            if model_path.endswith('.pkl'):
                # Load scikit-learn/XGBoost model
                model_data = joblib.load(model_path)
                self.loaded_models[model_name] = model_data
                logger.info(f"Loaded {model_name} model from {model_path}")
                
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                # Load PyTorch model
                model_data = torch.load(model_path, map_location=self.device)
                self.loaded_models[model_name] = model_data
                logger.info(f"Loaded {model_name} PyTorch model from {model_path}")
                
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False
            
            # Load metadata if available
            metadata_path = model_path.replace('.pkl', '_metadata.json').replace('.pth', '_metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
            except FileNotFoundError:
                self.model_metadata[model_name] = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def load_all_models(self, model_configs: Dict[str, str]) -> Dict[str, bool]:
        """Load multiple models"""
        results = {}
        for model_name, model_path in model_configs.items():
            results[model_name] = self.load_model(model_name, model_path)
        return results
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model"""
        return self.loaded_models.get(model_name)
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        return self.model_metadata.get(model_name, {})

class FeatureEngineer:
    """Feature engineering for predictions"""
    
    def __init__(self):
        self.feature_scalers = {}
        self.feature_encoders = {}
        self.feature_importance = {}
    
    def prepare_features(self, data: pd.DataFrame, model_name: str, 
                        model_metadata: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        try:
            # Get required features from metadata
            required_features = model_metadata.get('features', [])
            
            if not required_features:
                logger.warning(f"No features specified for model {model_name}")
                return np.array([])
            
            # Select and validate features
            available_features = [f for f in required_features if f in data.columns]
            missing_features = [f for f in required_features if f not in data.columns]
            
            if missing_features:
                logger.warning(f"Missing features for {model_name}: {missing_features}")
            
            if not available_features:
                logger.error(f"No available features for model {model_name}")
                return np.array([])
            
            # Extract features
            X = data[available_features].values
            
            # Handle missing values
            X = np.nan_to_num(X)
            
            # Apply scaling if scaler is available
            scaler_key = f"{model_name}_scaler"
            if scaler_key in self.feature_scalers:
                X = self.feature_scalers[scaler_key].transform(X)
            elif 'scaler' in model_metadata:
                # Load scaler from model metadata
                scaler = model_metadata['scaler']
                X = scaler.transform(X)
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing features for {model_name}: {e}")
            return np.array([])
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as features"""
        try:
            df = data.copy()
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std_val = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume features
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_price_trend'] = df['volume'] * df['price_change']
            
            # Volatility features
            df['volatility_5'] = df['close'].rolling(5).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # Fill NaN values
            df = df.ffill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return data

class PredictionEngine:
    """Main prediction engine"""
    
    def __init__(self, model_loader: ModelLoader, feature_engineer: FeatureEngineer):
        self.model_loader = model_loader
        self.feature_engineer = feature_engineer
        self.prediction_history = []
        self.model_performance = {}
        self.confidence_threshold = 0.6
    
    def predict_price(self, data: pd.DataFrame, symbol: str, 
                     model_name: str = None) -> Optional[PredictionResult]:
        """Predict future price"""
        try:
            # Create technical features
            data_with_features = self.feature_engineer.create_technical_features(data)
            
            # Get latest data point
            latest_data = data_with_features.iloc[-1:].copy()
            
            # Prepare features
            model_metadata = self.model_loader.get_model_metadata(model_name)
            X = self.feature_engineer.prepare_features(latest_data, model_name, model_metadata)
            
            if X.size == 0:
                logger.error(f"No features available for prediction")
                return None
            
            # Get model
            model_data = self.model_loader.get_model(model_name)
            if not model_data:
                logger.error(f"Model {model_name} not loaded")
                return None
            
            # Make prediction
            if isinstance(model_data, dict) and 'model' in model_data:
                # XGBoost/scikit-learn model
                prediction = model_data['model'].predict(X)[0]
                prediction_proba = model_data['model'].predict_proba(X)[0]
                confidence = max(prediction_proba)
                
            elif isinstance(model_data, dict) and 'model_state_dict' in model_data:
                # PyTorch model
                model = self._reconstruct_pytorch_model(model_data)
                model.eval()
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.model_loader.device)
                    prediction = model(X_tensor).cpu().numpy()[0]
                    confidence = abs(prediction - 0.5) * 2  # Convert to confidence
                
            else:
                logger.error(f"Unknown model format for {model_name}")
                return None
            
            # Create prediction result
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=float(prediction),
                confidence=float(confidence),
                model_name=model_name,
                features_used=model_metadata.get('features', []),
                prediction_type='price',
                metadata={
                    'input_features': X.tolist(),
                    'model_type': type(model_data).__name__ if not isinstance(model_data, dict) else 'dict'
                }
            )
            
            # Store prediction
            self.prediction_history.append(result)
            
            logger.info(f"Price prediction for {symbol}: {prediction:.4f} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return None
    
    def predict_direction(self, data: pd.DataFrame, symbol: str, 
                         model_name: str = None) -> Optional[PredictionResult]:
        """Predict price direction (up/down)"""
        try:
            # Similar to price prediction but for direction
            data_with_features = self.feature_engineer.create_technical_features(data)
            latest_data = data_with_features.iloc[-1:].copy()
            
            model_metadata = self.model_loader.get_model_metadata(model_name)
            X = self.feature_engineer.prepare_features(latest_data, model_name, model_metadata)
            
            if X.size == 0:
                return None
            
            model_data = self.model_loader.get_model(model_name)
            if not model_data:
                return None
            
            # Make prediction
            if isinstance(model_data, dict) and 'model' in model_data:
                prediction = model_data['model'].predict(X)[0]
                prediction_proba = model_data['model'].predict_proba(X)[0]
                confidence = max(prediction_proba)
                
            elif isinstance(model_data, dict) and 'model_state_dict' in model_data:
                model = self._reconstruct_pytorch_model(model_data)
                model.eval()
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.model_loader.device)
                    prediction = model(X_tensor).cpu().numpy()[0]
                    confidence = abs(prediction - 0.5) * 2
            
            # Convert to direction
            direction = 1 if prediction > 0.5 else 0  # 1 = up, 0 = down
            
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=float(direction),
                confidence=float(confidence),
                model_name=model_name,
                features_used=model_metadata.get('features', []),
                prediction_type='direction',
                metadata={
                    'raw_prediction': float(prediction),
                    'direction': 'up' if direction == 1 else 'down'
                }
            )
            
            self.prediction_history.append(result)
            
            logger.info(f"Direction prediction for {symbol}: {'UP' if direction == 1 else 'DOWN'} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting direction for {symbol}: {e}")
            return None
    
    def predict_volatility(self, data: pd.DataFrame, symbol: str, 
                          model_name: str = None) -> Optional[PredictionResult]:
        """Predict future volatility"""
        try:
            # Calculate historical volatility
            returns = data['close'].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Use volatility prediction model if available
            data_with_features = self.feature_engineer.create_technical_features(data)
            latest_data = data_with_features.iloc[-1:].copy()
            
            model_metadata = self.model_loader.get_model_metadata(model_name)
            X = self.feature_engineer.prepare_features(latest_data, model_name, model_metadata)
            
            if X.size == 0:
                # Fallback to historical volatility
                prediction = historical_vol
                confidence = 0.5
            else:
                model_data = self.model_loader.get_model(model_name)
                if model_data and isinstance(model_data, dict) and 'model' in model_data:
                    prediction = model_data['model'].predict(X)[0]
                    prediction_proba = model_data['model'].predict_proba(X)[0]
                    confidence = max(prediction_proba)
                else:
                    prediction = historical_vol
                    confidence = 0.5
            
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=float(prediction),
                confidence=float(confidence),
                model_name=model_name,
                features_used=model_metadata.get('features', []),
                prediction_type='volatility',
                metadata={
                    'historical_volatility': float(historical_vol),
                    'prediction_method': 'model' if X.size > 0 else 'historical'
                }
            )
            
            self.prediction_history.append(result)
            
            logger.info(f"Volatility prediction for {symbol}: {prediction:.4f} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting volatility for {symbol}: {e}")
            return None
    
    def _reconstruct_pytorch_model(self, model_data: Dict[str, Any]) -> nn.Module:
        """Reconstruct PyTorch model from saved data"""
        try:
            model_config = model_data.get('model_config', {})
            
            # Define LSTM model class
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=dropout if num_layers > 1 else 0)
                    self.fc = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            
            # Define Transformer model class
            class TransformerModel(nn.Module):
                def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.1):
                    super().__init__()
                    self.input_projection = nn.Linear(input_size, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 
                                                              dropout, batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    self.fc = nn.Linear(d_model, 1)
                
                def forward(self, x):
                    x = self.input_projection(x)
                    transformer_out = self.transformer(x)
                    return self.fc(transformer_out[:, -1, :])
            
            # Create appropriate model
            if 'input_size' in model_config and 'hidden_size' in model_config:
                model = LSTMModel(**model_config)
            elif 'input_size' in model_config and 'd_model' in model_config:
                model = TransformerModel(**model_config)
            else:
                raise ValueError("Unknown model configuration")
            
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.model_loader.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Error reconstructing PyTorch model: {e}")
            return None

class EnsemblePredictor:
    """Ensemble prediction combining multiple models"""
    
    def __init__(self, prediction_engine: PredictionEngine):
        self.prediction_engine = prediction_engine
        self.ensemble_weights = {}
        self.model_performance_tracker = {}
    
    def predict_ensemble(self, data: pd.DataFrame, symbol: str, 
                        prediction_type: str = 'price') -> Optional[PredictionResult]:
        """Make ensemble prediction using multiple models"""
        try:
            # Get all available models
            available_models = list(self.prediction_engine.model_loader.loaded_models.keys())
            
            if not available_models:
                logger.error("No models available for ensemble prediction")
                return None
            
            # Get predictions from all models
            predictions = []
            confidences = []
            model_names = []
            
            for model_name in available_models:
                try:
                    if prediction_type == 'price':
                        result = self.prediction_engine.predict_price(data, symbol, model_name)
                    elif prediction_type == 'direction':
                        result = self.prediction_engine.predict_direction(data, symbol, model_name)
                    elif prediction_type == 'volatility':
                        result = self.prediction_engine.predict_volatility(data, symbol, model_name)
                    else:
                        logger.error(f"Unknown prediction type: {prediction_type}")
                        continue
                    
                    if result and result.confidence >= self.prediction_engine.confidence_threshold:
                        predictions.append(result.prediction)
                        confidences.append(result.confidence)
                        model_names.append(model_name)
                        
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if not predictions:
                logger.error("No valid predictions for ensemble")
                return None
            
            # Calculate ensemble weights based on model performance
            weights = self._calculate_ensemble_weights(model_names)
            
            # Weighted ensemble prediction
            weighted_prediction = sum(p * w for p, w in zip(predictions, weights))
            ensemble_confidence = np.average(confidences, weights=weights)
            
            # Create ensemble result
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=float(weighted_prediction),
                confidence=float(ensemble_confidence),
                model_name='ensemble',
                features_used=list(set().union(*[self.prediction_engine.model_loader.get_model_metadata(name).get('features', []) for name in model_names])),
                prediction_type=prediction_type,
                metadata={
                    'individual_predictions': dict(zip(model_names, predictions)),
                    'individual_confidences': dict(zip(model_names, confidences)),
                    'ensemble_weights': dict(zip(model_names, weights)),
                    'model_count': len(model_names)
                }
            )
            
            logger.info(f"Ensemble prediction for {symbol}: {weighted_prediction:.4f} (confidence: {ensemble_confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return None
    
    def _calculate_ensemble_weights(self, model_names: List[str]) -> List[float]:
        """Calculate ensemble weights based on model performance"""
        try:
            weights = []
            
            for model_name in model_names:
                if model_name in self.model_performance_tracker:
                    # Use performance-based weights
                    performance = self.model_performance_tracker[model_name]
                    weight = performance.get('f1_score', 0.5)
                else:
                    # Default equal weights
                    weight = 1.0 / len(model_names)
                
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
            return [1.0 / len(model_names)] * len(model_names)
    
    def update_model_performance(self, model_name: str, actual_value: float, 
                               predicted_value: float, prediction_type: str):
        """Update model performance tracking"""
        try:
            if model_name not in self.model_performance_tracker:
                self.model_performance_tracker[model_name] = {
                    'predictions': [],
                    'actuals': [],
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'prediction_count': 0,
                    'correct_predictions': 0
                }
            
            tracker = self.model_performance_tracker[model_name]
            tracker['predictions'].append(predicted_value)
            tracker['actuals'].append(actual_value)
            tracker['prediction_count'] += 1
            
            # Calculate accuracy
            if prediction_type == 'direction':
                predicted_direction = 1 if predicted_value > 0.5 else 0
                actual_direction = 1 if actual_value > 0.5 else 0
                
                if predicted_direction == actual_direction:
                    tracker['correct_predictions'] += 1
                
                tracker['accuracy'] = tracker['correct_predictions'] / tracker['prediction_count']
            
            # Keep only recent predictions (last 100)
            if len(tracker['predictions']) > 100:
                tracker['predictions'] = tracker['predictions'][-100:]
                tracker['actuals'] = tracker['actuals'][-100:]
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

class PredictionMonitor:
    """Monitor prediction performance and model health"""
    
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize prediction monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                prediction REAL NOT NULL,
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                actual_value REAL,
                accuracy REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                prediction_count INTEGER NOT NULL,
                last_updated DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction: PredictionResult, actual_value: float = None):
        """Log prediction for monitoring"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            accuracy = None
            if actual_value is not None:
                if prediction.prediction_type == 'direction':
                    predicted_direction = 1 if prediction.prediction > 0.5 else 0
                    actual_direction = 1 if actual_value > 0.5 else 0
                    accuracy = 1.0 if predicted_direction == actual_direction else 0.0
                else:
                    # For price/volatility predictions, calculate relative accuracy
                    error = abs(prediction.prediction - actual_value) / actual_value
                    accuracy = max(0, 1 - error)
            
            cursor.execute('''
                INSERT INTO predictions (symbol, timestamp, prediction, confidence, model_name, prediction_type, actual_value, accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (prediction.symbol, prediction.timestamp, prediction.prediction, 
                 prediction.confidence, prediction.model_name, prediction.prediction_type, 
                 actual_value, accuracy))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def get_model_performance(self, model_name: str, days: int = 30) -> Dict[str, float]:
        """Get model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(accuracy) as avg_accuracy, COUNT(*) as prediction_count,
                       AVG(confidence) as avg_confidence
                FROM predictions 
                WHERE model_name = ? AND timestamp >= datetime('now', '-{} days')
            '''.format(days), (model_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] is not None:
                return {
                    'accuracy': result[0],
                    'prediction_count': result[1],
                    'avg_confidence': result[2]
                }
            else:
                return {
                    'accuracy': 0.0,
                    'prediction_count': 0,
                    'avg_confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'accuracy': 0.0, 'prediction_count': 0, 'avg_confidence': 0.0}

def main():
    """Main function to demonstrate prediction system"""
    try:
        # Initialize components
        model_loader = ModelLoader()
        feature_engineer = FeatureEngineer()
        prediction_engine = PredictionEngine(model_loader, feature_engineer)
        ensemble_predictor = EnsemblePredictor(prediction_engine)
        monitor = PredictionMonitor()
        
        # Load models (example paths - adjust based on your actual model files)
        model_configs = {
            'xgboost_model': 'trained_models/xgboost_model.pkl',
            'lstm_model': 'trained_models/lstm_model.pth',
            'transformer_model': 'trained_models/transformer_model.pth'
        }
        
        # Load models
        load_results = model_loader.load_all_models(model_configs)
        logger.info(f"Model loading results: {load_results}")
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98
        })
        
        # Test predictions
        symbol = 'SOLUSDT'
        
        # Individual model predictions
        for model_name in model_loader.loaded_models.keys():
            logger.info(f"Testing {model_name}...")
            
            # Price prediction
            price_result = prediction_engine.predict_price(sample_data, symbol, model_name)
            if price_result:
                monitor.log_prediction(price_result)
            
            # Direction prediction
            direction_result = prediction_engine.predict_direction(sample_data, symbol, model_name)
            if direction_result:
                monitor.log_prediction(direction_result)
        
        # Ensemble prediction
        logger.info("Testing ensemble prediction...")
        ensemble_result = ensemble_predictor.predict_ensemble(sample_data, symbol, 'price')
        if ensemble_result:
            monitor.log_prediction(ensemble_result)
        
        # Get performance metrics
        for model_name in model_loader.loaded_models.keys():
            performance = monitor.get_model_performance(model_name)
            logger.info(f"{model_name} performance: {performance}")
        
        logger.info("Prediction system test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main prediction function: {e}")

if __name__ == "__main__":
    main()
