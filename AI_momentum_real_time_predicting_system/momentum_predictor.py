#!/usr/bin/env python3
"""
AI Momentum Real-time Predicting System
Advanced momentum prediction system including:
- Real-time momentum calculation
- AI-based momentum forecasting
- Multi-timeframe momentum analysis
- Momentum-based trading signals
- Performance monitoring and optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not installed. Technical indicators may be limited.")
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MomentumData:
    """Momentum data structure"""
    timestamp: datetime
    symbol: str
    timeframe: str
    price: float
    momentum_value: float
    momentum_direction: str
    momentum_strength: float
    technical_indicators: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class MomentumPrediction:
    """Momentum prediction result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    predicted_momentum: float
    confidence: float
    prediction_horizon: int  # periods ahead
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any]

class MomentumDataset(Dataset):
    """PyTorch dataset for momentum prediction"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 10):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of features
        feature_sequence = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]
        
        return {
            'features': torch.FloatTensor(feature_sequence),
            'target': torch.FloatTensor([target])
        }

class MomentumLSTM(nn.Module):
    """LSTM model for momentum prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super(MomentumLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class MomentumTransformer(nn.Module):
    """Transformer model for momentum prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super(MomentumTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use the last output
        last_output = transformer_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class MomentumCalculator:
    """Calculate various momentum indicators"""
    
    def __init__(self):
        self.momentum_indicators = {}
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive momentum indicators"""
        try:
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for momentum calculation")
                return df
            
            df_copy = df.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_copy.columns:
                    logger.error(f"Required column {col} not found")
                    return df
            
            # Convert to numpy arrays for talib
            open_prices = df_copy['open'].values
            high_prices = df_copy['high'].values
            low_prices = df_copy['low'].values
            close_prices = df_copy['close'].values
            volume = df_copy['volume'].values
            
            # Price momentum
            df_copy = self._calculate_price_momentum(df_copy, close_prices)
            
            # Volume momentum
            df_copy = self._calculate_volume_momentum(df_copy, volume)
            
            # Technical momentum indicators
            df_copy = self._calculate_technical_momentum(df_copy, high_prices, low_prices, close_prices)
            
            # Custom momentum indicators
            df_copy = self._calculate_custom_momentum(df_copy)
            
            logger.info(f"Calculated {len([col for col in df_copy.columns if col not in df.columns])} momentum indicators")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return df
    
    def _calculate_price_momentum(self, df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate price-based momentum indicators"""
        try:
            if not TALIB_AVAILABLE:
                logger.warning("TA-Lib not available - using basic momentum calculations")
                # Basic fallback calculations
                periods = [1, 3, 5, 10, 20]
                for period in periods:
                    df[f'roc_{period}'] = ((close_prices / pd.Series(close_prices).shift(period)) - 1) * 100
                    df[f'momentum_{period}'] = close_prices - pd.Series(close_prices).shift(period)
                
                df['price_velocity'] = df['roc_5'].diff()
                df['price_acceleration'] = df['price_velocity'].diff()
                df['exp_momentum_5'] = close_prices / pd.Series(close_prices).ewm(span=5).mean() - 1
                df['exp_momentum_20'] = close_prices / pd.Series(close_prices).ewm(span=20).mean() - 1
                return df
            
            # Rate of Change
            periods = [1, 3, 5, 10, 20]
            for period in periods:
                df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
            
            # Momentum
            for period in periods:
                df[f'momentum_{period}'] = talib.MOM(close_prices, timeperiod=period)
            
            # Price velocity (rate of change of rate of change)
            df['price_velocity'] = df['roc_5'].diff()
            
            # Price acceleration
            df['price_acceleration'] = df['price_velocity'].diff()
            
            # Exponential momentum
            df['exp_momentum_5'] = close_prices / talib.EMA(close_prices, timeperiod=5) - 1
            df['exp_momentum_20'] = close_prices / talib.EMA(close_prices, timeperiod=20) - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return df
    
    def _calculate_volume_momentum(self, df: pd.DataFrame, volume: np.ndarray) -> pd.DataFrame:
        """Calculate volume-based momentum indicators"""
        try:
            if not TALIB_AVAILABLE:
                # Basic fallback for volume momentum
                periods = [5, 10, 20]
                for period in periods:
                    df[f'volume_roc_{period}'] = ((volume / pd.Series(volume).shift(period)) - 1) * 100
                    df[f'volume_momentum_{period}'] = volume - pd.Series(volume).shift(period)
                
                df['volume_ma_ratio_5'] = volume / pd.Series(volume).rolling(5).mean()
                df['volume_ma_ratio_20'] = volume / pd.Series(volume).rolling(20).mean()
                df['volume_velocity'] = df['volume_roc_5'].diff()
                return df
            
            # Volume Rate of Change
            periods = [5, 10, 20]
            for period in periods:
                df[f'volume_roc_{period}'] = talib.ROC(volume, timeperiod=period)
            
            # Volume momentum
            for period in periods:
                df[f'volume_momentum_{period}'] = talib.MOM(volume, timeperiod=period)
            
            # Volume moving average ratio
            df['volume_ma_ratio_5'] = volume / talib.SMA(volume, timeperiod=5)
            df['volume_ma_ratio_20'] = volume / talib.SMA(volume, timeperiod=20)
            
            # Volume velocity
            df['volume_velocity'] = df['volume_roc_5'].diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {e}")
            return df
    
    def _calculate_technical_momentum(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """Calculate technical momentum indicators"""
        try:
            if not TALIB_AVAILABLE:
                # Skip technical indicators if TA-Lib not available
                logger.warning("Technical indicators skipped - TA-Lib not available")
                return df
            
            # RSI momentum
            rsi_14 = talib.RSI(close_prices, timeperiod=14)
            df['rsi_14'] = rsi_14
            df['rsi_momentum'] = rsi_14.diff()
            df['rsi_acceleration'] = df['rsi_momentum'].diff()
            
            # MACD momentum
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            df['macd_momentum'] = macd_hist.diff()
            
            # Stochastic momentum
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            df['stoch_momentum'] = slowk.diff()
            
            # Williams %R momentum
            williams_r = talib.WILLR(high_prices, low_prices, close_prices)
            df['williams_r'] = williams_r
            df['williams_momentum'] = williams_r.diff()
            
            # CCI momentum
            cci = talib.CCI(high_prices, low_prices, close_prices)
            df['cci'] = cci
            df['cci_momentum'] = cci.diff()
            
            # ADX momentum
            adx = talib.ADX(high_prices, low_prices, close_prices)
            df['adx'] = adx
            df['adx_momentum'] = adx.diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical momentum: {e}")
            return df
    
    def _calculate_custom_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom momentum indicators"""
        try:
            # Composite momentum score
            momentum_indicators = [
                'roc_5', 'roc_10', 'momentum_5', 'momentum_10',
                'rsi_momentum', 'macd_momentum', 'stoch_momentum'
            ]
            
            available_indicators = [col for col in momentum_indicators if col in df.columns]
            
            if available_indicators:
                # Normalize indicators
                normalized_indicators = df[available_indicators].apply(
                    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x
                )
                
                # Calculate composite momentum
                df['composite_momentum'] = normalized_indicators.mean(axis=1)
                
                # Momentum strength
                df['momentum_strength'] = normalized_indicators.std(axis=1)
            
            # Momentum divergence
            df['momentum_divergence'] = self._calculate_momentum_divergence(df)
            
            # Momentum regime
            df['momentum_regime'] = self._calculate_momentum_regime(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating custom momentum: {e}")
            return df
    
    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum divergence"""
        try:
            if 'close' not in df.columns or 'composite_momentum' not in df.columns:
                return pd.Series([0] * len(df), index=df.index)
            
            # Calculate price and momentum trends
            price_trend = df['close'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            momentum_trend = df['composite_momentum'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Calculate divergence
            divergence = price_trend - momentum_trend
            
            return divergence.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum divergence: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum regime"""
        try:
            if 'composite_momentum' not in df.columns:
                return pd.Series(['neutral'] * len(df), index=df.index)
            
            momentum = df['composite_momentum']
            
            # Define regimes based on momentum values
            regime = pd.Series(['neutral'] * len(df), index=df.index)
            regime[momentum > 0.5] = 'strong_bullish'
            regime[(momentum > 0.1) & (momentum <= 0.5)] = 'weak_bullish'
            regime[(momentum >= -0.1) & (momentum <= 0.1)] = 'neutral'
            regime[(momentum >= -0.5) & (momentum < -0.1)] = 'weak_bearish'
            regime[momentum < -0.5] = 'strong_bearish'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating momentum regime: {e}")
            return pd.Series(['neutral'] * len(df), index=df.index)

class MomentumPredictor:
    """AI-based momentum predictor"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.model_configs = {
            'lstm': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3},
            'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1},
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'lightgbm': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'random_forest': {'n_estimators': 100, 'max_depth': 10}
        }
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'composite_momentum') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        try:
            # Select feature columns
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', target_column]]
            
            if not feature_columns:
                logger.warning("No feature columns found")
                return np.array([]), np.array([])
            
            # Remove rows with NaN values
            df_clean = df[feature_columns + [target_column]].dropna()
            
            if len(df_clean) < 50:
                logger.warning("Insufficient clean data for training")
                return np.array([]), np.array([])
            
            # Separate features and target
            X = df_clean[feature_columns].values
            y = df_clean[target_column].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.scalers['features'] = scaler
            self.feature_columns = feature_columns
            
            logger.info(f"Prepared {len(X_scaled)} samples with {len(feature_columns)} features")
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), np.array([])
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10) -> Dict[str, float]:
        """Train LSTM model for momentum prediction"""
        try:
            logger.info("Training LSTM model...")
            
            # Prepare sequence data
            X_seq, y_seq = self._create_sequences(X, y, sequence_length)
            
            if len(X_seq) == 0:
                logger.warning("No sequence data available for LSTM training")
                return {}
            
            # Create dataset and dataloader
            dataset = MomentumDataset(X_seq, y_seq, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = X.shape[1]
            model = MomentumLSTM(
                input_size=input_size,
                hidden_size=self.model_configs['lstm']['hidden_size'],
                num_layers=self.model_configs['lstm']['num_layers'],
                dropout=self.model_configs['lstm']['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(100):
                # Training
                model.train()
                train_loss = 0
                
                for batch in train_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        features = batch['features'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_lstm_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(torch.load('best_lstm_model.pth'))
            
            # Evaluate model
            metrics = self._evaluate_model(model, val_loader)
            
            self.models['lstm'] = model
            
            logger.info(f"LSTM training completed. Validation MSE: {metrics.get('mse', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}
    
    def train_transformer_model(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10) -> Dict[str, float]:
        """Train Transformer model for momentum prediction"""
        try:
            logger.info("Training Transformer model...")
            
            # Prepare sequence data
            X_seq, y_seq = self._create_sequences(X, y, sequence_length)
            
            if len(X_seq) == 0:
                logger.warning("No sequence data available for Transformer training")
                return {}
            
            # Create dataset and dataloader
            dataset = MomentumDataset(X_seq, y_seq, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = X.shape[1]
            model = MomentumTransformer(
                input_size=input_size,
                d_model=self.model_configs['transformer']['d_model'],
                nhead=self.model_configs['transformer']['nhead'],
                num_layers=self.model_configs['transformer']['num_layers'],
                dropout=self.model_configs['transformer']['dropout']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(100):
                # Training
                model.train()
                train_loss = 0
                
                for batch in train_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        features = batch['features'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_transformer_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(torch.load('best_transformer_model.pth'))
            
            # Evaluate model
            metrics = self._evaluate_model(model, val_loader)
            
            self.models['transformer'] = model
            
            logger.info(f"Transformer training completed. Validation MSE: {metrics.get('mse', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return {}
    
    def train_sklearn_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train sklearn-based models"""
        try:
            logger.info("Training sklearn models...")
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            models = {}
            metrics = {}
            
            # XGBoost
            try:
                xgb_model = xgb.XGBRegressor(**self.model_configs['xgboost'])
                xgb_model.fit(X_train, y_train)
                
                y_pred = xgb_model.predict(X_val)
                xgb_metrics = self._calculate_metrics(y_val, y_pred)
                
                models['xgboost'] = xgb_model
                metrics['xgboost'] = xgb_metrics
                
                logger.info(f"XGBoost trained. MSE: {xgb_metrics['mse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training XGBoost: {e}")
            
            # LightGBM
            try:
                lgb_model = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
                lgb_model.fit(X_train, y_train)
                
                y_pred = lgb_model.predict(X_val)
                lgb_metrics = self._calculate_metrics(y_val, y_pred)
                
                models['lightgbm'] = lgb_model
                metrics['lightgbm'] = lgb_metrics
                
                logger.info(f"LightGBM trained. MSE: {lgb_metrics['mse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training LightGBM: {e}")
            
            # Random Forest
            try:
                rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
                rf_model.fit(X_train, y_train)
                
                y_pred = rf_model.predict(X_val)
                rf_metrics = self._calculate_metrics(y_val, y_pred)
                
                models['random_forest'] = rf_model
                metrics['random_forest'] = rf_metrics
                
                logger.info(f"Random Forest trained. MSE: {rf_metrics['mse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training Random Forest: {e}")
            
            self.models.update(models)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training sklearn models: {e}")
            return {}
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        try:
            if len(X) < sequence_length:
                return np.array([]), np.array([])
            
            X_seq = []
            y_seq = []
            
            for i in range(len(X) - sequence_length + 1):
                X_seq.append(X[i:i + sequence_length])
                y_seq.append(y[i + sequence_length - 1])
            
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate PyTorch model"""
        try:
            model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    features = batch['features'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    outputs = model(features)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    targets.extend(target.cpu().numpy().flatten())
            
            return self._calculate_metrics(np.array(targets), np.array(predictions))
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def predict_momentum(self, df: pd.DataFrame, model_name: str = 'lstm', 
                        prediction_horizon: int = 1) -> Optional[MomentumPrediction]:
        """Predict momentum using trained model"""
        try:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found")
                return None
            
            if not self.feature_columns:
                logger.warning("Feature columns not defined")
                return None
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if not available_features:
                logger.warning("No available features for prediction")
                return None
            
            # Get latest features
            latest_features = df[available_features].iloc[-1:].values
            
            # Scale features
            if 'features' in self.scalers:
                latest_features_scaled = self.scalers['features'].transform(latest_features)
            else:
                latest_features_scaled = latest_features
            
            # Make prediction
            if model_name in ['lstm', 'transformer']:
                # For sequence models, we need to create a sequence
                sequence_length = 10  # Default sequence length
                
                if len(df) < sequence_length:
                    logger.warning("Insufficient data for sequence prediction")
                    return None
                
                # Get sequence of features
                feature_sequence = df[available_features].iloc[-sequence_length:].values
                
                if 'features' in self.scalers:
                    feature_sequence_scaled = self.scalers['features'].transform(feature_sequence)
                else:
                    feature_sequence_scaled = feature_sequence
                
                # Convert to tensor
                feature_tensor = torch.FloatTensor(feature_sequence_scaled).unsqueeze(0).to(self.device)
                
                # Make prediction
                model = self.models[model_name]
                model.eval()
                
                with torch.no_grad():
                    prediction = model(feature_tensor)
                    predicted_momentum = prediction.cpu().numpy().flatten()[0]
                
            else:
                # For sklearn models
                model = self.models[model_name]
                predicted_momentum = model.predict(latest_features_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = 0.8  # This would be calculated based on model uncertainty
            
            return MomentumPrediction(
                timestamp=datetime.now(),
                symbol='SYMBOL',  # Would be passed from caller
                timeframe='1h',   # Would be passed from caller
                predicted_momentum=predicted_momentum,
                confidence=confidence,
                prediction_horizon=prediction_horizon,
                model_name=model_name,
                features_used=available_features,
                metadata={'scaled_features': latest_features_scaled.tolist()}
            )
            
        except Exception as e:
            logger.error(f"Error predicting momentum: {e}")
            return None

class MomentumManager:
    """Main momentum prediction manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.momentum_calculator = MomentumCalculator()
        self.momentum_predictor = MomentumPredictor(config)
        
        # Settings
        self.symbols = self.config.get('symbols', ['SOLUSDT', 'BTCUSDT', 'ETHUSDT'])
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h'])
        
        # Running state
        self.running = False
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all momentum prediction models"""
        try:
            logger.info("Starting momentum model training...")
            
            # Calculate momentum indicators
            df_with_momentum = self.momentum_calculator.calculate_momentum_indicators(df)
            
            if df_with_momentum.empty:
                logger.error("Failed to calculate momentum indicators")
                return {}
            
            # Prepare features
            X, y = self.momentum_predictor.prepare_features(df_with_momentum)
            
            if len(X) == 0:
                logger.error("Failed to prepare features")
                return {}
            
            # Train models
            results = {}
            
            # Train LSTM
            lstm_metrics = self.momentum_predictor.train_lstm_model(X, y)
            if lstm_metrics:
                results['lstm'] = lstm_metrics
            
            # Train Transformer
            transformer_metrics = self.momentum_predictor.train_transformer_model(X, y)
            if transformer_metrics:
                results['transformer'] = transformer_metrics
            
            # Train sklearn models
            sklearn_metrics = self.momentum_predictor.train_sklearn_models(X, y)
            results.update(sklearn_metrics)
            
            logger.info("Momentum model training completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training momentum models: {e}")
            return {}
    
    def predict_momentum(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[MomentumPrediction]:
        """Predict momentum for a symbol"""
        try:
            # Calculate momentum indicators
            df_with_momentum = self.momentum_calculator.calculate_momentum_indicators(df)
            
            if df_with_momentum.empty:
                logger.warning(f"No momentum data available for {symbol}")
                return None
            
            # Use the best model (LSTM by default)
            prediction = self.momentum_predictor.predict_momentum(df_with_momentum, 'lstm')
            
            if prediction:
                prediction.symbol = symbol
                prediction.timeframe = timeframe
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting momentum for {symbol}: {e}")
            return None

def main():
    """Main function to demonstrate momentum prediction system"""
    try:
        # Initialize momentum manager
        config = {
            'symbols': ['SOLUSDT', 'BTCUSDT', 'ETHUSDT'],
            'timeframes': ['1m', '5m', '15m', '1h']
        }
        
        manager = MomentumManager(config)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        base_price = 100
        prices = [base_price]
        
        for i in range(n_samples - 1):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.index = pd.date_range(start=datetime.now() - timedelta(days=n_samples), periods=n_samples, freq='1H')
        
        # Train models
        logger.info("Training momentum prediction models...")
        training_results = manager.train_models(df)
        
        for model_name, metrics in training_results.items():
            logger.info(f"{model_name}: MSE={metrics.get('mse', 0):.4f}, R²={metrics.get('r2', 0):.4f}")
        
        # Test predictions
        logger.info("Testing momentum predictions...")
        for symbol in config['symbols']:
            prediction = manager.predict_momentum(df, symbol, '1h')
            
            if prediction:
                logger.info(f"{symbol}: Predicted Momentum={prediction.predicted_momentum:.4f}, "
                           f"Confidence={prediction.confidence:.2f}")
        
        logger.info("Momentum prediction system test completed!")
        
    except Exception as e:
        logger.error(f"Error in main momentum prediction function: {e}")

if __name__ == "__main__":
    main()
