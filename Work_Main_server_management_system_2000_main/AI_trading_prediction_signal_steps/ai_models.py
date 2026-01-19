

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Import existing AI modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from AI_trading_TimeGPT.ai_skills.forecasting import TimeSeriesForecaster
    from AI_trading_TimeGPT.ai_skills.pattern_detection import ChartPatternDetector
    from AI_trading_TimeGPT.ai_skills.sentiment_analysis import TransformerSentimentAnalyzer
except ImportError:
    print("Warning: Could not import existing AI modules. Using mock implementations.")
    TimeSeriesForecaster = None
    ChartPatternDetector = None
    TransformerSentimentAnalyzer = None

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("Warning: Could not import scikit-learn. ML predictions will be disabled.")
    joblib = None
    RandomForestClassifier = None

from config import Config

class AIModelManager:
    """Manages all AI models and their predictions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models
        self.forecaster = None
        self.pattern_detector = None
        self.sentiment_analyzer = None
        self.ml_model = None
        
        # Model initialization flags
        self.models_initialized = {
            'forecasting': False,
            'pattern_detection': False,
            'sentiment_analysis': False,
            'ml_model': False
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            # Initialize forecasting models
            if self.config.ai.TIMEGPT_ENABLED or self.config.ai.CHRONOS_T5_ENABLED:
                if TimeSeriesForecaster:
                    self.forecaster = TimeSeriesForecaster(
                        context_len=self.config.ai.CONTEXT_LENGTH,
                        horizon_len=self.config.ai.HORIZON_LENGTH
                    )
                    self.models_initialized['forecasting'] = True
                    self.logger.info("Forecasting models initialized")
                else:
                    self.logger.warning("TimeSeriesForecaster not available")
            
            # Initialize pattern detection
            if self.config.ai.PATTERN_DETECTION_ENABLED:
                if ChartPatternDetector:
                    self.pattern_detector = ChartPatternDetector()
                    self.models_initialized['pattern_detection'] = True
                    self.logger.info("Pattern detection model initialized")
                else:
                    self.logger.warning("ChartPatternDetector not available")
            
            # Initialize sentiment analysis
            if self.config.ai.SENTIMENT_ENABLED:
                if TransformerSentimentAnalyzer:
                    self.sentiment_analyzer = TransformerSentimentAnalyzer()
                    self.models_initialized['sentiment_analysis'] = True
                    self.logger.info("Sentiment analysis model initialized")
                else:
                    self.logger.warning("TransformerSentimentAnalyzer not available")
            
            # Initialize ML model
            if self.config.ai.ML_MODEL_ENABLED:
                self._load_ml_model()
            
        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
    
    def _load_ml_model(self):
        """Load the trained ML model"""
        try:
            model_path = self.config.ai.ML_MODEL_PATH
            
            if os.path.exists(f"{model_path}/trading_model.pkl"):
                if joblib:
                    self.ml_model = joblib.load(f"{model_path}/trading_model.pkl")
                    self.models_initialized['ml_model'] = True
                    self.logger.info("ML model loaded successfully")
                else:
                    self.logger.warning("joblib not available for ML model loading")
            else:
                self.logger.warning(f"ML model not found at {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
    
    def get_forecasts(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get forecasts from all available forecasting models"""
        forecasts = {}
        
        if not self.models_initialized['forecasting'] or not self.forecaster:
            return forecasts
        
        try:
            # Prepare data for forecasting
            df_forecast = self._prepare_forecast_data(market_data)
            
            # Get forecasts from different models
            if self.config.ai.TIMEGPT_ENABLED:
                try:
                    forecasts['TimeGPT'] = self.forecaster.timegpt_forecast(df_forecast)
                except Exception as e:
                    self.logger.warning(f"TimeGPT forecast failed: {e}")
            
            if self.config.ai.CHRONOS_T5_ENABLED:
                try:
                    forecasts['ChronosT5'] = self.forecaster.chronos_t5_forecast(df_forecast)
                except Exception as e:
                    self.logger.warning(f"ChronosT5 forecast failed: {e}")
            
            if self.config.ai.CHRONOS_BOLT_ENABLED:
                try:
                    forecasts['ChronosBolt'] = self.forecaster.chronos_bolt_forecast(df_forecast)
                except Exception as e:
                    self.logger.warning(f"ChronosBolt forecast failed: {e}")
            
            if self.config.ai.TIMESFM1_ENABLED:
                try:
                    forecasts['TimesFM1'] = self.forecaster.timesfm1_forecast(df_forecast)
                except Exception as e:
                    self.logger.warning(f"TimesFM1 forecast failed: {e}")
            
            if self.config.ai.TIMESFM2_ENABLED:
                try:
                    forecasts['TimesFM2'] = self.forecaster.timesfm2_forecast(df_forecast)
                except Exception as e:
                    self.logger.warning(f"TimesFM2 forecast failed: {e}")
            
            self.logger.info(f"Generated {len(forecasts)} forecasts")
            
        except Exception as e:
            self.logger.error(f"Error getting forecasts: {e}")
        
        return forecasts
    
    def detect_patterns(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns in market data"""
        patterns = {}
        
        if not self.models_initialized['pattern_detection'] or not self.pattern_detector:
            return patterns
        
        try:
            # Detect patterns using the existing pattern detector
            detected_patterns = self.pattern_detector.detect_patterns(market_data)
            
            if detected_patterns:
                patterns = {
                    'patterns': detected_patterns,
                    'confidence': self._calculate_pattern_confidence(detected_patterns),
                    'timestamp': datetime.now()
                }
            
            self.logger.info(f"Detected {len(detected_patterns) if detected_patterns else 0} patterns")
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def analyze_sentiment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment"""
        sentiment_data = {}
        
        if not self.models_initialized['sentiment_analysis'] or not self.sentiment_analyzer:
            return sentiment_data
        
        try:
            # Get symbol for sentiment analysis
            symbol = self.config.data.DEFAULT_SYMBOL.replace('USDT', '')
            
            # Fetch and analyze articles
            articles = self.sentiment_analyzer.fetch_articles(f"{symbol} stock")
            
            if articles:
                report, sentiment_score = self.sentiment_analyzer.analyze_articles(articles)
                
                sentiment_data = {
                    'sentiment_score': sentiment_score,
                    'articles_analyzed': len(articles),
                    'report': report,
                    'timestamp': datetime.now()
                }
            
            self.logger.info(f"Sentiment analysis completed: {sentiment_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
        
        return sentiment_data
    
    def get_ml_prediction(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get prediction from ML model"""
        prediction = {}
        
        if not self.models_initialized['ml_model'] or not self.ml_model:
            return prediction
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(market_data)
            
            if features is not None and len(features) > 0:
                # Get prediction
                prediction_proba = self.ml_model.predict_proba(features)
                prediction_class = self.ml_model.predict(features)
                
                prediction = {
                    'prediction': prediction_class[0],
                    'probabilities': prediction_proba[0],
                    'confidence': np.max(prediction_proba[0]),
                    'features_used': len(features.columns),
                    'timestamp': datetime.now()
                }
            
            self.logger.info(f"ML prediction: {prediction.get('prediction', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction: {e}")
        
        return prediction
    
    def _prepare_forecast_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for forecasting models"""
        try:
            # Convert to the format expected by forecasting models
            df_forecast = market_data[['close']].copy()
            df_forecast = df_forecast.rename(columns={'close': 'y'})
            df_forecast['ds'] = df_forecast.index
            df_forecast['unique_id'] = self.config.data.DEFAULT_SYMBOL
            
            return df_forecast
            
        except Exception as e:
            self.logger.error(f"Error preparing forecast data: {e}")
            return pd.DataFrame()
    
    def _prepare_ml_features(self, market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML model"""
        try:
            # Load feature scaler and feature list
            model_path = self.config.ai.ML_MODEL_PATH
            
            if not os.path.exists(f"{model_path}/scaler.pkl") or not os.path.exists(f"{model_path}/features.pkl"):
                self.logger.warning("ML model scaler or features not found")
                return None
            
            if joblib:
                scaler = joblib.load(f"{model_path}/scaler.pkl")
                feature_list = joblib.load(f"{model_path}/features.pkl")
            else:
                return None
            
            # Calculate technical indicators
            features_df = self._calculate_technical_features(market_data)
            
            # Select only the features used in training
            if feature_list and all(feat in features_df.columns for feat in feature_list):
                features_df = features_df[feature_list]
                
                # Scale features
                features_scaled = scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=feature_list)
                
                return features_df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            return None
    
    def _calculate_technical_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""
        try:
            features_df = market_data.copy()
            
            # Calculate RSI
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            
            # Calculate moving averages
            features_df['sma_5'] = features_df['close'].rolling(5).mean()
            features_df['sma_20'] = features_df['close'].rolling(20).mean()
            features_df['ema_12'] = features_df['close'].ewm(span=12).mean()
            features_df['ema_26'] = features_df['close'].ewm(span=26).mean()
            
            # Calculate Bollinger Bands
            bb_period = self.config.ai.BB_PERIOD
            bb_std = self.config.ai.BB_STD
            bb_middle = features_df['close'].rolling(bb_period).mean()
            bb_std_val = features_df['close'].rolling(bb_period).std()
            features_df['bb_upper'] = bb_middle + (bb_std_val * bb_std)
            features_df['bb_lower'] = bb_middle - (bb_std_val * bb_std)
            features_df['bb_middle'] = bb_middle
            
            # Calculate MACD
            macd_fast = self.config.ai.MACD_FAST
            macd_slow = self.config.ai.MACD_SLOW
            macd_signal = self.config.ai.MACD_SIGNAL
            
            ema_fast = features_df['close'].ewm(span=macd_fast).mean()
            ema_slow = features_df['close'].ewm(span=macd_slow).mean()
            features_df['macd'] = ema_fast - ema_slow
            features_df['macd_signal'] = features_df['macd'].ewm(span=macd_signal).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Calculate price changes
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['price_change_5'] = features_df['close'].pct_change(5)
            
            # Calculate volume indicators
            features_df['volume_sma_5'] = features_df['volume'].rolling(5).mean()
            features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return market_data.copy()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.config.ai.RSI_PERIOD
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_pattern_confidence(self, patterns: List[Dict]) -> float:
        """Calculate confidence score for detected patterns"""
        if not patterns:
            return 0.0
        
        # Simple confidence calculation based on pattern strength
        total_confidence = sum(pattern.get('confidence', 0.5) for pattern in patterns)
        return min(total_confidence / len(patterns), 1.0)
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return self.models_initialized.copy()
    
    def reload_models(self):
        """Reload all models"""
        self.logger.info("Reloading AI models...")
        self._initialize_models()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'forecasting': {
                'initialized': self.models_initialized['forecasting'],
                'models_enabled': {
                    'TimeGPT': self.config.ai.TIMEGPT_ENABLED,
                    'ChronosT5': self.config.ai.CHRONOS_T5_ENABLED,
                    'ChronosBolt': self.config.ai.CHRONOS_BOLT_ENABLED,
                    'TimesFM1': self.config.ai.TIMESFM1_ENABLED,
                    'TimesFM2': self.config.ai.TIMESFM2_ENABLED
                }
            },
            'pattern_detection': {
                'initialized': self.models_initialized['pattern_detection'],
                'enabled': self.config.ai.PATTERN_DETECTION_ENABLED
            },
            'sentiment_analysis': {
                'initialized': self.models_initialized['sentiment_analysis'],
                'enabled': self.config.ai.SENTIMENT_ENABLED
            },
            'ml_model': {
                'initialized': self.models_initialized['ml_model'],
                'enabled': self.config.ai.ML_MODEL_ENABLED,
                'model_path': self.config.ai.ML_MODEL_PATH
            }
        }
        
        return info

if __name__ == "__main__":
    # Example usage
    from config import config
    
    # Initialize AI model manager
    ai_manager = AIModelManager(config)
    
    # Check model status
    status = ai_manager.get_model_status()
    print("Model Status:")
    for model, initialized in status.items():
        print(f"  {model}: {'✓' if initialized else '✗'}")
    
    # Get model info
    info = ai_manager.get_model_info()
    print("\nModel Information:")
    print(info)
