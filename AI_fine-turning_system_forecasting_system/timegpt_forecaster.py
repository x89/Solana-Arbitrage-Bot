#!/usr/bin/env python3
"""
TimeGPT Forecasting and Fine-tuning Module
Advanced TimeGPT integration for time series forecasting including:
- TimeGPT API integration
- Custom fine-tuning
- Multi-horizon forecasting
- Ensemble with Chronos models
- Model comparison and selection
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from nixtlats import NixtlaClient
    from nixtlats.models import TimeGPT
except ImportError:
    print("Warning: nixtlats not installed. TimeGPT features will be limited.")
    NixtlaClient = None
    TimeGPT = None

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeGPTForecast:
    """TimeGPT forecast result"""
    forecast: pd.DataFrame
    confidence_intervals: pd.DataFrame
    model_info: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ForecastComparison:
    """Compare forecasts from multiple models"""
    timegpt: Optional[pd.DataFrame]
    chronos_t5: Optional[pd.DataFrame]
    chronos_bolt: Optional[pd.DataFrame]
    ensemble: pd.DataFrame
    metrics: Dict[str, float]

class TimeGPTForecaster:
    """TimeGPT forecasting integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TIMEGPT_API_KEY')
        
        if not self.api_key:
            logger.warning("TimeGPT API key not provided. Set TIMEGPT_API_KEY environment variable.")
            self.client = None
        else:
            try:
                self.client = NixtlaClient(api_key=self.api_key)
                logger.info("TimeGPT client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TimeGPT client: {e}")
                self.client = None
    
    def forecast(
        self,
        df: pd.DataFrame,
        h: int = 24,
        freq: str = '15min',
        level: List[float] = [80, 90],
        finetune_steps: int = 0,
        finetune_loss: str = 'mse',
        verbose: bool = True
    ) -> TimeGPTForecast:
        """
        Generate forecast using TimeGPT
        
        Args:
            df: DataFrame with 'unique_id', 'ds', 'y' columns
            h: Forecast horizon
            freq: Data frequency
            level: Confidence levels
            finetune_steps: Number of fine-tuning steps
            finetune_loss: Loss function for fine-tuning
            verbose: Print progress
            
        Returns:
            TimeGPTForecast object
        """
        try:
            if not self.client:
                raise ValueError("TimeGPT client not initialized")
            
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Prepare data
            prepared_df = self._prepare_dataframe(df)
            
            # Generate forecast
            forecast_df = self.client.forecast(
                df=prepared_df,
                h=h,
                freq=freq,
                level=level,
                finetune_steps=finetune_steps,
                finetune_loss=finetune_loss,
                verbose=verbose
            )
            
            # Extract confidence intervals
            confidence_intervals = self._extract_confidence_intervals(forecast_df, level)
            
            return TimeGPTForecast(
                forecast=forecast_df,
                confidence_intervals=confidence_intervals,
                model_info={
                    'model': 'TimeGPT',
                    'horizon': h,
                    'frequency': freq,
                    'confidence_levels': level,
                    'finetune_steps': finetune_steps
                },
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'input_length': len(df),
                    'forecast_points': len(forecast_df)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating TimeGPT forecast: {e}")
            return None
    
    def fine_tune(
        self,
        df: pd.DataFrame,
        base_model: Optional[str] = None,
        validation_steps: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 5
    ) -> Dict[str, Any]:
        """
        Fine-tune TimeGPT model on custom data
        
        Args:
            df: Training data
            base_model: Base model to fine-tune
            validation_steps: Steps for validation
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            
        Returns:
            Fine-tuning results
        """
        try:
            if not self.client:
                raise ValueError("TimeGPT client not initialized")
            
            # Prepare data
            prepared_df = self._prepare_dataframe(df)
            
            # Fine-tune model
            finetune_result = self.client.finetune(
                df=prepared_df,
                base_model=base_model,
                validation_steps=validation_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs
            )
            
            logger.info(f"Fine-tuning completed: {finetune_result}")
            
            return finetune_result
            
        except Exception as e:
            logger.error(f"Error fine-tuning TimeGPT: {e}")
            return {}
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for TimeGPT"""
        try:
            # Ensure required columns exist
            required_cols = ['unique_id', 'ds', 'y']
            
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Required columns missing: {required_cols}")
                return pd.DataFrame()
            
            # Convert ds to datetime if needed
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Sort by unique_id and timestamp
            df = df.sort_values(['unique_id', 'ds'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {e}")
            return pd.DataFrame()
    
    def _extract_confidence_intervals(self, forecast_df: pd.DataFrame, levels: List[float]) -> pd.DataFrame:
        """Extract confidence intervals from forecast"""
        try:
            intervals = {}
            
            for level in levels:
                level_pct = int(level)
                upper_col = f'TimeGPT-Upper-{level_pct}'
                lower_col = f'TimeGPT-Lower-{level_pct}'
                
                if upper_col in forecast_df.columns and lower_col in forecast_df.columns:
                    intervals[f'upper_{level_pct}'] = forecast_df[upper_col]
                    intervals[f'lower_{level_pct}'] = forecast_df[lower_col]
            
            if intervals:
                return pd.DataFrame(intervals)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error extracting confidence intervals: {e}")
            return pd.DataFrame()
    
    def save_forecast(self, forecast: TimeGPTForecast, filepath: str):
        """Save forecast to file"""
        try:
            forecast_data = {
                'forecast': forecast.forecast.to_dict('records') if not forecast.forecast.empty else [],
                'confidence_intervals': forecast.confidence_intervals.to_dict('records') if not forecast.confidence_intervals.empty else [],
                'model_info': forecast.model_info,
                'metadata': forecast.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(forecast_data, f, indent=4, default=str)
            
            logger.info(f"Forecast saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
    
    def load_forecast(self, filepath: str) -> Optional[TimeGPTForecast]:
        """Load forecast from file"""
        try:
            with open(filepath, 'r') as f:
                forecast_data = json.load(f)
            
            forecast = TimeGPTForecast(
                forecast=pd.DataFrame(forecast_data['forecast']),
                confidence_intervals=pd.DataFrame(forecast_data['confidence_intervals']),
                model_info=forecast_data['model_info'],
                metadata=forecast_data['metadata']
            )
            
            logger.info(f"Forecast loaded from {filepath}")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error loading forecast: {e}")
            return None

class EnsembleForecaster:
    """Ensemble forecasting combining TimeGPT with Chronos models"""
    
    def __init__(self, timegpt_forecaster: TimeGPTForecaster, 
                 chronos_t5_forecaster: Any = None,
                 chronos_bolt_forecaster: Any = None):
        self.timegpt = timegpt_forecaster
        self.chronos_t5 = chronos_t5_forecaster
        self.chronos_bolt = chronos_bolt_forecaster
        self.weights = {
            'timegpt': 0.4,
            'chronos_t5': 0.35,
            'chronos_bolt': 0.25
        }
    
    def forecast_ensemble(
        self,
        df: pd.DataFrame,
        h: int = 24,
        freq: str = '15min',
        method: str = 'weighted_average'
    ) -> ForecastComparison:
        """
        Generate ensemble forecast using multiple models
        
        Args:
            df: Input data
            h: Forecast horizon
            freq: Data frequency
            method: Ensemble method ('weighted_average', 'median', 'voting')
            
        Returns:
            ForecastComparison object
        """
        try:
            forecasts = {}
            
            # TimeGPT forecast
            if self.timegpt and self.timegpt.client:
                logger.info("Generating TimeGPT forecast...")
                timegpt_result = self.timegpt.forecast(df, h=h, freq=freq)
                if timegpt_result:
                    forecasts['timegpt'] = timegpt_result.forecast['TimeGPT']
                else:
                    forecasts['timegpt'] = None
            else:
                forecasts['timegpt'] = None
                logger.warning("TimeGPT not available")
            
            # Chronos T5 forecast
            if self.chronos_t5:
                logger.info("Generating Chronos T5 forecast...")
                try:
                    chronos_t5_result = self.chronos_t5.predict(df, h=h)
                    if chronos_t5_result is not None and not chronos_t5_result.empty:
                        forecasts['chronos_t5'] = chronos_t5_result.iloc[:, 0]
                    else:
                        forecasts['chronos_t5'] = None
                except Exception as e:
                    logger.error(f"Chronos T5 forecast error: {e}")
                    forecasts['chronos_t5'] = None
            else:
                forecasts['chronos_t5'] = None
            
            # Chronos Bolt forecast
            if self.chronos_bolt:
                logger.info("Generating Chronos Bolt forecast...")
                try:
                    chronos_bolt_result = self.chronos_bolt.predict(df, h=h)
                    if chronos_bolt_result is not None and not chronos_bolt_result.empty:
                        forecasts['chronos_bolt'] = chronos_bolt_result.iloc[:, 0]
                    else:
                        forecasts['chronos_bolt'] = None
                except Exception as e:
                    logger.error(f"Chronos Bolt forecast error: {e}")
                    forecasts['chronos_bolt'] = None
            else:
                forecasts['chronos_bolt'] = None
            
            # Create ensemble forecast
            ensemble_forecast = self._create_ensemble(forecasts, method)
            
            # Calculate metrics
            metrics = self._calculate_metrics(forecasts, ensemble_forecast)
            
            return ForecastComparison(
                timegpt=pd.DataFrame({'TimeGPT': forecasts['timegpt']}) if forecasts['timegpt'] is not None else None,
                chronos_t5=pd.DataFrame({'ChronosT5': forecasts['chronos_t5']}) if forecasts['chronos_t5'] is not None else None,
                chronos_bolt=pd.DataFrame({'ChronosBolt': forecasts['chronos_bolt']}) if forecasts['chronos_bolt'] is not None else None,
                ensemble=pd.DataFrame({'Ensemble': ensemble_forecast}),
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {e}")
            return ForecastComparison(
                timegpt=None,
                chronos_t5=None,
                chronos_bolt=None,
                ensemble=pd.DataFrame(),
                metrics={}
            )
    
    def _create_ensemble(self, forecasts: Dict[str, pd.Series], method: str) -> pd.Series:
        """Create ensemble forecast from individual forecasts"""
        try:
            # Remove None forecasts
            valid_forecasts = {k: v for k, v in forecasts.items() if v is not None}
            
            if not valid_forecasts:
                logger.warning("No valid forecasts for ensemble")
                return pd.Series()
            
            # Get the length from the first valid forecast
            first_key = list(valid_forecasts.keys())[0]
            forecast_length = len(valid_forecasts[first_key])
            
            # Initialize ensemble
            if method == 'weighted_average':
                ensemble = np.zeros(forecast_length)
                total_weight = 0
                
                for key, forecast in valid_forecasts.items():
                    weight = self.weights.get(key, 1.0 / len(valid_forecasts))
                    if len(forecast) == forecast_length:
                        ensemble += forecast.values * weight
                        total_weight += weight
                
                ensemble /= total_weight if total_weight > 0 else 1
                
            elif method == 'median':
                forecast_matrix = np.stack([
                    f.values for f in valid_forecasts.values() 
                    if len(f) == forecast_length
                ])
                ensemble = np.median(forecast_matrix, axis=0)
                
            elif method == 'mean':
                forecast_matrix = np.stack([
                    f.values for f in valid_forecasts.values() 
                    if len(f) == forecast_length
                ])
                ensemble = np.mean(forecast_matrix, axis=0)
                
            else:
                logger.warning(f"Unknown ensemble method: {method}. Using weighted_average.")
                ensemble = self._create_ensemble(forecasts, 'weighted_average')
            
            return pd.Series(ensemble)
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return pd.Series()
    
    def _calculate_metrics(self, forecasts: Dict[str, pd.Series], ensemble: pd.Series) -> Dict[str, float]:
        """Calculate forecast metrics"""
        try:
            metrics = {}
            
            valid_forecasts = {k: v for k, v in forecasts.items() if v is not None}
            
            if not valid_forecasts or ensemble.empty:
                return {'error': 'No forecasts available'}
            
            # Calculate variance across forecasts
            forecast_matrix = np.stack([f.values for f in valid_forecasts.values() if len(f) == len(ensemble)])
            
            if forecast_matrix.shape[0] > 1:
                metrics['forecast_variance'] = float(np.var(forecast_matrix, axis=0).mean())
                metrics['forecast_std'] = float(np.std(forecast_matrix, axis=0).mean())
            else:
                metrics['forecast_variance'] = 0.0
                metrics['forecast_std'] = 0.0
            
            # Calculate ensemble statistics
            metrics['ensemble_mean'] = float(ensemble.mean())
            metrics['ensemble_std'] = float(ensemble.std())
            metrics['forecast_horizon'] = len(ensemble)
            metrics['num_models'] = len(valid_forecasts)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

def main():
    """Main function to demonstrate TimeGPT integration"""
    try:
        # Initialize TimeGPT forecaster
        timegpt_forecaster = TimeGPTForecaster()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        data = {
            'unique_id': ['SOLUSDT'] * 100,
            'ds': dates,
            'y': np.random.randn(100).cumsum() + 100
        }
        df = pd.DataFrame(data)
        
        logger.info("Generating TimeGPT forecast...")
        
        # Generate forecast
        forecast_result = timegpt_forecaster.forecast(
            df=df,
            h=24,
            freq='15min',
            level=[80, 90],
            finetune_steps=0,
            verbose=True
        )
        
        if forecast_result:
            logger.info(f"Forecast generated successfully")
            logger.info(f"Forecast shape: {forecast_result.forecast.shape}")
            logger.info(f"Model info: {forecast_result.model_info}")
            
            # Save forecast
            output_file = 'AI_forecasting_fine-turning_system/timegpt_forecast.json'
            timegpt_forecaster.save_forecast(forecast_result, output_file)
        
        logger.info("TimeGPT forecasting completed!")
        
    except Exception as e:
        logger.error(f"Error in TimeGPT forecasting: {e}")

if __name__ == "__main__":
    main()

