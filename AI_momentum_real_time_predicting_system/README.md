# AI Momentum Real-time Predicting System

## Overview

Advanced momentum prediction system using multiple AI models including LSTM, Transformer, XGBoost, LightGBM, and ensemble methods. Features comprehensive momentum indicators, regime detection, and prediction validation.

## Features

- ✅ Multi-model ensemble predictions (LSTM, Transformer, XGBoost, LightGBM)
- ✅ **Chronos T5 & Bolt integration** for time series momentum forecasting
- ✅ Comprehensive momentum indicators calculation
- ✅ LGMM regime detection for market state identification
- ✅ Prediction validation and confidence scoring
- ✅ Model inference pipeline with caching
- ✅ Real-time momentum prediction

## Files

| File | Description |
|------|-------------|
| `config.py` | System configuration and model settings |
| `prediction_engine.py` | Main prediction engine with ensemble support |
| `ensemble_manager.py` | Ensemble prediction management |
| `prediction_validator.py` | Prediction validation |
| `model_loader.py` | Model loading utilities |
| `inference_pipeline.py` | Inference pipeline |
| `momentum_predictor.py` | Momentum prediction models |
| `momentum_calculator.py` | Momentum indicators |
| `momentum_trainer.py` | Model training system |
| `lgmm_regime_detector.py` | Market regime detection |
| **`chronos_momentum_predictor.py`** | **Chronos T5 & Bolt integration for momentum prediction** |
| `example_chronos_integration.py` | Example usage of Chronos models |

## Installation

### 1. Install System Dependencies

**Windows:**
```bash
# Download TA-Lib from: https://ta-lib.org/install/
# Install the .whl file matching your Python version
```

**Linux/Mac:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Check Installation

```bash
python check_installation.py
```

This will verify all required dependencies are installed.

### 4. Test Modules

```bash
python test_modules.py
```

This will verify all modules can be imported without errors.

### 5. Run System Demo

```bash
python run_main.py
```

This will run a complete demonstration of all system components.

## Quick Start

### 1. Calculate Momentum Indicators

```python
from momentum_calculator import MomentumCalculator
import pandas as pd

calculator = MomentumCalculator()

# Load your market data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Calculate all momentum indicators
indicators = calculator.calculate_all_indicators(df)
print(indicators)
```

### 2. Train Momentum Models

```python
from momentum_trainer import MomentumTrainer
import pandas as pd

trainer = MomentumTrainer()

# Prepare data
train_loader, test_loader, scaler = trainer.prepare_data(df, target_column='momentum')

# Train LSTM model
model, metrics = trainer.train_lstm_model(
    train_loader, 
    test_loader, 
    input_dim=num_features
)
```

### 3. Generate Predictions

```python
from momentum_predictor import MomentumManager

manager = MomentumManager(config={'symbols': ['SOLUSDT']})

# Predict momentum
prediction = manager.predict_momentum(df, symbol='SOLUSDT', timeframe='1h')

if prediction:
    print(f"Predicted Momentum: {prediction.predicted_momentum:.4f}")
    print(f"Confidence: {prediction.confidence:.2f}")
```

### 4. Regime Detection

```python
from lgmm_regime_detector import LGMMRegimeDetector
import numpy as np

detector = LGMMRegimeDetector(n_components=3)

# Load market features
features = np.array([...])  # Your feature array

# Fit and detect regimes
result = detector.fit(features)
regime_labels = detector.regime_labels

print(f"Identified {len(set(regime_labels))} market regimes")
```

### 5. Use Chronos Models for Real-Time Momentum Prediction

```python
from chronos_momentum_predictor import ChronosMomentumPredictor

# Initialize Chronos predictor
predictor = ChronosMomentumPredictor(
    context_length=64,
    prediction_horizon=24,
    model_type='bolt'  # 'bolt' for fast, 't5' for accurate
)

# Predict momentum using Chronos
predictions = predictor.predict_momentum(
    df=df_with_momentum,
    target_column='composite_momentum',
    prediction_type='both'  # Use both Bolt and T5
)

# Get ensemble prediction from both Chronos models
ensemble = predictor.ensemble_predict(
    df=df_with_momentum,
    target_column='composite_momentum',
    weights={'bolt': 0.3, 't5': 0.7}
)

print(f"Predicted momentum: {ensemble['ensemble_prediction']:.4f}")
print(f"Confidence: {ensemble['confidence']:.2f}")

# Quick real-time prediction
recent_momentum = df['composite_momentum'].tail(64).tolist()
next_momentum = predictor.predict_next_momentum(recent_momentum)
print(f"Next momentum: {next_momentum:.4f}")
```

### 6. Validate Predictions

```python
from prediction_validator import PredictionValidator

validator = PredictionValidator(confidence_threshold=0.6)

# Validate a prediction
result = validator.validate(
    prediction=0.85,
    confidence=0.75,
    historical_data=[0.7, 0.75, 0.8, 0.82, 0.84]
)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.validation_score:.2f}")
```

## Configuration

Edit `config.py` to customize:
- Model hyperparameters
- Ensemble weights
- Feature engineering settings
- Inference parameters
- Performance thresholds

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- torch>=2.0.0
- pandas>=1.5.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- lightgbm>=3.3.0
- TA-Lib>=0.4.25
- **chronos-forecasting>=0.1.0** (for Chronos models)

## Chronos Models

This system now includes **Amazon Chronos** time series forecasting models:

- **Chronos Bolt**: Fast, lightweight model for real-time predictions (~200ms)
- **Chronos T5**: Larger, more accurate model for detailed forecasting (~400ms)

### Installing Chronos

```bash
pip install chronos-forecasting
```

### Using Your Fine-Tuned Models

If you have fine-tuned Chronos models in `AI_fine-turning_system_forecasting_system/`, the system will automatically detect and use them:

1. Set environment variables (optional):
   ```bash
   # Windows
   set CHRONOS_T5_DIR=AI_fine-turning_system_forecasting_system\chronos_t5_ft
   set CHRONOS_BOLT_DIR=AI_fine-turning_system_forecasting_system\chronos_bolt_ft
   ```

2. Or edit `.env` file:
   ```
   CHRONOS_T5_DIR=AI_fine-turning_system_forecasting_system/chronos_t5_ft
   CHRONOS_BOLT_DIR=AI_fine-turning_system_forecasting_system/chronos_bolt_ft
   ```

The system will automatically use your fine-tuned checkpoints if available, otherwise fall back to pre-trained models.

### Quick Start with Chronos

```bash
# Run the complete example
python example_chronos_integration.py

# Or run the Chronos predictor demo
python chronos_momentum_predictor.py
```

## Real-Time Usage

### Start Real-Time Predictions

**Windows:**
```bash
# Double-click or run:
START_REALTIME.bat

# Or from command line:
python run_realtime_momentum.py
```

**Linux/Mac:**
```bash
python run_realtime_momentum.py
```

**Stop the system:** Press `Ctrl+C`

### What It Does

- ✅ Continuously predicts momentum every 15 seconds
- ✅ Shows real-time momentum values
- ✅ Uses Chronos Bolt for fast predictions
- ✅ Provides ensemble predictions
- ✅ Gracefully stops with Ctrl+C

### Output Example

```
[12:34:56] Iteration #64
  Current momentum: 0.123456
  Buffer size: 64
  Chronos Bolt prediction: 0.136123
  ✓ Ensemble prediction: 0.135890
  ✓ Confidence: 0.82
```

See `REALTIME_USAGE.md` for detailed documentation.

