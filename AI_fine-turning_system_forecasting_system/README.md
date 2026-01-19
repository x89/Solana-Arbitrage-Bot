# AI Forecasting System

## Overview

Advanced time series forecasting system using Chronos T5 and Bolt models, fine-tuned specifically for cryptocurrency price prediction (SOL/USDT).

## Features

- ✅ Fine-tuned Chronos T5 and Bolt models
- ✅ Multiple forecasting models ensemble
- ✅ 15-minute timeframe predictions
- ✅ Automatic model evaluation
- ✅ Forecast validation and accuracy testing
- ✅ Pattern-based predictions
- ✅ Sentiment-integrated forecasting

## Models Available

| Model | Status | Description |
|-------|--------|-------------|
| **Chronos T5** | ✅ Trained | Fine-tuned for SOL/USDT 15m predictions |
| **Chronos Bolt** | ✅ Trained | Lightweight, fast predictions |
| **TimeGPT** | ⚠️ Disabled | External API removed |
| **TimesFM** | ✅ Available | Multi-scale forecasting |

## Files

| File | Description |
|------|-------------|
| `forecasting.py` | Main forecasting interface |
| `forecasting_15min.py` | 15-minute specific forecasting |
| `forecasting_sol.py` | SOL/USDT specialized forecaster |
| `forecasting_config.py` | Configuration settings |
| `forecasting_data.py` | Data preprocessing |
| `forecasting_validator.py` | Forecast validation |
| `timegpt_forecaster.py` | (Disabled) TimeGPT integration |
| `pattern_detection.py` | Chart pattern integration |
| `sentiment_analysis.py` | Sentiment-based forecasting |

## Trained Models

### Chronos T5
- **Location**: `chronos_t5_ft/`
- **Checkpoint**: `checkpoint-2000/`
- **Context**: 128 timesteps
- **Horizon**: 64 predictions
- **Training**: 2000 steps

### Chronos Bolt
- **Location**: `chronos_bolt_ft/`
- **Checkpoint**: `checkpoint-1460/`
- **Context**: 128 timesteps
- **Horizon**: 64 predictions
- **Training**: 1460 steps

## Quick Start

### 1. Basic Forecast

```python
from forecasting import ForecastEngine

engine = ForecastEngine()

# Generate forecast
forecast = engine.forecast(
    symbol='SOLUSDT',
    timeframe='15m',
    horizon=64  # Predict 64 periods ahead
)

print(forecast)
```

### 2. 15-Minute Specific

```python
from forecasting_15min import Forecasting15Min

forecaster = Forecasting15Min()

# Get 15-minute forecast
prediction = forecaster.predict_next_hour()
print(f"Predicted price: {prediction}")
```

### 3. With Pattern Detection

```python
from pattern_detection import PatternAwareForecast

# Combine patterns with forecasting
forecast = PatternAwareForecast()
result = forecast.predict_with_patterns(data)
```

## Usage

### Single Model Forecast

```python
from forecasting_sol import SOFForecaster

forecaster = SOFForecaster()

# Load SOL/USDT data
data = load_data('solusdt_15m_1months.json')

# Generate forecast
prediction = forecaster.predict(data)
```

### Ensemble Forecast

```python
# Multiple models consensus
from forecasting import ensemble_forecast

predictions = ensemble_forecast(
    models=['chronos_t5', 'chronos_bolt'],
    data=data,
    horizon=64
)
```

### Forecast Validation

```python
from forecasting_validator import validate_forecast

# Validate forecast accuracy
metrics = validate_forecast(
    predictions=forecast,
    actual=actual_data,
    horizon=64
)

print(f"MAE: {metrics['mae']}")
print(f"RMSE: {metrics['rmse']}")
```

## Configuration

Edit `forecasting_config.py`:

```python
# Model Settings
CONTEXT_LENGTH = 128
HORIZON_LENGTH = 64
FORECAST_MODELS = ['chronos_t5', 'chronos_bolt']

# Data Settings
TIMEFRAME = '15m'
SYMBOL = 'SOLUSDT'

# Validation
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
```

## Training

### Fine-tune Models

```python
# Train Chronos T5
python train_chronos_t5.py \
    --data solusdt_15m_1months.json \
    --epochs 10 \
    --output_dir chronos_t5_ft/

# Train Chronos Bolt
python train_chronos_bolt.py \
    --data solusdt_15m_1months.json \
    --epochs 10 \
    --output_dir chronos_bolt_ft/
```

## Data Requirements

- Historical OHLCV data
- Minimum 1000 candles for training
- 15-minute timeframe preferred
- SOL/USDT or similar pair

## Output Format

```json
{
    "horizon": 64,
    "forecast": [95.5, 96.2, 97.0, ...],
    "confidence": [0.85, 0.82, 0.80, ...],
    "model": "chronos_t5",
    "timestamp": "2025-01-26T10:00:00"
}
```

## Model Performance

### Chronos T5
- MAE: ~0.5%
- Training Time: ~4 hours
- Inference: ~50ms
- Best for: Medium-term predictions

### Chronos Bolt
- MAE: ~0.6%
- Training Time: ~2 hours
- Inference: ~20ms
- Best for: Fast predictions

## Integration

### With Trading Signals

```python
from forecasting import get_trading_signal

signal = get_trading_signal(
    forecast=forecast,
    current_price=95.00,
    confidence_threshold=0.7
)

# signal: 'BUY', 'SELL', or 'HOLD'
```

### With Risk Management

```python
from forecasting import forecast_with_risk

forecast = forecast_with_risk(
    data=data,
    risk_params={'max_loss': 0.05}
)
```

## Requirements

```
torch>=1.12.0
transformers>=4.30.0
chronos-forecasting>=0.1.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Troubleshooting

**Issue**: Model not loading
- Check model files exist in respective folders
- Verify model path in config

**Issue**: Poor predictions
- Ensure sufficient training data
- Check data quality
- Try retraining models

**Issue**: Slow inference
- Use Chronos Bolt for faster predictions
- Reduce context length
- Use GPU acceleration

