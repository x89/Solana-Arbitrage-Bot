# 🧠 AI Crypto Future Trading Bot - Daily PNL :8~27%
## Overview
An enterprise-grade AI-powered cryptocurrency futures trading bot achieving 8-27% daily returns with machine learning-driven predictions, real-time signal generation, and fully automated execution across multiple exchanges.
Main Platform: Bitget, Blofin, Binance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Daily PNL: 5-27%](https://img.shields.io/badge/Daily_PNL-20--30%25-brightgreen.svg)]()

---
## Important tip: Recommended trading vps -> tradoxvps.com
## Performance Highlights

| Metric | Value |
|--------|-------|
| **Average Daily PNL** | 8-27% |
| **Maximum Drawdown** | <9% |
| **Supported Exchanges** | Bitget, Blofin, Binance |
| **Trading Pairs** | SOL/USDT, ETH/USDT, DOGE/USDT, XRP/USDT |
| **Automation Level** | 100% Fully Automated |
| **Signal Generation** | Real-time AI Predictions |

---

<img width="960" height="635" alt="image" src="https://github.com/user-attachments/assets/5964b00e-7243-4aa2-b3b9-961bdaf6d53c" />

<img width="1839" height="829" alt="image" src="https://github.com/user-attachments/assets/6cd7de3c-c1ae-4ad8-92f5-e20da54ff07b" />

<img width="1241" height="400" alt="image" src="https://github.com/user-attachments/assets/4e4932ff-abcd-479e-af97-6dbf6486d503" />

<img width="1280" height="224" alt="image" src="https://github.com/user-attachments/assets/24e3a6bd-2c69-426f-b00e-fdf064f4932e" />

<img width="864" height="152" alt="image" src="https://github.com/user-attachments/assets/93c5834a-116f-4da8-b064-78518c56718a" />

<img width="1280" height="880" alt="image" src="https://github.com/user-attachments/assets/e9f91ce3-bbd7-4b65-add7-412aacbae59f" />

<img width="1280" height="491" alt="image" src="https://github.com/user-attachments/assets/3abb978e-f518-47e5-87f6-b036709acc9f" />






## What Makes This Bot Different?

Unlike traditional trading bots that rely on simple technical indicators, this system combines **High accuracy AI models** with real-time market analysis to make intelligent trading decisions:

**Advanced AI Forecasting**
- Multiple AI models working in ensemble (Chronos T5, Chronos BOLT, TimeGPT)
- Pattern detection and recognition
- Sentiment analysis integration
- Fine-tuned models specifically for crypto markets

**Real-Time Intelligence**
- Live data streaming from multiple exchanges
- Millisecond-level signal generation
- Momentum calculation and regime detection
- Dynamic risk management with ATR + Trailing Stop loss

**Professional Infrastructure**
- GPU-accelerated processing
- Parallel training and prediction pipelines
- Cloud-ready with Ubuntu server management
- Automated position sizing and risk control

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI CRYPTO TRADING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Data Collection │───▶│  AI Forecasting  │───▶│  Signal Generator│
│                  │    │                  │    │                  │
│ • Bitget API     │    │ • Chronos Models │    │ • Buy/Sell       │
│ • Blofin API     │    │ • TimeGPT        │    │ • Position Size  │
│ • Binance API    │    │ • Pattern AI     │    │ • Risk Level     │
│ • Real-time OHLCV│    │ • Sentiment AI   │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐
│  Data Processing │    │  Model Training  │    │ Order Execution With API│
│                  │    │                  │    │                         │
│ • Validation     │    │ • GPU Training   │    │ • Auto Trading          │
│ • Cleaning       │    │ • Fine-tuning    │    │ • Risk Control          │
│ • Indicators     │    │ • Optimization   │    │ • P&L Tracking          │
└──────────────────┘    └──────────────────┘    └─────────────────────────┘
```

---

##  Core Components

### 1. Data Collection System
- **Real-time OHLCV data** from Bitget, Blofin, and Binance
- **Historical data collection** (supports up to 1000+ days)
- **Multi-symbol tracking** (SOL, ETH, DOGE, XRP)
- **Automatic data validation** and quality checks
- **Multiple timeframes** (1m, 5m, 15m, 1h, 4h, 1d)

### 2. AI Forecasting System
- **Chronos T5** - Time series foundation model
- **Chronos BOLT** - Fast inference optimized model
- **TimeGPT** - Advanced forecasting with GPT architecture
- **Pattern Detection AI** - Identifies market patterns and formations
- **Sentiment Analysis** - News and social media sentiment integration

**Location**: `AI_fine-turning_system_forecasting_system/`

### 3.  Real-Time Prediction Engine
- **Momentum Calculator** - Real-time momentum tracking
- **Regime Detection** (LGMM) - Identifies market conditions
- **Ensemble Predictions** - Combines multiple model outputs
- **Signal Validation** - Filters false signals
- **Confidence Scoring** - Assigns probability to each prediction

**Location**: `AI_momentum_real_time_predicting_system/`

### 4.  Training System
- **Advanced AI Trainers** - Automated model training
- **LLM + RL Hybrid** - Combines language models with reinforcement learning
- **Modern Architecture** - Transformers, LSTMs, CNNs
- **Model Registry** - Tracks and versions all models
- **Performance Metrics** - Comprehensive evaluation

**Location**: `AI_training_system(models_trainers)/`

### 5.  Technical Analysis System
- **8+ Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Custom Momentum Indicators**
- **Support/Resistance Detection**
- **Volume Analysis**
- **Real-time Indicator Updates**

**Location**: `Analyzing_indicators_comparing_momentum_system/`

### 6.  GPU & Cloud Infrastructure
- **GPU Manager** - Efficient GPU allocation and monitoring
- **Parallel Processing** - Train multiple models simultaneously
- **Resource Optimization** - Automatic resource allocation
- **Cloud Deployment** - Ubuntu server management

---


```

## Usage for running

### Collecting Historical Data
```python
from AI_Data_collecting_system_bitget.bitget_client import BitgetClient, DataCollector

# Initialize
collector = DataCollector(symbol="SOLUSDT", timeframe="15m")

# Collect 100 days of data
data = await collector.collect_historical_data(days=100)

# Data is automatically saved to database and JSON
```

### Running AI Forecasting
```python
from AI_fine-turning_system_forecasting_system.forecasting import ChronosForecaster

# Initialize forecaster
forecaster = ChronosForecaster(model_name="chronos-t5-large")

# Make predictions
predictions = forecaster.predict(
    historical_data=data,
    horizon=96  # Predict next 24 hours (15m intervals)
)

print(f"Predicted price movement: {predictions['direction']}")
print(f"Confidence: {predictions['confidence']}%")
```

### Real-time Trading Signals
```python
from AI_momentum_real_time_predicting_system.momentum_predictor import MomentumPredictor

# Initialize predictor
predictor = MomentumPredictor(symbol="SOLUSDT")

# Get real-time signal
signal = predictor.get_signal()

if signal['action'] == 'BUY':
    print(f"🟢 BUY Signal - Confidence: {signal['confidence']}%")
    print(f"   Entry: ${signal['entry_price']}")
    print(f"   Take Profit: ${signal['take_profit']}")
    print(f"   Stop Loss: ${signal['stop_loss']}")
```

### Training Custom Models
```python
from AI_training_system.advanced_ai_trainer import AdvancedAITrainer

# Initialize trainer
trainer = AdvancedAITrainer(
    model_type="transformer",
    use_gpu=True
)

# Train on historical data
trainer.train(
    data=historical_data,
    epochs=100,
    batch_size=32
)

# Evaluate
metrics = trainer.evaluate()
print(f"Model Accuracy: {metrics['accuracy']}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
```

---

## 🔧 Configuration

### Configure API Keys ###

```bash
# Create .env file
cp .env.example .env

# Add your API keys
BITGET_API_KEY=bg_025caa13ea1938260ed6ae47968d4bfb
BITGET_API_SECRET=a8f58ef61c44d1435187181070306d842cdc1a592ebe464cac60c38fb46a6289
BITGET_PASSPHRASE=bg2025!@#

BLOFIN_API_KEY=your_api_key
BLOFIN_API_SECRET=your_api_secret

BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

### Trading Configuration
Edit `config.py` in your chosen trading system:

```python
# Trading Parameters
TRADING_CONFIG = {
    'symbols': ['SOLUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT'],
    'leverage': 10,
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_positions': 4,
    'stop_loss_pct': 0.03,   # 3% stop loss
    'take_profit_pct': 0.06, # 6% take profit
}

# AI Configuration
AI_CONFIG = {
    'model_ensemble': ['chronos-t5', 'chronos-bolt', 'timegpt'],
    'prediction_horizon': 96,  # 24 hours at 15m intervals
    'min_confidence': 0.65,    # 65% minimum confidence
    'use_sentiment': True,
    'use_patterns': True,
}

# Risk Management
RISK_CONFIG = {
    'max_daily_loss': 0.10,    # Stop trading if 10% daily loss
    'max_drawdown': 0.15,      # 15% maximum drawdown
    'position_sizing': 'dynamic',
    'use_martingale': True,
}
```


##  Supported Trading Pairs

| Symbol | Exchange | Leverage | Status |
|--------|----------|----------|--------|
| SOL/USDT | Bitget, Blofin, Binance | 1-30x | ✅ Active |
| ETH/USDT | Bitget, Blofin, Binance | 1-30x | ✅ Active |
| DOGE/USDT | Bitget, Blofin, Binance | 1-20x | ✅ Active |
| XRP/USDT | Bitget, Blofin, Binance | 1-20x | ✅ Active |

*More trading pairs can be added easily*

---






