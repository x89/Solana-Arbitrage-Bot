# Real-Time AI Crypto Trading Bot

## Architecture Overview

```
┌─────────────────┐
│  Market Data    │  ← Bitget WebSocket
│     Feeds       │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Stream Ingest  │  ← WebSocket → Candle Generator
└────────┬────────┘
         ↓
┌─────────────────┐
│ Feature Engine  │  ← Real-time feature computation
│  + Feature      │  ← Feature cache (Redis)
│     Cache       │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Model Server   │  ← Chronos, TimesFM, Transformers
│ (TorchScript)   │  ← Low-latency inference
└────────┬────────┘
         ↓
┌─────────────────┐
│ Signal Generator│  ← Ensemble signals
│  + Risk Mgr     │  ← Position sizing, stop-loss
└────────┬────────┘
         ↓
┌─────────────────┐
│    Executor     │  ← Signal execution (stub)
│  + Monitoring   │  ← Logs, metrics, alerts
└─────────────────┘
```

## Key Features

- **Real-time Processing**: <100ms latency target
- **Modern Models**: Chronos T5/Bolt, TimesFM, Transformers (NO LSTM)
- **Incremental Features**: Fast feature computation, no recomputation
- **Ensemble Signals**: Multi-model consensus
- **Production Ready**: Model versioning, monitoring, automated retuning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your Bitget API keys

# Start the service
python scripts/start_service.py

# Run backtest
python backtest/backtest_engine.py

# Monitor
python monitoring/dashboard.py
```

## Directory Structure

- `ingest/` - Market data ingestion (WebSocket, candles)
- `features/` - Feature engineering (real-time computation)
- `models/` - Model definitions (Chronos, TimesFM, etc.)
- `serving/` - Model serving (low-latency inference)
- `strategy/` - Signal generation & risk management
- `backtest/` - Backtesting engine
- `train/` - Training & fine-tuning scripts
- `monitoring/` - Metrics, logs, alerts
- `config/` - Configuration files

## Models Used

1. **Chronos T5/Bolt** - Multi-horizon time series forecasting
2. **TimesFM** - Foundation model for time series
3. **Custom Transformers** - Temporal attention models
4. **LightGBM/XGBoost** - Fast, robust baseline

## Architecture Benefits

✅ Modular and testable
✅ Low-latency real-time inference
✅ Automated retraining/fine-tuning
✅ Proper monitoring and observability
✅ Production-ready deployment

