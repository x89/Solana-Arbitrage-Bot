# Real-Time Momentum Prediction Guide

## Quick Start

### Option 1: Simple Simulated Real-Time Loop

Run continuous predictions with simulated data:

```bash
python run_realtime_momentum.py
```

**Features:**
- ✅ Continuous predictions every 15 seconds
- ✅ Shows current momentum
- ✅ Chronos Bolt predictions
- ✅ Ensemble predictions (when buffer is full)
- ✅ Press **Ctrl+C** to stop gracefully

**Output Example:**
```
[2025-01-27 12:34:56] Iteration #42
  Current momentum: 0.123456
  Buffer size: 64
  Chronos Bolt prediction: 0.124789
  ✓ Ensemble prediction: 0.124123
  ✓ Confidence: 0.82
```

### Option 2: Real-Time with Live Data

Run with live data fetching:

```bash
python run_realtime_with_data.py
```

**Features:**
- ✅ Fetches live market data
- ✅ Calculates real momentum indicators
- ✅ Updates predictions continuously
- ✅ Shows bullish/bearish direction
- ✅ Press **Ctrl+C** to stop

## Configuration

### Adjust Prediction Interval

Edit the script to change update frequency:

```python
prediction_interval = 15  # seconds
```

### Change Symbol or Timeframe

Edit `run_realtime_with_data.py`:

```python
symbol = 'BTCUSDT'  # Change symbol
timeframe = '5m'    # Change timeframe
interval = 10       # Change update interval
```

## Stopping the System

Press **Ctrl+C** to stop gracefully. The system will:
1. Complete current prediction
2. Save any pending data
3. Shut down cleanly

## Advanced Usage

### Run in Background (Linux/Mac)

```bash
nohup python run_realtime_momentum.py > momentum.log 2>&1 &
```

### View Logs

```bash
tail -f momentum.log
```

### Run on Windows Background

```powershell
Start-Process python -ArgumentList "run_realtime_momentum.py" -WindowStyle Hidden
```

## Integration with Real Data Feeds

To use real market data, modify `fetch_live_data()` in `run_realtime_with_data.py`:

### Option A: Using Bitget (Your Current Setup)

```python
def fetch_live_data(self):
    from bitget_client import BitgetClient
    
    client = BitgetClient()
    data = client.get_latest_candles(
        symbol=self.symbol,
        timeframe=self.timeframe,
        limit=1
    )
    
    return data
```

### Option B: Using yfinance

```python
def fetch_live_data(self):
    import yfinance as yf
    
    ticker = yf.Ticker(self.symbol)
    data = ticker.history(period='1d', interval=self.timeframe)
    
    return data.tail(1)
```

### Option C: Using WebSocket (Real-Time)

```python
def fetch_live_data(self):
    # Connect to WebSocket feed
    # Process incoming tick data
    # Return DataFrame
    pass
```

## Monitoring Performance

### Check Prediction Accuracy

The system logs all predictions. Compare with actual values:

```python
# Actual vs Predicted
for i in range(len(actual_values)):
    error = abs(predicted[i] - actual_values[i])
    print(f"Error: {error:.6f}")
```

### Adjust Confidence Threshold

```python
validator = PredictionValidator(confidence_threshold=0.7)
```

## Troubleshooting

### Issue: "Buffer building..." keeps showing

**Solution**: Wait until buffer reaches 64 points. Or reduce `min_buffer_size`:

```python
min_buffer_size = 32  # Use smaller buffer
```

### Issue: Predictions are too slow

**Solution**: Use Chronos Bolt only (not T5):

```python
predictor = ChronosMomentumPredictor(model_type='bolt')
```

### Issue: CPU usage too high

**Solution**: Increase interval:

```python
prediction_interval = 30  # Wait 30 seconds between predictions
```

## Production Deployment

### 1. Create Windows Service

```powershell
# Install NSSM (Non-Sucking Service Manager)
# Create service
nssm install MomentumPredictor python.exe run_realtime_momentum.py
```

### 2. Use Systemd (Linux)

```bash
# Create service file
sudo nano /etc/systemd/system/momentum-predictor.service

[Unit]
Description=Real-Time Momentum Predictor
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/AI_momentum_real_time_predicting_system
ExecStart=/usr/bin/python3 run_realtime_momentum.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable momentum-predictor
sudo systemctl start momentum-predictor
```

## Output Interpretation

### Prediction Signals

- **BULLISH ↑**: Predicted momentum is rising (buy signal)
- **BEARISH ↓**: Predicted momentum is falling (sell signal)
- **NEUTRAL -**: Momentum is stable (hold signal)

### Confidence Levels

- **High (0.8+)**: Strong prediction, take action
- **Medium (0.6-0.8)**: Moderate prediction, be cautious
- **Low (<0.6)**: Weak prediction, wait for more confirmation

## Example Session

```bash
$ python run_realtime_momentum.py

======================================================================
Real-Time Momentum Prediction System
======================================================================

INFO: Loading modules...
INFO: ✓ Modules loaded
INFO: Initializing Chronos predictor...
INFO: ✓ Chronos predictor initialized
INFO: Prediction interval: 15 seconds
INFO: Starting real-time predictions...
======================================================================

[12:34:56] Iteration #1
  Current momentum: 0.123456
  Buffer size: 1
  ⏳ Buffer building... need 63 more points

[12:35:11] Iteration #2
  Current momentum: 0.124567
  Buffer size: 2
  ⏳ Buffer building... need 62 more points

...

[12:37:45] Iteration #64
  Current momentum: 0.135678
  Buffer size: 64
  Chronos Bolt prediction: 0.136123
  ✓ Ensemble prediction: 0.135890
  ✓ Confidence: 0.82
  Time taken: 0.45s

^C

INFO: Shutting down gracefully...
======================================================================
INFO: Real-time prediction stopped after 64 iterations
======================================================================
```

## Next Steps

1. **Run the system**: `python run_realtime_momentum.py`
2. **Monitor predictions**: Watch for bullish/bearish signals
3. **Integrate with trading**: Use predictions for automated trading
4. **Optimize**: Adjust intervals and parameters for your needs
5. **Backtest**: Test predictions against historical data

