# AI Trading Signal Generation

## Overview

Comprehensive AI-powered trading signal generation system that combines multiple AI models, technical analysis, and risk management to generate high-confidence trading signals.

## Features

- ✅ Multi-model AI ensemble predictions
- ✅ Technical indicator integration
- ✅ Risk-adjusted signal scoring
- ✅ Confidence-based filtering
- ✅ Real-time signal generation
- ✅ Risk management integration
- ✅ Position sizing

## Files

| File | Description |
|------|-------------|
| `signal_generator.py` | Main signal generation engine |
| `ai_models.py` | AI model integration |
| `technical_analysis.py` | Technical indicator calculations |
| `risk_manager.py` | Risk management and position sizing |
| `config.py` | Configuration settings |
| `setup.py` | Setup and installation |

## Signal Types

| Signal | Strength | Description |
|--------|----------|-------------|
| **BUY** | STRONG | High confidence upward prediction |
| **SELL** | STRONG | High confidence downward prediction |
| **HOLD** | - | Wait for better opportunity |

## Quick Start

### 1. Initialize Signal Generator

```python
from signal_generator import SignalGenerator
from config import Config

config = Config()
generator = SignalGenerator(config)
```

### 2. Generate Signal

```python
# Generate trading signal
signal = generator.generate_signal(
    symbol='SOLUSDT',
    timeframe='15m'
)

print(f"Signal: {signal.action}")
print(f"Confidence: {signal.confidence}")
print(f"Strength: {signal.strength}")
```

### 3. With Risk Management

```python
from risk_manager import RiskManager

risk_manager = RiskManager(config)

signal = generator.generate_signal('SOLUSDT', '15m')

# Get position size based on risk
position_size = risk_manager.calculate_position_size(
    signal=signal,
    account_balance=10000,
    risk_per_trade=0.02  # 2% risk
)

print(f"Position size: {position_size}")
```

## Usage

### Signal Generation

```python
from signal_generator import generate_signal

# Generate signal for current market
signal = generate_signal(
    symbol='SOLUSDT',
    timeframe='15m',
    use_all_models=True
)

# Signal properties
signal.action         # 'BUY', 'SELL', 'HOLD'
signal.confidence     # 0.0 - 1.0
signal.strength       # 'WEAK', 'MODERATE', 'STRONG'
signal.models_used    # List of models
signal.reasons        # Explanation
```

### Multi-Timeframe Analysis

```python
# Analyze multiple timeframes
timeframes = ['15m', '1h', '4h']

signals = []
for tf in timeframes:
    signal = generator.generate_signal('SOLUSDT', tf)
    signals.append(signal)

# Consensus signal
consensus = generator.get_consensus(signals)
```

### Technical Analysis Integration

```python
from technical_analysis import TechnicalAnalyzer

ta = TechnicalAnalyzer(data)

# Calculate indicators
rsi = ta.rsi(period=14)
macd = ta.macd()
bb = ta.bollinger_bands()

# Combine with AI signals
signal = generator.generate_signal_with_ta(data)
```

## Signal Components

### AI Model Signals
- Forecasting predictions
- Pattern detection
- Sentiment analysis
- Momentum indicators

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Fisher Transform
- CCI (Commodity Channel Index)

### Risk Factors
- Volatility assessment
- Trend strength
- Support/resistance levels
- Volume analysis

## Configuration

Edit `config.py`:

```python
@dataclass
class SignalConfig:
    # Signal Weights
    WEIGHTS = {
        'forecast': 0.25,
        'pattern': 0.20,
        'sentiment': 0.15,
        'technical': 0.25,
        'ml': 0.15
    }
    
    # Thresholds
    MIN_CONFIDENCE = 0.6
    MIN_STRENGTH = 'MODERATE'
    
    # Models
    USE_FORECAST = True
    USE_PATTERNS = True
    USE_SENTIMENT = True
```

## Risk Management

### Position Sizing

```python
from risk_manager import calculate_position

position = calculate_position(
    signal=signal,
    account_value=10000,
    risk_per_trade=0.02,
    stop_loss_pct=0.05
)

# position = {
#     'quantity': 95.0,
#     'value': 9500,
#     'stop_loss': 0.05,
#     'take_profit': 0.125
# }
```

### Risk Monitoring

```python
from risk_manager import RiskMonitor

monitor = RiskMonitor()

# Check daily limits
if monitor.check_daily_loss() > 0.05:
    print("Daily loss limit reached")

# Check drawdown
if monitor.check_drawdown() > 0.15:
    print("Max drawdown reached")
```

## Signal Validation

```python
from signal_generator import validate_signal

# Validate before executing
validation = validate_signal(signal)

if validation.valid:
    print(f"Signal is valid: {validation.reason}")
else:
    print(f"Signal rejected: {validation.reason}")
```

## Integration Example

### Complete Trading Workflow

```python
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from backtesting import BacktestEngine

# 1. Generate signal
generator = SignalGenerator(config)
signal = generator.generate_signal('SOLUSDT', '15m')

# 2. Calculate risk-adjusted position
risk_manager = RiskManager(config)
position = risk_manager.calculate_position_size(signal, 10000, 0.02)

# 3. Validate signal
if signal.confidence > 0.7 and position.valid:
    # 4. Execute (in backtesting)
    backtester = BacktestEngine()
    results = backtester.execute_signal(signal, position)
    
    print(f"P&L: {results.pnl}")
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
ta>=0.10.0
transformers>=4.30.0
torch>=1.12.0
```

## Performance

- **Signal Generation**: <100ms
- **Multi-Model Analysis**: <500ms
- **Risk Calculation**: <50ms
- **Total Latency**: <1 second

## Best Practices

1. **Always use risk management** - Never risk more than 2% per trade
2. **Wait for strong signals** - Minimum confidence 0.7
3. **Use stop losses** - Never trade without stop loss
4. **Monitor drawdown** - Stop if drawdown exceeds 15%
5. **Backtest first** - Test strategies before live trading

## Troubleshooting

**Issue**: Low confidence signals
- Check data quality
- Verify AI models are loaded
- Ensure sufficient historical data

**Issue**: Signals too frequent
- Increase minimum confidence threshold
- Add confirmation period
- Filter by signal strength

**Issue**: Poor signal quality
- Retrain AI models
- Adjust signal weights
- Use multi-timeframe confirmation

