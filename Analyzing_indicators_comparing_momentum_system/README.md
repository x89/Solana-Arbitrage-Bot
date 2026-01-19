# Indicator Analysis & Momentum Comparison

## Overview

Comprehensive technical indicator analysis system that calculates, visualizes, and compares momentum across multiple timeframes and indicators.

## Features

- ✅ Multiple technical indicators
- ✅ Momentum comparison across timeframes
- ✅ Visual indicator analysis
- ✅ Signal generation
- ✅ Indicator performance tracking

## Files

| File | Description |
|------|-------------|
| `indicator_calculator.py` | Technical indicator calculations |
| `indicator_analyzer.py` | Indicator analysis and comparison |
| `momentum_comparator.py` | Multi-timeframe momentum comparison |
| `indicator_visualizer.py` | Visualization utilities |
| `config.py` | Configuration settings |

## Supported Indicators

### Trend Indicators
- Moving Averages (SMA, EMA, WMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Ichimoku Cloud

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- Momentum

### Volatility Indicators
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Standard Deviation

### Volume Indicators
- On-Balance Volume (OBV)
- Volume Profile
- Accumulation/Distribution

## Quick Start

### 1. Calculate Indicators

```python
from indicator_calculator import IndicatorCalculator

calculator = IndicatorCalculator()

# Calculate all indicators
indicators = calculator.calculate_all(data)

print(f"RSI: {indicators['rsi']}")
print(f"MACD: {indicators['macd']}")
print(f"Bollinger Bands: {indicators['bb']}")
```

### 2. Compare Momentum

```python
from momentum_comparator import MomentumComparator

comparator = MomentumComparator()

# Compare across timeframes
comparison = comparator.compare(
    symbol='SOLUSDT',
    timeframes=['15m', '1h', '4h']
)

print(comparison)
```

### 3. Visualize

```python
from indicator_visualizer import IndicatorVisualizer

visualizer = IndicatorVisualizer()

# Create indicator charts
visualizer.plot_indicators(
    data=data,
    indicators=['rsi', 'macd', 'bollinger'],
    output='charts.png'
)
```

## Usage

### Analyze Multiple Indicators

```python
from indicator_analyzer import IndicatorAnalyzer

analyzer = IndicatorAnalyzer()

# Comprehensive analysis
analysis = analyzer.analyze(data)

print(f"""
Indicator Analysis:
- RSI: {analysis['rsi']['value']} ({analysis['rsi']['signal']})
- MACD: {analysis['macd']['signal']}
- Bollinger: {analysis['bb']['signal']}
- Overall: {analysis['overall_signal']}
""")
```

### Multi-Timeframe Comparison

```python
from momentum_comparator import compare_all_timeframes

# Compare momentum across all timeframes
comparison = compare_all_timeframes(
    symbol='SOLUSDT',
    timeframes=['5m', '15m', '1h', '4h', '1d']
)

# Get consensus
consensus = comparison['consensus']
print(f"Consensus: {consensus}")
```

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
ta>=0.10.0
pandas_ta>=0.3.14b0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Best Practices

1. **Use multiple indicators** - Don't rely on one indicator
2. **Confirm across timeframes** - Higher timeframe for trend
3. **Watch for divergences** - Early reversal signals
4. **Combine with price action** - Price is most important
5. **Monitor overbought/oversold** - Extreme readings

## Example Output

```python
{
    'rsi': {
        'value': 65.5,
        'signal': 'BULLISH',
        'overbought': False,
        'oversold': False
    },
    'macd': {
        'signal': 'BULLISH_CROSS',
        'histogram': 0.5
    },
    'bb': {
        'position': 'UPPER_BAND',
        'signal': 'OVERBOUGHT'
    },
    'overall_signal': 'BUY',
    'confidence': 0.75
}
```

