# How to Run the Indicator Analysis System

## Quick Start

### Run the Demo
```bash
cd "Analyzing_indicators_comparing_momentum_system"
python run_indicators_demo.py
```

This will:
- Generate sample data
- Calculate technical indicators (with fallbacks if TA-Lib not installed)
- Analyze momentum
- Compare momentum across timeframes
- Test visualization components

## Installation

### Basic Requirements (Required)
```bash
pip install pandas numpy matplotlib seaborn
```

### Optional TA-Lib Installation

**Windows:**
```bash
# Download pre-built wheel from:
# https://github.com/cgohlke/talib-build/releases
pip install TA_Lib-0.4.25-cp312-cp312-win_amd64.whl
```

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**Mac:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Note:** Without TA-Lib, the system uses fallback calculations. Most functionality will work, but some advanced indicators may be limited.

## Usage Examples

### Example 1: Calculate Basic Indicators

```python
from indicator_calculator import IndicatorCalculator
import pandas as pd

calculator = IndicatorCalculator()

# Your OHLCV data
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Calculate all indicators
indicators = calculator.calculate_all(data)

# Access specific indicators
rsi = calculator.calculate_rsi(data, period=14)
macd = calculator.calculate_macd(data, fast=12, slow=26, signal=9)

print(f"RSI: {rsi[-1]}")
print(f"MACD: {macd['macd'][-1]}")
```

### Example 2: Momentum Analysis

```python
from indicator_analyzer import TechnicalIndicators, MomentumAnalyzer

# Initialize
indicators_calc = TechnicalIndicators()
momentum_analyzer = MomentumAnalyzer()

# Calculate indicators
data_with_indicators = indicators_calc.calculate_all_indicators(data)

# Analyze momentum
analysis = momentum_analyzer.analyze_momentum(
    data_with_indicators, 
    symbol='SOLUSDT', 
    timeframe='1h'
)

print(f"Momentum Score: {analysis.momentum_score}")
print(f"Trend: {analysis.trend_direction}")
print(f"Strength: {analysis.momentum_strength}")

for rec in analysis.recommendations:
    print(f"  - {rec}")
```

### Example 3: Compare Momentum Across Timeframes

```python
from momentum_comparator import MomentumComparator

comparator = MomentumComparator()

# Sample momentum data for different timeframes
momentum_data = {
    '1m': {'momentum_score': 0.3, 'trend_direction': 'bullish'},
    '5m': {'momentum_score': 0.5, 'trend_direction': 'bullish'},
    '15m': {'momentum_score': 0.7, 'trend_direction': 'bullish'},
    '1h': {'momentum_score': 0.2, 'trend_direction': 'bearish'}
}

comparison = comparator.compare_timeframes(momentum_data, 'SOLUSDT')

print(f"Timeframes: {comparison.timeframes}")
print(f"Divergence signals: {comparison.divergence_signals}")
```

### Example 4: Visualize Indicators

```python
from indicator_visualizer import IndicatorVisualizer

visualizer = IndicatorVisualizer()

# Plot price with indicators
fig = visualizer.plot_price_with_indicators(
    df=data_with_indicators,
    symbol='SOLUSDT',
    indicators=['sma_20', 'sma_50', 'bb_upper', 'bb_lower']
)

# Save or show
visualizer.save_plot(fig, 'indicators_chart.png')
```

## Available Features

### Technical Indicators

**Trend Indicators:**
- Moving Averages (SMA, EMA, WMA, DEMA, TEMA)
- MACD
- ADX (Average Directional Index)
- Parabolic SAR
- Aroon Oscillator

**Momentum Indicators:**
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)
- Momentum

**Volatility Indicators:**
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels

**Volume Indicators:**
- OBV (On-Balance Volume)
- A/D Line (Accumulation/Distribution)
- Chaikin A/D Oscillator
- Volume Profile

### Momentum Analysis

- Calculate momentum scores (-1 to 1)
- Determine trend direction (bullish/bearish/neutral)
- Assess momentum strength
- Generate trading recommendations
- Compare across timeframes

### Visualization

- Price charts with indicators
- Momentum analysis charts
- Correlation heatmaps
- Multi-timeframe comparisons
- Indicator comparisons

## Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    'periods': {
        'short': [5, 9, 12, 14],
        'medium': [20, 21, 26, 50],
        'long': [100, 200]
    },
    'analysis': {
        'momentum_score_range': (-1.0, 1.0),
        'signal_strength_thresholds': {
            'weak': 0.3,
            'moderate': 0.5,
            'strong': 0.7
        }
    },
    'symbols': ['SOLUSDT', 'BTCUSDT', 'ETHUSDT'],
    'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
}
```

## File Structure

```
Analyzing_indicators_comparing_momentum_system/
├── config.py                    # Configuration settings
├── indicator_calculator.py      # Indicator calculations
├── indicator_analyzer.py        # Momentum analysis
├── momentum_comparator.py       # Multi-timeframe comparison
├── indicator_visualizer.py      # Visualization
├── run_indicators_demo.py       # Demo script
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'talib'`
- **Solution:** Install TA-Lib or the system will use fallback calculations

**Issue:** `Cannot calculate indicators`
- **Solution:** Ensure data has 'open', 'high', 'low', 'close', 'volume' columns

**Issue:** `Too many NaN values`
- **Solution:** Use at least 200 data points for accurate indicator calculations

**Issue:** `Visualization errors`
- **Solution:** Install matplotlib and seaborn: `pip install matplotlib seaborn`

## Performance Tips

1. **Use sample data size:** 200+ points for best results
2. **Multiple timeframes:** Compare signals across different timeframes
3. **Combine indicators:** Use multiple indicators for confirmation
4. **Monitor divergence:** Watch for divergences across timeframes

## Next Steps

1. Run the demo: `python run_indicators_demo.py`
2. Integrate with your data source
3. Customize indicator periods in config.py
4. Test with your trading strategy

