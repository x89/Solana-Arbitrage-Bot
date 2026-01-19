# Pattern Detection System

## Overview

AI-powered chart pattern detection system using YOLO (You Only Look Once) models to identify trading patterns in candlestick charts.

## Features

- ✅ YOLO-based pattern recognition
- ✅ Real-time pattern detection
- ✅ Multiple pattern types supported
- ✅ Chart generation and visualization
- ✅ Pattern confidence scoring
- ✅ Historical pattern matching

## Files

| File | Description |
|------|-------------|
| `pattern_detector.py` | Main pattern detection engine |
| `pattern_trainer.py` | YOLO model training |
| `chart_generator.py` | Chart generation utilities |
| `yolo_config.py` | YOLO configuration |

## Supported Patterns

| Pattern | Type | Description |
|---------|------|-------------|
| **Head and Shoulders** | Reversal | Bearish reversal pattern |
| **Double Top** | Reversal | Bearish reversal pattern |
| **Double Bottom** | Reversal | Bullish reversal pattern |
| **Triangles** | Continuation | Consolidation patterns |
| **Flags** | Continuation | Short continuation |
| **Wedges** | Continuation | Trend continuation |

## Quick Start

### 1. Basic Pattern Detection

```python
from pattern_detector import PatternDetector

detector = PatternDetector()

# Detect patterns in data
patterns = detector.detect(
    data=chart_data,
    timeframe='15m'
)

for pattern in patterns:
    print(f"{pattern.name}: {pattern.confidence:.2%}")
```

### 2. Generate Chart

```python
from chart_generator import ChartGenerator

generator = ChartGenerator()

# Create candlestick chart
chart = generator.create_chart(
    data=data,
    width=800,
    height=600,
    pattern_overlay=True
)

# Save chart
chart.save('candlestick_chart.png')
```

### 3. Train Custom Model

```python
from pattern_trainer import PatternTrainer

trainer = PatternTrainer()

# Train on custom data
trainer.train(
    data_dir='training_data/',
    epochs=50,
    batch_size=16
)

# Save trained model
trainer.save_model('custom_pattern_model.pt')
```

## Usage

### Real-time Detection

```python
from pattern_detector import RealTimePatternDetector

detector = RealTimePatternDetector()

# Monitor real-time patterns
def on_pattern_detected(pattern):
    print(f"Pattern detected: {pattern.name}")
    print(f"Confidence: {pattern.confidence}")

detector.start_monitoring(
    symbol='SOLUSDT',
    timeframe='15m',
    callback=on_pattern_detected
)
```

### Pattern Analysis

```python
from pattern_detector import analyze_patterns

# Analyze detected patterns
analysis = analyze_patterns(
    patterns=detected_patterns,
    historical_data=data
)

print(f"Pattern Success Rate: {analysis['success_rate']}")
print(f"Average Confidence: {analysis['avg_confidence']}")
```

### Multi-Pattern Detection

```python
# Detect multiple pattern types
patterns = detector.detect_all_patterns(
    data=data,
    pattern_types=['head_shoulders', 'double_top', 'triangles']
)

# Get strongest pattern
strongest = max(patterns, key=lambda p: p.confidence)
print(f"Strongest: {strongest.name}")
```

## Model Training

### Prepare Training Data

```python
from chart_generator import prepare_training_data

# Generate training data
prepare_training_data(
    data=historical_data,
    output_dir='training_data/',
    pattern_types=['head_shoulders', 'double_top'],
    num_samples=1000
)
```

### Train Model

```python
from pattern_trainer import train_yolo_model

# Train custom YOLO model
model = train_yolo_model(
    data_dir='training_data/',
    config='yolo_config.py',
    epochs=100,
    batch_size=32,
    img_size=640
)
```

### Evaluate Model

```python
from pattern_trainer import evaluate_model

# Evaluate trained model
metrics = evaluate_model(
    model=model,
    test_data='test_data/'
)

print(f"mAP50: {metrics['map50']}")
print(f"mAP50-95: {metrics['map50_95']}")
```

## Configuration

Edit `yolo_config.py`:

```python
class YOLOConfig:
    # Model Settings
    MODEL_SIZE = 'yolov8n'  # nano
    IMG_SIZE = 640
    
    # Training Settings
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Pattern Settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Classes
    CLASSES = [
        'head_shoulders',
        'double_top',
        'double_bottom',
        'triangles'
    ]
```

## Integration

### With Trading System

```python
from pattern_detector import PatternDetector
from trading_system import TradingSystem

# Initialize systems
detector = PatternDetector()
trader = TradingSystem()

# Detect patterns
patterns = detector.detect(data)

# Use patterns in trading
for pattern in patterns:
    if pattern.confidence > 0.7:
        if pattern.is_bullish():
            trader.place_buy_order()
        elif pattern.is_bearish():
            trader.place_sell_order()
```

### With Forecasting

```python
from pattern_detector import PatternAwareForecaster

# Combine patterns with forecasting
forecaster = PatternAwareForecaster()

# Forecast with pattern context
forecast = forecaster.forecast(
    data=data,
    patterns=detected_patterns,
    horizon=64
)
```

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.12.0
torchvision>=0.13.0
matplotlib>=3.5.0
Pillow>=9.0.0
```

## Model Performance

- **mAP50**: 0.85+
- **Inference Speed**: <50ms per image
- **Accuracy**: 80-90% for common patterns
- **Model Size**: ~6MB (YOLOv8n)

## Best Practices

1. **Use high-quality data** - Clean, accurate OHLCV data
2. **Normalize charts** - Consistent chart appearance
3. **Label accurately** - Precise pattern annotations
4. **Augment data** - More training samples
5. **Validate patterns** - Check detected patterns manually
6. **Monitor performance** - Track detection accuracy

## Troubleshooting

**Issue**: Low detection accuracy
- Use higher quality training data
- Increase model size
- Train for more epochs
- Add data augmentation

**Issue**: False positives
- Adjust confidence threshold
- Increase IOU threshold
- Add validation filters

**Issue**: Slow inference
- Use smaller model (nano)
- Optimize image size
- Use GPU acceleration

