# Sentiment Analysis System

## Overview

Advanced sentiment analysis system using transformer models to analyze news, social media, and text data for trading decisions.

## Features

- ✅ Transformer-based sentiment analysis
- ✅ News sentiment processing
- ✅ Real-time sentiment monitoring
- ✅ Sentiment scoring and classification
- ✅ Model training and evaluation

## Files

| File | Description |
|------|-------------|
| `sentiment_analyzer.py` | Main sentiment analysis |
| `sentiment_trainer.py` | Model training |
| `sentiment_evaluator.py` | Model evaluation |
| `sentiment_collector.py` | Data collection |

## Quick Start

### 1. Analyze Sentiment

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze text
result = analyzer.analyze(
    text="Bitcoin breaks new all-time high!",
    model='transformer'
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

### 2. Train Model

```python
from sentiment_trainer import SentimentTrainer

trainer = SentimentTrainer()

# Train on custom data
model = trainer.train(
    data=training_data,
    epochs=10,
    learning_rate=2e-5
)
```

## Requirements

```
transformers>=4.30.0
torch>=1.12.0
nltk>=3.8.0
textblob>=0.17.0
```

