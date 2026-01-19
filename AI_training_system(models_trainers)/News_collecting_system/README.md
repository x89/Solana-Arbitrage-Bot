# News Collection System

## Overview

Automated news collection and processing system for gathering financial news, articles, and sentiment data from multiple sources.

## Features

- ✅ Multi-source news aggregation
- ✅ Automated news collection
- ✅ Content processing and parsing
- ✅ Duplicate detection
- ✅ Sentiment analysis integration
- ✅ Scheduled collection

## Files

| File | Description |
|------|-------------|
| `news_collector.py` | Main news collection engine |
| `news_processor.py` | News processing and filtering |
| `config.py` | Configuration settings |

## Quick Start

### 1. Collect News

```python
from news_collector import NewsCollector

collector = NewsCollector()

# Collect news for symbol
news = collector.collect(
    symbol='SOLUSDT',
    limit=50
)

for article in news:
    print(f"{article['title']}: {article['sentiment']}")
```

### 2. Process News

```python
from news_processor import NewsProcessor

processor = NewsProcessor()

# Process and filter news
processed = processor.process(
    news=raw_news,
    filter_sentiment='positive',
    min_confidence=0.7
)
```

### 3. Scheduled Collection

```python
from news_collector import schedule_collection

# Collect news every hour
schedule_collection(
    symbol='SOLUSDT',
    interval=3600  # seconds
)
```

## Requirements

```
requests>=2.28.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
newspaper3k>=0.2.8
```

