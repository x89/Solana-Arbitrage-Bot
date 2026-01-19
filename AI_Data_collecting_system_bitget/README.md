# Data Collection System - Bitget

## Overview

Real-time and historical cryptocurrency data collection from Bitget exchange. Provides high-quality market data for trading analysis and AI model training.

## Features

- ✅ Real-time OHLCV data collection
- ✅ Historical data retrieval
- ✅ WebSocket streaming support
- ✅ Data validation and processing
- ✅ Multiple timeframe support (1m, 5m, 15m, 1h, 1d)
- ✅ Automatic data storage
- ✅ Data quality checks

## Files

| File | Description |
|------|-------------|
| `bitget_client.py` | Bitget API client wrapper |
| `advanced_data_collector.py` | Advanced data collection with retry logic |
| `new_data_collector.py` | Simplified data collector |
| `data_processor.py` | Data cleaning and processing |
| `data_storage.py` | Data persistence layer |
| `data_validator.py` | Data quality validation |
| `config.py` | Configuration settings |

## Quick Start

### 1. Setup API Keys

```python
# .env file
BITGET_API_KEY=your_api_key
BITGET_API_SECRET=your_api_secret
```

### 2. Collect Data

```python
from bitget_client import BitgetClient

# Initialize client
client = BitgetClient(
    api_key=os.getenv('BITGET_API_KEY'),
    api_secret=os.getenv('BITGET_API_SECRET')
)

# Get klines (candlestick data)
data = client.get_klines('SOLUSDT', '15m', limit=500)
print(data.head())
```

### 3. Advanced Collection

```python
from advanced_data_collector import AdvancedDataCollector

collector = AdvancedDataCollector()

# Collect 1000 days of data
collector.collect_historical_data(
    symbol='SOLUSDT',
    timeframe='15m',
    days=1000
)
```

## Data Format

Collected data includes:

```python
{
    'timestamp': datetime,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'symbol': 'SOLUSDT'
}
```

## Usage

### Historical Data Collection

```python
# Collect last 30 days
collector = AdvancedDataCollector()
data = collector.get_historical_data('SOLUSDT', '15m', days=30)

# Save to JSON
collector.save_to_json(data, 'solusdt_15m_30days.json')
```

### Real-time Data Stream

```python
# Start real-time collection
collector.start_stream('SOLUSDT', '1m', callback=process_data)

def process_data(candle):
    print(f"New candle: {candle}")
    # Process data...
```

## Supported Timeframes

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

## Configuration

Edit `config.py`:

```python
# API Configuration
API_BASE_URL = "https://api.bitget.com"

# Collection Settings
MAX_RETRIES = 3
TIMEOUT = 30
COLLECTION_INTERVAL = 60  # seconds

# Data Storage
DATA_DIR = "Datas/financial_data"
```

## Data Validation

Automatic quality checks:

- ✅ Missing data detection
- ✅ Price anomaly detection
- ✅ Volume validation
- ✅ Timestamp verification
- ✅ Gap detection

## Error Handling

- Automatic retry on failures
- Exponential backoff
- Rate limit handling
- Connection recovery

## Requirements

```
requests>=2.28.0
python-dotenv>=0.19.0
pandas>=1.3.0
websocket-client>=1.0.0
```

## Storage

Data is stored in `Datas/` directory:

- `financial_data/` - Basic OHLCV data
- `enhanced_financial_data/` - Enriched data
- JSON format for easy access

## API Rate Limits

Bitget API limits:
- Public endpoints: 20 requests/second
- Data endpoints: 10 requests/second

The system automatically handles rate limits.

## Example Output

```json
{
    "timestamp": "2025-01-26T10:00:00",
    "open": 95.50,
    "high": 96.20,
    "low": 95.30,
    "close": 96.00,
    "volume": 1250000,
    "symbol": "SOLUSDT"
}
```

