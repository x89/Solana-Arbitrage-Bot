# Signal Testing System

## Overview

Comprehensive signal testing framework for validating trading signals, tracking performance, and conducting A/B tests on different strategies.

## Features

- ✅ Signal quality validation
- ✅ Performance tracking
- ✅ A/B testing capabilities
- ✅ Database storage
- ✅ Statistical analysis

## Quick Start

### 1. Test Signal

```python
from signal_tester import SignalTester

tester = SignalTester()

# Test trading signal
result = tester.test_signal(
    signal=signal,
    historical_data=data
)

print(f"Win Rate: {result['win_rate']}")
print(f"Average Return: {result['avg_return']}")
```

### 2. A/B Testing

```python
# Compare two strategies
results = tester.ab_test(
    strategy_a=strategy_v1,
    strategy_b=strategy_v2,
    data=test_data
)
```

## Requirements

```
pandas>=1.3.0
sqlalchemy>=2.0.0
pytest>=7.0.0
```

