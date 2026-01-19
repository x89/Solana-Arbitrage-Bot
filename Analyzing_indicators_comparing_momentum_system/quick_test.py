"""
Quick test of the real-time system - runs once then exits
"""
import time
import sys
import pandas as pd
from datetime import datetime
from indicator_calculator import IndicatorCalculator
from indicator_analyzer import MomentumAnalyzer

def generate_live_data(n_samples=200):
    """Generate continuous live data for testing"""
    import numpy as np
    import pandas as pd
    
    np.random.seed(int(time.time()))
    base_price = 100 + np.random.uniform(-10, 10)
    prices = [base_price]
    
    for i in range(n_samples - 1):
        change = np.random.normal(0, 0.02)
        trend = np.sin(i / 50) * 0.01
        new_price = prices[-1] * (1 + change + trend)
        prices.append(new_price)
    
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start=datetime.now() - pd.Timedelta(days=n_samples),
                             periods=n_samples, freq='1h')
    
    return df

print("\n" + "="*70)
print("Quick Test - Real-Time Indicator System")
print("="*70 + "\n")

# Generate data
print("[1] Generating test data...")
data = generate_live_data(200)
print(f"    Generated {len(data)} data points")
print(f"    Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

# Calculate indicators
print("\n[2] Calculating all indicators...")
calculator = IndicatorCalculator()
indicators = calculator.calculate_all(data, include_advanced=True)
print(f"    Total indicators: {len(indicators.columns) - len(data.columns)}")
print(f"    Columns: {list(indicators.columns)[5:15]}...")  # Show first few

# Get advanced indicators
print("\n[3] Checking advanced indicators...")

# DTFX Zones
try:
    zones = calculator.get_dtfx_zones(data, structure_len=10)
    print(f"    [OK] DTFX Zones: {len(zones)} zones found")
except Exception as e:
    print(f"    [FAIL] DTFX Zones: {e}")

# ZigZag
try:
    zigzag = calculator.get_zigzag_points(data, momentum_type='macd')
    print(f"    [OK] ZigZag: {len(zigzag)} points found")
except Exception as e:
    print(f"    [FAIL] ZigZag: {e}")

# Momentum Analysis
print("\n[4] Momentum Analysis:")
try:
    momentum_analyzer = MomentumAnalyzer()
    momentum = momentum_analyzer.analyze_momentum(indicators, 'SOLUSDT', '1h')
    if momentum:
        print(f"    [OK] Momentum Score: {momentum.momentum_score:.3f}")
        print(f"    [OK] Trend: {momentum.trend_direction}")
        print(f"    [OK] Strength: {momentum.momentum_strength}")
except Exception as e:
    print(f"    [FAIL] Momentum Analysis: {e}")

# Current values
print("\n[5] Current Indicator Values:")
current = indicators.iloc[-1]
indicators_to_check = ['cci', 'rsi_14', 'macd', 'supertrend_direction', 'bb_position_advanced']

for ind in indicators_to_check:
    if ind in indicators.columns:
        val = current[ind]
        if pd.notna(val):
            print(f"    {ind}: {val:.4f}")
        else:
            print(f"    {ind}: N/A")
    else:
        print(f"    {ind}: Not calculated")

print("\n" + "="*70)
print("[SUCCESS] Quick test complete!")
print("="*70)
print("\nTo run in REAL-TIME mode, use:")
print("  python main.py")
print("\nPress Ctrl+C to stop when running in real-time mode.")
print("="*70 + "\n")

