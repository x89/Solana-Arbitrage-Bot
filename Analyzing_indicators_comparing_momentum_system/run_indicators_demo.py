#!/usr/bin/env python3
"""
Indicator Analysis Demo
Demonstrates how to use the indicator analysis system
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=200):
    """Generate sample OHLCV data"""
    np.random.seed(42)
    
    base_price = 100
    prices = [base_price]
    
    for i in range(n_samples - 1):
        change = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
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
    df.index = pd.date_range(start=datetime.now() - timedelta(days=n_samples), 
                             periods=n_samples, freq='1h')
    
    return df

def main():
    print("=" * 70)
    print("Indicator Analysis & Momentum Comparison Demo")
    print("=" * 70)
    
    # Generate sample data
    print("\n[1] Generating sample data...")
    data = generate_sample_data(200)
    print(f"Generated {len(data)} data points")
    print(data.head())
    
    # Test indicator analyzer
    print("\n[2] Testing indicator analyzer...")
    try:
        from indicator_analyzer import TechnicalIndicators, MomentumAnalyzer, IndicatorManager
        
        # Initialize components
        indicators_calc = TechnicalIndicators()
        momentum_analyzer = MomentumAnalyzer()
        
        print("[OK] Components initialized")
        
        # Calculate indicators (will use fallbacks if TA-Lib not available)
        print("\n[3] Calculating technical indicators...")
        if hasattr(indicators_calc, 'calculate_all_indicators'):
            data_with_indicators = indicators_calc.calculate_all_indicators(data)
        else:
            data_with_indicators = indicators_calc.calculate_all(data, include_advanced=True)
        
        print(f"[OK] Indicators calculated")
        print(f"Original columns: {len(data.columns)}")
        print(f"Total columns now: {len(data_with_indicators.columns)}")
        print(f"New indicators: {len(data_with_indicators.columns) - len(data.columns)}")
        
        # Check for advanced indicators
        advanced_cols = [col for col in data_with_indicators.columns if 'supertrend' in col or 'cci' in col or 'advanced' in col]
        if advanced_cols:
            print(f"\nAdvanced indicators found: {advanced_cols[:5]}...")
        
        # Test advanced indicators
        print("\n[3.1] Testing advanced indicators...")
        try:
            from indicator_calculator import IndicatorCalculator
            
            advanced_calc = IndicatorCalculator()
            
            # Test DTFX zones
            print("  - Testing DTFX zones...")
            zones = advanced_calc.get_dtfx_zones(data, structure_len=10)
            if zones:
                print(f"    Found {len(zones)} zones")
                if zones[-1]:
                    print(f"    Latest zone: {zones[-1].direction} ({zones[-1].bottom:.2f} - {zones[-1].top:.2f})")
            
            # Test ZigZag
            print("  - Testing ZigZag momentum...")
            zigzag_points = advanced_calc.get_zigzag_points(data, momentum_type='macd')
            if zigzag_points:
                print(f"    Found {len(zigzag_points)} ZigZag points")
            
        except Exception as e:
            print(f"    [WARNING] Advanced indicators not fully available: {e}")
        
        # Test momentum analysis
        print("\n[4] Testing momentum analysis...")
        config = {
            'symbols': ['SOLUSDT'],
            'timeframes': ['1h']
        }
        manager = IndicatorManager(config)
        
        analysis = manager.analyze_symbol(data, 'SOLUSDT', '1h')
        
        if analysis:
            print("[OK] Momentum analysis completed")
            print(f"Momentum Score: {analysis.momentum_score:.3f}")
            print(f"Trend Direction: {analysis.trend_direction}")
            print(f"Momentum Strength: {analysis.momentum_strength}")
            
            if analysis.recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(analysis.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        else:
            print("[INFO] Momentum analysis returned None (may be due to missing TA-Lib)")
        
    except ImportError as e:
        print(f"[WARNING] Import error: {e}")
        print("Some features may not be available without TA-Lib")
    except Exception as e:
        print(f"[ERROR] Error in indicator analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Test momentum comparator
    print("\n[5] Testing momentum comparator...")
    try:
        from momentum_comparator import MomentumComparator
        
        comparator = MomentumComparator()
        print("[OK] MomentumComparator initialized")
        
        # Create sample momentum data
        sample_momentum_data = {
            '1m': {'momentum_score': 0.5, 'trend_direction': 'bullish', 'momentum_strength': 'moderate'},
            '5m': {'momentum_score': 0.6, 'trend_direction': 'bullish', 'momentum_strength': 'strong'},
            '15m': {'momentum_score': 0.7, 'trend_direction': 'bullish', 'momentum_strength': 'strong'},
            '1h': {'momentum_score': 0.4, 'trend_direction': 'bullish', 'momentum_strength': 'moderate'}
        }
        
        comparison = comparator.compare_timeframes(sample_momentum_data, 'SOLUSDT')
        
        if comparison:
            print("[OK] Timeframe comparison completed")
            print(f"Timeframes: {', '.join(comparison.timeframes)}")
            print(f"Divergence signals: {len(comparison.divergence_signals)}")
            if comparison.recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(comparison.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
    except Exception as e:
        print(f"[ERROR] Error in momentum comparator: {e}")
    
    # Test visualizer
    print("\n[6] Testing visualizer...")
    try:
        from indicator_visualizer import IndicatorVisualizer
        
        visualizer = IndicatorVisualizer()
        print("[OK] IndicatorVisualizer initialized")
        
        # Note: Don't actually show plots in demo
        print("[INFO] Visualization components ready")
        
    except Exception as e:
        print(f"[WARNING] Visualization error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print("\n[SUCCESS] Basic tests completed!")
    print("\nNote: Full functionality requires TA-Lib installation.")
    print("To install TA-Lib:")
    print("  Windows: Download wheel from https://github.com/cgohlke/talib-build/releases")
    print("  Linux/Mac: pip install TA-Lib")
    print("\nThe system will use fallback calculations when TA-Lib is not available.")
    print("=" * 70)

if __name__ == "__main__":
    main()

