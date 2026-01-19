#!/usr/bin/env python3
"""
Real-Time Indicators & Momentum Analysis
Runs continuously and analyzes indicators in real-time
Press Ctrl+C to stop
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_live_data(n_samples=200):
    """Generate continuous live data for testing"""
    np.random.seed(int(time.time()))  # Use time as seed for variety
    
    base_price = 100 + np.random.uniform(-10, 10)
    prices = [base_price]
    
    for i in range(n_samples - 1):
        # Simulate more realistic price movements
        change = np.random.normal(0, 0.02)
        trend = np.sin(i / 50) * 0.01  # Add some trend
        new_price = prices[-1] * (1 + change + trend)
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

def analyze_signals(indicators_df: pd.DataFrame, current_price: float) -> Dict:
    """Analyze current signals from indicators"""
    try:
        latest = indicators_df.iloc[-1]
        signals = []
        signal_strength = []
        
        # RSI Signal
        if 'rsi_14' in indicators_df.columns and not pd.isna(latest['rsi_14']):
            rsi = latest['rsi_14']
            if rsi > 70:
                signals.append("RSI: OVERBOUGHT ⚠️")
                signal_strength.append(-1.0)
            elif rsi < 30:
                signals.append("RSI: OVERSOLD 🟢")
                signal_strength.append(1.0)
        
        # MACD Signal
        if 'macd' in indicators_df.columns and not pd.isna(latest['macd']):
            macd = latest['macd']
            macd_signal = latest['macd_signal'] if 'macd_signal' in indicators_df.columns else 0
            if macd > macd_signal:
                signals.append("MACD: BULLISH 📈")
                signal_strength.append(0.5)
            else:
                signals.append("MACD: BEARISH 📉")
                signal_strength.append(-0.5)
        
        # Supertrend Signal
        if 'supertrend_direction' in indicators_df.columns and not pd.isna(latest['supertrend_direction']):
            st_dir = latest['supertrend_direction']
            if st_dir > 0:
                signals.append("Supertrend: BULLISH ✅")
                signal_strength.append(0.7)
            elif st_dir < 0:
                signals.append("Supertrend: BEARISH ❌")
                signal_strength.append(-0.7)
        
        # Bollinger Position
        if 'bb_position_advanced' in indicators_df.columns and not pd.isna(latest['bb_position_advanced']):
            bb_pos = latest['bb_position_advanced']
            if bb_pos > 0.8:
                signals.append("BB: Near Upper Band (SELL) 📊")
                signal_strength.append(-0.6)
            elif bb_pos < 0.2:
                signals.append("BB: Near Lower Band (BUY) 📊")
                signal_strength.append(0.6)
        
        # CCI Signal
        if 'cci' in indicators_df.columns and not pd.isna(latest['cci']):
            cci = latest['cci']
            if cci > 100:
                signals.append("CCI: VERY BULLISH 🚀")
                signal_strength.append(1.0)
            elif cci < -100:
                signals.append("CCI: VERY BEARISH 🔻")
                signal_strength.append(-1.0)
            elif cci > 0:
                signals.append("CCI: BULLISH")
                signal_strength.append(0.3)
            elif cci < 0:
                signals.append("CCI: BEARISH")
                signal_strength.append(-0.3)
        
        # Calculate overall signal
        if signal_strength:
            total_strength = sum(signal_strength) / len(signal_strength)
            
            if total_strength > 0.5:
                overall = "STRONG BUY 🟢🟢🟢"
            elif total_strength > 0:
                overall = "BUY 🟢"
            elif total_strength < -0.5:
                overall = "STRONG SELL 🔴🔴🔴"
            elif total_strength < 0:
                overall = "SELL 🔴"
            else:
                overall = "NEUTRAL ⚪"
        else:
            overall = "NO SIGNAL"
            total_strength = 0
        
        return {
            'signals': signals,
            'overall': overall,
            'strength': total_strength,
            'signal_count': len(signals)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing signals: {e}")
        return {'signals': [], 'overall': 'NO SIGNAL', 'strength': 0, 'signal_count': 0}

def run_realtime_analysis(symbol='SOLUSDT', update_interval=60):
    """Run real-time indicator analysis"""
    try:
        print("=" * 70)
        print("Real-Time Indicator Analysis System")
        print("=" * 70)
        print(f"\nSymbol: {symbol}")
        print(f"Update Interval: {update_interval} seconds")
        print(f"Press Ctrl+C to stop\n")
        
        from indicator_calculator import IndicatorCalculator
        from indicator_analyzer import MomentumAnalyzer
        
        calculator = IndicatorCalculator()
        momentum_analyzer = MomentumAnalyzer()
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n{'='*70}")
                print(f"Update #{iteration} - {current_time}")
                print(f"{'='*70}")
                
                # Generate new data (in production, fetch from API)
                print("\n[1] Fetching market data...")
                data = generate_live_data(200)
                print(f"    Data points: {len(data)}")
                print(f"    Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                
                # Calculate all indicators
                print("\n[2] Calculating indicators...")
                try:
                    indicators_df = calculator.calculate_all(data, include_advanced=True)
                    print(f"    Indicators calculated: {len(indicators_df.columns) - len(data.columns)}")
                except Exception as e:
                    print(f"    Error: {e}")
                    time.sleep(update_interval)
                    continue
                
                # Get current price
                current_price = data['close'].iloc[-1]
                previous_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
                
                print(f"\n[3] Current Market Status:")
                print(f"    Price: ${current_price:.2f}")
                if price_change >= 0:
                    print(f"    Change: +${price_change:.2f} (+{price_change_pct:.2f}%) 🟢")
                else:
                    print(f"    Change: ${price_change:.2f} ({price_change_pct:.2f}%) 🔴")
                
                # Analyze signals
                print("\n[4] Trading Signals:")
                signal_data = analyze_signals(indicators_df, current_price)
                
                print(f"    Overall Signal: {signal_data['overall']}")
                print(f"    Signal Strength: {signal_data['strength']:.2f}")
                print(f"    Active Signals: {signal_data['signal_count']}")
                
                if signal_data['signals']:
                    print("\n    Signal Details:")
                    for signal in signal_data['signals']:
                        print(f"      • {signal}")
                
                # Advanced indicators
                print("\n[5] Advanced Indicators:")
                
                # DTFX Zones
                try:
                    zones = calculator.get_dtfx_zones(data, structure_len=10)
                    print(f"    DTFX Zones: {len(zones)} zones found")
                    if zones:
                        latest_zone = zones[-1]
                        zone_type = "BULLISH" if latest_zone.direction == 1 else "BEARISH"
                        print(f"    Latest zone: {zone_type} ({latest_zone.bottom:.2f} - {latest_zone.top:.2f})")
                except Exception as e:
                    print(f"    DTFX Zones: Error - {e}")
                
                # ZigZag
                try:
                    zigzag = calculator.get_zigzag_points(data, momentum_type='macd')
                    print(f"    ZigZag Points: {len(zigzag)} points")
                    if zigzag:
                        latest_zz = zigzag[-1]
                        direction = "UP" if latest_zz.direction == 1 else "DOWN"
                        print(f"    Latest ZigZag: {direction} at ${latest_zz.price:.2f}")
                except Exception as e:
                    print(f"    ZigZag: Error - {e}")
                
                # Momentum Analysis
                print("\n[6] Momentum Analysis:")
                try:
                    momentum_analysis = momentum_analyzer.analyze_momentum(
                        indicators_df, symbol, '1h'
                    )
                    if momentum_analysis:
                        print(f"    Momentum Score: {momentum_analysis.momentum_score:.3f}")
                        print(f"    Trend: {momentum_analysis.trend_direction.upper()}")
                        print(f"    Strength: {momentum_analysis.momentum_strength.upper()}")
                        
                        if momentum_analysis.recommendations:
                            print(f"\n    Recommendations:")
                            for i, rec in enumerate(momentum_analysis.recommendations[:2], 1):
                                print(f"      {i}. {rec}")
                except Exception as e:
                    print(f"    Error: {e}")
                
                # Summary
                print(f"\n[7] Summary:")
                print(f"    Overall Signal: {signal_data['overall']}")
                print(f"    Current Price: ${current_price:.2f}")
                print(f"    24h Change: {price_change_pct:.2f}%")
                print(f"    Next Update: {update_interval} seconds")
                
                print(f"\n{'='*70}")
                print(f"Sleeping for {update_interval} seconds... (Press Ctrl+C to stop)")
                print(f"{'='*70}")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Stopping real-time analysis...")
            print("="*70)
            print(f"\nAnalysis ran for {iteration} updates")
            print("Thank you for using Real-Time Indicator Analysis!")
            print("="*70)
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error in real-time analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Indicator Analysis')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Trading symbol')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     Real-Time Indicator Analysis System                         ║
    ║     Analyzing Indicators & Comparing Momentum                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nStarting real-time analysis...")
    print("This system will:")
    print("  • Calculate 50+ technical indicators")
    print("  • Analyze momentum across timeframes")
    print("  • Detect DTFX zones and ZigZag points")
    print("  • Generate trading signals in real-time")
    print("\nPress Ctrl+C to stop at any time\n")
    
    run_realtime_analysis(symbol=args.symbol, update_interval=args.interval)

if __name__ == "__main__":
    main()

