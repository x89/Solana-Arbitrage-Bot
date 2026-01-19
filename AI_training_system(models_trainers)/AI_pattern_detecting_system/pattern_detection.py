import pandas as pd
import numpy as np
import mplfinance as mpf
import os
import sys
import matplotlib
matplotlib.use('Agg')



# Load YOLO models for pattern detection
class ChartPatternDetector:
    def __init__(self):
        pass

    def detect_patterns(self, df: pd.DataFrame, trend_score: int = 0):
        """
        Detect chart and candlestick patterns in the given price data using code logic only.
        Returns:
            tuple: (chart_pattern_score, candlestick_score)
        """
        data = df.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            print("Warning: Missing required OHLCV columns")
            return 0, 0
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            elif 'timestamp' in data.columns:
                data = data.set_index('timestamp')
        if data.empty:
            print("Warning: DataFrame is empty; no chart to analyze.")
            return 0, 0

        # Chart pattern detection
        chart_patterns = self.detect_chart_patterns_code(data, trend_score)
        print(f"Chart patterns: {chart_patterns}")
        last_chart_score = 0
        prev_chart_score = 0
        for _, name in chart_patterns:
            last_chart_score = self.pattern_score_cal(name, prev_chart_score)
            prev_chart_score = last_chart_score
        chart_pattern_score = last_chart_score if chart_patterns else prev_chart_score

        # Candlestick pattern detection
        candlestick_patterns = self.detect_candlestick_patterns_code(data)
        print(f"Candlestick patterns: {candlestick_patterns}")
        last_candle_score = 0
        prev_candle_score = 0
        for _, name in candlestick_patterns:
            last_candle_score = self.candlestick_score_cal(name, prev_candle_score)
            prev_candle_score = last_candle_score
        candlestick_score = last_candle_score if candlestick_patterns else prev_candle_score

        return chart_pattern_score, candlestick_score

    def detect_candlestick_patterns_code(self, df):
        """
        Detect candlestick patterns in the given OHLCV data using code logic (not YOLO).
        Returns a list of (timestamp, pattern_name) tuples.
        """
        data = df.copy()
        # --- BEGIN: Candlestick pattern functions from pattern_manual.py ---
        def marubozu(row, threshold=0.01):
            if row['body'] <= 0:
                return False
            return (row['upper_shadow'] <= threshold * row['body'] and 
                    row['lower_shadow'] <= threshold * row['body'])

        def three_advancing_white_soldiers(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'],
                c1['close'] > c1['open'] * 1.005,
                c2['close'] > c2['open'] * 1.005,
                c3['close'] > c3['open'] * 1.005,
                c1['close'] < c2['open'],
                c2['close'] < c3['open'],
                c1['body'] > c1['upper_shadow'] + c1['lower_shadow'],
                c2['body'] > c2['upper_shadow'] + c2['lower_shadow'],
                c3['body'] > c3['upper_shadow'] + c3['lower_shadow']
            ])

        def three_black_crows(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'] and not c2['is_bullish'] and not c3['is_bullish'],
                c1['close'] < c1['open'] * 0.995,
                c2['close'] < c2['open'] * 0.995,
                c3['close'] < c3['open'] * 0.995,
                c1['close'] > c2['open'],
                c2['close'] > c3['open'],
                c1['body'] > c1['upper_shadow'] + c1['lower_shadow'],
                c2['body'] > c2['upper_shadow'] + c2['lower_shadow'],
                c3['body'] > c3['upper_shadow'] + c3['lower_shadow']
            ])

        def identical_three_crows(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                three_black_crows(df, i),
                abs(c1['body'] - c2['body']) < 0.2 * c1['body'],
                abs(c2['body'] - c3['body']) < 0.2 * c1['body'],
                abs(c1['upper_shadow'] - c2['upper_shadow']) < 0.2 * c1['upper_shadow'],
                abs(c2['upper_shadow'] - c3['upper_shadow']) < 0.2 * c1['upper_shadow']
            ])

        def morning_star(df, i, threshold=0.05):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'],  # First candle bearish
                c3['is_bullish'],      # Third candle bullish
                c1['body'] > threshold * df['close'].mean(),
                c3['body'] > threshold * df['close'].mean(),
                c2['body'] < abs(c1['close'] - c1['open']) * 0.5,  # Small body
                c2['low'] < c1['low'],  # New low
                c3['close'] > c1['body_high']  # Closes above first candle's body
            ])

        def hammer(row, threshold=0.3):
            return all([
                row['is_bullish'],
                row['lower_shadow'] > 2 * row['body'],
                row['upper_shadow'] < row['body'] * 0.1,
                row['body_size'] > 0.005  # Filter tiny candles
            ])

        def morning_doji_star(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'],  # Bearish first candle
                abs(c2['close'] - c2['open']) < 0.02 * c2['open'],  # Doji condition
                c3['is_bullish'],      # Bullish third candle
                c2['low'] < c1['low'],  # New low
                c3['close'] > c1['mid_point']  # Closes above midpoint of first candle
            ])

        def unique_three_river(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'],  # Long bearish
                not c2['is_bullish'],  # Bearish with long lower shadow
                c3['is_bullish'],      # Small bullish
                c1['body'] > c1['upper_shadow'] + c1['lower_shadow'],
                c2['lower_shadow'] > 2 * c2['body'],
                c3['close'] < c2['close'],  # Closes below previous close
                c3['close'] > c1['close']   # But above first candle's close
            ])

        def ladder_bottom(df, i):
            if i < 4: 
                return False
            c1, c2, c3, c4, c5 = df.iloc[i-4], df.iloc[i-3], df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'], not c2['is_bullish'], not c3['is_bullish'], not c4['is_bullish'],
                c5['is_bullish'],
                c1['close'] > c2['open'],  # Consecutive down candles
                c2['close'] > c3['open'],
                c3['close'] > c4['open'],
                c5['open'] < c4['close'],  # Last candle gaps down
                c5['close'] > c1['open']   # Closes above first open
            ])

        def stick_sandwich(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                not c1['is_bullish'],  # First bearish
                c2['is_bullish'],      # Second bullish
                not c3['is_bullish'],  # Third bearish
                abs(c1['close'] - c3['close']) < 0.01 * c1['close'],  # Same close price
                c2['close'] > c1['open']  # Second closes above first open
            ])

        def evening_star(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                c1['is_bullish'],    # First bullish
                not c3['is_bullish'], # Third bearish
                c1['body'] > 0.005 * df['close'].mean(),
                c3['body'] > 0.005 * df['close'].mean(),
                c2['body'] < abs(c1['close'] - c1['open']) * 0.5,
                c2['high'] > c1['high'],  # New high
                c3['close'] < c1['body_low']  # Closes below first candle's body
            ])

        def hanging_man(row, threshold=0.3):
            return all([
                not row['is_bullish'],
                row['lower_shadow'] > 2 * row['body'],
                row['upper_shadow'] < row['body'] * 0.1,
                row['body_size'] > 0.005
            ])

        def evening_doji_star(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                c1['is_bullish'],    # Bullish first candle
                abs(c2['close'] - c2['open']) < 0.02 * c2['open'],  # Doji
                not c3['is_bullish'], # Bearish third candle
                c2['high'] > c1['high'],  # New high
                c3['close'] < c1['mid_point']  # Closes below midpoint of first candle
            ])

        def gravestone_doji(row):
            return all([
                abs(row['close'] - row['open']) < 0.01 * row['open'],  # Doji condition
                row['upper_shadow'] > 2 * abs(row['close'] - row['open']),
                row['lower_shadow'] < 0.1 * row['upper_shadow']
            ])

        def tristar_pattern(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                abs(c1['close'] - c1['open']) < 0.02 * c1['open'],  # Doji
                abs(c2['close'] - c2['open']) < 0.02 * c2['open'],  # Doji
                abs(c3['close'] - c3['open']) < 0.02 * c3['open'],  # Doji
                c2['high'] < c1['low'] and c2['high'] < c3['low'],  # Middle doji gaps
                c2['low'] > c1['high'] and c2['low'] > c3['high']
            ])

        def tasuki_gap(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                c1['is_bullish'],  # First bullish
                c2['is_bullish'],  # Second bullish (gap up)
                not c3['is_bullish'],  # Third bearish
                c2['low'] > c1['high'],  # Gap up
                c3['open'] > c1['high'],  # Opens above previous high
                c3['close'] < c2['low'],  # Closes in the gap
                c3['close'] > c1['close']  # But above first close
            ])

        def three_line_strike(df, i):
            if i < 3: 
                return False
            c1, c2, c3, c4 = df.iloc[i-3], df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            # Bullish version
            if all([
                not c1['is_bullish'], not c2['is_bullish'], not c3['is_bullish'],
                c4['is_bullish'],
                c1['close'] > c2['open'], c2['close'] > c3['open'],  # Consecutive
                c4['open'] < c3['close'],  # Opens below previous close
                c4['close'] > c1['open']   # Closes above first open
            ]):
                return True
            # Bearish version
            if all([
                c1['is_bullish'], c2['is_bullish'], c3['is_bullish'],
                not c4['is_bullish'],
                c1['close'] < c2['open'], c2['close'] < c3['open'],  # Consecutive
                c4['open'] > c3['close'],  # Opens above previous close
                c4['close'] < c1['open']   # Closes below first open
            ]):
                return True
            return False

        def dragonfly_doji(row):
            return all([
                abs(row['close'] - row['open']) < 0.01 * row['open'],
                row['lower_shadow'] > 3 * abs(row['close'] - row['open']),
                row['upper_shadow'] < 0.1 * row['lower_shadow']
            ])

        def hikkake_pattern(df, i):
            if i < 3: 
                return False
            c1, c2, c3, c4 = df.iloc[i-3], df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            # Bullish Hikkake (false bearish breakout)
            if all([
                c1['body'] > 0,  # First has normal body
                c2['body'] < c1['body'] * 0.5,  # Inside day
                c2['high'] < c1['high'] and c2['low'] > c1['low'],
                not c3['is_bullish'] and c3['low'] < c2['low'],  # Breaks low
                c4['is_bullish'] and c4['close'] > c2['high']  # Closes above inside high
            ]):
                return True
            # Bearish Hikkake (false bullish breakout)
            if all([
                c1['body'] > 0,  # First has normal body
                c2['body'] < c1['body'] * 0.5,  # Inside day
                c2['high'] < c1['high'] and c2['low'] > c1['low'],
                c3['is_bullish'] and c3['high'] > c2['high'],  # Breaks high
                not c4['is_bullish'] and c4['close'] < c2['low']  # Closes below inside low
            ]):
                return True
            return False

        def advance_block(df, i):
            if i < 2: 
                return False
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            return all([
                c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'],
                c1['body'] > df['body'].mean() * 0.8,
                c2['body'] < c1['body'] * 0.8,
                c3['body'] < c2['body'] * 0.8,
                c1['upper_shadow'] < c1['body'] * 0.2,
                c2['upper_shadow'] > c2['body'] * 0.5,
                c3['upper_shadow'] > c3['body'] * 0.5,
                c3['close'] > c2['close']  # Still making higher closes
            ])

        def spinning_top(row):
            return all([
                row['body'] < 0.3 * (row['high'] - row['low']),  # Small body
                row['upper_shadow'] > row['body'] * 1.5,
                row['lower_shadow'] > row['body'] * 1.5
            ])
        # --- END: Candlestick pattern functions ---
        # Calculate candle properties
        data['body'] = abs(data['close'] - data['open'])
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        data['is_bullish'] = data['close'] > data['open']
        data['body_high'] = data[['open', 'close']].max(axis=1)
        data['body_low'] = data[['open', 'close']].min(axis=1)
        data['mid_point'] = (data['high'] + data['low']) / 2
        data['body_size'] = data['body'] / data['high'].rolling(20).mean()
        patterns = []
        pattern_funcs = {
            'Marubozu': lambda df, i: marubozu(df.iloc[i]),
            'Three Advancing White Soldiers': three_advancing_white_soldiers,
            'Three Black Crows': three_black_crows,
            'Identical Three Crows': identical_three_crows,
            'Morning Star': morning_star,
            'Hammer': lambda df, i: hammer(df.iloc[i]),
            'Morning Doji Star': morning_doji_star,
            'Unique 3 River': unique_three_river,
            'Ladder Bottom': ladder_bottom,
            'Stick Sandwich': stick_sandwich,
            'Evening Star': evening_star,
            'Hanging Man': lambda df, i: hanging_man(df.iloc[i]),
            'Evening Doji Star': evening_doji_star,
            'Gravestone Doji': lambda df, i: gravestone_doji(df.iloc[i]),
            'Tristar Pattern': tristar_pattern,
            'Tasuki Gap': tasuki_gap,
            'Three Line Strike': three_line_strike,
            'Dragonfly Doji': lambda df, i: dragonfly_doji(df.iloc[i]),
            'Hikkake Pattern': hikkake_pattern,
            'Advance Block': advance_block,
            'Spinning Top': lambda df, i: spinning_top(df.iloc[i]),
        }
        for i in range(len(data)):
            for pattern_name, pattern_func in pattern_funcs.items():
                try:
                    if pattern_func(data, i):
                        patterns.append((data.index[i], pattern_name))
                        break  # Only the first detected pattern per timestamp
                except Exception:
                    continue
        return patterns

    def candlestick_score_cal(self, candlestick_name, current_score):
        score_list = {
            # Strong Bullish/Bearish (Fixed)
            'Marubozu': 100 if current_score > 20 else (-100 if current_score < -20 else 0),
            'Three Advancing White Soldiers': 100,
            'Three Black Crows': -100,
            'Identical Three Crows': -100,
            
            # Bullish Reversal Patterns
            'Morning Star': 70,
            'Hammer': 70,
            'Morning Doji Star': 70,
            'Unique 3 River': 70,
            'Ladder Bottom': 70,
            'Stick Sandwich': 70,
            
            # Bearish Reversal Patterns
            'Evening Star': -70,
            'Hanging Man': -70,
            'Evening Doji Star': -70,
            'Gravestone Doji': -70,
            'Tristar Pattern': -70,
            
            # Trend-Dependent Patterns
            'Tasuki Gap': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Three Inside Up-Down': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Three Outside Up-Down': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Rising-Falling Three Methods': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Upside-Downside Gap Three Methods': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Up-Down Gap Side-by-slide White Lines': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            'Three Line Strike': 70 if current_score > 20 else (-70 if current_score < -20 else 0),
            
            # Small Bullish/Bearish Reversal
            'Dragonfly Doji': 40 if current_score < -20 else (-40 if current_score > 20 else 0),
            'Hikkake Pattern': 40 if current_score < -20 else (-40 if current_score > 20 else 0),
            
            # Neutral Patterns
            'Advance Block': 0,
            'Spinning Top': 0,
        }
        return score_list.get(candlestick_name, current_score)

    def pattern_score_cal(self, pattern_name, trend_score):
        score_dict = {
            # Strong Bullish Patterns (100)
            'Cup-and-handle': 100,
            'Resistance-breakout': 100,
            'Double-Bottom': 100 if trend_score < -20 else trend_score,
            'Inverse-Head-Shoulders': 100 if trend_score < -20 else trend_score,
            'Triple-Bottom': 100 if trend_score < -20 else trend_score,
            'Rounding-Bottom': 100 if trend_score < -20 else trend_score,
            
            # Strong Bearish Patterns (-100)
            'Support-breakout': -100,
            'Double-Top': -100 if trend_score > 20 else trend_score,
            'Head-Shoulders': -100 if trend_score > 20 else trend_score,
            'Triple-Top': -100 if trend_score > 20 else trend_score,
            'Rounding-Top': -100 if trend_score > 20 else trend_score,
            
            # Trend-Continuation Patterns (70/-70)
            'Ascending-Triangle': 70 if trend_score > 20 else trend_score,
            'Channel-up': 70 if trend_score > 20 else trend_score,
            'Channel-down': -70 if trend_score < -20 else trend_score,
            'Descending-Triangle': -70 if trend_score < -20 else trend_score,
            'Triangle': 70 if trend_score > 20 else (-70 if trend_score < -20 else trend_score),
            
            # Counter-Trend Patterns (70/-70)
            'Falling-Wedge': 70 if trend_score < -20 else trend_score,
            'Rising-Wedge': -70 if trend_score > 20 else trend_score,
            
            # Warning Signals (-70)
            'Resistance-Emerging': -70 if trend_score > 20 else trend_score,
            
            # Neutral Patterns
            'Rectangle': trend_score
        }
        return score_dict.get(pattern_name, trend_score)

    def detect_chart_patterns_code(self, data, trend_score):
        # Ensure timestamp column is datetime
        if not np.issubdtype(data['timestamp'].dtype, np.datetime64):
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.copy()
        data = self.find_pivots(data, window=3)
        pivot_highs = data[data['pivot_high']]
        pivot_lows = data[data['pivot_low']]
        patterns = []
        last_pattern_time = None

        # Helper to avoid duplicate/overlapping patterns
        def is_far_enough(ts, min_days=5):
            nonlocal last_pattern_time
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if last_pattern_time is None:
                return True
            if isinstance(last_pattern_time, str):
                last_pattern_time_dt = pd.to_datetime(last_pattern_time)
            else:
                last_pattern_time_dt = last_pattern_time
            # If time difference is enough, allow pattern
            if (ts - last_pattern_time_dt).days >= min_days:
                return True
            return False

        # Double Top
        for i in range(1, len(pivot_highs)):
            price1 = pivot_highs['high'].iloc[i-1]
            price2 = pivot_highs['high'].iloc[i]
            time2 = pivot_highs.index[i]
            if abs(price1 - price2) < 0.01 * price1 and is_far_enough(time2):
                patterns.append((time2, 'Double-Top'))
                last_pattern_time = time2

        # Double Bottom
        for i in range(1, len(pivot_lows)):
            price1 = pivot_lows['low'].iloc[i-1]
            price2 = pivot_lows['low'].iloc[i]
            time2 = pivot_lows.index[i]
            if abs(price1 - price2) < 0.01 * price1 and is_far_enough(time2):
                patterns.append((time2, 'Double-Bottom'))
                last_pattern_time = time2

        # Triple Top
        for i in range(2, len(pivot_highs)):
            p1 = pivot_highs['high'].iloc[i-2]
            p2 = pivot_highs['high'].iloc[i-1]
            p3 = pivot_highs['high'].iloc[i]
            t3 = pivot_highs.index[i]
            if (abs(p1 - p2) < 0.01 * p1 and abs(p2 - p3) < 0.01 * p2 and is_far_enough(t3)):
                patterns.append((t3, 'Triple-Top'))
                last_pattern_time = t3

        # Triple Bottom
        for i in range(2, len(pivot_lows)):
            p1 = pivot_lows['low'].iloc[i-2]
            p2 = pivot_lows['low'].iloc[i-1]
            p3 = pivot_lows['low'].iloc[i]
            t3 = pivot_lows.index[i]
            if (abs(p1 - p2) < 0.01 * p1 and abs(p2 - p3) < 0.01 * p2 and is_far_enough(t3)):
                patterns.append((t3, 'Triple-Bottom'))
                last_pattern_time = t3

        # Head and Shoulders
        for i in range(2, len(pivot_highs)):
            l = pivot_highs['high'].iloc[i-2]
            h = pivot_highs['high'].iloc[i-1]
            r = pivot_highs['high'].iloc[i]
            t = pivot_highs.index[i]
            if h > l and h > r and abs(l - r) < 0.01 * h and is_far_enough(t):
                patterns.append((t, 'Head-Shoulders'))
                last_pattern_time = t

        # Inverse Head and Shoulders
        for i in range(2, len(pivot_lows)):
            l = pivot_lows['low'].iloc[i-2]
            h = pivot_lows['low'].iloc[i-1]
            r = pivot_lows['low'].iloc[i]
            t = pivot_lows.index[i]
            if h < l and h < r and abs(l - r) < 0.01 * h and is_far_enough(t):
                patterns.append((t, 'Inverse-Head-Shoulders'))
                last_pattern_time = t

        # Rectangle, Ascending, Descending Triangles: look for alternating pivots
        pivots = pd.concat([
            pd.DataFrame({'type': 'high', 'price': pivot_highs['high']}, index=pivot_highs.index),
            pd.DataFrame({'type': 'low', 'price': pivot_lows['low']}, index=pivot_lows.index)
        ]).sort_index()

        for i in range(4, len(pivots)):
            window = pivots.iloc[i-4:i+1]
            types = window['type'].values
            prices = window['price'].values
            t = window.index[-1]
            # Check for alternating types
            if all(types[j] != types[j+1] for j in range(len(types)-1)):
                # Rectangle: alternating highs/lows, similar prices
                highs = prices[types == 'high']
                lows = prices[types == 'low']
                if (len(highs) > 1 and len(lows) > 1 and
                    np.max(highs) - np.min(highs) < 0.01 * np.mean(highs) and
                    np.max(lows) - np.min(lows) < 0.01 * np.mean(lows) and
                    is_far_enough(t)):
                    patterns.append((t, 'Rectangle'))
                    last_pattern_time = t
                # Ascending Triangle: flat highs, rising lows
                if (len(highs) > 1 and len(lows) > 1 and
                    np.max(highs) - np.min(highs) < 0.01 * np.mean(highs) and
                    np.all(np.diff(lows) > 0) and
                    is_far_enough(t)):
                    patterns.append((t, 'Ascending-Triangle'))
                    last_pattern_time = t
                # Descending Triangle: flat lows, falling highs
                if (len(highs) > 1 and len(lows) > 1 and
                    np.max(lows) - np.min(lows) < 0.01 * np.mean(lows) and
                    np.all(np.diff(highs) < 0) and
                    is_far_enough(t)):
                    patterns.append((t, 'Descending-Triangle'))
                    last_pattern_time = t

        return patterns

    def find_pivots(self, df, window=3):
        highs = df['high']
        lows = df['low']
        pivots_high = (highs.shift(window) < highs) & (highs.shift(-window) < highs)
        pivots_low = (lows.shift(window) > lows) & (lows.shift(-window) > lows)
        df['pivot_high'] = pivots_high
        df['pivot_low'] = pivots_low
        return df

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # Example usage
    print("Running pattern detection example...")
    # Load sample data from CSV
    df = pd.read_csv('data/aapl_1min_5yrs.csv')
    print("After loading:", df.shape, df.columns)
    df.columns = [c.strip().lower() for c in df.columns]
    print("Columns after lower/strip:", df.columns)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    print("After setting index:", df.shape)
    required = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required if c in df.columns]]
    print("After selecting columns:", df.shape)
    print(df.tail())
    print("Rows before dropna:", len(df))
    df = df.dropna()
    print("Rows after dropna:", len(df))
    sample_data = df.tail(500)
    detector = ChartPatternDetector()
    patterns, candlesticks = detector.detect_patterns(sample_data, "SAMPLE")
    print(f"Detected patterns: {patterns}")
    print(f"Candlestick patterns: {candlesticks}")
