import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_technical_indicators(df):
    """
    Generate technical indicators for the sample data
    """
    # RSI calculation
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMA calculation
    df['EMA'] = df['CLOSE'].ewm(span=20, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_MIDDLE'] = df['CLOSE'].rolling(window=20).mean()
    bb_std = df['CLOSE'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (bb_std * 2)
    
    return df

def generate_sample_data(start_date='2023-01-01', periods=1000):
    """
    Generate realistic sample trading data
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='1H')
    
    # Generate base price data with realistic patterns
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 100.0
    prices = [base_price]
    
    # Generate price movements with trend and volatility
    for i in range(1, periods):
        # Add some trend component
        trend = 0.0001 * i  # Slight upward trend
        
        # Add random walk component
        random_walk = np.random.normal(0, 0.01)
        
        # Add some mean reversion
        mean_reversion = -0.001 * (prices[-1] - base_price) / base_price
        
        # Calculate new price
        price_change = trend + random_walk + mean_reversion
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price stays positive
        new_price = max(new_price, 1.0)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = 0.02  # 2% daily volatility
        
        # Random high and low
        high_offset = np.random.uniform(0, volatility)
        low_offset = np.random.uniform(-volatility, 0)
        
        high = close_price * (1 + high_offset)
        low = close_price * (1 + low_offset)
        
        # Ensure high >= close >= low
        if high < close_price:
            high = close_price * 1.01
        if low > close_price:
            low = close_price * 0.99
        
        # Open price (previous close with some noise)
        if i == 0:
            open_price = close_price * np.random.uniform(0.98, 1.02)
        else:
            open_price = prices[i-1] * np.random.uniform(0.99, 1.01)
        
        # Volume (correlated with price movement)
        base_volume = 1000000
        volume_multiplier = 1 + abs(close_price - open_price) / open_price * 10
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'TIME': date,
            'OPEN': round(open_price, 4),
            'HIGH': round(high, 4),
            'LOW': round(low, 4),
            'CLOSE': round(close_price, 4),
            'VOLUME': volume
        })
    
    df = pd.DataFrame(data)
    
    # Add technical indicators
    df = generate_technical_indicators(df)
    
    # Clean up any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    """
    Generate and save sample training and testing data
    """
    print("Generating sample trading data...")
    
    # Generate training data (first 800 periods)
    train_data = generate_sample_data(start_date='2023-01-01', periods=800)
    
    # Generate testing data (next 200 periods)
    test_data = generate_sample_data(start_date='2023-04-01', periods=200)
    
    # Save the data
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    print(f"Generated training data: {len(train_data)} rows")
    print(f"Generated testing data: {len(test_data)} rows")
    print("\nSample training data:")
    print(train_data.head())
    print("\nColumns in the data:")
    print(train_data.columns.tolist())
    
    print("\nFiles created:")
    print("- train.csv")
    print("- test.csv")
    print("\nYou can now run: python model.py")

if __name__ == "__main__":
    main() 