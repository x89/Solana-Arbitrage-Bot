import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import pickle
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def clean_data(df):
    """
    Handle infinite values, NaNs, and extreme outliers in the dataframe
    """
    df = df.copy()
    
    # Replace infinite values with NaN first
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Forward fill then backfill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Cap extreme values at reasonable percentiles
    for col in df.select_dtypes(include=[np.number]).columns:
        # Skip binary/dummy variables
        if df[col].nunique() > 2:
            upper_limit = df[col].quantile(0.995)
            lower_limit = df[col].quantile(0.005)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
    
    return df

def load_data(train_path, test_path):
    """
    Load and preprocess training and testing data
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Convert TIME to datetime and set as index
    for df in [train_data, test_data]:
        if 'TIME' in df.columns:
            df['TIME'] = pd.to_datetime(df['TIME'])
            df.set_index('TIME', inplace=True)
    
    # Clean data to handle infinity/large values
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    return train_data, test_data

def add_features(df):
    """
    Create enhanced technical indicators with safe calculations
    """
    df = df.copy()
    
    # 1. RSI-Based Features (Core Focus)
    df['RSI_OVERBOUGHT'] = (df['RSI'] > 70).astype(int)
    df['RSI_OVERSOLD'] = (df['RSI'] < 30).astype(int)
    df['RSI_TREND'] = df['RSI'].rolling(5, min_periods=3).mean() - df['RSI'].rolling(20, min_periods=10).mean()
    df['RSI_CROSS_50'] = (df['RSI'] > 50).astype(int) - (df['RSI'].shift(1) > 50).astype(int)
    
    # 2. Price Action Features
    df['PRICE_TREND'] = df['CLOSE'].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df['CLOSE_SMA_5'] = df['CLOSE'].rolling(5, min_periods=3).mean()
    df['CLOSE_SMA_20'] = df['CLOSE'].rolling(20, min_periods=10).mean()
    df['PRICE_SMA_CROSS'] = (df['CLOSE_SMA_5'] > df['CLOSE_SMA_20']).astype(int)
    
    # 3. Volume Features
    df['VOLUME_TREND'] = df['VOLUME'].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df['VOLUME_SMA_5'] = df['VOLUME'].rolling(5, min_periods=3).mean()
    df['VOLUME_SMA_20'] = df['VOLUME'].rolling(20, min_periods=10).mean()
    
    # 4. Bollinger Band Features
    with np.errstate(divide='ignore', invalid='ignore'):
        df['BB_WIDTH'] = np.where(
            df['BB_MIDDLE'] != 0,
            (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE'],
            0
        )
        df['BB_PERCENT'] = np.where(
            (df['BB_UPPER'] - df['BB_LOWER']) != 0,
            (df['CLOSE'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER']),
            0.5
        )
    df['BB_CROSS_UPPER'] = (df['CLOSE'] > df['BB_UPPER']).astype(int)
    df['BB_CROSS_LOWER'] = (df['CLOSE'] < df['BB_LOWER']).astype(int)
    
    # 5. EMA Features
    df['EMA_12'] = df['CLOSE'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['CLOSE'].ewm(span=26, adjust=False).mean()
    df['EMA_CROSS'] = (df['EMA_12'] > df['EMA_26']).astype(int)
    
    # 6. Enhanced Features for Better Signal Generation
    df['PRICE_MOMENTUM'] = df['CLOSE'].pct_change(3)
    df['VOLUME_MOMENTUM'] = df['VOLUME'].pct_change(3)
    df['RSI_MOMENTUM'] = df['RSI'].diff(3)
    df['BB_SQUEEZE'] = (df['BB_WIDTH'] < df['BB_WIDTH'].rolling(20).mean()).astype(int)
    
    # Final cleaning of any remaining issues
    df = clean_data(df)
    
    return df

def create_targets_enhanced(df, future_periods=6, profit_threshold=0.01, stop_loss=-0.008):
    """
    Create target labels with more balanced signal generation
    """
    df = df.copy()
    
    # Calculate future returns
    future_prices = df['CLOSE'].shift(-future_periods)
    returns = (future_prices - df['CLOSE']) / df['CLOSE']
    
    # Initialize targets
    df['TARGET'] = 0  # Default to hold
    
    # Buy signals - more relaxed conditions
    buy_condition = (returns > profit_threshold) & (df['RSI'] < 70)  # Less restrictive RSI
    df.loc[buy_condition, 'TARGET'] = 1
    
    # Sell signals - more relaxed conditions
    sell_condition = (returns < stop_loss) & (df['RSI'] > 30)  # Less restrictive RSI
    df.loc[sell_condition, 'TARGET'] = -1
    
    # Remove rows with NaN values (end of dataset)
    df.dropna(subset=['TARGET'], inplace=True)
    
    return df

def train_model_enhanced(X_train, y_train):
    """
    Train Random Forest model with enhanced parameters
    """
    # Calculate balanced class weights
    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    class_weights = {
        1: total_samples / (3 * class_counts.get(1, 1)),  # Buy
        -1: total_samples / (3 * class_counts.get(-1, 1)),  # Sell
        0: total_samples / (3 * class_counts.get(0, 1))    # Hold
    }
    
    # Enhanced Random Forest parameters
    model = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=20,       # Deeper trees
        min_samples_split=3, # More splits
        min_samples_leaf=1,  # Smaller leaves
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def backtest_enhanced(df, predictions, initial_capital=10000, commission=0.0005, max_position_size=0.8):
    """
    Enhanced backtesting with position sizing and risk management
    """
    df = df.copy()
    df['SIGNAL'] = predictions
    
    # Initialize trading variables
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    max_capital = initial_capital
    drawdown = 0
    
    # Calculate buy-and-hold return
    if len(df) > 0:
        buy_hold_return = (df['CLOSE'].iloc[-1] - df['CLOSE'].iloc[0]) / df['CLOSE'].iloc[0] * 100
        buy_hold_value = initial_capital * (1 + buy_hold_return/100)
    else:
        buy_hold_return = 0
        buy_hold_value = initial_capital
    
    for i in range(len(df)):
        current_price = df['CLOSE'].iloc[i]
        
        # Safety check for invalid prices
        if current_price <= 0 or np.isinf(current_price) or np.isnan(current_price):
            continue
            
        # Execute sell signal (only if we have a position)
        if df['SIGNAL'].iloc[i] == -1 and position > 0:
            # Calculate proceeds after commission
            proceeds = position * current_price * (1 - commission)
            
            # Calculate PnL for this trade
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_dollar = proceeds - (position * entry_price)
            
            trades.append({
                'type': 'SELL',
                'entry_price': entry_price,
                'exit_price': current_price,
                'quantity': position,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'duration': i - trades[-1]['entry_index'] if trades else 0
            })
            
            capital = proceeds
            position = 0
            entry_price = 0
        
        # Execute buy signal with position sizing
        elif (df['SIGNAL'].iloc[i] == 1 and 
              position == 0 and 
              capital > 0 and 
              current_price > 0):
            
            # Position sizing based on available capital
            position_size = capital * max_position_size
            position = (position_size * (1 - commission)) / current_price
            entry_price = current_price
            
            trades.append({
                'type': 'BUY',
                'entry_price': current_price,
                'exit_price': np.nan,
                'quantity': position,
                'pnl_pct': np.nan,
                'pnl_dollar': np.nan,
                'entry_index': i
            })
            
            capital -= position_size
        
        # Calculate current portfolio value
        portfolio_value = capital + (position * current_price if position > 0 else 0)
        portfolio_values.append(portfolio_value)
        
        # Track maximum drawdown
        if portfolio_value > max_capital:
            max_capital = portfolio_value
        current_drawdown = (portfolio_value - max_capital) / max_capital * 100
        if current_drawdown < drawdown:
            drawdown = current_drawdown
    
    # Calculate final strategy value
    final_value = portfolio_values[-1] if portfolio_values else initial_capital
    
    # Calculate trade statistics
    closed_trades = [t for t in trades if t['type'] == 'SELL']
    winning_trades = [t for t in closed_trades if t['pnl_pct'] > 0]
    losing_trades = [t for t in closed_trades if t['pnl_pct'] <= 0]
    
    # Prepare comprehensive results
    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'strategy_return': (final_value - initial_capital) / initial_capital * 100,
        'buy_hold_return': buy_hold_return,
        'buy_hold_value': buy_hold_value,
        'num_trades': len(closed_trades),
        'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
        'avg_win_pct': np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0,
        'avg_loss_pct': np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0,
        'max_drawdown': drawdown,
        'portfolio_values': portfolio_values,
        'trades': trades,
        'trade_details': {
            'total': len(closed_trades),
            'winners': len(winning_trades),
            'losers': len(losing_trades),
            'win_pct': len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
            'profit_factor': sum(t['pnl_dollar'] for t in winning_trades) / 
                           abs(sum(t['pnl_dollar'] for t in losing_trades)) if losing_trades else float('inf')
        }
    }
    
    return results

def print_enhanced_results(results):
    """Print enhanced backtest results with additional metrics"""
    print("\n=== Enhanced Backtest Results ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Strategy Value: ${results['final_value']:,.2f}")
    print(f"Final Buy & Hold Value: ${results['buy_hold_value']:,.2f}")
    print(f"\nStrategy Return: {results['strategy_return']:.2f}%")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
    print(f"Outperformance: {results['strategy_return'] - results['buy_hold_return']:.2f}%")
    
    print("\n=== Enhanced Trade Statistics ===")
    print(f"Total Trades: {results['trade_details']['total']}")
    print(f"Winning Trades: {results['trade_details']['winners']} ({results['trade_details']['win_pct']:.1f}%)")
    print(f"Losing Trades: {results['trade_details']['losers']}")
    print(f"Average Win: {results['avg_win_pct']:.2f}%")
    print(f"Average Loss: {results['avg_loss_pct']:.2f}%")
    print(f"Profit Factor: {results['trade_details']['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    
    # Additional metrics
    if results['trade_details']['total'] > 0:
        print(f"\n=== Additional Metrics ===")
        print(f"Trade Frequency: {results['trade_details']['total']} trades")
        print(f"Risk-Adjusted Return: {results['strategy_return'] / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 'N/A'}")
        print(f"Sharpe Ratio (approximate): {results['strategy_return'] / (abs(results['max_drawdown']) + 1):.2f}")

def plot_enhanced_results(test_data, results):
    """
    Visualize enhanced backtest results
    """
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(test_data.index, results['portfolio_values'], label='Strategy', color='blue')
    plt.plot(test_data.index, 
             results['initial_capital'] * (test_data['CLOSE'] / test_data['CLOSE'].iloc[0]), 
             label='Buy & Hold', color='green', linestyle='--')
    plt.title('Enhanced Portfolio Value Over Time')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot RSI with overbought/oversold levels
    plt.subplot(2, 1, 2)
    plt.plot(test_data.index, test_data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', alpha=0.3)
    plt.axhline(30, color='green', linestyle='--', alpha=0.3)
    plt.title('RSI Indicator')
    plt.ylabel('RSI')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results_enhanced.png')
    plt.close()

def main_enhanced():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data, test_data = load_data('train.csv', 'test.csv')
    
    # Feature engineering
    print("Creating enhanced features...")
    train_data = add_features(train_data)
    test_data = add_features(test_data)
    
    # Create targets with enhanced parameters
    print("Creating enhanced target labels...")
    train_data = create_targets_enhanced(train_data)
    test_data = create_targets_enhanced(test_data)
    
    # Prepare features and targets
    feature_cols = [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'RSI', 'EMA', 
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'RSI_OVERBOUGHT', 
        'RSI_OVERSOLD', 'RSI_TREND', 'RSI_CROSS_50', 'PRICE_TREND',
        'CLOSE_SMA_5', 'CLOSE_SMA_20', 'PRICE_SMA_CROSS', 'VOLUME_TREND',
        'BB_WIDTH', 'BB_PERCENT', 'BB_CROSS_UPPER', 'BB_CROSS_LOWER',
        'EMA_12', 'EMA_26', 'EMA_CROSS', 'PRICE_MOMENTUM', 'VOLUME_MOMENTUM',
        'RSI_MOMENTUM', 'BB_SQUEEZE'
    ]
    
    X_train = train_data[feature_cols]
    y_train = train_data['TARGET']
    
    X_test = test_data[feature_cols]
    y_test = test_data['TARGET']
    
    # Scale features
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train enhanced model
    print("Training enhanced model...")
    model = train_model_enhanced(X_train_scaled, y_train)
    
    # Evaluate on training data
    print("\nEnhanced Training Performance:")
    train_pred = model.predict(X_train_scaled)
    print(classification_report(y_train, train_pred))
    
    # Evaluate on test data
    print("\nEnhanced Test Performance:")
    test_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, test_pred))
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Enhanced Feature Importance (Top 20)")
    plt.bar(range(20), importances[indices][:20], align='center')
    plt.xticks(range(20), [feature_cols[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance_enhanced.png')
    plt.close()
    
    # Enhanced backtest
    print("\nRunning enhanced backtest...")
    initial_capital = 10000
    test_data = test_data.iloc[:len(test_pred)]
    results = backtest_enhanced(test_data, test_pred, initial_capital=10000)

    print_enhanced_results(results)
    
    # Plot enhanced results
    plot_enhanced_results(test_data, {**results, 'initial_capital': initial_capital})
    print("\nSaved enhanced backtest visualization to 'backtest_results_enhanced.png'")
    
    # Save enhanced model
    os.makedirs('model_enhanced', exist_ok=True)
    with open('model_enhanced/trading_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_enhanced/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model_enhanced/features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("\nEnhanced model saved to 'model_enhanced' directory")

if __name__ == "__main__":
    main_enhanced() 