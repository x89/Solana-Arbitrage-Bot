import argparse
import os
import sys
import logging
import time as time_module
import pandas as pd
import csv  # for report CSV
from datetime import datetime, time, timedelta
from dotenv import load_dotenv
from ai_skills.forecasting import TimeSeriesForecaster
from ai_skills.pattern_detection import ChartPatternDetector
from ai_skills.sentiment_analysis import TransformerSentimentAnalyzer
import pytz

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IBKR Import Disabled - Use Bitget instead
try:
    from data_collect.data import IBDataCollector
except ImportError:
    IBDataCollector = None

# ib_insync disabled - not needed
# from ib_insync import IB, Stock, util
from trading_algos.ai_trading_algo import fisher_transform, rsi, cci, get_combined_algo_signal, get_total_algo_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AI Trading Bot')

# Load environment variables
TRANSACTION_COST_PCT = 0.001 # 0.1% per trade, consistent with optimization scripts
load_dotenv()

class TradingBot:
    def __init__(self, symbol: str, paper: bool = True, client_id: int = 1):
        """
        Initialize the trading bot.
        
        ⚠️ NOTE: IBKR integration is DISABLED. Use Bitget for data collection.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            paper: Whether to use paper trading (default: True) - DEPRECATED
            client_id: Client ID for IB connection - DEPRECATED
        """
        self.symbol = symbol.upper()
        self.paper = paper  # Deprecated - not used
        self.client_id = client_id  # Deprecated - not used
        
        # IBKR collector disabled
        if IBDataCollector:
            self.data_collector = IBDataCollector()
        else:
            self.data_collector = None
            logger.warning("⚠️ IBDataCollector not available. Use Bitget for data collection.")
        
        self.ib = None
        
    def connect(self) -> bool:
        """
        Connect to TWS/IB Gateway - DISABLED
        
        ⚠️ IBKR integration is disabled. This method always returns False.
        Use Bitget for data collection instead.
        
        Returns:
            False (IBKR is disabled)
        """
        logger.error("❌ IBKR connection is disabled. Use Bitget data collector.")
        logger.info("See: Data_collecting_system_bitget/advanced_data_collector.py")
        return False
    
    def get_historical_data(self, days: int = 30, bar_size: str = '1 hour') -> pd.DataFrame:
        """
        Get historical data - DISABLED (IBKR removed)
        
        ⚠️ This method is disabled. Use Bitget data collector instead.
        
        Returns:
            Empty DataFrame
        """
        logger.error("❌ IBKR data fetching is disabled. Use Bitget.")
        return pd.DataFrame()  # Return empty DataFrame
    
    def disconnect(self):
        """Disconnect - DISABLED (no action needed)"""
        if self.data_collector and hasattr(self.data_collector, 'disconnect'):
            self.data_collector.disconnect()
            logger.info("Disconnected (IBKR disabled)")
    
    def __del__(self):
        """Cleanup"""
        try:
            if self.data_collector and hasattr(self.data_collector, 'disconnect'):
                self.data_collector.disconnect()
        except:
            pass
    
    def place_order(self, action: str, quantity: int, order_type: str = 'MKT') -> bool:
        """
        Place an order - DISABLED (IBKR removed)
        
        ⚠️ IBKR live trading is disabled. Use backtesting instead.
        """
        logger.error("❌ IBKR order placement is disabled. Use backtesting.")
        return False
    
    def get_account_info(self) -> dict:
        """Get account information - DISABLED (IBKR removed)"""
        logger.error("❌ IBKR account info is disabled.")
        return {}

def main():
    parser = argparse.ArgumentParser(description='AI Trading Bot (IBKR Integration Disabled)')
    parser.add_argument('--symbol', type=str, required=True, help='Ticker symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch (default: 30)')
    parser.add_argument('--quantity', type=int, default=100, help='Number of shares to trade (default: 100)')
    parser.add_argument('--paper', action='store_true', default=True, help='Use paper trading (default: True)')
    parser.add_argument('--live', dest='paper', action='store_false', help='Use live trading (overrides --paper)')
    parser.add_argument('--client-id', type=int, default=1, help='Client ID for IB connection (default: 1)')
    args = parser.parse_args()
    
    # Initialize and run the trading bot
    bot = TradingBot(symbol=args.symbol, paper=args.paper, client_id=args.client_id)
    
    try:
        # Connect to TWS/IB Gateway
        if not bot.connect():
            logger.error("Failed to connect to TWS/IB Gateway. Exiting...")
            return
        
        # Get account information
        account_info = bot.get_account_info()
        if account_info:
            logger.info(f"Account: {account_info.get('account')}")
            logger.info("Account Summary:")
            for key, value in account_info.get('values', {}).items():
                logger.info(f"  {key}: {value}")
        
        # Get historical data
        df_full = bot.get_historical_data(days=args.days, bar_size='1 min')
        if df_full is None or df_full.empty:
            logger.error(f"Failed to fetch 1-minute data for {args.symbol}")
            return
        logger.info(f"\nLatest 1-min data points:")
        logger.info(df_full.tail())
        
        # After fetching and preprocessing df_full, ensure 'timestamp' column exists and is datetime
        if 'timestamp' not in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full.index)
        else:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')
        
        # Compute all indicator columns and get the combined signal using the trading_algos function
        df_full, algo_signal = get_total_algo_signal(
            df_full,
            fisher_period=FISHER_PERIOD,
            fisher_signal_period=FISHER_SIGNAL_PERIOD,
            rsi_period=RSI_PERIOD,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_oversold=RSI_OVERSOLD,
            cci_period=CCI_PERIOD,
            cci_level=CCI_LEVEL
        )

    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Ensure we disconnect cleanly
        bot.disconnect()
        logger.info("Trading bot stopped")
        logger.warning("Running in simulation mode (no actual trades will be placed)")
    
    # Rename and format columns to match expected format
    df_full = df_full.rename(columns={
        'date': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    # For forecasting: generate point forecasts with horizon=64 (fast)
    df_forecast = (
        df_full[['timestamp','close']]
        .rename(columns={'timestamp':'ds','close':'y'})
        .assign(unique_id=args.symbol)
    )
    forecaster = TimeSeriesForecaster(context_len=128, horizon_len=64)
    preds = {
        'TimeGPT': forecaster.timegpt_forecast(df_forecast),
        'ChronosT5': forecaster.chronos_t5_forecast(df_forecast),
        'ChronosBolt': forecaster.chronos_bolt_forecast(df_forecast),
        'TimesFM1': forecaster.timesfm1_forecast(df_forecast),
        'TimesFM2': forecaster.timesfm2_forecast(df_forecast),
    }
    for name, df_pred in preds.items():
        logger.info(f"{name} forecast:\n{df_pred}")

    previous_score = 0
    current_position = None  # track open position (None, 'long', 'short')
    forecast_signal = 0.0  # initialize forecast signal for reporting
    est = pytz.timezone('US/Eastern')
    market_open = time(8, 30)
    market_close = time(15, 55)
    # Prepare CSV report file
    report_file = f"{args.symbol}_report.csv"
    if not os.path.isfile(report_file):
        with open(report_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Include current portfolio size
            writer.writerow(['timestamp','symbol','position','forecast_signal','pattern_score','position_size'])
    # Real-time trading loop, polling every 5 minutes
    import numpy as np

    # Add parameters
    STOP_LOSS_PCT = 0.048  # 4.8%
    TAKE_PROFIT_PCT = 0.125  # 12.5%

    # Initialize tracking variables before the trading loop
    entry_price = None

    # Inside the trading loop, replace or augment the signal logic with:

    while True:
        now_est = datetime.now(est)
        if market_open < now_est.time() < market_close:
            latest = df_full.iloc[-1]
            prev = df_full.iloc[-2]

            # Fisher cross signals
            fisher_cross_up = (prev['fisher'] < prev['fisher_signal']) and (latest['fisher'] > latest['fisher_signal'])
            fisher_cross_down = (prev['fisher'] > prev['fisher_signal']) and (latest['fisher'] < latest['fisher_signal'])

            # RSI signals
            rsi_oversold = latest['rsi'] < RSI_OVERSOLD
            rsi_overbought = latest['rsi'] > RSI_OVERBOUGHT

            # CCI signals
            cci_cross_up = (prev['cci'] < CCI_LEVEL) and (latest['cci'] > CCI_LEVEL)
            cci_cross_down = (prev['cci'] > -CCI_LEVEL) and (latest['cci'] < -CCI_LEVEL)

            # Determine buy/sell signals
            buy_signal = fisher_cross_up and rsi_oversold and cci_cross_up
            sell_signal = fisher_cross_down and rsi_overbought and cci_cross_down

            current_price = latest['close']

            # Manage open positions with stop loss / take profit
            if current_position == 'long' and entry_price is not None:
                if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                    logger.info("Stop loss triggered on long position")
                    if bot.ib and bot.ib.isConnected():
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    current_position = None
                    entry_price = None
                elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                    logger.info("Take profit triggered on long position")
                    if bot.ib and bot.ib.isConnected():
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    current_position = None
                    entry_price = None

            elif current_position == 'short' and entry_price is not None:
                if current_price >= entry_price * (1 + STOP_LOSS_PCT):
                    logger.info("Stop loss triggered on short position")
                    if bot.ib and bot.ib.isConnected():
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                    current_position = None
                    entry_price = None
                elif current_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                    logger.info("Take profit triggered on short position")
                    if bot.ib and bot.ib.isConnected():
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                    current_position = None
                    entry_price = None

            # Entry signals
            if buy_signal and current_position != 'long':
                logger.info("Signal: Enter long position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'short':
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                    execute_trade(bot.ib, args.symbol, 'buy', args.quantity)
                current_position = 'long'
                entry_price = current_price

            elif sell_signal and current_position != 'short':
                logger.info("Signal: Enter short position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'long':
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    execute_trade(bot.ib, args.symbol, 'sell', args.quantity)
                current_position = 'short'
                entry_price = current_price

            # No clear signal: optionally exit positions or hold
            if not buy_signal and not sell_signal and current_position is not None:
                logger.info("No clear signal - exiting position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'long':
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    elif current_position == 'short':
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                current_position = None
                entry_price = None

            # --- INSERT: Combined Algo Trading Strategy ---
            # The algo_signal is now directly available from get_total_algo_signal
            if algo_signal == 'long' and current_position != 'long':
                logger.info("[ALGO] All methods agree: Enter long position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'short':
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                    execute_trade(bot.ib, args.symbol, 'buy', args.quantity)
                current_position = 'long'
                entry_price = current_price
            elif algo_signal == 'short' and current_position != 'short':
                logger.info("[ALGO] All methods agree: Enter short position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'long':
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    execute_trade(bot.ib, args.symbol, 'sell', args.quantity)
                current_position = 'short'
                entry_price = current_price
            elif algo_signal == 'hold' and current_position is not None:
                logger.info("[ALGO] All methods agree: Exit position")
                if bot.ib and bot.ib.isConnected():
                    if current_position == 'long':
                        execute_trade(bot.ib, args.symbol, 'exit_buy')
                    elif current_position == 'short':
                        execute_trade(bot.ib, args.symbol, 'exit_sell')
                current_position = None
                entry_price = None

        else:
            logger.info('Market is closed or pre-open')
            try:
                analyzer = TransformerSentimentAnalyzer()
                articles = analyzer.fetch_articles(f"{args.symbol} stock")
                if articles:
                    report, sentiment = analyzer.analyze_articles(articles)
                    logger.info(f"Sentiment Analysis - Average: {sentiment:.2f}")
                    
                    # Only trade on strong sentiment outside market hours
                    if sentiment > 0.3 and current_position != 'long':
                        logger.info("Strong positive sentiment - Opening long position")
                        if bot.ib and bot.ib.isConnected():
                            if current_position == 'short':
                                execute_trade(bot.ib, args.symbol, 'exit_sell')
                            execute_trade(bot.ib, args.symbol, 'buy', args.quantity)
                        current_position = 'long'
                    elif sentiment < -0.3 and current_position != 'short':
                        logger.info("Strong negative sentiment - Opening short position")
                        if bot.ib and bot.ib.isConnected():
                            if current_position == 'long':
                                execute_trade(bot.ib, args.symbol, 'exit_buy')
                            execute_trade(bot.ib, args.symbol, 'sell', args.quantity)
                        current_position = 'short'
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
            
            # Longer sleep when market is closed
            time_module.sleep(900)  # 15 minutes
        # Wait before next cycle (5 minutes)
        next_run = datetime.now() + timedelta(minutes=5)
        logger.info(f"Next update at {next_run.strftime('%H:%M:%S')}")
        # Periodic report
        report_time = datetime.now(est).strftime('%Y-%m-%d %H:%M')
        logger.info(f"Periodic Report - {report_time} | Symbol: {args.symbol} | Position: {current_position} | ForecastSignal: {forecast_signal:.2f} | PatternScore: {previous_score}")
        # Append to CSV report, including current portfolio size
        account_info = bot.get_account_info()
        positions = account_info.get('positions', [])
        position_size = next((p['position'] for p in positions if p['contract'].symbol == args.symbol), 0)
        with open(report_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([report_time, args.symbol, current_position, f"{forecast_signal:.2f}", previous_score, position_size])
        time_module.sleep(300)

# --- Helper: execute_trade (if missing) ---
def execute_trade(ib, symbol, action, quantity=None):
    """Unified trade execution for the bot trading loop."""
    if action in ['buy', 'sell'] and quantity is not None:
        bot.place_order(action.upper(), quantity)
    elif action == 'exit_buy':
        # Sell all position
        bot.place_order('SELL', args.quantity)
    elif action == 'exit_sell':
        # Buy to cover all position
        bot.place_order('BUY', args.quantity)
    else:
        logger.warning(f"Unknown action for execute_trade: {action}")

# --- Optimized Parameters (from Optuna) ---
FISHER_PERIOD = 25
FISHER_SIGNAL_PERIOD = 28
RSI_PERIOD = 8
RSI_OVERBOUGHT = 76
RSI_OVERSOLD = 19
CCI_PERIOD = 23
CCI_LEVEL = 123
STOP_LOSS_PCT = 0.048  # 4.8%
TAKE_PROFIT_PCT = 0.125  # 12.5%

if __name__ == '__main__':
    main()