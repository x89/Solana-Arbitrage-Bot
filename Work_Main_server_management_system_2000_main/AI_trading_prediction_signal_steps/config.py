

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class DataConfig:
    """Data collection and processing configuration"""
    # API Settings
    BITGET_API_URL: str = "https://api.bitget.com"
    # IBKR integration removed - use Bitget instead
    # IBKR_HOST: str = "127.0.0.1"
    # IBKR_PAPER_PORT: int = 7497
    # IBKR_LIVE_PORT: int = 7496
    
    # Data Collection
    DEFAULT_SYMBOL: str = "SOLUSDT"
    DEFAULT_TIMEFRAME: str = "15min"
    HISTORICAL_DAYS: int = 30
    MAX_CANDLES: int = 200
    
    # Data Storage
    DATA_DIR: str = "data"
    CHARTS_DIR: str = "charts"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"

@dataclass
class TradingConfig:
    """Trading execution configuration"""
    # Trading Mode
    MODE: TradingMode = TradingMode.PAPER
    
    # Position Management
    DEFAULT_QUANTITY: int = 100
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    MIN_POSITION_SIZE: float = 0.01  # 1% of portfolio
    
    # Risk Management
    STOP_LOSS_PCT: float = 0.048  # 4.8%
    TAKE_PROFIT_PCT: float = 0.125  # 12.5%
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_DRAWDOWN: float = 0.15  # 15% max drawdown
    
    # Transaction Costs
    TRANSACTION_COST_PCT: float = 0.001  # 0.1% per trade
    
    # Trading Hours
    MARKET_OPEN: str = "08:30"
    MARKET_CLOSE: str = "15:55"
    TIMEZONE: str = "US/Eastern"

@dataclass
class AIConfig:
    """AI models and prediction configuration"""
    # Time Series Forecasting
    TIMEGPT_ENABLED: bool = True
    CHRONOS_T5_ENABLED: bool = True
    CHRONOS_BOLT_ENABLED: bool = True
    TIMESFM1_ENABLED: bool = True
    TIMESFM2_ENABLED: bool = True
    
    # Forecasting Parameters
    CONTEXT_LENGTH: int = 128
    HORIZON_LENGTH: int = 64
    FORECAST_PERIODS: int = 6
    
    # Pattern Detection
    PATTERN_DETECTION_ENABLED: bool = True
    YOLO_MODEL_PATH: str = "chart_pattern/pattern_yolo_12x.pt"
    CANDLESTICK_MODEL_PATH: str = "chart_pattern/candlestick_yolo_12x.pt"
    
    # Sentiment Analysis
    SENTIMENT_ENABLED: bool = True
    SENTIMENT_THRESHOLD: float = 0.3
    NEWS_SOURCES: List[str] = None
    
    # Technical Indicators
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    # Machine Learning
    ML_MODEL_ENABLED: bool = True
    ML_MODEL_PATH: str = "AI_news_training_system/Model_enhanced_system/model_enhanced"
    FEATURE_ENGINEERING_ENABLED: bool = True

@dataclass
class SignalConfig:
    """Signal generation and filtering configuration"""
    # Signal Weights (must sum to 1.0)
    WEIGHTS: Dict[str, float] = None
    
    # Signal Thresholds
    MIN_SIGNAL_CONFIDENCE: float = 0.6
    MIN_SIGNAL_STRENGTH: SignalStrength = SignalStrength.MODERATE
    
    # Signal Validation
    CONFIRMATION_PERIODS: int = 3
    SIGNAL_TIMEOUT_MINUTES: int = 30
    
    # Signal Types
    ENABLE_FORECAST_SIGNALS: bool = True
    ENABLE_PATTERN_SIGNALS: bool = True
    ENABLE_SENTIMENT_SIGNALS: bool = True
    ENABLE_TECHNICAL_SIGNALS: bool = True
    ENABLE_ML_SIGNALS: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "trading_bot.log"
    LOG_ROTATION_SIZE: str = "10MB"
    LOG_BACKUP_COUNT: int = 5
    
    # Alerts
    ENABLE_EMAIL_ALERTS: bool = False
    ENABLE_SLACK_ALERTS: bool = False
    ENABLE_DISCORD_ALERTS: bool = False
    
    # Performance Monitoring
    PERFORMANCE_TRACKING_ENABLED: bool = True
    PORTFOLIO_TRACKING_ENABLED: bool = True
    RISK_MONITORING_ENABLED: bool = True
    
    # Reporting
    DAILY_REPORT_ENABLED: bool = True
    WEEKLY_REPORT_ENABLED: bool = True
    MONTHLY_REPORT_ENABLED: bool = True

@dataclass
class BotConfig:
    """Main bot configuration"""
    # Core Settings
    BOT_NAME: str = "AI Trading Prediction Signal Bot"
    VERSION: str = "1.0.0"
    
    # Update Intervals
    DATA_UPDATE_INTERVAL: int = 15  # seconds
    SIGNAL_CHECK_INTERVAL: int = 60  # seconds
    PORTFOLIO_UPDATE_INTERVAL: int = 300  # seconds
    
    # Safety Settings
    MAX_CONCURRENT_POSITIONS: int = 3
    EMERGENCY_STOP_ENABLED: bool = True
    EMERGENCY_STOP_THRESHOLD: float = 0.20  # 20% loss
    
    # Development
    DEBUG_MODE: bool = False
    BACKTEST_MODE: bool = False
    SIMULATION_MODE: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data = DataConfig()
        self.trading = TradingConfig()
        self.ai = AIConfig()
        self.signal = SignalConfig()
        self.monitoring = MonitoringConfig()
        self.bot = BotConfig()
        
        # Set default signal weights
        if self.signal.WEIGHTS is None:
            self.signal.WEIGHTS = {
                'forecast': 0.25,
                'pattern': 0.20,
                'sentiment': 0.15,
                'technical': 0.25,
                'ml': 0.15
            }
        
        # Set default news sources
        if self.ai.NEWS_SOURCES is None:
            self.ai.NEWS_SOURCES = [
                "yahoo_finance",
                "marketwatch",
                "bloomberg",
                "reuters"
            ]
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        import json
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section, values in config_data.items():
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
        except Exception as e:
            print(f"Error loading config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        import json
        try:
            config_data = {
                'data': self.data.__dict__,
                'trading': self.trading.__dict__,
                'ai': self.ai.__dict__,
                'signal': self.signal.__dict__,
                'monitoring': self.monitoring.__dict__,
                'bot': self.bot.__dict__
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check signal weights sum to 1.0
        if abs(sum(self.signal.WEIGHTS.values()) - 1.0) > 0.01:
            print("Error: Signal weights must sum to 1.0")
            return False
        
        # Check required directories exist
        required_dirs = [
            self.data.DATA_DIR,
            self.data.CHARTS_DIR,
            self.data.MODELS_DIR,
            self.data.LOGS_DIR
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        return True
    
    def get_trading_hours(self) -> tuple:
        """Get trading hours as time objects"""
        from datetime import time
        
        market_open = time.fromisoformat(self.trading.MARKET_OPEN)
        market_close = time.fromisoformat(self.trading.MARKET_CLOSE)
        
        return market_open, market_close
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        import pytz
        from datetime import datetime, time
        
        tz = pytz.timezone(self.trading.TIMEZONE)
        now = datetime.now(tz)
        current_time = now.time()
        
        market_open, market_close = self.get_trading_hours()
        
        return market_open <= current_time <= market_close

# Global configuration instance
config = Config()

if __name__ == "__main__":
    # Example usage
    print("AI Trading Prediction Signal Bot Configuration")
    print("=" * 50)
    print(f"Bot Name: {config.bot.BOT_NAME}")
    print(f"Version: {config.bot.VERSION}")
    print(f"Trading Mode: {config.trading.MODE.value}")
    print(f"Default Symbol: {config.data.DEFAULT_SYMBOL}")
    print(f"Signal Weights: {config.signal.WEIGHTS}")
    print(f"Market Open: {config.trading.MARKET_OPEN}")
    print(f"Market Close: {config.trading.MARKET_CLOSE}")
    print(f"Is Market Open: {config.is_market_open()}")
