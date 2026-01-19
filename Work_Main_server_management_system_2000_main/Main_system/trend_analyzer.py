#!/usr/bin/env python3
"""
Specialized Trend Analyzer for Financial Markets
Real-time monitoring of Gold, Dollar, Interest Rates, Oil, IOF trends
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from threading import Thread, Lock
from collections import defaultdict

@dataclass
class TrendData:
    """Data structure for trend analysis"""
    symbol: str
    current_value: float
    previous_value: float
    change: float
    change_pct: float
    trend_direction: str  # up, down, sideways
    trend_strength: float  # 0-1
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class MarketAlert:
    """Data structure for market alerts"""
    symbol: str
    alert_type: str  # breakout, breakdown, reversal, news_impact
    message: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    price_level: Optional[float] = None

class TrendAnalyzer:
    def __init__(self, data_dir: str = "trend_data"):
        self.data_dir = data_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/trends", exist_ok=True)
        os.makedirs(f"{data_dir}/alerts", exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Data storage
        self.trend_data = {}
        self.historical_data = defaultdict(list)
        self.alerts = []
        self.lock = Lock()
        
        # Market symbols for trend analysis
        self.market_symbols = {
            'gold': ['XAU/USD', 'XAU/BRL', 'GLD'],
            'dollar': ['USD/BRL', 'DXY', 'EUR/USD', 'GBP/USD'],
            'interest_rates': ['SELIC', 'FED_RATE', 'ECB_RATE', 'LIBOR'],
            'oil': ['WTI/USD', 'BRENT/USD', 'USOIL'],
            'iof': ['IOF_RATE', 'BRAZIL_TAX'],
            'crypto': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
            'stocks': ['^BVSP', 'PETR4.SA', 'VALE3.SA']
        }
        
        # Trend analysis parameters
        self.trend_params = {
            'short_period': 5,    # 5 periods for short-term trend
            'medium_period': 20,  # 20 periods for medium-term trend
            'long_period': 50,    # 50 periods for long-term trend
            'volatility_threshold': 0.02,  # 2% volatility threshold
            'trend_strength_threshold': 0.6  # 60% trend strength threshold
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.data_dir}/trend_analyzer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_gold_trend_data(self) -> Dict[str, TrendData]:
        """Get gold trend data from multiple sources"""
        try:
            gold_data = {}
            
            # Mock gold data (replace with real API calls)
            current_gold_usd = 2345.67
            previous_gold_usd = 2333.22
            change = current_gold_usd - previous_gold_usd
            change_pct = (change / previous_gold_usd) * 100
            
            gold_data['XAU/USD'] = TrendData(
                symbol='XAU/USD',
                current_value=current_gold_usd,
                previous_value=previous_gold_usd,
                change=change,
                change_pct=change_pct,
                trend_direction='up' if change_pct > 0 else 'down',
                trend_strength=0.75,
                support_level=2320.0,
                resistance_level=2360.0,
                timestamp=datetime.now()
            )
            
            # Gold in BRL
            current_gold_brl = 12680.50
            previous_gold_brl = 12620.30
            change_brl = current_gold_brl - previous_gold_brl
            change_pct_brl = (change_brl / previous_gold_brl) * 100
            
            gold_data['XAU/BRL'] = TrendData(
                symbol='XAU/BRL',
                current_value=current_gold_brl,
                previous_value=previous_gold_brl,
                change=change_brl,
                change_pct=change_pct_brl,
                trend_direction='up' if change_pct_brl > 0 else 'down',
                trend_strength=0.68,
                support_level=12550.0,
                resistance_level=12800.0,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Retrieved gold trend data for {len(gold_data)} symbols")
            return gold_data
            
        except Exception as e:
            self.logger.error(f"Error getting gold trend data: {e}")
            return {}
    
    def get_dollar_trend_data(self) -> Dict[str, TrendData]:
        """Get dollar trend data"""
        try:
            dollar_data = {}
            
            # USD/BRL, 
            current_usd_brl = 5.4050
            previous_usd_brl = 5.4200
            change = current_usd_brl - previous_usd_brl
            change_pct = (change / previous_usd_brl) * 100
            
            dollar_data['USD/BRL'] = TrendData(
                symbol='USD/BRL',
                current_value=current_usd_brl,
                previous_value=previous_usd_brl,
                change=change,
                change_pct=change_pct,
                trend_direction='down' if change_pct < 0 else 'up',
                trend_strength=0.45,
                support_level=5.35,
                resistance_level=5.45,
                timestamp=datetime.now()
            )
            
            # DXY (Dollar Index)
            current_dxy = 104.25
            previous_dxy = 104.50
            change_dxy = current_dxy - previous_dxy
            change_pct_dxy = (change_dxy / previous_dxy) * 100
            
            dollar_data['DXY'] = TrendData(
                symbol='DXY',
                current_value=current_dxy,
                previous_value=previous_dxy,
                change=change_dxy,
                change_pct=change_pct_dxy,
                trend_direction='down' if change_pct_dxy < 0 else 'up',
                trend_strength=0.52,
                support_level=103.80,
                resistance_level=104.80,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Retrieved dollar trend data for {len(dollar_data)} symbols")
            return dollar_data
            
        except Exception as e:
            self.logger.error(f"Error getting dollar trend data: {e}")
            return {}
    
    def get_interest_rate_trend_data(self) -> Dict[str, TrendData]:
        """Get interest rate trend data"""
        try:
            rate_data = {}
            
            # SELIC (Brazil)
            current_selic = 13.75
            previous_selic = 13.75  # No change
            change = current_selic - previous_selic
            change_pct = 0.0
            
            rate_data['SELIC'] = TrendData(
                symbol='SELIC',
                current_value=current_selic,
                previous_value=previous_selic,
                change=change,
                change_pct=change_pct,
                trend_direction='sideways',
                trend_strength=0.0,
                support_level=13.50,
                resistance_level=14.00,
                timestamp=datetime.now()
            )
            
            # FED Rate
            current_fed_rate = 5.50
            previous_fed_rate = 5.50  # No change
            change_fed = current_fed_rate - previous_fed_rate
            change_pct_fed = 0.0
            
            rate_data['FED_RATE'] = TrendData(
                symbol='FED_RATE',
                current_value=current_fed_rate,
                previous_value=previous_fed_rate,
                change=change_fed,
                change_pct=change_pct_fed,
                trend_direction='sideways',
                trend_strength=0.0,
                support_level=5.25,
                resistance_level=5.75,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Retrieved interest rate trend data for {len(rate_data)} symbols")
            return rate_data
            
        except Exception as e:
            self.logger.error(f"Error getting interest rate trend data: {e}")
            return {}
    
    def get_oil_trend_data(self) -> Dict[str, TrendData]:
        """Get oil trend data"""
        try:
            oil_data = {}
            
            # WTI
            current_wti = 78.45
            previous_wti = 77.20
            change = current_wti - previous_wti
            change_pct = (change / previous_wti) * 100
            
            oil_data['WTI/USD'] = TrendData(
                symbol='WTI/USD',
                current_value=current_wti,
                previous_value=previous_wti,
                change=change,
                change_pct=change_pct,
                trend_direction='up' if change_pct > 0 else 'down',
                trend_strength=0.62,
                support_level=76.50,
                resistance_level=80.00,
                timestamp=datetime.now()
            )
            
            # Brent
            current_brent = 82.30
            previous_brent = 81.15
            change_brent = current_brent - previous_brent
            change_pct_brent = (change_brent / previous_brent) * 100
            
            oil_data['BRENT/USD'] = TrendData(
                symbol='BRENT/USD',
                current_value=current_brent,
                previous_value=previous_brent,
                change=change_brent,
                change_pct=change_pct_brent,
                trend_direction='up' if change_pct_brent > 0 else 'down',
                trend_strength=0.58,
                support_level=80.00,
                resistance_level=84.00,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Retrieved oil trend data for {len(oil_data)} symbols")
            return oil_data
            
        except Exception as e:
            self.logger.error(f"Error getting oil trend data: {e}")
            return {}
    
    def get_iof_trend_data(self) -> Dict[str, TrendData]:
        """Get IOF trend data"""
        try:
            iof_data = {}
            
            # IOF Rate (Brazilian Financial Transaction Tax)
            current_iof = 0.38  # 0.38% for forex transactions
            previous_iof = 0.38  # No change
            change = current_iof - previous_iof
            change_pct = 0.0
            
            iof_data['IOF_RATE'] = TrendData(
                symbol='IOF_RATE',
                current_value=current_iof,
                previous_value=previous_iof,
                change=change,
                change_pct=change_pct,
                trend_direction='sideways',
                trend_strength=0.0,
                support_level=0.38,
                resistance_level=0.38,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Retrieved IOF trend data for {len(iof_data)} symbols")
            return iof_data
            
        except Exception as e:
            self.logger.error(f"Error getting IOF trend data: {e}")
            return {}
    
    def analyze_trend_strength(self, prices: List[float]) -> float:
        """Analyze trend strength using linear regression"""
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared (trend strength)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return abs(r_squared)
    
    def detect_breakout_breakdown(self, current_price: float, support: float, resistance: float) -> Optional[str]:
        """Detect breakout or breakdown"""
        if current_price > resistance:
            return 'breakout'
        elif current_price < support:
            return 'breakdown'
        return None
    
    def generate_trend_alerts(self, trend_data: Dict[str, TrendData]) -> List[MarketAlert]:
        """Generate alerts based on trend analysis"""
        alerts = []
        
        for symbol, data in trend_data.items():
            # Check for significant moves
            if abs(data.change_pct) > 2.0:  # 2% move
                alert_type = 'significant_move'
                severity = 'high' if abs(data.change_pct) > 5.0 else 'medium'
                message = f"{symbol} moved {data.change_pct:+.2f}% - {data.trend_direction.upper()} trend"
                
                alerts.append(MarketAlert(
                    symbol=symbol,
                    alert_type=alert_type,
                    message=message,
                    severity=severity,
                    timestamp=datetime.now(),
                    price_level=data.current_value
                ))
            
            # Check for breakouts/breakdowns
            if data.support_level and data.resistance_level:
                breakout_type = self.detect_breakout_breakdown(
                    data.current_value, data.support_level, data.resistance_level
                )
                
                if breakout_type:
                    alert_type = 'breakout' if breakout_type == 'breakout' else 'breakdown'
                    severity = 'high'
                    message = f"{symbol} {breakout_type.upper()} at {data.current_value:.2f}"
                    
                    alerts.append(MarketAlert(
                        symbol=symbol,
                        alert_type=alert_type,
                        message=message,
                        severity=severity,
                        timestamp=datetime.now(),
                        price_level=data.current_value
                    ))
            
            # Check for strong trends
            if data.trend_strength > 0.7:
                alert_type = 'strong_trend'
                severity = 'medium'
                message = f"{symbol} showing strong {data.trend_direction} trend (strength: {data.trend_strength:.2f})"
                
                alerts.append(MarketAlert(
                    symbol=symbol,
                    alert_type=alert_type,
                    message=message,
                    severity=severity,
                    timestamp=datetime.now(),
                    price_level=data.current_value
                ))
        
        return alerts
    
    def collect_all_trend_data(self):
        """Collect trend data from all sources"""
        with self.lock:
            # Collect data from different sources
            gold_data = self.get_gold_trend_data()
            dollar_data = self.get_dollar_trend_data()
            rate_data = self.get_interest_rate_trend_data()
            oil_data = self.get_oil_trend_data()
            iof_data = self.get_iof_trend_data()
            
            # Combine all data
            self.trend_data = {
                'gold': gold_data,
                'dollar': dollar_data,
                'interest_rates': rate_data,
                'oil': oil_data,
                'iof': iof_data
            }
            
            # Generate alerts
            all_alerts = []
            for category, data in self.trend_data.items():
                alerts = self.generate_trend_alerts(data)
                all_alerts.extend(alerts)
            
            self.alerts = all_alerts
            
            # Save data
            self.save_trend_data()
            self.save_alerts()
            
            total_symbols = sum(len(data) for data in self.trend_data.values())
            self.logger.info(f"Collected trend data for {total_symbols} symbols, generated {len(all_alerts)} alerts")
    
    def save_trend_data(self):
        """Save trend data to file"""
        try:
            trend_data = {}
            for category, data in self.trend_data.items():
                trend_data[category] = {}
                for symbol, item in data.items():
                    trend_data[category][symbol] = {
                        'symbol': item.symbol,
                        'current_value': item.current_value,
                        'previous_value': item.previous_value,
                        'change': item.change,
                        'change_pct': item.change_pct,
                        'trend_direction': item.trend_direction,
                        'trend_strength': item.trend_strength,
                        'support_level': item.support_level,
                        'resistance_level': item.resistance_level,
                        'timestamp': item.timestamp.isoformat() if item.timestamp else None
                    }
            
            filename = f"{self.data_dir}/trends/trend_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w') as f:
                json.dump(trend_data, f, indent=2)
            
            self.logger.info(f"Saved trend data to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving trend data: {e}")
    
    def save_alerts(self):
        """Save alerts to file"""
        try:
            alert_data = []
            for alert in self.alerts:
                alert_data.append({
                    'symbol': alert.symbol,
                    'alert_type': alert.alert_type,
                    'message': alert.message,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat(),
                    'price_level': alert.price_level
                })
            
            filename = f"{self.data_dir}/alerts/alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            self.logger.info(f"Saved {len(alert_data)} alerts to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alerts: {e}")
    
    def print_trend_summary(self):
        """Print comprehensive trend summary"""
        print("\n" + "="*100)
        print("TREND ANALYSIS SUMMARY")
        print("="*100)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Print trends by category
        for category, data in self.trend_data.items():
            if data:
                print(f"{category.upper()} TRENDS:")
                print("-" * 50)
                
                for symbol, item in data.items():
                    change_symbol = "🟢" if item.change_pct > 0 else "🔴" if item.change_pct < 0 else "⚪"
                    trend_emoji = "📈" if item.trend_direction == 'up' else "📉" if item.trend_direction == 'down' else "➡️"
                    
                    print(f"  {change_symbol} {trend_emoji} {symbol}: {item.current_value:.2f} ({item.change_pct:+.2f}%)")
                    print(f"     Trend: {item.trend_direction.upper()} | Strength: {item.trend_strength:.2f}")
                    
                    if item.support_level and item.resistance_level:
                        print(f"     Support: {item.support_level:.2f} | Resistance: {item.resistance_level:.2f}")
                    print()
        
        # Print alerts
        if self.alerts:
            print("ALERTS:")
            print("-" * 50)
            
            # Group alerts by severity
            high_alerts = [a for a in self.alerts if a.severity == 'high']
            medium_alerts = [a for a in self.alerts if a.severity == 'medium']
            
            if high_alerts:
                print("  🚨 HIGH PRIORITY:")
                for alert in high_alerts[:5]:
                    print(f"    • {alert.message}")
                print()
            
            if medium_alerts:
                print("  ⚠️  MEDIUM PRIORITY:")
                for alert in medium_alerts[:5]:
                    print(f"    • {alert.message}")
                print()
        
        print("="*100)
    
    def start_monitoring(self):
        """Start continuous trend monitoring"""
        self.logger.info("Starting Trend Analyzer monitoring...")
        
        # Schedule tasks
        try:
            import schedule
            schedule.every(5).minutes.do(self.collect_all_trend_data)
            schedule.every(1).hours.do(self.print_trend_summary)
        except ImportError:
            self.logger.warning("Schedule library not available, running without scheduling")
        
        # Initial collection
        self.collect_all_trend_data()
        self.print_trend_summary()
        
        # Run scheduled tasks
        while True:
            try:
                import schedule
                schedule.run_pending()
            except ImportError:
                # If schedule is not available, just sleep
                pass
            time.sleep(60)

def main():
    print("Trend Analyzer - Real-time Market Trend Monitoring")
    print("Monitoring: Gold, Dollar, Interest Rates, Oil, IOF Trends")
    print("="*100)
    
    analyzer = TrendAnalyzer()
    
    try:
        analyzer.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping Trend Analyzer...")
        analyzer.logger.info("Trend Analyzer stopped by user")

if __name__ == "__main__":
    main() 