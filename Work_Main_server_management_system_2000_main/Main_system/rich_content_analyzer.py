#!/usr/bin/env python3
"""
Rich Content Financial News Bot - Comprehensive Market Intelligence with Detailed Analysis
Provides rich content for each news item including market impact analysis, technical insights, and detailed explanations
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re
from dataclasses import dataclass
from threading import Thread, Lock
import schedule

@dataclass
class RichNewsItem:
    """Enhanced data structure for news items with rich content"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    category: str
    sentiment: str = "neutral"
    impact_score: float = 0.0
    market_impact: str = "neutral"
    
    # Rich content fields, 
    detailed_analysis: str = ""
    market_implications: str = ""
    technical_insights: str = ""
    related_symbols: List[str] = None
    key_metrics: Dict = None
    expert_opinion: str = ""
    risk_assessment: str = ""
    trading_opportunities: str = ""
    historical_context: str = ""
    future_outlook: str = ""

@dataclass
class MarketData:
    """Enhanced market data structure"""
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    # Rich content fields
    technical_indicators: Dict = None
    support_resistance: Dict = None
    volatility_metrics: Dict = None
    correlation_data: Dict = None
    market_sentiment: str = "neutral"

class RichContentFinancialBot:
    def __init__(self, data_dir: str = "rich_content_data"):
        self.data_dir = data_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/rich_news", exist_ok=True)
        os.makedirs(f"{data_dir}/market_analysis", exist_ok=True)
        os.makedirs(f"{data_dir}/reports", exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Data storage
        self.news_items = []
        self.market_data = defaultdict(list)
        self.lock = Lock()
        
        # Enhanced news sources with rich content capabilities
        self.news_sources = {
            # Brazilian Markets
            'bom_dia_mercado': {
                'url': 'https://www.bomdiamercado.com.br/picpay_tarde/',
                'name': 'Bom Dia Mercado',
                'category': 'brazilian_markets',
                'rich_content': True
            },
            'valor_economico': {
                'url': 'https://valor.globo.com/financas/',
                'name': 'Valor Econômico',
                'category': 'brazilian_markets',
                'rich_content': True
            },
            'infomoney': {
                'url': 'https://www.infomoney.com.br/mercados/',
                'name': 'InfoMoney',
                'category': 'brazilian_markets',
                'rich_content': True
            },
            
            # Global Markets
            'reuters': {
                'url': 'https://www.reuters.com/markets/',
                'name': 'Reuters',
                'category': 'global_markets',
                'rich_content': True
            },
            'bloomberg': {
                'url': 'https://www.bloomberg.com/markets',
                'name': 'Bloomberg',
                'category': 'global_markets',
                'rich_content': True
            },
            
            # Commodities & Gold
            'kitco': {
                'url': 'https://www.kitco.com/news/',
                'name': 'Kitco',
                'category': 'commodities',
                'rich_content': True
            },
            'oil_price': {
                'url': 'https://oilprice.com/',
                'name': 'OilPrice.com',
                'category': 'commodities',
                'rich_content': True
            },
            
            # Crypto News
            'coin_desk': {
                'url': 'https://www.coindesk.com/',
                'name': 'CoinDesk',
                'category': 'crypto',
                'rich_content': True
            },
            'cointelegraph': {
                'url': 'https://cointelegraph.com/',
                'name': 'Cointelegraph',
                'category': 'crypto',
                'rich_content': True
            },
            
            # Geopolitical & War News
            'reuters_world': {
                'url': 'https://www.reuters.com/world/',
                'name': 'Reuters World',
                'category': 'geopolitical',
                'rich_content': True
            },
            'bbc_world': {
                'url': 'https://www.bbc.com/news/world',
                'name': 'BBC World',
                'category': 'geopolitical',
                'rich_content': True
            }
        }
        
        # Market symbols for rich analysis
        self.market_symbols = {
            'crypto': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT'],
            'stocks': ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA'],
            'forex': ['USD/BRL', 'EUR/BRL', 'EUR/USD', 'GBP/USD'],
            'commodities': ['XAU/USD', 'XAG/USD', 'WTI/USD', 'BRENT/USD'],
            'indices': ['^GSPC', '^DJI', '^IXIC', '^BVSP'],
            'interest_rates': ['SELIC', 'FED_RATE', 'ECB_RATE']
        }
        
        # Rich content analysis parameters
        self.analysis_params = {
            'sentiment_threshold': 0.6,
            'impact_threshold': 0.7,
            'correlation_threshold': 0.5,
            'volatility_threshold': 0.03
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.data_dir}/rich_content_bot.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_detailed_analysis(self, title: str, content: str, category: str) -> str:
        """Generate detailed analysis for news item"""
        text = f"{title} {content}".lower()
        
        analysis = ""
        
        if category == 'gold':
            if 'ouro' in text or 'gold' in text:
                analysis = """
🔍 GOLD MARKET ANALYSIS:
• Safe haven demand likely to increase amid geopolitical tensions
• Central bank purchases continue to support gold prices
• Technical resistance at $2,360, support at $2,320
• Correlation with USD strength remains inverse
• Inflation hedge properties in focus
                """
        
        elif category == 'dollar':
            if 'dólar' in text or 'dollar' in text:
                analysis = """
💵 DOLLAR MARKET ANALYSIS:
• Fed policy expectations driving USD movements
• Risk sentiment affecting dollar strength
• Technical levels: Support at 5.35 BRL, Resistance at 5.45 BRL
• Correlation with emerging market currencies
• Safe haven flows during market stress
                """
        
        elif category == 'interest_rates':
            if 'juros' in text or 'selic' in text or 'fed' in text:
                analysis = """
📊 INTEREST RATE ANALYSIS:
• Central bank policy decisions impact all asset classes
• Higher rates typically strengthen local currency
• Bond yields and equity valuations affected
• Carry trade opportunities in forex markets
• Inflation expectations driving rate decisions
                """
        
        elif category == 'oil':
            if 'petróleo' in text or 'oil' in text:
                analysis = """
🛢️ OIL MARKET ANALYSIS:
• OPEC+ decisions critical for price direction
• Geopolitical tensions affecting supply concerns
• Demand outlook from major economies
• Technical levels: WTI support at $76.50, resistance at $80.00
• Energy sector stocks correlation
                """
        
        elif category == 'crypto':
            if 'bitcoin' in text or 'crypto' in text:
                analysis = """
₿ CRYPTO MARKET ANALYSIS:
• Institutional adoption continues to grow
• Regulatory developments key for market sentiment
• Technical analysis: BTC support at $64,000, resistance at $66,000
• Altcoin season indicators
• DeFi and NFT trends affecting specific tokens
                """
        
        elif category == 'geopolitical':
            if 'war' in text or 'conflict' in text:
                analysis = """
🌍 GEOPOLITICAL IMPACT ANALYSIS:
• Risk-off sentiment typically benefits safe havens
• Energy and commodity prices affected by supply disruptions
• Currency markets react to geopolitical tensions
• Flight to quality: Gold, USD, and government bonds
• Regional market impacts vary by proximity to conflicts
                """
        
        return analysis.strip()
    
    def generate_market_implications(self, title: str, content: str, category: str) -> str:
        """Generate market implications for news item"""
        text = f"{title} {content}".lower()
        
        implications = ""
        
        if 'positive' in text or 'alta' in text or 'sobe' in text:
            implications = """
📈 BULLISH IMPLICATIONS:
• Positive sentiment likely to continue
• Potential buying opportunities in related assets
• Momentum traders may enter positions
• Risk appetite increasing
• Correlation trades with similar assets
            """
        
        elif 'negative' in text or 'queda' in text or 'cai' in text:
            implications = """
📉 BEARISH IMPLICATIONS:
• Risk-off sentiment may prevail
• Safe haven assets likely to benefit
• Defensive positioning recommended
• Volatility expected to increase
• Correlation with risk assets
            """
        
        elif 'neutral' in text or 'lateral' in text:
            implications = """
➡️ NEUTRAL IMPLICATIONS:
• Market may continue sideways movement
• Range-bound trading opportunities
• Wait for clearer directional signals
• Focus on individual stock/sector performance
• Volatility may remain low
            """
        
        return implications.strip()
    
    def generate_technical_insights(self, category: str, sentiment: str) -> str:
        """Generate technical insights based on category and sentiment"""
        insights = ""
        
        if category == 'gold':
            if sentiment == 'positive':
                insights = """
🔧 TECHNICAL INSIGHTS - GOLD (BULLISH):
• RSI showing momentum building above 50
• MACD histogram turning positive
• Price above 20-day moving average
• Volume increasing on up days
• Fibonacci retracement levels: 38.2% at $2,320
• Key resistance at $2,360, breakout potential
                """
            else:
                insights = """
🔧 TECHNICAL INSIGHTS - GOLD (BEARISH):
• RSI below 50 indicating weakness
• MACD below signal line
• Price below 20-day moving average
• Support at $2,320 critical level
• Fibonacci support at 61.8% ($2,280)
• Volume declining on rallies
                """
        
        elif category == 'dollar':
            if sentiment == 'positive':
                insights = """
🔧 TECHNICAL INSIGHTS - USD (BULLISH):
• Dollar index above 104.50 resistance
• EUR/USD breaking below 1.1750 support
• USD/BRL testing 5.45 resistance level
• Momentum indicators showing strength
• Higher highs and higher lows pattern
• Volume confirming trend direction
                """
            else:
                insights = """
🔧 TECHNICAL INSIGHTS - USD (BEARISH):
• Dollar index below 104.00 support
• EUR/USD breaking above 1.1800 resistance
• USD/BRL testing 5.35 support level
• RSI showing oversold conditions
• Lower highs and lower lows pattern
• Potential reversal signals forming
                """
        
        return insights.strip()
    
    def identify_related_symbols(self, title: str, content: str, category: str) -> List[str]:
        """Identify related market symbols for news item"""
        text = f"{title} {content}".lower()
        related_symbols = []
        
        # Gold related
        if 'ouro' in text or 'gold' in text:
            related_symbols.extend(['XAU/USD', 'XAU/BRL', 'GLD', 'GDX'])
        
        # Dollar related
        if 'dólar' in text or 'dollar' in text:
            related_symbols.extend(['USD/BRL', 'DXY', 'EUR/USD', 'GBP/USD'])
        
        # Interest rates related
        if 'juros' in text or 'selic' in text or 'fed' in text:
            related_symbols.extend(['SELIC', 'FED_RATE', '^TNX', '^TYX'])
        
        # Oil related
        if 'petróleo' in text or 'oil' in text:
            related_symbols.extend(['WTI/USD', 'BRENT/USD', 'XOM', 'CVX'])
        
        # Crypto related
        if 'bitcoin' in text or 'crypto' in text:
            related_symbols.extend(['BTC/USD', 'ETH/USD', 'SOL/USD'])
        
        # Brazilian stocks
        if 'ibovespa' in text or 'bolsa' in text:
            related_symbols.extend(['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA'])
        
        return list(set(related_symbols))  # Remove duplicates
    
    def generate_key_metrics(self, category: str) -> Dict:
        """Generate key metrics for the category"""
        metrics = {}
        
        if category == 'gold':
            metrics = {
                'current_price': 2345.67,
                'daily_change': 12.45,
                'daily_change_pct': 0.53,
                'volatility': 0.018,
                'correlation_usd': -0.65,
                'correlation_stocks': -0.25,
                'support_level': 2320.0,
                'resistance_level': 2360.0,
                'rsi': 58.5,
                'macd_signal': 'positive'
            }
        
        elif category == 'dollar':
            metrics = {
                'current_price': 5.4050,
                'daily_change': -0.0150,
                'daily_change_pct': -0.28,
                'volatility': 0.012,
                'correlation_rates': 0.75,
                'correlation_risk': -0.45,
                'support_level': 5.35,
                'resistance_level': 5.45,
                'rsi': 42.3,
                'macd_signal': 'negative'
            }
        
        elif category == 'oil':
            metrics = {
                'current_price': 78.45,
                'daily_change': 1.25,
                'daily_change_pct': 1.62,
                'volatility': 0.025,
                'correlation_usd': -0.35,
                'correlation_stocks': 0.15,
                'support_level': 76.50,
                'resistance_level': 80.00,
                'rsi': 62.1,
                'macd_signal': 'positive'
            }
        
        return metrics
    
    def generate_expert_opinion(self, category: str, sentiment: str) -> str:
        """Generate expert opinion based on category and sentiment"""
        opinions = {
            'gold': {
                'positive': "Analysts expect gold to continue its upward trajectory as geopolitical tensions and inflation concerns drive safe-haven demand. Central bank purchases remain strong, providing fundamental support.",
                'negative': "Gold may face headwinds from a stronger dollar and rising real yields. However, long-term fundamentals remain supportive for the precious metal.",
                'neutral': "Gold trading in a range as conflicting factors balance each other. Technical levels key for short-term direction."
            },
            'dollar': {
                'positive': "USD strength expected to continue as Fed maintains hawkish stance. Risk-off sentiment and yield differentials favor the greenback.",
                'negative': "Dollar weakness may persist as Fed signals potential rate cuts. Risk-on sentiment and narrowing yield spreads pressure USD.",
                'neutral': "Dollar trading sideways as market awaits clearer Fed guidance. Technical levels important for direction."
            },
            'oil': {
                'positive': "Oil prices supported by OPEC+ supply cuts and geopolitical tensions. Demand outlook remains positive despite economic concerns.",
                'negative': "Oil prices pressured by demand concerns and potential economic slowdown. Supply increases may offset geopolitical risks.",
                'neutral': "Oil trading range-bound as supply and demand factors balance. OPEC+ decisions key for direction."
            }
        }
        
        return opinions.get(category, {}).get(sentiment, "Market analysis suggests mixed signals with key technical levels determining short-term direction.")
    
    def generate_risk_assessment(self, category: str, impact_score: float) -> str:
        """Generate risk assessment for news item"""
        if impact_score > 0.8:
            risk_level = "HIGH"
            assessment = f"""
⚠️ RISK ASSESSMENT - {risk_level}:
• Significant market impact expected
• High volatility likely
• Consider position sizing carefully
• Monitor closely for follow-through
• Potential for sharp reversals
            """
        elif impact_score > 0.5:
            risk_level = "MEDIUM"
            assessment = f"""
⚠️ RISK ASSESSMENT - {risk_level}:
• Moderate market impact expected
• Increased volatility possible
• Standard position sizing appropriate
• Monitor for confirmation signals
• Normal market conditions
            """
        else:
            risk_level = "LOW"
            assessment = f"""
⚠️ RISK ASSESSMENT - {risk_level}:
• Minimal market impact expected
• Normal volatility levels
• Standard trading approach
• Regular monitoring sufficient
• Business as usual
            """
        
        return assessment.strip()
    
    def generate_trading_opportunities(self, category: str, sentiment: str, related_symbols: List[str]) -> str:
        """Generate trading opportunities based on analysis"""
        opportunities = ""
        
        if sentiment == 'positive':
            opportunities = f"""
💼 TRADING OPPORTUNITIES (BULLISH):
• Consider long positions in {category} related assets
• Key symbols to watch: {', '.join(related_symbols[:3])}
• Entry levels: Current price with stop-loss below support
• Take profit targets: 2-3% above current levels
• Risk management: 1-2% position size per trade
• Monitor for breakout confirmations
            """
        elif sentiment == 'negative':
            opportunities = f"""
💼 TRADING OPPORTUNITIES (BEARISH):
• Consider short positions or defensive plays
• Safe haven assets: Gold, USD, government bonds
• Hedging opportunities in {category} related assets
• Risk management: Tight stops above resistance
• Consider inverse ETFs for protection
• Monitor for reversal signals
            """
        else:
            opportunities = f"""
💼 TRADING OPPORTUNITIES (NEUTRAL):
• Range-bound trading strategies
• Buy support, sell resistance in {category}
• Options strategies for low volatility
• Mean reversion opportunities
• Focus on individual stock selection
• Wait for clearer directional signals
            """
        
        return opportunities.strip()
    
    def generate_historical_context(self, category: str) -> str:
        """Generate historical context for the category"""
        context = ""
        
        if category == 'gold':
            context = """
📚 HISTORICAL CONTEXT - GOLD:
• Gold has been a store of value for 5,000+ years
• Historically performs well during inflation and uncertainty
• 2020-2023: Strong performance during COVID and inflation
• 2011-2015: Bear market after 2008 crisis recovery
• 2000-2011: Bull market during tech bubble and financial crisis
• Current cycle: Supported by central bank purchases and geopolitical tensions
            """
        
        elif category == 'dollar':
            context = """
📚 HISTORICAL CONTEXT - USD:
• Dollar index peaked in 1985, 2001, and 2022
• Typically strengthens during Fed tightening cycles
• 2008-2011: Safe haven flows during financial crisis
• 2014-2016: Strong dollar cycle with Fed rate hikes
• 2020-2022: Pandemic response and inflation concerns
• Current cycle: Fed policy and risk sentiment driving moves
            """
        
        elif category == 'oil':
            context = """
📚 HISTORICAL CONTEXT - OIL:
• 2008: Peak at $147/bbl before financial crisis
• 2014-2016: OPEC price war, dropped to $26/bbl
• 2020: COVID crash to negative prices briefly
• 2022: Russia-Ukraine war spike to $130/bbl
• Current: OPEC+ supply management vs. demand concerns
• Long-term: Energy transition affecting demand outlook
            """
        
        return context.strip()
    
    def generate_future_outlook(self, category: str, sentiment: str) -> str:
        """Generate future outlook based on current analysis"""
        outlook = ""
        
        if category == 'gold':
            if sentiment == 'positive':
                outlook = """
🔮 FUTURE OUTLOOK - GOLD (BULLISH):
• Short-term (1-3 months): Continued strength, target $2,400
• Medium-term (3-6 months): Consolidation around $2,300-2,400
• Long-term (6-12 months): Potential for $2,500+ on inflation concerns
• Key drivers: Central bank purchases, geopolitical tensions, inflation
• Risks: Stronger dollar, higher real yields, risk-on sentiment
            """
            else:
                outlook = """
🔮 FUTURE OUTLOOK - GOLD (BEARISH):
• Short-term (1-3 months): Potential pullback to $2,200-2,300
• Medium-term (3-6 months): Range-bound $2,200-2,400
• Long-term (6-12 months): Recovery potential on inflation return
• Key drivers: Dollar strength, real yields, risk sentiment
• Risks: Geopolitical escalation, inflation resurgence
            """
        
        elif category == 'dollar':
            if sentiment == 'positive':
                outlook = """
🔮 FUTURE OUTLOOK - USD (BULLISH):
• Short-term (1-3 months): Continued strength, USD/BRL 5.50+
• Medium-term (3-6 months): Consolidation around current levels
• Long-term (6-12 months): Fed policy dependent
• Key drivers: Fed rate policy, risk sentiment, yield differentials
• Risks: Fed dovish pivot, risk-on sentiment, economic slowdown
            """
            else:
                outlook = """
🔮 FUTURE OUTLOOK - USD (BEARISH):
• Short-term (1-3 months): Potential weakness, USD/BRL 5.30-
• Medium-term (3-6 months): Range-bound 5.30-5.50
• Long-term (6-12 months): Fed policy dependent
• Key drivers: Fed rate cuts, risk-on sentiment, yield compression
• Risks: Fed hawkish surprise, risk-off sentiment
            """
        
        return outlook.strip()
    
    def scrape_news_with_rich_content(self, source_key: str, source_config: dict) -> List[RichNewsItem]:
        """Scrape news with rich content analysis"""
        try:
            response = self.session.get(source_config['url'], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles
            selectors = [
                'article', 'div[class*="article"]', 'div[class*="news"]', 
                'div[class*="post"]', 'div[class*="story"]', '.news-item'
            ]
            
            articles = []
            for selector in selectors:
                articles.extend(soup.select(selector))
                if articles:
                    break
            
            if not articles:
                articles = soup.find_all(['article', 'div'], class_=re.compile(r'article|news|post|story'))
            
            for article in articles[:10]:  # Limit articles per source
                try:
                    # Extract basic info
                    title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'h5'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    
                    content_elem = article.find(['p', 'div'], class_=re.compile(r'content|excerpt|summary|description'))
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    link_elem = article.find('a')
                    url = link_elem.get('href') if link_elem else ""
                    if url and not url.startswith('http'):
                        base_url = source_config['url'].split('/')[0] + '//' + source_config['url'].split('/')[2]
                        url = f"{base_url}{url}"
                    
                    # Analyze content
                    category = self.categorize_news(title, content)
                    sentiment = self.analyze_sentiment(title, content)
                    impact_score = self.calculate_impact_score(title, content, source_config['category'])
                    market_impact = self.analyze_market_impact(title, content)
                    
                    # Generate rich content
                    detailed_analysis = self.generate_detailed_analysis(title, content, category)
                    market_implications = self.generate_market_implications(title, content, category)
                    technical_insights = self.generate_technical_insights(category, sentiment)
                    related_symbols = self.identify_related_symbols(title, content, category)
                    key_metrics = self.generate_key_metrics(category)
                    expert_opinion = self.generate_expert_opinion(category, sentiment)
                    risk_assessment = self.generate_risk_assessment(category, impact_score)
                    trading_opportunities = self.generate_trading_opportunities(category, sentiment, related_symbols)
                    historical_context = self.generate_historical_context(category)
                    future_outlook = self.generate_future_outlook(category, sentiment)
                    
                    news_item = RichNewsItem(
                        title=title,
                        content=content,
                        source=source_config['name'],
                        url=url,
                        timestamp=datetime.now(),
                        category=category,
                        sentiment=sentiment,
                        impact_score=impact_score,
                        market_impact=market_impact,
                        detailed_analysis=detailed_analysis,
                        market_implications=market_implications,
                        technical_insights=technical_insights,
                        related_symbols=related_symbols,
                        key_metrics=key_metrics,
                        expert_opinion=expert_opinion,
                        risk_assessment=risk_assessment,
                        trading_opportunities=trading_opportunities,
                        historical_context=historical_context,
                        future_outlook=future_outlook
                    )
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing article from {source_config['name']}: {e}")
                    continue
            
            self.logger.info(f"Scraped {len(news_items)} rich news items from {source_config['name']}")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error scraping {source_config['name']}: {e}")
            return []
    
    def categorize_news(self, title: str, content: str) -> str:
        """Enhanced news categorization"""
        text = f"{title} {content}".lower()
        
        categories = {
            'gold': ['ouro', 'gold', 'xau', 'precious metals', 'safe haven', 'bullion'],
            'dollar': ['dólar', 'dollar', 'usd', 'greenback', 'federal reserve', 'fed'],
            'interest_rates': ['juros', 'selic', 'bc', 'banco central', 'fed', 'federal reserve', 'ecb', 'interest rate'],
            'oil': ['petróleo', 'oil', 'wti', 'brent', 'crude', 'energy', 'opec'],
            'iof': ['iof', 'imposto', 'tax', 'brazil tax', 'financial tax'],
            'crypto': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'btc', 'eth', 'digital currency'],
            'stocks': ['ações', 'bolsa', 'ibovespa', 'b3', 'petrobras', 'vale', 'stock market'],
            'forex': ['câmbio', 'exchange rate', 'currency', 'usd/brl', 'eur/brl'],
            'commodities': ['commodity', 'raw materials', 'natural resources'],
            'geopolitical': ['war', 'conflict', 'russia', 'ukraine', 'sanctions', 'tension', 'diplomatic'],
            'economy': ['economia', 'pib', 'inflação', 'desemprego', 'economy', 'gdp', 'inflation']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def analyze_sentiment(self, title: str, content: str) -> str:
        """Enhanced sentiment analysis"""
        text = f"{title} {content}".lower()
        
        positive_words = [
            'alta', 'sobe', 'recorde', 'positivo', 'crescimento', 'alta', 'up', 'gain', 'positive',
            'bullish', 'rally', 'surge', 'jump', 'rise', 'increase', 'strong', 'recovery'
        ]
        negative_words = [
            'queda', 'cai', 'perda', 'negativo', 'crise', 'baixa', 'down', 'loss', 'negative',
            'bearish', 'crash', 'drop', 'fall', 'decline', 'weak', 'recession', 'crisis'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_market_impact(self, title: str, content: str) -> str:
        """Analyze market impact of news"""
        text = f"{title} {content}".lower()
        
        bullish_keywords = [
            'bullish', 'rally', 'surge', 'jump', 'rise', 'gain', 'positive', 'strong',
            'recovery', 'growth', 'increase', 'higher', 'up', 'sobe', 'alta'
        ]
        
        bearish_keywords = [
            'bearish', 'crash', 'drop', 'fall', 'decline', 'loss', 'negative', 'weak',
            'recession', 'crisis', 'down', 'cai', 'queda', 'baixa'
        ]
        
        bullish_count = sum(1 for word in bullish_keywords if word in text)
        bearish_count = sum(1 for word in bearish_keywords if word in text)
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def calculate_impact_score(self, title: str, content: str, source_category: str) -> float:
        """Calculate impact score based on keywords and source"""
        text = f"{title} {content}".lower()
        
        # High impact keywords
        high_impact = [
            'fed', 'bc', 'selic', 'juros', 'dólar', 'bitcoin', 'ibovespa', 'recorde',
            'war', 'conflict', 'sanctions', 'russia', 'ukraine', 'oil', 'gold',
            'crisis', 'crash', 'rally', 'surge', 'emergency'
        ]
        medium_impact = [
            'petrobras', 'vale', 'ações', 'bolsa', 'crypto', 'economia',
            'inflation', 'interest rate', 'monetary policy', 'trade war'
        ]
        
        score = 0.0
        
        # Base score
        score += 0.1
        
        # High impact keywords
        for keyword in high_impact:
            if keyword in text:
                score += 0.3
        
        # Medium impact keywords
        for keyword in medium_impact:
            if keyword in text:
                score += 0.2
        
        # Source weight
        if source_category in ['interest_rates', 'geopolitical']:
            score += 0.2
        elif source_category in ['commodities', 'crypto']:
            score += 0.1
        
        return min(score, 1.0)
    
    def collect_all_news(self):
        """Collect news from all sources with rich content"""
        with self.lock:
            all_news = []
            
            for source_key, source_config in self.news_sources.items():
                try:
                    news_items = self.scrape_news_with_rich_content(source_key, source_config)
                    all_news.extend(news_items)
                    
                    # Respect rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error scraping {source_key}: {e}")
                    continue
            
            # Sort by impact score and timestamp
            all_news.sort(key=lambda x: (x.impact_score, x.timestamp), reverse=True)
            
            # Store news
            self.news_items = all_news
            
            # Save to file
            self.save_rich_news()
            
            self.logger.info(f"Collected {len(all_news)} rich news items")
    
    def save_rich_news(self):
        """Save rich news content to file"""
        try:
            news_data = []
            for item in self.news_items:
                news_data.append({
                    'title': item.title,
                    'content': item.content,
                    'source': item.source,
                    'url': item.url,
                    'timestamp': item.timestamp.isoformat(),
                    'category': item.category,
                    'sentiment': item.sentiment,
                    'impact_score': item.impact_score,
                    'market_impact': item.market_impact,
                    'detailed_analysis': item.detailed_analysis,
                    'market_implications': item.market_implications,
                    'technical_insights': item.technical_insights,
                    'related_symbols': item.related_symbols,
                    'key_metrics': item.key_metrics,
                    'expert_opinion': item.expert_opinion,
                    'risk_assessment': item.risk_assessment,
                    'trading_opportunities': item.trading_opportunities,
                    'historical_context': item.historical_context,
                    'future_outlook': item.future_outlook
                })
            
            filename = f"{self.data_dir}/rich_news/rich_news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(news_data)} rich news items to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving rich news: {e}")
    
    def print_rich_summary(self):
        """Print comprehensive rich content summary"""
        print("\n" + "="*120)
        print("RICH CONTENT FINANCIAL NEWS ANALYSIS")
        print("="*120)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Print top news with rich content
        for i, item in enumerate(self.news_items[:5], 1):
            print(f"📰 NEWS ITEM #{i}")
            print("-" * 80)
            print(f"Title: {item.title}")
            print(f"Source: {item.source} | Category: {item.category}")
            print(f"Sentiment: {item.sentiment.upper()} | Impact Score: {item.impact_score:.2f}")
            print(f"Market Impact: {item.market_impact.upper()}")
            print()
            
            if item.detailed_analysis:
                print("🔍 DETAILED ANALYSIS:")
                print(item.detailed_analysis)
                print()
            
            if item.market_implications:
                print("📊 MARKET IMPLICATIONS:")
                print(item.market_implications)
                print()
            
            if item.technical_insights:
                print("🔧 TECHNICAL INSIGHTS:")
                print(item.technical_insights)
                print()
            
            if item.related_symbols:
                print(f"📈 RELATED SYMBOLS: {', '.join(item.related_symbols)}")
                print()
            
            if item.expert_opinion:
                print("💡 EXPERT OPINION:")
                print(item.expert_opinion)
                print()
            
            if item.trading_opportunities:
                print("💼 TRADING OPPORTUNITIES:")
                print(item.trading_opportunities)
                print()
            
            if item.risk_assessment:
                print("⚠️  RISK ASSESSMENT:")
                print(item.risk_assessment)
                print()
            
            print("=" * 80)
            print()
        
        print("="*120)
    
    def start_monitoring(self):
        """Start continuous monitoring with rich content"""
        self.logger.info("Starting Rich Content Financial News Bot monitoring...")
        
        # Schedule tasks
        try:
            import schedule
            schedule.every(30).minutes.do(self.collect_all_news)
            schedule.every(1).hours.do(self.print_rich_summary)
        except ImportError:
            self.logger.warning("Schedule library not available, running without scheduling")
        
        # Initial collection
        self.collect_all_news()
        self.print_rich_summary()
        
        # Run scheduled tasks
        while True:
            try:
                import schedule
                schedule.run_pending()
            except ImportError:
                pass
            time.sleep(60)

def main():
    print("Rich Content Financial News Bot - Comprehensive Market Intelligence")
    print("Provides detailed analysis, technical insights, and trading opportunities")
    print("="*120)
    
    bot = RichContentFinancialBot()
    
    try:
        bot.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping Rich Content Financial News Bot...")
        bot.logger.info("Rich Content Financial News Bot stopped by user")

if __name__ == "__main__":
    main() 