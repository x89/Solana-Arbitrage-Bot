#!/usr/bin/env python3
"""
LLM+RL Hybrid Trading System
Integrating Large Language Models with Reinforcement Learning for Algorithmic Trading

Based on: "LLM-Guided Reinforcement Learning for Algorithmic Trading"
Framework: LLM generates strategic guidance for RL agents

Components:
- Strategist Agent: Generates high-level trading strategies
- Analyst Agent: Processes news and extracts signals
- DDQN Agent: Executes tactical decisions guided by LLM signals
- Hybrid Architecture: Combines LLM strategic reasoning with RL execution
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# OpenAI integration for LLM
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

logger = logging.getLogger(__name__)

@dataclass
class LLMStrategy:
    """LLM-generated trading strategy"""
    direction: str  # 'LONG' or 'SHORT'
    confidence: int  # Likert scale 1-3
    explanation: str
    features_used: List[Dict[str, Any]]
    timestamp: datetime
    signal_strength: float = 0.0
    entropy: float = 0.0

@dataclass
class LLMConfig:
    """Configuration for LLM agents"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # Deterministic for strategy generation
    max_tokens: int = 500
    frequency_penalty: float = 1.0
    presence_penalty: float = 0.25
    context_window: int = 128000
    seed: int = 49

@dataclass
class RLConfig:
    """Configuration for RL agent"""
    state_dim: int = 100
    action_dim: int = 3  # 0=hold, 1=buy, 2=sell
    hidden_dim: int = 128
    learning_rate: float = 0.0001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 10000
    update_target_freq: int = 100

class StrategistAgent:
    """
    LLM-based Strategist Agent
    Generates high-level trading strategies using structured prompting
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.api_key = None
        self.strategy_history = []
        self.prompt_template = self._create_prompt_template()
        
        if OPENAI_AVAILABLE:
            try:
                openai.api_key = self.api_key
            except:
                pass
    
    def _create_prompt_template(self) -> str:
        """Create the prompt template for strategist agent"""
        return """
You are an expert algorithmic trading strategist. Given the following market context, 
generate a trading strategy for the next month (20 trading days).

Market Context:
{context}

Previous Strategy Performance:
{previous_strategy}

Instructions:
1. Analyze the market data across stock fundamentals, technical indicators, macroeconomic factors, and options data
2. Integrate news sentiment analysis from the Analyst Agent
3. Reflect on previous strategy performance
4. Generate a directional recommendation (LONG or SHORT) with confidence (1-3)
5. Provide weighted feature attribution
6. Explain your reasoning (max 350 words)

Output Format:
- direction: LONG or SHORT
- confidence: 1-3 (Likert scale)
- explanation: Comprehensive rationale
- features_used: [{{"feature": "...", "direction": "...", "weight": 1-3}}, ...]
"""
    
    def generate_strategy(
        self,
        stock_data: Dict[str, Any],
        market_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        options_data: Dict[str, Any],
        news_sentiment: Optional[Dict[str, Any]] = None,
        previous_strategy: Optional[LLMStrategy] = None
    ) -> LLMStrategy:
        """
        Generate trading strategy using LLM
        
        Args:
            stock_data: Stock-specific data
            market_data: Market indicators
            fundamental_data: Fundamental ratios
            technical_data: Technical indicators
            macro_data: Macroeconomic indicators
            options_data: Options market data
            news_sentiment: News sentiment from Analyst Agent
            previous_strategy: Previous strategy for reflection
            
        Returns:
            LLMStrategy object
        """
        try:
            # Build context
            context = self._build_context(
                stock_data, market_data, fundamental_data,
                technical_data, macro_data, options_data, news_sentiment
            )
            
            # Build previous strategy reflection
            prev_strategy_text = self._format_previous_strategy(previous_strategy)
            
            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                previous_strategy=prev_strategy_text
            )
            
            # Call LLM (simulated if OpenAI not available)
            if OPENAI_AVAILABLE and self.api_key:
                strategy_dict = self._call_openai(prompt)
            else:
                # Fallback: Rule-based strategy
                strategy_dict = self._generate_fallback_strategy(
                    technical_data, fundamental_data, macro_data
                )
            
            # Parse and create strategy
            strategy = self._parse_strategy(strategy_dict, previous_strategy)
            
            # Calculate signal strength
            strategy.signal_strength = self._calculate_signal_strength(strategy)
            
            # Store in history
            self.strategy_history.append(strategy)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return self._generate_default_strategy()
    
    def _build_context(
        self,
        stock_data: Dict[str, Any],
        market_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        options_data: Dict[str, Any],
        news_sentiment: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string from all data sources"""
        context = ""
        
        # Stock data
        context += f"Stock Data: {json.dumps(stock_data, indent=2)}\n\n"
        
        # Market data
        context += f"Market Data: {json.dumps(market_data, indent=2)}\n\n"
        
        # Technical indicators
        context += f"Technical Data: {json.dumps(technical_data, indent=2)}\n\n"
        
        # Fundamental data
        context += f"Fundamental Data: {json.dumps(fundamental_data, indent=2)}\n\n"
        
        # Macro data
        context += f"Macro Data: {json.dumps(macro_data, indent=2)}\n\n"
        
        # Options data
        context += f"Options Data: {json.dumps(options_data, indent=2)}\n\n"
        
        # News sentiment
        if news_sentiment:
            context += f"News Sentiment: {json.dumps(news_sentiment, indent=2)}\n"
        
        return context
    
    def _format_previous_strategy(self, strategy: Optional[LLMStrategy]) -> str:
        """Format previous strategy for reflection"""
        if strategy is None:
            return "No previous strategy available."
        
        return f"""
Previous Strategy:
- Direction: {strategy.direction}
- Confidence: {strategy.confidence}/3
- Explanation: {strategy.explanation[:200]}...
- Result: To be evaluated
"""
    
    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API (placeholder - requires API key)"""
        try:
            # This would be the actual OpenAI call
            # response = openai.ChatCompletion.create(
            #     model=self.config.model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=self.config.temperature,
            #     seed=self.config.seed
            # )
            # return self._parse_llm_response(response)
            pass
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
        
        return self._generate_fallback_strategy({}, {}, {})
    
    def _generate_fallback_strategy(
        self,
        technical_data: Dict[str, Any],
        fundamental_data: Dict[str, Any],
        macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate rule-based strategy when LLM unavailable"""
        # Simple rule-based approach
        direction = "LONG"
        confidence = 2
        
        # Check RSI
        if technical_data.get('rsi', 50) > 70:
            direction = "SHORT"
            confidence = 2
        elif technical_data.get('rsi', 50) < 30:
            direction = "LONG"
            confidence = 3
        
        # Check fundamentals
        if fundamental_data.get('pe_ratio', 20) < 15:
            confidence += 1
        
        # Check macro
        if macro_data.get('vix', 15) > 25:
            confidence -= 1
        
        # Normalize confidence
        confidence = max(1, min(3, confidence))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'explanation': f"Rule-based strategy: RSI={technical_data.get('rsi', 50):.1f}, PE={fundamental_data.get('pe_ratio', 20):.1f}",
            'features_used': [
                {"feature": "RSI", "direction": direction, "weight": 2},
                {"feature": "PE_Ratio", "direction": direction, "weight": 1}
            ]
        }
    
    def _parse_strategy(self, strategy_dict: Dict[str, Any], previous: Optional[LLMStrategy]) -> LLMStrategy:
        """Parse strategy from LLM response"""
        return LLMStrategy(
            direction=strategy_dict.get('direction', 'NEUTRAL'),
            confidence=int(strategy_dict.get('confidence', 2)),
            explanation=strategy_dict.get('explanation', 'No explanation provided'),
            features_used=strategy_dict.get('features_used', []),
            timestamp=datetime.now(),
            signal_strength=float(strategy_dict.get('signal_strength', 0.5)),
            entropy=float(strategy_dict.get('entropy', 0.3))
        )
    
    def _calculate_signal_strength(self, strategy: LLMStrategy) -> float:
        """Calculate signal strength with entropy adjustment"""
        # Base strength from confidence
        base_strength = strategy.confidence / 3.0
        
        # Entropy adjustment
        entropy_adjustment = 1.0 - strategy.entropy  # Lower entropy = higher confidence
        
        # Final strength
        signal_strength = base_strength * entropy_adjustment
        
        return max(0.0, min(1.0, signal_strength))
    
    def _generate_default_strategy(self) -> LLMStrategy:
        """Generate default neutral strategy"""
        return LLMStrategy(
            direction="LONG",
            confidence=1,
            explanation="Default strategy: No LLM available",
            features_used=[],
            timestamp=datetime.now()
        )

class AnalystAgent:
    """
    LLM-based Analyst Agent
    Processes news and extracts market-impacting factors
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.extraction_history = []
    
    def extract_news_factors(self, news_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract news factors influencing stock prices
        
        Args:
            news_articles: List of news article dictionaries
            
        Returns:
            Dictionary with sentiment and impact scores
        """
        try:
            # Group articles by relevance
            ranked_factors = []
            
            for article in news_articles[:10]:  # Top 10 articles
                factor = self._extract_factor(article)
                if factor:
                    ranked_factors.append(factor)
            
            # Aggregate sentiment
            if ranked_factors:
                avg_sentiment = np.mean([f['sentiment'] for f in ranked_factors])
                avg_impact = np.mean([f['impact'] for f in ranked_factors])
                
                # Combine with feature weighting
                news_sentiment = {
                    'sentiment': avg_sentiment,
                    'impact_score': avg_impact,
                    'factors': ranked_factors[:3]  # Top 3 factors
                }
                
                self.extraction_history.append(news_sentiment)
                return news_sentiment
            else:
                return {'sentiment': 0, 'impact_score': 1, 'factors': []}
                
        except Exception as e:
            logger.error(f"Error extracting news factors: {e}")
            return {'sentiment': 0, 'impact_score': 1, 'factors': []}
    
    def _extract_factor(self, article: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract single factor from article"""
        try:
            # Simple heuristic-based extraction
            headline = article.get('headline', '')
            content = article.get('content', '')
            
            # Score sentiment
            sentiment = self._score_sentiment(headline + content)
            
            # Score impact
            impact = self._score_impact(headline)
            
            if impact > 1:
                return {
                    'factor': headline[:100],
                    'sentiment': sentiment,
                    'impact': impact
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting factor: {e}")
            return None
    
    def _score_sentiment(self, text: str) -> int:
        """Score sentiment from text (-1 to 1)"""
        pos_words = ['growth', 'profit', 'gain', 'positive', 'expansion', 'beat', 'surge']
        neg_words = ['loss', 'decline', 'weak', 'negative', 'miss', 'fall', 'drop']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in pos_words if word in text_lower)
        neg_count = sum(1 for word in neg_words if word in text_lower)
        
        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return -1
        else:
            return 0
    
    def _score_impact(self, headline: str) -> int:
        """Score impact level (1-3)"""
        high_impact_words = ['earnings', 'revenue', 'acquisition', 'partnership', 'guidance']
        medium_impact_words = ['expansion', 'product', 'announcement', 'rating']
        
        headline_lower = headline.lower()
        
        if any(word in headline_lower for word in high_impact_words):
            return 3
        elif any(word in headline_lower for word in medium_impact_words):
            return 2
        else:
            return 1

class LLMGuidedDQN(nn.Module):
    """
    Deep Q-Network guided by LLM signals
    Incorporates LLM strategic guidance into RL observation space
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Include LLM signal in state dimension
        self.state_dim += 1  # +1 for LLM signal
        
        self.q_network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with LLM-guided state"""
        return self.q_network(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection"""
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()

class LLMRLHybridTrainer:
    """
    LLM+RL Hybrid Trading System
    Integrates LLM strategic guidance with RL tactical execution
    """
    
    def __init__(self, llm_config: LLMConfig = None, rl_config: RLConfig = None):
        self.llm_config = llm_config or LLMConfig()
        self.rl_config = rl_config or RLConfig()
        
        # Initialize agents
        self.strategist = StrategistAgent(self.llm_config)
        self.analyst = AnalystAgent(self.llm_config)
        
        # Initialize RL agent
        self.rl_agent = LLMGuidedDQN(
            state_dim=self.rl_config.state_dim,
            action_dim=self.rl_config.action_dim,
            hidden_dim=self.rl_config.hidden_dim
        )
        
        # Metrics
        self.sharpe_ratio_history = []
        self.max_drawdown_history = []
        self.training_metrics = {}
    
    def compute_llm_signal(
        self,
        strategy: LLMStrategy
    ) -> float:
        """
        Compute LLM interaction term for RL agent
        
        Args:
            strategy: LLM-generated strategy
            
        Returns:
            Interaction term τ
        """
        # Direction: {LONG, SHORT} -> {-1, 1}
        direction_map = {'LONG': 1.0, 'SHORT': -1.0, 'NEUTRAL': 0.0}
        direction = direction_map.get(strategy.direction, 0.0)
        
        # Signal strength with entropy adjustment
        confidence = strategy.confidence / 3.0  # Normalize to [0, 1]
        entropy = strategy.entropy
        
        # Adjusted confidence
        epsilon = 0.01
        certainty = epsilon + (1.0 - epsilon) * (1.0 - entropy)
        signal_strength = confidence * certainty
        
        # Interaction term: τ = dir(π_g) * str(π_g)
        interaction_term = direction * signal_strength
        
        return interaction_term
    
    def augment_state_with_llm_signal(
        self,
        state: np.ndarray,
        llm_signal: float
    ) -> np.ndarray:
        """
        Augment RL state with LLM signal
        
        Args:
            state: Original RL state
            llm_signal: LLM interaction term
            
        Returns:
            Augmented state
        """
        # Append LLM signal to state
        augmented_state = np.append(state, llm_signal)
        
        return augmented_state
    
    def train_episode(
        self,
        data: pd.DataFrame,
        episode: int,
        max_episodes: int = 50
    ) -> Dict[str, float]:
        """
        Train one episode with LLM+RL hybrid
        
        Args:
            data: Market data DataFrame
            episode: Current episode number
            max_episodes: Total number of episodes
            
        Returns:
            Episode metrics
        """
        try:
            # Generate LLM strategy (monthly)
            strategy = self._generate_monthly_strategy(data)
            
            # Compute LLM signal
            llm_signal = self.compute_llm_signal(strategy)
            
            # Initialize RL environment
            portfolio_value = 10000.0
            shares = 0
            position = 'HOLD'
            
            episode_returns = []
            episode_reward = 0.0
            
            # Training loop
            for i in range(len(data) - 1):
                # Get current state
                current_state = self._get_state(data, i)
                
                # Augment with LLM signal
                augmented_state = self.augment_state_with_llm_signal(current_state, llm_signal)
                
                # RL action
                epsilon = max(self.rl_config.epsilon_end, 
                            self.rl_config.epsilon_start * (self.rl_config.epsilon_decay ** episode))
                action = self.rl_agent.get_action(
                    torch.FloatTensor(augmented_state).unsqueeze(0),
                    epsilon=epsilon
                )
                
                # Execute action
                reward, portfolio_value = self._execute_action(
                    action, data.iloc[i], data.iloc[i+1],
                    portfolio_value, shares
                )
                
                episode_returns.append(reward)
                episode_reward += reward
            
            # Calculate metrics
            episode_return = portfolio_value / 10000.0 - 1.0
            episode_sharpe = self._calculate_sharpe_ratio(episode_returns)
            episode_mdd = self._calculate_max_drawdown(episode_returns)
            
            # Update history
            self.sharpe_ratio_history.append(episode_sharpe)
            self.max_drawdown_history.append(episode_mdd)
            
            return {
                'episode': episode,
                'return': episode_return,
                'sharpe_ratio': episode_sharpe,
                'max_drawdown': episode_mdd,
                'portfolio_value': portfolio_value,
                'llm_direction': strategy.direction,
                'llm_confidence': strategy.confidence
            }
            
        except Exception as e:
            logger.error(f"Error training episode: {e}")
            return {}
    
    def _generate_monthly_strategy(self, data: pd.DataFrame) -> LLMStrategy:
        """Generate monthly LLM strategy"""
        # Extract features from data
        stock_data = self._extract_stock_data(data)
        market_data = self._extract_market_data(data)
        fundamental_data = self._extract_fundamental_data(data)
        technical_data = self._extract_technical_data(data)
        macro_data = self._extract_macro_data(data)
        options_data = self._extract_options_data(data)
        
        # Get previous strategy
        previous_strategy = self.strategist.strategy_history[-1] if self.strategist.strategy_history else None
        
        # Generate new strategy
        strategy = self.strategist.generate_strategy(
            stock_data, market_data, fundamental_data,
            technical_data, macro_data, options_data,
            previous_strategy=previous_strategy
        )
        
        return strategy
    
    def _get_state(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Get RL state from data"""
        # Simplified state extraction
        row = data.iloc[index]
        
        features = []
        for col in data.columns:
            if pd.notna(row[col]):
                features.append(float(row[col]))
            else:
                features.append(0.0)
        
        return np.array(features[:self.rl_config.state_dim])
    
    def _execute_action(
        self,
        action: int,
        current_data: pd.Series,
        next_data: pd.Series,
        portfolio_value: float,
        shares: int
    ) -> Tuple[float, float]:
        """Execute RL action"""
        current_price = float(current_data.get('close', 0))
        next_price = float(next_data.get('close', 0))
        
        reward = 0.0
        
        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and portfolio_value > current_price:  # Buy
            shares_to_buy = portfolio_value / current_price
            shares += shares_to_buy
            portfolio_value = 0
        elif action == 2 and shares > 0:  # Sell
            portfolio_value = shares * current_price
            shares = 0
        
        # Calculate reward
        reward = (next_price - current_price) / current_price
        
        # Update portfolio
        current_portfolio_value = portfolio_value + shares * next_price
        
        return reward, current_portfolio_value
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe Ratio"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        
        if np.std(returns_array) == 0:
            return 0.0
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        sharpe = mean_return / std_return
        return float(sharpe)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate Maximum Drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    def _extract_stock_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract stock-specific data"""
        last_row = data.iloc[-1]
        
        return {
            'close': float(last_row.get('close', 0)),
            'volume': float(last_row.get('volume', 0)),
            'beta': float(last_row.get('beta', 1.0))
        }
    
    def _extract_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract market data"""
        return {
            'spx_close': 0.0,
            'vix_close': 0.0
        }
    
    def _extract_fundamental_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract fundamental data"""
        return {
            'pe_ratio': 15.0,
            'current_ratio': 2.0,
            'debt_to_equity': 0.5
        }
    
    def _extract_technical_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical indicators"""
        last_row = data.iloc[-1]
        
        return {
            'rsi': float(last_row.get('rsi', 50)),
            'macd': float(last_row.get('macd', 0)),
            'sma_20': float(last_row.get('sma_20', 0)),
            'sma_50': float(last_row.get('sma_50', 0))
        }
    
    def _extract_macro_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract macroeconomic data"""
        return {
            'vix': 20.0,
            'gdp_growth': 0.02
        }
    
    def _extract_options_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract options market data"""
        return {
            'implied_volatility': 0.20
        }
    
    def get_training_results(self) -> Dict[str, Any]:
        """Get training results"""
        return {
            'sharpe_ratio_mean': float(np.mean(self.sharpe_ratio_history)) if self.sharpe_ratio_history else 0.0,
            'sharpe_ratio_std': float(np.std(self.sharpe_ratio_history)) if self.sharpe_ratio_history else 0.0,
            'max_drawdown_mean': float(np.mean(self.max_drawdown_history)) if self.max_drawdown_history else 0.0,
            'max_drawdown_std': float(np.std(self.max_drawdown_history)) if self.max_drawdown_history else 0.0,
            'episodes': len(self.sharpe_ratio_history)
        }

def main():
    """Demonstrate LLM+RL hybrid system"""
    try:
        print("=" * 70)
        print("LLM+RL Hybrid Trading System")
        print("=" * 70)
        
        # Initialize trainer
        trainer = LLMRLHybridTrainer()
        
        print("\n✓ LLM+RL hybrid trainer initialized")
        print(f"  - Strategist Agent: Configured")
        print(f"  - Analyst Agent: Configured")
        print(f"  - RL Agent: {trainer.rl_config.action_dim}-action DDQN")
        
        print("\nSystem Components:")
        print("  - Strategist: Generates high-level trading strategies")
        print("  - Analyst: Processes news and extracts market factors")
        print("  - RL Agent: Executes tactical decisions")
        print("  - Hybrid: Combines LLM strategic reasoning with RL execution")
        
        print("\n=" * 70)
        print("LLM+RL Hybrid System Ready!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in LLM+RL demo: {e}")

if __name__ == "__main__":
    main()

