#!/usr/bin/env python3
"""
Martingale Strategy Optimizer
Optimize martingale parameters for best performance
"""

import logging
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import optuna

from martingale_calculator import MartingaleCalculator, MartingaleConfig, MartingaleType, MartingaleBacktester

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Optimization result"""
    best_params: Dict[str, float]
    best_score: float
    optimization_time: float
    trials: int
    convergence_data: List[float]

class MartingaleOptimizer:
    """Optimize martingale strategy parameters"""
    
    def __init__(self, strategy_type: MartingaleType):
        self.strategy_type = strategy_type
        self.optimization_results = []
        
        logger.info(f"MartingaleOptimizer initialized for strategy: {strategy_type.value}")
    
    def optimize_parameters(self, historical_data: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio',
                          n_trials: int = 100) -> OptimizationResult:
        """Optimize martingale parameters using Optuna"""
        
        def objective(trial):
            # Suggest parameters
            initial_bet = trial.suggest_float('initial_bet', 10, 1000)
            multiplier = trial.suggest_float('multiplier', 1.5, 5.0)
            max_bet = trial.suggest_float('max_bet', 100, 5000)
            max_consecutive_losses = trial.suggest_int('max_consecutive_losses', 3, 10)
            
            # Create config
            config = MartingaleConfig(
                strategy_type=self.strategy_type,
                initial_bet=initial_bet,
                multiplier=multiplier,
                max_bet=max_bet,
                max_consecutive_losses=max_consecutive_losses
            )
            
            # Run backtest
            calculator = MartingaleCalculator(config)
            backtester = MartingaleBacktester(calculator)
            
            result = backtester.backtest_strategy(historical_data)
            
            if 'error' in result:
                return -1000.0
            
            # Return metric based on optimization type
            if optimization_metric == 'sharpe_ratio':
                return self._calculate_sharpe_ratio(result)
            elif optimization_metric == 'total_return':
                return result['total_return']
            elif optimization_metric == 'win_rate':
                return result['win_rate']
            else:
                return result.get('total_return', -1000.0)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_time=0.0,  # Would need to track time
            trials=n_trials,
            convergence_data=[]
        )
    
    def _calculate_sharpe_ratio(self, backtest_result: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio"""
        if 'equity_curve' not in backtest_result:
            return -1000.0
        
        equity_curve = backtest_result['equity_curve']
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return -1000.0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        return sharpe
    
    def grid_search_optimization(self, historical_data: pd.DataFrame,
                                 param_grid: Dict[str, List[float]]) -> OptimizationResult:
        """Perform grid search optimization"""
        
        best_score = -np.inf
        best_params = {}
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            
            config = MartingaleConfig(
                strategy_type=self.strategy_type,
                **params
            )
            
            # Run backtest
            calculator = MartingaleCalculator(config)
            backtester = MartingaleBacktester(calculator)
            
            result = backtester.backtest_strategy(historical_data)
            
            if 'error' not in result:
                score = result.get('total_return', -1000.0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i+1}/{len(param_combinations)} trials")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_time=0.0,
            trials=len(param_combinations),
            convergence_data=[]
        )
    
    def monte_carlo_optimization(self, historical_data: pd.DataFrame,
                                n_simulations: int = 1000,
                                confidence_level: float = 0.95) -> Dict[str, Any]:
        """Run Monte Carlo simulations to optimize strategy"""
        
        results = []
        
        for i in range(n_simulations):
            # Random parameter selection
            initial_bet = np.random.uniform(10, 500)
            multiplier = np.random.uniform(1.5, 4.0)
            max_bet = np.random.uniform(100, 3000)
            max_consecutive_losses = np.random.randint(3, 10)
            
            config = MartingaleConfig(
                strategy_type=self.strategy_type,
                initial_bet=initial_bet,
                multiplier=multiplier,
                max_bet=max_bet,
                max_consecutive_losses=max_consecutive_losses
            )
            
            calculator = MartingaleCalculator(config)
            backtester = MartingaleBacktester(calculator)
            
            result = backtester.backtest_strategy(historical_data)
            
            if 'error' not in result:
                results.append({
                    'params': config,
                    'total_return': result.get('total_return', 0),
                    'win_rate': result.get('win_rate', 0),
                    'max_drawdown': result.get('max_drawdown', 1.0)
                })
        
        # Calculate statistics
        returns = [r['total_return'] for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'confidence_interval': np.percentile(returns, 
                                                 [(1-confidence_level)/2 * 100, 
                                                  (1+confidence_level)/2 * 100]),
            'n_simulations': n_simulations,
            'best_result': max(results, key=lambda x: x['total_return'])
        }

class RiskAnalyzer:
    """Analyze risk for martingale strategies"""
    
    def __init__(self):
        self.risk_metrics = {}
        
    def calculate_value_at_risk(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(returns) == 0:
            return 0.0
        
        var = returns.quantile(1 - confidence_level)
        return abs(var)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        if len(returns) == 0:
            return 0.0
        
        threshold = returns.quantile(1 - confidence_level)
        tail_losses = returns[returns <= threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        return abs(tail_losses.mean())
    
    def calculate_kelly_criterion(self, win_probability: float, win_loss_ratio: float) -> float:
        """Calculate Kelly Criterion for optimal bet sizing"""
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Kelly fraction should be between 0 and 1
        return max(0.0, min(1.0, kelly))
    
    def analyze_portfolio_risk(self, backtest_result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze portfolio risk metrics"""
        if 'equity_curve' not in backtest_result:
            return {}
        
        equity_curve = pd.Series(backtest_result['equity_curve'])
        returns = equity_curve.pct_change().dropna()
        
        risk_metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': self.calculate_value_at_risk(returns, 0.95),
            'cvar_95': self.calculate_expected_shortfall(returns, 0.95),
            'max_drawdown': backtest_result.get('max_drawdown', 0.0)
        }
        
        return risk_metrics

def main():
    """Example usage"""
    # Create optimizer
    optimizer = MartingaleOptimizer(MartingaleType.CLASSIC)
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'ma_short': np.random.randn(100).cumsum() + 100,
        'ma_long': np.random.randn(100).cumsum() + 99
    }, index=dates)
    
    print("Running optimization...")
    
    # Run optimization
    result = optimizer.optimize_parameters(data, n_trials=50)
    
    print("\nOptimization Results:")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Best Parameters: {result.best_params}")

if __name__ == "__main__":
    main()

