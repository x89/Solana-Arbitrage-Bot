#!/usr/bin/env python3
"""
Indicator Visualizer Module
Comprehensive visualization for technical indicators and momentum analysis
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IndicatorVisualizer:
    """Visualize technical indicators and momentum analysis"""
    
    def __init__(self, style: str = 'dark_background', figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
            figsize: Figure size tuple
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(self.style)
        
        # Color schemes
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#ffa726',
            'signal': '#42a5f5',
            'background': '#1e1e1e'
        }
    
    def plot_price_with_indicators(self, df: pd.DataFrame, symbol: str, 
                                  indicators: List[str] = None) -> plt.Figure:
        """Plot price with technical indicators"""
        try:
            fig, axes = plt.subplots(4, 1, figsize=self.figsize, facecolor=self.colors['background'])
            fig.suptitle(f'{symbol} - Technical Analysis', fontsize=16, color='white')
            
            # Plot 1: Price with MA
            ax1 = axes[0]
            ax1.plot(df.index, df['close'], label='Close', color=self.colors['signal'], linewidth=2)
            
            if 'sma_20' in df.columns:
                ax1.plot(df.index, df['sma_20'], label='SMA 20', color='orange', alpha=0.7)
            if 'sma_50' in df.columns:
                ax1.plot(df.index, df['sma_50'], label='SMA 50', color='red', alpha=0.7)
            
            ax1.set_ylabel('Price', color='white')
            ax1.tick_params(colors='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Bollinger Bands
            if 'bb_upper' in df.columns:
                ax2 = axes[1]
                ax2.plot(df.index, df['close'], label='Close', color=self.colors['signal'])
                ax2.plot(df.index, df['bb_upper'], label='BB Upper', color='green', alpha=0.5)
                ax2.plot(df.index, df['bb_middle'], label='BB Middle', color='gray', alpha=0.5)
                ax2.plot(df.index, df['bb_lower'], label='BB Lower', color='red', alpha=0.5)
                ax2.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
                ax2.set_ylabel('Price & BB', color='white')
                ax2.tick_params(colors='white')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: RSI
            if 'rsi_14' in df.columns:
                ax3 = axes[2]
                ax3.plot(df.index, df['rsi_14'], label='RSI', color=self.colors['signal'], linewidth=2)
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
                ax3.set_ylabel('RSI', color='white')
                ax3.set_ylim(0, 100)
                ax3.tick_params(colors='white')
                ax3.legend(loc='upper left')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: MACD
            if 'macd' in df.columns:
                ax4 = axes[3]
                ax4.plot(df.index, df['macd'], label='MACD', color=self.colors['signal'])
                if 'macd_signal' in df.columns:
                    ax4.plot(df.index, df['macd_signal'], label='Signal', color='orange')
                if 'macd_histogram' in df.columns:
                    colors_hist = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
                    ax4.bar(df.index, df['macd_histogram'], label='Histogram', color=colors_hist, alpha=0.6)
                ax4.set_ylabel('MACD', color='white')
                ax4.tick_params(colors='white')
                ax4.legend(loc='upper left')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting indicators: {e}")
            return None
    
    def plot_momentum_analysis(self, df: pd.DataFrame, momentum_data: Dict[str, Any], 
                               symbol: str) -> plt.Figure:
        """Plot momentum analysis"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=self.figsize, facecolor=self.colors['background'])
            fig.suptitle(f'{symbol} - Momentum Analysis', fontsize=16, color='white')
            
            # Plot 1: Momentum Score
            ax1 = axes[0]
            momentum_score = momentum_data.get('momentum_score', 0)
            colors_momentum = ['red' if momentum_score < 0 else 'green' if momentum_score > 0 else 'gray']
            ax1.barh([0], [momentum_score], color=colors_momentum, alpha=0.7)
            ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            ax1.set_xlim(-1, 1)
            ax1.set_xlabel('Momentum Score', color='white')
            ax1.set_title(f'Overall Momentum: {momentum_score:.3f}', color='white')
            ax1.tick_params(colors='white')
            
            # Plot 2: Trend Direction
            ax2 = axes[1]
            trend = momentum_data.get('trend_direction', 'neutral')
            strength = momentum_data.get('momentum_strength', 'weak')
            trend_colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
            ax2.bar([0], [1], color=trend_colors.get(trend, 'gray'), alpha=0.7)
            ax2.set_title(f'Trend: {trend.upper()} | Strength: {strength.upper()}', color='white')
            ax2.set_ylim(0, 1.2)
            ax2.tick_params(colors='white')
            ax2.set_xticks([])
            
            # Plot 3: Indicator Signals
            ax3 = axes[2]
            indicators = momentum_data.get('indicators', {})
            
            buy_signals = sum(1 for ind in indicators.values() if ind.get('signal') == 'buy')
            sell_signals = sum(1 for ind in indicators.values() if ind.get('signal') == 'sell')
            hold_signals = sum(1 for ind in indicators.values() if ind.get('signal') == 'hold')
            
            categories = ['Buy', 'Sell', 'Hold']
            values = [buy_signals, sell_signals, hold_signals]
            colors_signals = [self.colors['bullish'], self.colors['bearish'], self.colors['neutral']]
            
            ax3.bar(categories, values, color=colors_signals, alpha=0.7)
            ax3.set_ylabel('Signal Count', color='white')
            ax3.set_title('Indicator Signals Distribution', color='white')
            ax3.tick_params(colors='white')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting momentum analysis: {e}")
            return None
    
    def plot_correlation_heatmap(self, correlation_matrix: Dict[str, Dict[str, float]], 
                                title: str = "Indicator Correlation") -> plt.Figure:
        """Plot correlation heatmap"""
        try:
            # Convert to DataFrame
            df_corr = pd.DataFrame(correlation_matrix)
            
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.colors['background'])
            
            # Create heatmap
            sns.heatmap(df_corr, annot=True, cmap='RdYlGn', center=0, 
                       fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            
            ax.set_title(title, color='white', fontsize=14)
            ax.tick_params(colors='white', labelsize=8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting correlation heatmap: {e}")
            return None
    
    def plot_indicator_comparison(self, df: pd.DataFrame, indicators: List[str], 
                                  title: str = "Indicator Comparison") -> plt.Figure:
        """Plot multiple indicators for comparison"""
        try:
            fig, axes = plt.subplots(len(indicators), 1, figsize=self.figsize, 
                                    facecolor=self.colors['background'])
            fig.suptitle(title, fontsize=16, color='white')
            
            if len(indicators) == 1:
                axes = [axes]
            
            for i, indicator in enumerate(indicators):
                if indicator in df.columns:
                    ax = axes[i]
                    ax.plot(df.index, df[indicator], label=indicator, color=self.colors['signal'])
                    ax.set_ylabel(indicator, color='white')
                    ax.tick_params(colors='white')
                    ax.legend(loc='upper left')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting indicator comparison: {e}")
            return None
    
    def plot_timeframe_comparison(self, momentum_data: Dict[str, Dict[str, Any]], 
                                  symbol: str) -> plt.Figure:
        """Plot momentum comparison across timeframes"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), facecolor=self.colors['background'])
            fig.suptitle(f'{symbol} - Multi-Timeframe Momentum Analysis', fontsize=16, color='white')
            
            # Extract data
            timeframes = list(momentum_data.keys())
            scores = [momentum_data[tf].get('momentum_score', 0) for tf in timeframes]
            trends = [momentum_data[tf].get('trend_direction', 'neutral') for tf in timeframes]
            
            # Plot 1: Momentum Scores
            colors_scores = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
            ax1.bar(timeframes, scores, color=colors_scores, alpha=0.7)
            ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Momentum Score', color='white')
            ax1.set_title('Momentum Scores Across Timeframes', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Trend Directions
            trend_colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
            colors_trends = [trend_colors.get(t, 'gray') for t in trends]
            ax2.bar(timeframes, [1] * len(timeframes), color=colors_trends, alpha=0.7)
            ax2.set_ylabel('Trend', color='white')
            ax2.set_title('Trend Direction Across Timeframes', color='white')
            ax2.tick_params(colors='white')
            ax2.set_xticklabels(timeframes)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting timeframe comparison: {e}")
            return None
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 150) -> bool:
        """Save plot to file"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=self.colors['background'])
            logger.info(f"Plot saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return False
    
    def show_plot(self, fig: plt.Figure):
        """Display plot"""
        try:
            plt.show()
        except Exception as e:
            logger.error(f"Error showing plot: {e}")
    
    @staticmethod
    def create_summary_chart(momentum_data: Dict[str, Any]) -> plt.Figure:
        """Create summary chart of momentum data"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1e1e1e')
            
            symbol = momentum_data.get('symbol', 'Unknown')
            momentum_score = momentum_data.get('momentum_score', 0)
            trend = momentum_data.get('trend_direction', 'neutral')
            strength = momentum_data.get('momentum_strength', 'weak')
            recommendations = momentum_data.get('recommendations', [])
            
            # Create text summary
            summary_text = f"""
            Momentum Analysis Summary
            ========================
            
            Symbol: {symbol}
            Momentum Score: {momentum_score:.3f}
            Trend Direction: {trend.upper()}
            Momentum Strength: {strength.upper()}
            
            Recommendations:
            """
            
            for i, rec in enumerate(recommendations[:5], 1):
                summary_text += f"\n{i}. {rec}"
            
            ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='center', color='white',
                   family='monospace')
            
            ax.axis('off')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary chart: {e}")
            return None

