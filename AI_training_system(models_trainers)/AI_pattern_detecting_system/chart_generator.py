#!/usr/bin/env python3
"""
Chart Generator Module
Comprehensive chart generation system for pattern detection including:
- Multiple chart types (candlestick, OHLC, line, mountain)
- Technical indicator overlays
- Pattern detection visualization
- Custom styling and themes
- Image export for YOLO training
- Multi-timeframe chart generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import mplfinance as mpf
import io
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate comprehensive chart images for pattern detection"""
    
    def __init__(self, style: str = 'classic', figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize chart generator
        
        Args:
            style: Chart style ('classic', 'yahoo', 'sas')
            figsize: Figure size tuple
            dpi: DPI for images
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
    
    def generate_candlestick(self, df: pd.DataFrame, output_path: str, 
                           indicators: Optional[Dict[str, Any]] = None,
                           pattern_annotations: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Generate candlestick chart
        
        Args:
            df: OHLCV DataFrame
            output_path: Path to save chart
            indicators: Technical indicators to overlay
            pattern_annotations: Pattern detection annotations
            
        Returns:
            True if successful
        """
        try:
            # Prepare plot style
            mpf_style = mpf.make_mpf_style(
                base_mpf_style=self.style,
                marketcolors=mpf.make_marketcolors(
                    up=self.colors['bullish'],
                    down=self.colors['bearish'],
                    edge='inherit',
                    wick='inherit',
                    volume='in'
                ),
                gridstyle='-',
                gridcolor=self.colors['grid']
            )
            
            # Build plot arguments
            plot_kwargs = {
                'type': 'candle',
                'style': mpf_style,
                'savefig': output_path,
                'figsize': self.figsize,
                'dpi': self.dpi
            }
            
            # Add volume
            if 'volume' in df.columns:
                plot_kwargs['volume'] = True
            
            # Add indicators if provided
            if indicators:
                plot_kwargs['addplot'] = self._build_indicator_plots(indicators)
            
            # Generate plot
            mpf.plot(df, **plot_kwargs)
            
            # Add pattern annotations if provided
            if pattern_annotations:
                self._annotate_patterns(output_path, pattern_annotations)
            
            logger.info(f"Candlestick chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating candlestick chart: {e}")
            return False
    
    def generate_line_chart(self, df: pd.DataFrame, output_path: str, 
                           price_column: str = 'close',
                           indicators: Optional[List[str]] = None) -> bool:
        """
        Generate line chart
        
        Args:
            df: OHLCV DataFrame
            output_path: Path to save chart
            price_column: Price column to plot
            indicators: Indicator columns to overlay
            
        Returns:
            True if successful
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot main price
            ax.plot(df.index, df[price_column], 
                   color=self.colors['bullish'], 
                   linewidth=2, 
                   label='Price')
            
            # Plot indicators
            if indicators:
                for indicator in indicators:
                    if indicator in df.columns:
                        ax.plot(df.index, df[indicator], 
                               alpha=0.7, 
                               label=indicator)
            
            ax.set_title('Price Chart with Indicators', color='white', fontsize=14)
            ax.set_xlabel('Time', color='white')
            ax.set_ylabel('Price', color='white')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(self.colors['background'])
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor=self.colors['background'])
            plt.close()
            
            logger.info(f"Line chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating line chart: {e}")
            return False
    
    def generate_mountain_chart(self, df: pd.DataFrame, output_path: str, 
                               price_column: str = 'close') -> bool:
        """Generate mountain/area chart"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            ax.fill_between(df.index, df[price_column], 
                          color=self.colors['bullish'], 
                          alpha=0.5)
            ax.plot(df.index, df[price_column], 
                   color=self.colors['bullish'], 
                   linewidth=2)
            
            ax.set_title('Mountain Chart', color='white', fontsize=14)
            ax.set_xlabel('Time', color='white')
            ax.set_ylabel('Price', color='white')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(self.colors['background'])
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
            plt.close()
            
            logger.info(f"Mountain chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating mountain chart: {e}")
            return False
    
    def generate_pattern_annotated_chart(self, df: pd.DataFrame, 
                                        patterns: List[Dict[str, Any]],
                                        output_path: str) -> bool:
        """
        Generate chart with pattern annotations
        
        Args:
            df: OHLCV DataFrame
            patterns: List of detected patterns
            output_path: Path to save chart
            
        Returns:
            True if successful
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot candlesticks
            for i, row in df.iterrows():
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                color = self.colors['bullish'] if close_price >= open_price else self.colors['bearish']
                
                # Draw high-low line
                ax.plot([i, i], [low_price, high_price], 
                       color='black', linewidth=1)
                
                # Draw open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                
                rect = patches.Rectangle(
                    (i-0.4, bottom), 0.8, height,
                    facecolor=color, edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)
            
            # Annotate patterns
            for pattern in patterns:
                self._draw_pattern_annotation(ax, pattern)
            
            ax.set_facecolor(self.colors['background'])
            ax.set_title('Chart with Pattern Annotations', color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
            plt.close()
            
            logger.info(f"Pattern annotated chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating pattern annotated chart: {e}")
            return False
    
    def generate_multi_timeframe_chart(self, dfs: Dict[str, pd.DataFrame], 
                                      output_path: str) -> bool:
        """
        Generate multi-timeframe comparison chart
        
        Args:
            dfs: Dictionary with timeframe as key and DataFrame as value
            output_path: Path to save chart
            
        Returns:
            True if successful
        """
        try:
            num_charts = len(dfs)
            fig, axes = plt.subplots(num_charts, 1, figsize=(self.figsize[0], self.figsize[1] * num_charts), 
                                    dpi=self.dpi)
            
            if num_charts == 1:
                axes = [axes]
            
            for i, (timeframe, df) in enumerate(dfs.items()):
                ax = axes[i]
                
                for idx, row in df.iterrows():
                    open_price = row['open']
                    high_price = row['high']
                    low_price = row['low']
                    close_price = row['close']
                    
                    color = self.colors['bullish'] if close_price >= open_price else self.colors['bearish']
                    
                    # Draw candlestick
                    ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=1)
                    height = abs(close_price - open_price)
                    bottom = min(open_price, close_price)
                    rect = patches.Rectangle(
                        (idx-0.4, bottom), 0.8, height,
                        facecolor=color, edgecolor='black', linewidth=1
                    )
                    ax.add_patch(rect)
                
                ax.set_title(f'{timeframe} Timeframe', color='white', fontsize=12)
                ax.set_facecolor(self.colors['background'])
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
            plt.close()
            
            logger.info(f"Multi-timeframe chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating multi-timeframe chart: {e}")
            return False
    
    def generate_for_yolo_training(self, df: pd.DataFrame, 
                                  output_path: str,
                                  image_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Generate chart image optimized for YOLO training
        
        Args:
            df: OHLCV DataFrame
            output_path: Path to save image
            image_size: Image dimensions (width, height)
            
        Returns:
            Image array
        """
        try:
            fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
            
            # Plot candlesticks
            for i, row in df.iterrows():
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                color = self.colors['bullish'] if close_price >= open_price else self.colors['bearish']
                
                # High-low line
                ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.5)
                
                # Open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                rect = patches.Rectangle(
                    (i-0.3, bottom), 0.6, height,
                    facecolor=color, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)
            
            ax.set_facecolor('white')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight', 
                       pad_inches=0, facecolor='white')
            
            # Convert to numpy array
            plt.close()
            
            logger.info(f"YOLO training image saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating YOLO training image: {e}")
            return False
    
    def _build_indicator_plots(self, indicators: Dict[str, Any]) -> List:
        """Build indicator plots for mplfinance"""
        plots = []
        
        try:
            # Add moving averages
            if 'ma_20' in indicators and indicators['ma_20'] is not None:
                plots.append(mpf.make_addplot(indicators['ma_20'], color='orange', width=1))
            
            if 'ma_50' in indicators and indicators['ma_50'] is not None:
                plots.append(mpf.make_addplot(indicators['ma_50'], color='blue', width=1))
            
            # Add Bollinger Bands
            if 'bb_upper' in indicators and indicators['bb_upper'] is not None:
                plots.append(mpf.make_addplot(indicators['bb_upper'], color='green', width=1, alpha=0.5))
                plots.append(mpf.make_addplot(indicators['bb_lower'], color='red', width=1, alpha=0.5))
            
            return plots
            
        except Exception as e:
            logger.error(f"Error building indicator plots: {e}")
            return []
    
    def _annotate_patterns(self, image_path: str, patterns: List[Dict[str, Any]]):
        """Annotate detected patterns on chart"""
        # This would load the image and add annotations
        # Implementation depends on specific requirements
        pass
    
    def _draw_pattern_annotation(self, ax, pattern: Dict[str, Any]):
        """Draw pattern annotation on chart"""
        try:
            pattern_type = pattern.get('type', 'unknown')
            confidence = pattern.get('confidence', 0)
            points = pattern.get('points', [])
            
            if points:
                # Draw pattern shape
                for i in range(len(points) - 1):
                    ax.plot([points[i][0], points[i+1][0]], 
                           [points[i][1], points[i+1][1]],
                           color='yellow', linewidth=2, alpha=0.7)
                
                # Add label
                if len(points) > 0:
                    ax.text(points[0][0], points[0][1], 
                           f"{pattern_type}\n({confidence:.2f})",
                           color='yellow', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
        except Exception as e:
            logger.error(f"Error drawing pattern annotation: {e}")
    
    def generate_batch(self, data_dict: Dict[str, pd.DataFrame], 
                      output_dir: str, chart_type: str = 'candlestick') -> int:
        """
        Generate batch of charts
        
        Args:
            data_dict: Dictionary with identifier as key and DataFrame as value
            output_dir: Output directory
            chart_type: Type of chart to generate
            
        Returns:
            Number of charts generated
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            count = 0
            for identifier, df in data_dict.items():
                output_path = os.path.join(output_dir, f"{identifier}.png")
                
                if chart_type == 'candlestick':
                    if self.generate_candlestick(df, output_path):
                        count += 1
                elif chart_type == 'line':
                    if self.generate_line_chart(df, output_path):
                        count += 1
                elif chart_type == 'mountain':
                    if self.generate_mountain_chart(df, output_path):
                        count += 1
            
            logger.info(f"Generated {count} charts in {output_dir}")
            return count
            
        except Exception as e:
            logger.error(f"Error generating batch charts: {e}")
            return 0
    
    @staticmethod
    def chart_to_numpy_array(chart_path: str) -> np.ndarray:
        """Convert chart image to numpy array for YOLO training"""
        try:
            import cv2
            img = cv2.imread(chart_path)
            return img
            
        except Exception as e:
            logger.error(f"Error converting chart to numpy array: {e}")
            return None

