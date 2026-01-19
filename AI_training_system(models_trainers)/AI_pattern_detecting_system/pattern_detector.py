#!/usr/bin/env python3
"""
AI Pattern Detecting System
Advanced pattern detection system using:
- YOLO-based chart pattern recognition
- Traditional technical pattern detection
- Machine learning pattern classification
- Real-time pattern monitoring
- Pattern confidence scoring
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Using traditional pattern detection only.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatternDetection:
    """Pattern detection result"""
    pattern_type: str
    confidence: float
    timestamp: datetime
    symbol: str
    timeframe: str
    price_level: float
    pattern_points: List[Tuple[float, float]]  # (price, time) coordinates
    pattern_metadata: Dict[str, Any]
    detection_method: str  # 'yolo', 'traditional', 'ml'

@dataclass
class ChartPattern:
    """Chart pattern structure"""
    name: str
    description: str
    bullish_bearish: str  # 'bullish', 'bearish', 'neutral'
    reliability: float  # 0-1
    target_ratio: float  # Expected move ratio
    stop_loss_ratio: float  # Stop loss ratio

class TraditionalPatternDetector:
    """Traditional technical pattern detection"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.min_pattern_length = 10
        self.max_pattern_length = 100
    
    def _initialize_patterns(self) -> Dict[str, ChartPattern]:
        """Initialize pattern definitions"""
        return {
            'head_and_shoulders': ChartPattern(
                name='Head and Shoulders',
                description='Reversal pattern with three peaks',
                bullish_bearish='bearish',
                reliability=0.8,
                target_ratio=1.0,
                stop_loss_ratio=0.05
            ),
            'inverse_head_and_shoulders': ChartPattern(
                name='Inverse Head and Shoulders',
                description='Reversal pattern with three troughs',
                bullish_bearish='bullish',
                reliability=0.8,
                target_ratio=1.0,
                stop_loss_ratio=0.05
            ),
            'double_top': ChartPattern(
                name='Double Top',
                description='Two peaks at similar levels',
                bullish_bearish='bearish',
                reliability=0.7,
                target_ratio=0.8,
                stop_loss_ratio=0.03
            ),
            'double_bottom': ChartPattern(
                name='Double Bottom',
                description='Two troughs at similar levels',
                bullish_bearish='bullish',
                reliability=0.7,
                target_ratio=0.8,
                stop_loss_ratio=0.03
            ),
            'triangle_ascending': ChartPattern(
                name='Ascending Triangle',
                description='Horizontal resistance, rising support',
                bullish_bearish='bullish',
                reliability=0.6,
                target_ratio=0.6,
                stop_loss_ratio=0.02
            ),
            'triangle_descending': ChartPattern(
                name='Descending Triangle',
                description='Horizontal support, falling resistance',
                bullish_bearish='bearish',
                reliability=0.6,
                target_ratio=0.6,
                stop_loss_ratio=0.02
            ),
            'triangle_symmetrical': ChartPattern(
                name='Symmetrical Triangle',
                description='Converging support and resistance',
                bullish_bearish='neutral',
                reliability=0.5,
                target_ratio=0.5,
                stop_loss_ratio=0.02
            ),
            'flag_bullish': ChartPattern(
                name='Bull Flag',
                description='Brief consolidation after strong move up',
                bullish_bearish='bullish',
                reliability=0.7,
                target_ratio=0.8,
                stop_loss_ratio=0.03
            ),
            'flag_bearish': ChartPattern(
                name='Bear Flag',
                description='Brief consolidation after strong move down',
                bullish_bearish='bearish',
                reliability=0.7,
                target_ratio=0.8,
                stop_loss_ratio=0.03
            ),
            'cup_and_handle': ChartPattern(
                name='Cup and Handle',
                description='U-shaped base with small handle',
                bullish_bearish='bullish',
                reliability=0.8,
                target_ratio=1.0,
                stop_loss_ratio=0.05
            )
        }
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect all patterns in the data"""
        try:
            patterns = []
            
            if len(data) < self.min_pattern_length:
                return patterns
            
            # Detect each pattern type
            for pattern_name in self.patterns.keys():
                detected = self._detect_specific_pattern(data, pattern_name, symbol, timeframe)
                if detected:
                    patterns.extend(detected)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_specific_pattern(self, data: pd.DataFrame, pattern_name: str, 
                               symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect specific pattern type"""
        try:
            if pattern_name == 'head_and_shoulders':
                return self._detect_head_and_shoulders(data, symbol, timeframe)
            elif pattern_name == 'inverse_head_and_shoulders':
                return self._detect_inverse_head_and_shoulders(data, symbol, timeframe)
            elif pattern_name == 'double_top':
                return self._detect_double_top(data, symbol, timeframe)
            elif pattern_name == 'double_bottom':
                return self._detect_double_bottom(data, symbol, timeframe)
            elif pattern_name.startswith('triangle'):
                return self._detect_triangle(data, pattern_name, symbol, timeframe)
            elif pattern_name.startswith('flag'):
                return self._detect_flag(data, pattern_name, symbol, timeframe)
            elif pattern_name == 'cup_and_handle':
                return self._detect_cup_and_handle(data, symbol, timeframe)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error detecting {pattern_name}: {e}")
            return []
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect head and shoulders pattern"""
        try:
            patterns = []
            
            if len(data) < 20:
                return patterns
            
            # Find peaks
            highs = data['high'].values
            peaks = self._find_peaks(highs, min_distance=5)
            
            if len(peaks) < 3:
                return patterns
            
            # Look for head and shoulders formation
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Check if head is higher than shoulders
                if (highs[head] > highs[left_shoulder] and 
                    highs[head] > highs[right_shoulder]):
                    
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[head]
                    
                    if shoulder_diff < 0.05:  # 5% tolerance
                        # Calculate confidence based on pattern quality
                        confidence = self._calculate_pattern_confidence(
                            highs[left_shoulder], highs[head], highs[right_shoulder]
                        )
                        
                        if confidence > 0.6:
                            pattern_points = [
                                (left_shoulder, highs[left_shoulder]),
                                (head, highs[head]),
                                (right_shoulder, highs[right_shoulder])
                            ]
                            
                            patterns.append(PatternDetection(
                                pattern_type='head_and_shoulders',
                                confidence=confidence,
                                timestamp=data.index[right_shoulder],
                                symbol=symbol,
                                timeframe=timeframe,
                                price_level=highs[right_shoulder],
                                pattern_points=pattern_points,
                                pattern_metadata={
                                    'left_shoulder_price': highs[left_shoulder],
                                    'head_price': highs[head],
                                    'right_shoulder_price': highs[right_shoulder],
                                    'neckline_level': min(highs[left_shoulder], highs[right_shoulder])
                                },
                                detection_method='traditional'
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return []
    
    def _detect_inverse_head_and_shoulders(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect inverse head and shoulders pattern"""
        try:
            patterns = []
            
            if len(data) < 20:
                return patterns
            
            # Find troughs
            lows = data['low'].values
            troughs = self._find_troughs(lows, min_distance=5)
            
            if len(troughs) < 3:
                return patterns
            
            # Look for inverse head and shoulders formation
            for i in range(len(troughs) - 2):
                left_shoulder = troughs[i]
                head = troughs[i + 1]
                right_shoulder = troughs[i + 2]
                
                # Check if head is lower than shoulders
                if (lows[head] < lows[left_shoulder] and 
                    lows[head] < lows[right_shoulder]):
                    
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder]) / lows[head]
                    
                    if shoulder_diff < 0.05:  # 5% tolerance
                        confidence = self._calculate_pattern_confidence(
                            lows[left_shoulder], lows[head], lows[right_shoulder]
                        )
                        
                        if confidence > 0.6:
                            pattern_points = [
                                (left_shoulder, lows[left_shoulder]),
                                (head, lows[head]),
                                (right_shoulder, lows[right_shoulder])
                            ]
                            
                            patterns.append(PatternDetection(
                                pattern_type='inverse_head_and_shoulders',
                                confidence=confidence,
                                timestamp=data.index[right_shoulder],
                                symbol=symbol,
                                timeframe=timeframe,
                                price_level=lows[right_shoulder],
                                pattern_points=pattern_points,
                                pattern_metadata={
                                    'left_shoulder_price': lows[left_shoulder],
                                    'head_price': lows[head],
                                    'right_shoulder_price': lows[right_shoulder],
                                    'neckline_level': max(lows[left_shoulder], lows[right_shoulder])
                                },
                                detection_method='traditional'
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting inverse head and shoulders: {e}")
            return []
    
    def _detect_double_top(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect double top pattern"""
        try:
            patterns = []
            
            if len(data) < 15:
                return patterns
            
            highs = data['high'].values
            peaks = self._find_peaks(highs, min_distance=5)
            
            if len(peaks) < 2:
                return patterns
            
            # Look for double top formation
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]
                
                # Check if peaks are roughly equal
                peak_diff = abs(highs[peak1] - highs[peak2]) / highs[peak1]
                
                if peak_diff < 0.03:  # 3% tolerance
                    confidence = self._calculate_pattern_confidence(
                        highs[peak1], highs[peak2], 0
                    )
                    
                    if confidence > 0.6:
                        pattern_points = [
                            (peak1, highs[peak1]),
                            (peak2, highs[peak2])
                        ]
                        
                        patterns.append(PatternDetection(
                            pattern_type='double_top',
                            confidence=confidence,
                            timestamp=data.index[peak2],
                            symbol=symbol,
                            timeframe=timeframe,
                            price_level=highs[peak2],
                            pattern_points=pattern_points,
                            pattern_metadata={
                                'first_peak_price': highs[peak1],
                                'second_peak_price': highs[peak2],
                                'resistance_level': highs[peak1]
                            },
                            detection_method='traditional'
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting double top: {e}")
            return []
    
    def _detect_double_bottom(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect double bottom pattern"""
        try:
            patterns = []
            
            if len(data) < 15:
                return patterns
            
            lows = data['low'].values
            troughs = self._find_troughs(lows, min_distance=5)
            
            if len(troughs) < 2:
                return patterns
            
            # Look for double bottom formation
            for i in range(len(troughs) - 1):
                trough1 = troughs[i]
                trough2 = troughs[i + 1]
                
                # Check if troughs are roughly equal
                trough_diff = abs(lows[trough1] - lows[trough2]) / lows[trough1]
                
                if trough_diff < 0.03:  # 3% tolerance
                    confidence = self._calculate_pattern_confidence(
                        lows[trough1], lows[trough2], 0
                    )
                    
                    if confidence > 0.6:
                        pattern_points = [
                            (trough1, lows[trough1]),
                            (trough2, lows[trough2])
                        ]
                        
                        patterns.append(PatternDetection(
                            pattern_type='double_bottom',
                            confidence=confidence,
                            timestamp=data.index[trough2],
                            symbol=symbol,
                            timeframe=timeframe,
                            price_level=lows[trough2],
                            pattern_points=pattern_points,
                            pattern_metadata={
                                'first_trough_price': lows[trough1],
                                'second_trough_price': lows[trough2],
                                'support_level': lows[trough1]
                            },
                            detection_method='traditional'
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return []
    
    def _detect_triangle(self, data: pd.DataFrame, pattern_name: str, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect triangle patterns"""
        try:
            patterns = []
            
            if len(data) < 20:
                return patterns
            
            highs = data['high'].values
            lows = data['low'].values
            
            # Calculate trend lines
            high_trend = self._calculate_trend_line(highs)
            low_trend = self._calculate_trend_line(lows)
            
            # Determine triangle type
            if high_trend > 0.001 and low_trend > 0.001:
                triangle_type = 'triangle_ascending'
            elif high_trend < -0.001 and low_trend < -0.001:
                triangle_type = 'triangle_descending'
            elif abs(high_trend) < 0.001 and abs(low_trend) < 0.001:
                triangle_type = 'triangle_symmetrical'
            else:
                return patterns
            
            if triangle_type == pattern_name:
                # Calculate confidence based on trend line quality
                confidence = self._calculate_triangle_confidence(highs, lows, high_trend, low_trend)
                
                if confidence > 0.5:
                    pattern_points = [
                        (0, highs[0]),
                        (len(highs)-1, highs[-1]),
                        (0, lows[0]),
                        (len(lows)-1, lows[-1])
                    ]
                    
                    patterns.append(PatternDetection(
                        pattern_type=triangle_type,
                        confidence=confidence,
                        timestamp=data.index[-1],
                        symbol=symbol,
                        timeframe=timeframe,
                        price_level=data['close'].iloc[-1],
                        pattern_points=pattern_points,
                        pattern_metadata={
                            'high_trend': high_trend,
                            'low_trend': low_trend,
                            'triangle_width': highs[-1] - lows[-1]
                        },
                        detection_method='traditional'
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting triangle: {e}")
            return []
    
    def _detect_flag(self, data: pd.DataFrame, pattern_name: str, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect flag patterns"""
        try:
            patterns = []
            
            if len(data) < 15:
                return patterns
            
            # Flag detection logic
            # This is a simplified implementation
            highs = data['high'].values
            lows = data['low'].values
            
            # Check for flag-like consolidation
            recent_high = np.max(highs[-10:])
            recent_low = np.min(lows[-10:])
            flag_height = recent_high - recent_low
            
            # Check if it's a flag (small consolidation)
            if flag_height < np.mean(highs) * 0.05:  # Less than 5% of average price
                confidence = 0.7
                
                pattern_points = [
                    (len(highs)-10, recent_high),
                    (len(highs)-1, recent_high),
                    (len(highs)-10, recent_low),
                    (len(highs)-1, recent_low)
                ]
                
                patterns.append(PatternDetection(
                    pattern_type=pattern_name,
                    confidence=confidence,
                    timestamp=data.index[-1],
                    symbol=symbol,
                    timeframe=timeframe,
                    price_level=data['close'].iloc[-1],
                    pattern_points=pattern_points,
                    pattern_metadata={
                        'flag_height': flag_height,
                        'consolidation_period': 10
                    },
                    detection_method='traditional'
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting flag: {e}")
            return []
    
    def _detect_cup_and_handle(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect cup and handle pattern"""
        try:
            patterns = []
            
            if len(data) < 30:
                return patterns
            
            # Cup and handle detection logic
            # This is a simplified implementation
            lows = data['low'].values
            
            # Find the lowest point (cup bottom)
            cup_bottom_idx = np.argmin(lows)
            cup_bottom_price = lows[cup_bottom_idx]
            
            # Check if there's a U-shaped recovery
            left_side = lows[:cup_bottom_idx]
            right_side = lows[cup_bottom_idx:]
            
            if len(left_side) > 5 and len(right_side) > 5:
                # Check for U-shape
                left_recovery = np.mean(left_side[-5:]) - cup_bottom_price
                right_recovery = np.mean(right_side[:5]) - cup_bottom_price
                
                if left_recovery > 0 and right_recovery > 0:
                    confidence = 0.8
                    
                    pattern_points = [
                        (0, lows[0]),
                        (cup_bottom_idx, cup_bottom_price),
                        (len(lows)-1, lows[-1])
                    ]
                    
                    patterns.append(PatternDetection(
                        pattern_type='cup_and_handle',
                        confidence=confidence,
                        timestamp=data.index[-1],
                        symbol=symbol,
                        timeframe=timeframe,
                        price_level=data['close'].iloc[-1],
                        pattern_points=pattern_points,
                        pattern_metadata={
                            'cup_bottom_price': cup_bottom_price,
                            'cup_depth': np.max(lows) - cup_bottom_price
                        },
                        detection_method='traditional'
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting cup and handle: {e}")
            return []
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find peaks in data"""
        peaks = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_peak = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find troughs in data"""
        troughs = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_trough = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] <= data[i]:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _calculate_trend_line(self, data: np.ndarray) -> float:
        """Calculate trend line slope"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _calculate_pattern_confidence(self, *values) -> float:
        """Calculate pattern confidence based on values"""
        try:
            if len(values) < 2:
                return 0.5
            
            # Calculate relative differences
            max_val = max(values)
            min_val = min(values)
            
            if max_val == 0:
                return 0.5
            
            relative_diff = (max_val - min_val) / max_val
            
            # Higher confidence for smaller relative differences (more symmetric patterns)
            confidence = max(0.1, 1.0 - relative_diff)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    def _calculate_triangle_confidence(self, highs: np.ndarray, lows: np.ndarray, 
                                     high_trend: float, low_trend: float) -> float:
        """Calculate triangle pattern confidence"""
        try:
            # Check convergence
            convergence = abs(high_trend - low_trend)
            
            # Check volume (if available)
            volume_factor = 1.0  # Placeholder
            
            # Calculate confidence
            confidence = min(convergence * 100, 1.0) * volume_factor
            
            return max(0.1, min(confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating triangle confidence: {e}")
            return 0.5

class YOLOPatternDetector:
    """YOLO-based pattern detection"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if YOLO_AVAILABLE and model_path:
            self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect patterns using YOLO"""
        try:
            if not self.model:
                logger.warning("YOLO model not loaded")
                return []
            
            # Convert data to chart image
            chart_image = self._data_to_image(data)
            
            if chart_image is None:
                return []
            
            # Run YOLO detection
            results = self.model(chart_image)
            
            patterns = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection information
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        if confidence > 0.5:  # Confidence threshold
                            # Convert bounding box to pattern points
                            pattern_points = self._box_to_pattern_points(box.xyxy[0], data)
                            
                            patterns.append(PatternDetection(
                                pattern_type=class_name,
                                confidence=confidence,
                                timestamp=data.index[-1],
                                symbol=symbol,
                                timeframe=timeframe,
                                price_level=data['close'].iloc[-1],
                                pattern_points=pattern_points,
                                pattern_metadata={
                                    'yolo_confidence': confidence,
                                    'bounding_box': box.xyxy[0].tolist()
                                },
                                detection_method='yolo'
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in YOLO pattern detection: {e}")
            return []
    
    def _data_to_image(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Convert OHLCV data to chart image"""
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot candlestick chart
            for i in range(len(data)):
                open_price = data['open'].iloc[i]
                high_price = data['high'].iloc[i]
                low_price = data['low'].iloc[i]
                close_price = data['close'].iloc[i]
                
                # Determine color
                color = 'green' if close_price >= open_price else 'red'
                
                # Draw high-low line
                ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
                
                # Draw open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                
                rect = patches.Rectangle((i-0.4, bottom), 0.8, height, 
                                       facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            # Set axis properties
            ax.set_xlim(-1, len(data))
            ax.set_ylim(data['low'].min() * 0.99, data['high'].max() * 1.01)
            ax.set_facecolor('white')
            
            # Convert to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            logger.error(f"Error converting data to image: {e}")
            return None
    
    def _box_to_pattern_points(self, box: torch.Tensor, data: pd.DataFrame) -> List[Tuple[float, float]]:
        """Convert YOLO bounding box to pattern points"""
        try:
            x1, y1, x2, y2 = box.tolist()
            
            # Convert image coordinates to price-time coordinates
            # This is a simplified conversion
            time_range = len(data)
            price_range = data['high'].max() - data['low'].min()
            
            # Convert to relative coordinates
            rel_x1 = x1 / 12  # Assuming 12 width
            rel_y1 = y1 / 8   # Assuming 8 height
            rel_x2 = x2 / 12
            rel_y2 = y2 / 8
            
            # Convert to actual coordinates
            time1 = int(rel_x1 * time_range)
            time2 = int(rel_x2 * time_range)
            price1 = data['low'].min() + (1 - rel_y1) * price_range
            price2 = data['low'].min() + (1 - rel_y2) * price_range
            
            return [(time1, price1), (time2, price2)]
            
        except Exception as e:
            logger.error(f"Error converting box to pattern points: {e}")
            return []

class MLPatternDetector:
    """Machine learning-based pattern detection"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load ML model"""
        try:
            import joblib
            model_data = joblib.load(model_path)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"ML pattern detection model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect patterns using ML model"""
        try:
            if not self.model:
                logger.warning("ML model not loaded")
                return []
            
            # Extract features
            features = self._extract_features(data)
            
            if features is None or len(features) == 0:
                return []
            
            # Scale features
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)
            prediction_proba = self.model.predict_proba(features)
            
            patterns = []
            
            for i, (pred, proba) in enumerate(zip(prediction, prediction_proba)):
                if proba.max() > 0.7:  # Confidence threshold
                    pattern_type = self.model.classes_[pred]
                    confidence = proba.max()
                    
                    patterns.append(PatternDetection(
                        pattern_type=pattern_type,
                        confidence=confidence,
                        timestamp=data.index[-1],
                        symbol=symbol,
                        timeframe=timeframe,
                        price_level=data['close'].iloc[-1],
                        pattern_points=[],  # ML doesn't provide specific points
                        pattern_metadata={
                            'ml_confidence': confidence,
                            'all_probabilities': proba.tolist(),
                            'feature_count': len(features[i])
                        },
                        detection_method='ml'
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in ML pattern detection: {e}")
            return []
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        try:
            features = []
            
            # Technical indicators
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'] = self._calculate_macd(data['close'])
            data['bb_position'] = self._calculate_bb_position(data['close'])
            
            # Price features
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = data['close'].rolling(20).std()
            
            # Volume features
            if 'volume' in data.columns:
                data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            else:
                data['volume_ratio'] = 1.0
            
            # Select features
            feature_columns = ['rsi', 'macd', 'bb_position', 'price_change', 'volatility', 'volume_ratio']
            available_features = [col for col in feature_columns if col in data.columns]
            
            if not available_features:
                return None
            
            # Extract latest values
            latest_features = data[available_features].iloc[-1:].values
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        return macd.fillna(0)
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        return bb_position.fillna(0.5)

class PatternDetectionManager:
    """Main pattern detection manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize detectors
        self.traditional_detector = TraditionalPatternDetector()
        
        # Initialize YOLO detector if available
        yolo_model_path = self.config.get('yolo_model_path')
        if yolo_model_path and YOLO_AVAILABLE:
            self.yolo_detector = YOLOPatternDetector(yolo_model_path)
        else:
            self.yolo_detector = None
        
        # Initialize ML detector
        ml_model_path = self.config.get('ml_model_path')
        if ml_model_path:
            self.ml_detector = MLPatternDetector(ml_model_path)
        else:
            self.ml_detector = None
        
        # Pattern database
        self.db_path = self.config.get('db_path', 'pattern_detections.db')
        self._init_database()
    
    def _init_database(self):
        """Initialize pattern detection database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                price_level REAL NOT NULL,
                pattern_points TEXT,
                pattern_metadata TEXT,
                detection_method TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_all_patterns(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternDetection]:
        """Detect patterns using all available methods"""
        try:
            all_patterns = []
            
            # Traditional pattern detection
            traditional_patterns = self.traditional_detector.detect_patterns(data, symbol, timeframe)
            all_patterns.extend(traditional_patterns)
            
            # YOLO pattern detection
            if self.yolo_detector:
                yolo_patterns = self.yolo_detector.detect_patterns(data, symbol, timeframe)
                all_patterns.extend(yolo_patterns)
            
            # ML pattern detection
            if self.ml_detector:
                ml_patterns = self.ml_detector.detect_patterns(data, symbol, timeframe)
                all_patterns.extend(ml_patterns)
            
            # Remove duplicates and low confidence patterns
            filtered_patterns = self._filter_patterns(all_patterns)
            
            # Save to database
            self._save_patterns(filtered_patterns)
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return []
    
    def _filter_patterns(self, patterns: List[PatternDetection]) -> List[PatternDetection]:
        """Filter and deduplicate patterns"""
        try:
            # Remove low confidence patterns
            filtered = [p for p in patterns if p.confidence > 0.5]
            
            # Group by pattern type and keep highest confidence
            pattern_groups = {}
            for pattern in filtered:
                key = f"{pattern.pattern_type}_{pattern.symbol}_{pattern.timeframe}"
                if key not in pattern_groups or pattern.confidence > pattern_groups[key].confidence:
                    pattern_groups[key] = pattern
            
            return list(pattern_groups.values())
            
        except Exception as e:
            logger.error(f"Error filtering patterns: {e}")
            return patterns
    
    def _save_patterns(self, patterns: List[PatternDetection]):
        """Save patterns to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pattern in patterns:
                cursor.execute('''
                    INSERT INTO pattern_detections 
                    (symbol, timeframe, pattern_type, confidence, timestamp, price_level, 
                     pattern_points, pattern_metadata, detection_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.symbol,
                    pattern.timeframe,
                    pattern.pattern_type,
                    pattern.confidence,
                    pattern.timestamp,
                    pattern.price_level,
                    json.dumps(pattern.pattern_points),
                    json.dumps(pattern.pattern_metadata),
                    pattern.detection_method
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def get_recent_patterns(self, symbol: str, timeframe: str, hours: int = 24) -> List[PatternDetection]:
        """Get recent patterns for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, timeframe, pattern_type, confidence, timestamp, price_level,
                       pattern_points, pattern_metadata, detection_method
                FROM pattern_detections
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours), (symbol, timeframe))
            
            rows = cursor.fetchall()
            conn.close()
            
            patterns = []
            for row in rows:
                patterns.append(PatternDetection(
                    symbol=row[0],
                    timeframe=row[1],
                    pattern_type=row[2],
                    confidence=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    price_level=row[5],
                    pattern_points=json.loads(row[6]) if row[6] else [],
                    pattern_metadata=json.loads(row[7]) if row[7] else {},
                    detection_method=row[8]
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting recent patterns: {e}")
            return []

def main():
    """Main function to demonstrate pattern detection"""
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        
        # Generate sample OHLCV data
        base_price = 100
        prices = [base_price]
        
        for i in range(n_samples - 1):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.index = pd.date_range(start=datetime.now() - timedelta(days=n_samples), periods=n_samples, freq='1H')
        
        # Initialize pattern detection manager
        config = {
            'yolo_model_path': 'chart_pattern/pattern_yolo_12x.pt',  # Adjust path as needed
            'ml_model_path': None,  # Add ML model path if available
            'db_path': 'pattern_detections.db'
        }
        
        detector = PatternDetectionManager(config)
        
        # Detect patterns
        logger.info("Detecting patterns...")
        patterns = detector.detect_all_patterns(df, 'SOLUSDT', '1H')
        
        logger.info(f"Detected {len(patterns)} patterns:")
        for pattern in patterns:
            logger.info(f"  {pattern.pattern_type}: {pattern.confidence:.3f} confidence "
                       f"({pattern.detection_method})")
        
        # Get recent patterns
        recent_patterns = detector.get_recent_patterns('SOLUSDT', '1H', 24)
        logger.info(f"Recent patterns: {len(recent_patterns)}")
        
        logger.info("Pattern detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pattern detection function: {e}")

if __name__ == "__main__":
    main()
