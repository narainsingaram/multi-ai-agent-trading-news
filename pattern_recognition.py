"""
Pattern Recognition Module for AI Trading System.
Detects classic chart patterns using technical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import json


def _local_extrema(values: np.ndarray, comparator: Callable[[np.ndarray, float], bool], order: int) -> np.ndarray:
    """
    Lightweight replacement for scipy.signal.argrelextrema.
    Compares each point against its surrounding window of size `order`.
    """
    idxs = []
    if values is None or len(values) == 0:
        return np.array([], dtype=int)
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        center = values[i]
        # Skip if NaN present
        if np.isnan(center) or np.isnan(window).any():
            continue
        before = window[:order]
        after = window[order + 1 :]
        if comparator(center, before.max()) and comparator(center, after.max()):
            idxs.append(i)
    return np.array(idxs, dtype=int)


class PatternRecognition:
    """
    Detect classic chart patterns in OHLCV data.
    Patterns: Head & Shoulders, Cup & Handle, Flags, Triangles, Double Top/Bottom
    """
    
    def __init__(self, min_pattern_bars: int = 20):
        """
        Initialize pattern recognition.
        
        Args:
            min_pattern_bars: Minimum bars required to form a pattern
        """
        self.min_pattern_bars = min_pattern_bars
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all patterns in the given OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns with details
        """
        if df is None or len(df) < self.min_pattern_bars:
            return []
        
        patterns = []
        
        # Detect each pattern type
        patterns.extend(self.detect_head_and_shoulders(df))
        patterns.extend(self.detect_inverse_head_and_shoulders(df))
        patterns.extend(self.detect_double_top(df))
        patterns.extend(self.detect_double_bottom(df))
        patterns.extend(self.detect_cup_and_handle(df))
        patterns.extend(self.detect_bull_flag(df))
        patterns.extend(self.detect_bear_flag(df))
        patterns.extend(self.detect_ascending_triangle(df))
        patterns.extend(self.detect_descending_triangle(df))
        
        # Sort by quality score
        patterns.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return patterns
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect bearish head and shoulders pattern.
        Pattern: Left Shoulder - Head - Right Shoulder with neckline
        """
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Find peaks (potential shoulders and head)
        peaks_idx = _local_extrema(df['High'].values, lambda c, m: c > m, order=5)
        
        if len(peaks_idx) < 3:
            return patterns
        
        # Check each combination of 3 peaks
        for i in range(len(peaks_idx) - 2):
            left_shoulder_idx = peaks_idx[i]
            head_idx = peaks_idx[i + 1]
            right_shoulder_idx = peaks_idx[i + 2]
            
            left_shoulder = df['High'].iloc[left_shoulder_idx]
            head = df['High'].iloc[head_idx]
            right_shoulder = df['High'].iloc[right_shoulder_idx]
            
            # Head should be highest
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal (within 3%)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                
                if shoulder_diff < 0.03:
                    # Find neckline (lows between peaks)
                    neckline_lows = df['Low'].iloc[left_shoulder_idx:right_shoulder_idx+1]
                    neckline_level = neckline_lows.min()
                    
                    # Calculate pattern measurements
                    pattern_height = head - neckline_level
                    target = neckline_level - pattern_height  # Bearish target
                    
                    # Quality score based on symmetry and clarity
                    quality_score = 1.0 - shoulder_diff
                    
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'subtype': 'bearish',
                        'quality_score': quality_score,
                        'coordinates': {
                            'left_shoulder': {'index': int(left_shoulder_idx), 'price': float(left_shoulder)},
                            'head': {'index': int(head_idx), 'price': float(head)},
                            'right_shoulder': {'index': int(right_shoulder_idx), 'price': float(right_shoulder)},
                            'neckline': float(neckline_level)
                        },
                        'measurements': {
                            'pattern_height': float(pattern_height),
                            'target_price': float(target),
                            'current_price': float(df['Close'].iloc[-1])
                        },
                        'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                    })
        
        return patterns
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect bullish inverse head and shoulders pattern.
        Pattern: Left Shoulder - Head - Right Shoulder (inverted) with neckline
        """
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Find troughs (potential shoulders and head)
        troughs_idx = _local_extrema(df['Low'].values, lambda c, m: c < m, order=5)
        
        if len(troughs_idx) < 3:
            return patterns
        
        # Check each combination of 3 troughs
        for i in range(len(troughs_idx) - 2):
            left_shoulder_idx = troughs_idx[i]
            head_idx = troughs_idx[i + 1]
            right_shoulder_idx = troughs_idx[i + 2]
            
            left_shoulder = df['Low'].iloc[left_shoulder_idx]
            head = df['Low'].iloc[head_idx]
            right_shoulder = df['Low'].iloc[right_shoulder_idx]
            
            # Head should be lowest
            if head < left_shoulder and head < right_shoulder:
                # Shoulders should be roughly equal (within 3%)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                
                if shoulder_diff < 0.03:
                    # Find neckline (highs between troughs)
                    neckline_highs = df['High'].iloc[left_shoulder_idx:right_shoulder_idx+1]
                    neckline_level = neckline_highs.max()
                    
                    # Calculate pattern measurements
                    pattern_height = neckline_level - head
                    target = neckline_level + pattern_height  # Bullish target
                    
                    # Quality score
                    quality_score = 1.0 - shoulder_diff
                    
                    patterns.append({
                        'type': 'inverse_head_and_shoulders',
                        'subtype': 'bullish',
                        'quality_score': quality_score,
                        'coordinates': {
                            'left_shoulder': {'index': int(left_shoulder_idx), 'price': float(left_shoulder)},
                            'head': {'index': int(head_idx), 'price': float(head)},
                            'right_shoulder': {'index': int(right_shoulder_idx), 'price': float(right_shoulder)},
                            'neckline': float(neckline_level)
                        },
                        'measurements': {
                            'pattern_height': float(pattern_height),
                            'target_price': float(target),
                            'current_price': float(df['Close'].iloc[-1])
                        },
                        'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                    })
        
        return patterns
    
    def detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bearish double top pattern."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        peaks_idx = _local_extrema(df['High'].values, lambda c, m: c > m, order=3)
        
        if len(peaks_idx) < 2:
            return patterns
        
        # Check each pair of peaks
        for i in range(len(peaks_idx) - 1):
            peak1_idx = peaks_idx[i]
            peak2_idx = peaks_idx[i + 1]
            
            peak1 = df['High'].iloc[peak1_idx]
            peak2 = df['High'].iloc[peak2_idx]
            
            # Peaks should be roughly equal (within 2%)
            peak_diff = abs(peak1 - peak2) / peak1
            
            if peak_diff < 0.02:
                # Find trough between peaks
                trough = df['Low'].iloc[peak1_idx:peak2_idx+1].min()
                
                # Pattern height
                pattern_height = ((peak1 + peak2) / 2) - trough
                target = trough - pattern_height
                
                quality_score = 1.0 - peak_diff
                
                patterns.append({
                    'type': 'double_top',
                    'subtype': 'bearish',
                    'quality_score': quality_score,
                    'coordinates': {
                        'peak1': {'index': int(peak1_idx), 'price': float(peak1)},
                        'peak2': {'index': int(peak2_idx), 'price': float(peak2)},
                        'trough': float(trough)
                    },
                    'measurements': {
                        'pattern_height': float(pattern_height),
                        'target_price': float(target),
                        'current_price': float(df['Close'].iloc[-1])
                    },
                    'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                })
        
        return patterns
    
    def detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish double bottom pattern."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        troughs_idx = argrelextrema(df['Low'].values, np.less, order=3)[0]
        
        if len(troughs_idx) < 2:
            return patterns
        
        # Check each pair of troughs
        for i in range(len(troughs_idx) - 1):
            trough1_idx = troughs_idx[i]
            trough2_idx = troughs_idx[i + 1]
            
            trough1 = df['Low'].iloc[trough1_idx]
            trough2 = df['Low'].iloc[trough2_idx]
            
            # Troughs should be roughly equal (within 2%)
            trough_diff = abs(trough1 - trough2) / trough1
            
            if trough_diff < 0.02:
                # Find peak between troughs
                peak = df['High'].iloc[trough1_idx:trough2_idx+1].max()
                
                # Pattern height
                pattern_height = peak - ((trough1 + trough2) / 2)
                target = peak + pattern_height
                
                quality_score = 1.0 - trough_diff
                
                patterns.append({
                    'type': 'double_bottom',
                    'subtype': 'bullish',
                    'quality_score': quality_score,
                    'coordinates': {
                        'trough1': {'index': int(trough1_idx), 'price': float(trough1)},
                        'trough2': {'index': int(trough2_idx), 'price': float(trough2)},
                        'peak': float(peak)
                    },
                    'measurements': {
                        'pattern_height': float(pattern_height),
                        'target_price': float(target),
                        'current_price': float(df['Close'].iloc[-1])
                    },
                    'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                })
        
        return patterns
    
    def detect_cup_and_handle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish cup and handle pattern."""
        patterns = []
        
        if len(df) < 40:  # Need longer period for cup and handle
            return patterns
        
        # Look for U-shaped bottom (cup) followed by small consolidation (handle)
        # This is a simplified detection - real implementation would be more sophisticated
        
        recent_data = df.tail(40)
        mid_point = len(recent_data) // 2
        
        # Cup: Check if price made a U-shape
        left_rim = recent_data['High'].iloc[:5].max()
        bottom = recent_data['Low'].iloc[mid_point-5:mid_point+5].min()
        right_rim = recent_data['High'].iloc[-10:-5].max()
        
        # Rims should be roughly equal
        rim_diff = abs(left_rim - right_rim) / left_rim
        
        if rim_diff < 0.05 and bottom < left_rim * 0.85:  # Cup depth at least 15%
            # Handle: Small pullback after right rim
            handle_low = recent_data['Low'].iloc[-5:].min()
            handle_high = recent_data['High'].iloc[-5:].max()
            
            # Handle should be shallow (less than 50% of cup depth)
            cup_depth = left_rim - bottom
            handle_depth = right_rim - handle_low
            
            if handle_depth < cup_depth * 0.5:
                target = right_rim + cup_depth
                
                patterns.append({
                    'type': 'cup_and_handle',
                    'subtype': 'bullish',
                    'quality_score': 0.8,
                    'coordinates': {
                        'left_rim': float(left_rim),
                        'bottom': float(bottom),
                        'right_rim': float(right_rim),
                        'handle_low': float(handle_low)
                    },
                    'measurements': {
                        'cup_depth': float(cup_depth),
                        'handle_depth': float(handle_depth),
                        'target_price': float(target),
                        'current_price': float(df['Close'].iloc[-1])
                    },
                    'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                })
        
        return patterns
    
    def detect_bull_flag(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish flag pattern (continuation)."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        recent = df.tail(20)
        
        # Flag: Strong uptrend (pole) followed by downward consolidation (flag)
        # Pole: First 10-15 bars should show strong uptrend
        pole_start = recent['Close'].iloc[0]
        pole_end = recent['Close'].iloc[10]
        pole_gain = (pole_end - pole_start) / pole_start
        
        if pole_gain > 0.05:  # At least 5% gain for pole
            # Flag: Last 5-10 bars should show slight downtrend or consolidation
            flag_data = recent.tail(10)
            flag_slope = (flag_data['Close'].iloc[-1] - flag_data['Close'].iloc[0]) / len(flag_data)
            
            if flag_slope < 0:  # Slight downtrend in flag
                target = pole_end + (pole_end - pole_start)  # Measured move
                
                patterns.append({
                    'type': 'bull_flag',
                    'subtype': 'bullish',
                    'quality_score': 0.75,
                    'coordinates': {
                        'pole_start': float(pole_start),
                        'pole_end': float(pole_end),
                        'flag_start': float(flag_data['Close'].iloc[0]),
                        'flag_end': float(flag_data['Close'].iloc[-1])
                    },
                    'measurements': {
                        'pole_height': float(pole_end - pole_start),
                        'target_price': float(target),
                        'current_price': float(df['Close'].iloc[-1])
                    },
                    'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                })
        
        return patterns
    
    def detect_bear_flag(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bearish flag pattern (continuation)."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        recent = df.tail(20)
        
        # Bear flag: Strong downtrend (pole) followed by upward consolidation (flag)
        pole_start = recent['Close'].iloc[0]
        pole_end = recent['Close'].iloc[10]
        pole_loss = (pole_start - pole_end) / pole_start
        
        if pole_loss > 0.05:  # At least 5% drop for pole
            # Flag: Last 5-10 bars should show slight uptrend or consolidation
            flag_data = recent.tail(10)
            flag_slope = (flag_data['Close'].iloc[-1] - flag_data['Close'].iloc[0]) / len(flag_data)
            
            if flag_slope > 0:  # Slight uptrend in flag
                target = pole_end - (pole_start - pole_end)  # Measured move down
                
                patterns.append({
                    'type': 'bear_flag',
                    'subtype': 'bearish',
                    'quality_score': 0.75,
                    'coordinates': {
                        'pole_start': float(pole_start),
                        'pole_end': float(pole_end),
                        'flag_start': float(flag_data['Close'].iloc[0]),
                        'flag_end': float(flag_data['Close'].iloc[-1])
                    },
                    'measurements': {
                        'pole_height': float(pole_start - pole_end),
                        'target_price': float(target),
                        'current_price': float(df['Close'].iloc[-1])
                    },
                    'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                })
        
        return patterns
    
    def detect_ascending_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish ascending triangle pattern."""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        recent = df.tail(30)
        
        # Ascending triangle: Flat resistance + rising support
        # Find highs (should be roughly equal - flat resistance)
        peaks_idx = argrelextrema(recent['High'].values, np.greater, order=3)[0]
        
        if len(peaks_idx) >= 2:
            peaks = recent['High'].iloc[peaks_idx]
            resistance = peaks.mean()
            resistance_std = peaks.std()
            
            # Check if resistance is flat (low std deviation)
            if resistance_std / resistance < 0.02:  # Within 2%
                # Check if lows are rising
                troughs_idx = argrelextrema(recent['Low'].values, np.less, order=3)[0]
                
                if len(troughs_idx) >= 2:
                    # Calculate slope of support line
                    support_prices = recent['Low'].iloc[troughs_idx]
                    support_slope = (support_prices.iloc[-1] - support_prices.iloc[0]) / len(troughs_idx)
                    
                    if support_slope > 0:  # Rising support
                        target = resistance + (resistance - support_prices.mean())
                        
                        patterns.append({
                            'type': 'ascending_triangle',
                            'subtype': 'bullish',
                            'quality_score': 0.8,
                            'coordinates': {
                                'resistance': float(resistance),
                                'support_start': float(support_prices.iloc[0]),
                                'support_end': float(support_prices.iloc[-1])
                            },
                            'measurements': {
                                'triangle_height': float(resistance - support_prices.mean()),
                                'target_price': float(target),
                                'current_price': float(df['Close'].iloc[-1])
                            },
                            'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                        })
        
        return patterns
    
    def detect_descending_triangle(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bearish descending triangle pattern."""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        recent = df.tail(30)
        
        # Descending triangle: Flat support + falling resistance
        # Find lows (should be roughly equal - flat support)
        troughs_idx = argrelextrema(recent['Low'].values, np.less, order=3)[0]
        
        if len(troughs_idx) >= 2:
            troughs = recent['Low'].iloc[troughs_idx]
            support = troughs.mean()
            support_std = troughs.std()
            
            # Check if support is flat
            if support_std / support < 0.02:
                # Check if highs are falling
                peaks_idx = argrelextrema(recent['High'].values, np.greater, order=3)[0]
                
                if len(peaks_idx) >= 2:
                    resistance_prices = recent['High'].iloc[peaks_idx]
                    resistance_slope = (resistance_prices.iloc[-1] - resistance_prices.iloc[0]) / len(peaks_idx)
                    
                    if resistance_slope < 0:  # Falling resistance
                        target = support - (resistance_prices.mean() - support)
                        
                        patterns.append({
                            'type': 'descending_triangle',
                            'subtype': 'bearish',
                            'quality_score': 0.8,
                            'coordinates': {
                                'support': float(support),
                                'resistance_start': float(resistance_prices.iloc[0]),
                                'resistance_end': float(resistance_prices.iloc[-1])
                            },
                            'measurements': {
                                'triangle_height': float(resistance_prices.mean() - support),
                                'target_price': float(target),
                                'current_price': float(df['Close'].iloc[-1])
                            },
                            'detected_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                        })
        
        return patterns
    
    def get_pattern_summary(self, patterns: List[Dict]) -> str:
        """Generate human-readable summary of detected patterns."""
        if not patterns:
            return "No significant chart patterns detected."
        
        summary_lines = [f"Detected {len(patterns)} pattern(s):"]
        
        for i, pattern in enumerate(patterns[:3], 1):  # Top 3
            ptype = pattern['type'].replace('_', ' ').title()
            subtype = pattern.get('subtype', '').title()
            quality = pattern.get('quality_score', 0) * 100
            target = pattern.get('measurements', {}).get('target_price', 0)
            
            summary_lines.append(
                f"{i}. {ptype} ({subtype}) - Quality: {quality:.0f}%, Target: ${target:.2f}"
            )
        
        return "\n".join(summary_lines)


# Global instance
_pattern_recognition = None

def get_pattern_recognition() -> PatternRecognition:
    """Get global pattern recognition instance."""
    global _pattern_recognition
    if _pattern_recognition is None:
        _pattern_recognition = PatternRecognition()
    return _pattern_recognition
