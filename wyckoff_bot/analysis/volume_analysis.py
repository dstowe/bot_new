# wyckoff_bot/analysis/volume_analysis.py
"""
Volume Analysis for Wyckoff Method
==================================
Specialized volume analysis following Wyckoff principles
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class VolumeAnalysis:
    """Volume analysis results"""
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    volume_strength: float  # 0-1 scale
    climax_detected: bool
    volume_confirmation: bool
    relative_volume: float  # vs average

class VolumeAnalyzer:
    """
    Volume analysis following Wyckoff principles
    Focus on volume-price relationships
    """
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def analyze_volume(self, df: pd.DataFrame) -> VolumeAnalysis:
        """
        Comprehensive volume analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            VolumeAnalysis: Volume analysis results
        """
        if len(df) < self.lookback_period:
            return self._create_default_analysis()
        
        recent_data = df.tail(self.lookback_period)
        
        # Volume trend analysis
        volume_trend = self._analyze_volume_trend(recent_data)
        
        # Volume strength relative to average
        volume_strength = self._calculate_volume_strength(recent_data)
        
        # Climax detection (unusually high volume)
        climax_detected = self._detect_volume_climax(recent_data)
        
        # Volume-price confirmation
        volume_confirmation = self._check_volume_price_confirmation(recent_data)
        
        # Relative volume
        relative_volume = self._calculate_relative_volume(recent_data)
        
        return VolumeAnalysis(
            volume_trend=volume_trend,
            volume_strength=volume_strength,
            climax_detected=climax_detected,
            volume_confirmation=volume_confirmation,
            relative_volume=relative_volume
        )
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend direction"""
        recent_avg = df['volume'].tail(5).mean()
        older_avg = df['volume'].iloc[-10:-5].mean()
        
        if recent_avg > older_avg * 1.2:
            return 'increasing'
        elif recent_avg < older_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_volume_strength(self, df: pd.DataFrame) -> float:
        """Calculate volume strength (0-1)"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        
        # Normalize to 0-1 scale
        strength = min(current_volume / (avg_volume * 3), 1.0)
        return strength
    
    def _detect_volume_climax(self, df: pd.DataFrame) -> bool:
        """Detect volume climax (unusually high volume)"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        std_volume = df['volume'].std()
        
        # Climax if volume > 2 standard deviations above mean
        return current_volume > (avg_volume + 2 * std_volume)
    
    def _check_volume_price_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms price movement"""
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
        
        # Volume should increase with significant price moves
        if abs(price_change) > 0.02:  # Significant price move
            return volume_change > 0.1  # Volume should increase
        
        return True  # No significant price move, no requirement
    
    def _calculate_relative_volume(self, df: pd.DataFrame) -> float:
        """Calculate current volume relative to average"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        return current_volume / avg_volume
    
    def _create_default_analysis(self) -> VolumeAnalysis:
        """Create default analysis for insufficient data"""
        return VolumeAnalysis(
            volume_trend='stable',
            volume_strength=0.5,
            climax_detected=False,
            volume_confirmation=True,
            relative_volume=1.0
        )
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volume-based indicators"""
        if len(df) < 20:
            return {}
        
        # Volume Moving Averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=5)
        
        # On-Balance Volume (OBV)
        df['price_change'] = df['close'].diff()
        df['obv'] = (df['price_change'] > 0).astype(int) * df['volume'] - \
                    (df['price_change'] < 0).astype(int) * df['volume']
        df['obv'] = df['obv'].cumsum()
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        return {
            'volume_sma_10': df['volume_sma_10'].iloc[-1],
            'volume_sma_20': df['volume_sma_20'].iloc[-1],
            'volume_roc': df['volume_roc'].iloc[-1],
            'obv': df['obv'].iloc[-1],
            'vpt': df['vpt'].iloc[-1]
        }