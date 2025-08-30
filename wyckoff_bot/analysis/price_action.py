# wyckoff_bot/analysis/price_action.py
"""
Price Action Analysis for Wyckoff Method
========================================
Analyzes price movements and patterns for Wyckoff signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PriceActionSignal:
    """Price action signal"""
    signal_type: str  # 'spring', 'upthrust', 'test', 'breakout'
    strength: float   # 0-1
    price_level: float
    confirmation: bool

class PriceActionAnalyzer:
    """
    Price action analysis for Wyckoff method
    Identifies key price patterns and levels
    """
    
    def __init__(self, swing_period: int = 10):
        self.swing_period = swing_period
    
    def analyze_price_action(self, df: pd.DataFrame) -> List[PriceActionSignal]:
        """
        Analyze price action for Wyckoff signals
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List[PriceActionSignal]: Detected signals
        """
        signals = []
        
        if len(df) < self.swing_period * 2:
            return signals
        
        # Find swing highs and lows
        swing_highs, swing_lows = self._find_swing_points(df)
        
        # Detect various Wyckoff patterns
        signals.extend(self._detect_springs(df, swing_lows))
        signals.extend(self._detect_upthrusts(df, swing_highs))
        signals.extend(self._detect_tests(df, swing_highs, swing_lows))
        signals.extend(self._detect_breakouts(df, swing_highs, swing_lows))
        
        return signals
    
    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        highs = []
        lows = []
        
        for i in range(self.swing_period, len(df) - self.swing_period):
            # Check for swing high
            if (df['high'].iloc[i] == df['high'].iloc[i-self.swing_period:i+self.swing_period+1].max()):
                highs.append(i)
            
            # Check for swing low
            if (df['low'].iloc[i] == df['low'].iloc[i-self.swing_period:i+self.swing_period+1].min()):
                lows.append(i)
        
        return highs, lows
    
    def _detect_springs(self, df: pd.DataFrame, swing_lows: List[int]) -> List[PriceActionSignal]:
        """Detect spring patterns (false breaks below support)"""
        signals = []
        
        if len(swing_lows) < 2:
            return signals
        
        # Look for recent price action that breaks below previous low briefly
        recent_low = df['low'].tail(5).min()
        
        for low_idx in swing_lows[-3:]:  # Check last 3 swing lows
            support_level = df['low'].iloc[low_idx]
            
            # Check if recent low broke below but recovered
            if (recent_low < support_level * 0.998 and  # Broke below by small amount
                df['close'].iloc[-1] > support_level):   # But recovered above
                
                strength = self._calculate_spring_strength(df, support_level)
                confirmation = df['volume'].tail(3).mean() > df['volume'].mean()
                
                signals.append(PriceActionSignal(
                    signal_type='spring',
                    strength=strength,
                    price_level=support_level,
                    confirmation=confirmation
                ))
        
        return signals
    
    def _detect_upthrusts(self, df: pd.DataFrame, swing_highs: List[int]) -> List[PriceActionSignal]:
        """Detect upthrust patterns (false breaks above resistance)"""
        signals = []
        
        if len(swing_highs) < 2:
            return signals
        
        # Look for recent price action that breaks above previous high briefly
        recent_high = df['high'].tail(5).max()
        
        for high_idx in swing_highs[-3:]:  # Check last 3 swing highs
            resistance_level = df['high'].iloc[high_idx]
            
            # Check if recent high broke above but failed
            if (recent_high > resistance_level * 1.002 and  # Broke above by small amount
                df['close'].iloc[-1] < resistance_level):    # But failed below
                
                strength = self._calculate_upthrust_strength(df, resistance_level)
                confirmation = df['volume'].tail(3).mean() > df['volume'].mean()
                
                signals.append(PriceActionSignal(
                    signal_type='upthrust',
                    strength=strength,
                    price_level=resistance_level,
                    confirmation=confirmation
                ))
        
        return signals
    
    def _detect_tests(self, df: pd.DataFrame, swing_highs: List[int], 
                     swing_lows: List[int]) -> List[PriceActionSignal]:
        """Detect test patterns (retests of support/resistance)"""
        signals = []
        
        # Test of highs (resistance)
        for high_idx in swing_highs[-2:]:
            resistance = df['high'].iloc[high_idx]
            recent_high = df['high'].tail(5).max()
            
            # Check if recent price tested resistance without breaking
            if (abs(recent_high - resistance) / resistance < 0.005 and  # Close to level
                df['close'].iloc[-1] < resistance):                      # But didn't break
                
                signals.append(PriceActionSignal(
                    signal_type='test',
                    strength=0.6,
                    price_level=resistance,
                    confirmation=True
                ))
        
        # Test of lows (support)
        for low_idx in swing_lows[-2:]:
            support = df['low'].iloc[low_idx]
            recent_low = df['low'].tail(5).min()
            
            # Check if recent price tested support without breaking
            if (abs(recent_low - support) / support < 0.005 and  # Close to level
                df['close'].iloc[-1] > support):                  # But held above
                
                signals.append(PriceActionSignal(
                    signal_type='test',
                    strength=0.6,
                    price_level=support,
                    confirmation=True
                ))
        
        return signals
    
    def _detect_breakouts(self, df: pd.DataFrame, swing_highs: List[int], 
                         swing_lows: List[int]) -> List[PriceActionSignal]:
        """Detect legitimate breakout patterns"""
        signals = []
        current_price = df['close'].iloc[-1]
        
        # Bullish breakouts above resistance
        for high_idx in swing_highs[-2:]:
            resistance = df['high'].iloc[high_idx]
            
            if current_price > resistance * 1.01:  # Clear break above
                strength = self._calculate_breakout_strength(df, resistance, 'bullish')
                volume_confirmation = df['volume'].tail(3).mean() > df['volume'].mean() * 1.5
                
                signals.append(PriceActionSignal(
                    signal_type='breakout',
                    strength=strength,
                    price_level=resistance,
                    confirmation=volume_confirmation
                ))
        
        # Bearish breakdowns below support
        for low_idx in swing_lows[-2:]:
            support = df['low'].iloc[low_idx]
            
            if current_price < support * 0.99:  # Clear break below
                strength = self._calculate_breakout_strength(df, support, 'bearish')
                volume_confirmation = df['volume'].tail(3).mean() > df['volume'].mean() * 1.5
                
                signals.append(PriceActionSignal(
                    signal_type='breakout',
                    strength=strength,
                    price_level=support,
                    confirmation=volume_confirmation
                ))
        
        return signals
    
    def _calculate_spring_strength(self, df: pd.DataFrame, support_level: float) -> float:
        """Calculate strength of spring pattern"""
        # How quickly did it recover?
        recovery_speed = (df['close'].iloc[-1] - df['low'].tail(5).min()) / support_level
        
        # Volume during recovery
        volume_strength = df['volume'].tail(3).mean() / df['volume'].mean()
        
        return min((recovery_speed + volume_strength) / 2, 1.0)
    
    def _calculate_upthrust_strength(self, df: pd.DataFrame, resistance_level: float) -> float:
        """Calculate strength of upthrust pattern"""
        # How quickly did it fail?
        failure_speed = (df['high'].tail(5).max() - df['close'].iloc[-1]) / resistance_level
        
        # Volume during failure
        volume_strength = df['volume'].tail(3).mean() / df['volume'].mean()
        
        return min((failure_speed + volume_strength) / 2, 1.0)
    
    def _calculate_breakout_strength(self, df: pd.DataFrame, level: float, direction: str) -> float:
        """Calculate breakout strength"""
        current_price = df['close'].iloc[-1]
        
        # Distance of breakout
        if direction == 'bullish':
            distance = (current_price - level) / level
        else:
            distance = (level - current_price) / level
        
        # Volume confirmation
        volume_strength = df['volume'].tail(3).mean() / df['volume'].mean()
        
        return min((distance * 10 + volume_strength) / 2, 1.0)
    
    def calculate_support_resistance_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate strength of key support/resistance levels"""
        swing_highs, swing_lows = self._find_swing_points(df)
        
        if not swing_highs or not swing_lows:
            return {}
        
        # Find most recent and significant levels
        recent_resistance = df['high'].iloc[swing_highs[-1]] if swing_highs else 0
        recent_support = df['low'].iloc[swing_lows[-1]] if swing_lows else 0
        
        # Calculate strength based on number of tests and volume
        resistance_strength = len([h for h in swing_highs if abs(df['high'].iloc[h] - recent_resistance) / recent_resistance < 0.02])
        support_strength = len([l for l in swing_lows if abs(df['low'].iloc[l] - recent_support) / recent_support < 0.02])
        
        return {
            'resistance_level': recent_resistance,
            'resistance_strength': min(resistance_strength / 3, 1.0),
            'support_level': recent_support,
            'support_strength': min(support_strength / 3, 1.0)
        }