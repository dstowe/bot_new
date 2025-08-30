# wyckoff_bot/analysis/wyckoff_analyzer.py
"""
Core Wyckoff Analysis Engine
============================
Identifies Wyckoff market phases and structures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class WyckoffPhase(Enum):
    """Wyckoff market phases"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

class WyckoffEvent(Enum):
    """Key Wyckoff events"""
    PS = "preliminary_support"  # Preliminary Support
    SC = "selling_climax"       # Selling Climax
    AR = "automatic_rally"      # Automatic Rally
    ST = "secondary_test"       # Secondary Test
    SOS = "sign_of_strength"    # Sign of Strength
    LPS = "last_point_support"  # Last Point of Support
    PSY = "preliminary_supply"  # Preliminary Supply
    BC = "buying_climax"        # Buying Climax
    AD = "automatic_reaction"   # Automatic Reaction
    UT = "upthrust"            # Upthrust
    SOW = "sign_of_weakness"   # Sign of Weakness
    LPSY = "last_point_supply" # Last Point of Supply

@dataclass
class WyckoffAnalysis:
    """Wyckoff analysis results"""
    symbol: str
    timeframe: str
    phase: WyckoffPhase
    confidence: float
    volume_confirmation: bool
    price_action_strength: float
    key_events: List[WyckoffEvent]
    support_resistance_levels: Dict[str, float]
    trend_strength: float

class WyckoffAnalyzer:
    """
    Main Wyckoff analysis engine
    Analyzes price and volume data to identify Wyckoff structures
    """
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.volume_analyzer = None  # Will be set by import
        self.price_analyzer = None   # Will be set by import
        
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str = "1D") -> WyckoffAnalysis:
        """
        Perform complete Wyckoff analysis on price data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            timeframe: Data timeframe
            
        Returns:
            WyckoffAnalysis: Complete analysis results
        """
        if len(df) < self.lookback_periods:
            return self._create_unknown_analysis(symbol, timeframe)
            
        # Identify current phase
        phase = self._identify_phase(df)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(df, phase)
        
        # Volume confirmation
        volume_confirmation = self._check_volume_confirmation(df, phase)
        
        # Price action strength
        price_strength = self._calculate_price_action_strength(df)
        
        # Key events detection
        key_events = self._detect_key_events(df, phase)
        
        # Support/resistance levels
        sr_levels = self._find_support_resistance_levels(df)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(df)
        
        return WyckoffAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            phase=phase,
            confidence=confidence,
            volume_confirmation=volume_confirmation,
            price_action_strength=price_strength,
            key_events=key_events,
            support_resistance_levels=sr_levels,
            trend_strength=trend_strength
        )
    
    def _identify_phase(self, df: pd.DataFrame) -> WyckoffPhase:
        """Identify current Wyckoff phase"""
        recent_data = df.tail(self.lookback_periods)
        
        # Calculate price volatility and volume patterns
        price_volatility = recent_data['close'].pct_change().std()
        volume_trend = self._calculate_volume_trend(recent_data)
        price_trend = self._calculate_price_trend(recent_data)
        
        # Phase identification logic
        if self._is_accumulation_phase(recent_data, price_volatility, volume_trend):
            return WyckoffPhase.ACCUMULATION
        elif self._is_markup_phase(recent_data, price_trend, volume_trend):
            return WyckoffPhase.MARKUP
        elif self._is_distribution_phase(recent_data, price_volatility, volume_trend):
            return WyckoffPhase.DISTRIBUTION
        elif self._is_markdown_phase(recent_data, price_trend, volume_trend):
            return WyckoffPhase.MARKDOWN
        else:
            return WyckoffPhase.UNKNOWN
    
    def _is_accumulation_phase(self, df: pd.DataFrame, volatility: float, volume_trend: float) -> bool:
        """Check if in accumulation phase"""
        # Low volatility, sideways movement, volume patterns
        recent_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[-1]
        return (volatility < 0.02 and  # Low volatility
                recent_range < 0.15 and  # Tight trading range
                volume_trend > 0)  # Increasing volume
    
    def _is_markup_phase(self, df: pd.DataFrame, price_trend: float, volume_trend: float) -> bool:
        """Check if in markup phase"""
        return (price_trend > 0.05 and  # Strong uptrend
                volume_trend > 0)  # Confirming volume
    
    def _is_distribution_phase(self, df: pd.DataFrame, volatility: float, volume_trend: float) -> bool:
        """Check if in distribution phase"""
        recent_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[-1]
        return (volatility > 0.01 and  # Increased volatility
                recent_range < 0.20 and  # Still rangebound
                volume_trend > 0)  # High volume
    
    def _is_markdown_phase(self, df: pd.DataFrame, price_trend: float, volume_trend: float) -> bool:
        """Check if in markdown phase"""
        return (price_trend < -0.05 and  # Strong downtrend
                volume_trend > 0)  # Confirming volume
    
    def _calculate_confidence(self, df: pd.DataFrame, phase: WyckoffPhase) -> float:
        """Calculate confidence in phase identification"""
        if phase == WyckoffPhase.UNKNOWN:
            return 0.0
            
        # Multiple confirmation factors
        volume_confirmation = self._check_volume_confirmation(df, phase)
        price_action_confirmation = self._check_price_action_confirmation(df, phase)
        
        confidence = 0.5  # Base confidence
        if volume_confirmation:
            confidence += 0.3
        if price_action_confirmation:
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    def _check_volume_confirmation(self, df: pd.DataFrame, phase: WyckoffPhase) -> bool:
        """Check if volume confirms the identified phase"""
        recent_volume = df['volume'].tail(10).mean()
        avg_volume = df['volume'].tail(50).mean()
        
        if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.DISTRIBUTION]:
            return recent_volume > avg_volume * 1.2  # Above average volume
        elif phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            return recent_volume > avg_volume  # Confirming volume
        return False
    
    def _check_price_action_confirmation(self, df: pd.DataFrame, phase: WyckoffPhase) -> bool:
        """Check if price action confirms the phase"""
        recent_closes = df['close'].tail(5)
        trend = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        
        if phase == WyckoffPhase.MARKUP:
            return trend > 0.02  # Uptrend confirmation
        elif phase == WyckoffPhase.MARKDOWN:
            return trend < -0.02  # Downtrend confirmation
        return True  # Less strict for accumulation/distribution
    
    def _calculate_price_action_strength(self, df: pd.DataFrame) -> float:
        """Calculate price action strength (0-1)"""
        recent_data = df.tail(20)
        
        # Calculate various strength metrics
        trend_strength = abs((recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) 
                            / recent_data['close'].iloc[0])
        volume_strength = (recent_data['volume'].tail(5).mean() / 
                          recent_data['volume'].mean())
        
        return min((trend_strength + volume_strength) / 2, 1.0)
    
    def _detect_key_events(self, df: pd.DataFrame, phase: WyckoffPhase) -> List[WyckoffEvent]:
        """Detect key Wyckoff events in recent data - Enhanced with all signals"""
        events = []
        recent_data = df.tail(30)
        
        if phase == WyckoffPhase.ACCUMULATION:
            # All accumulation phase signals
            if self._detect_preliminary_support(recent_data):
                events.append(WyckoffEvent.PS)
            if self._detect_selling_climax(recent_data):
                events.append(WyckoffEvent.SC)
            if self._detect_automatic_rally(recent_data):
                events.append(WyckoffEvent.AR)
            if self._detect_secondary_test(recent_data):
                events.append(WyckoffEvent.ST)
            if self._detect_sign_of_strength(recent_data):
                events.append(WyckoffEvent.SOS)
            if self._detect_last_point_support(recent_data):
                events.append(WyckoffEvent.LPS)
                
        elif phase == WyckoffPhase.DISTRIBUTION:
            # All distribution phase signals
            if self._detect_preliminary_supply(recent_data):
                events.append(WyckoffEvent.PSY)
            if self._detect_buying_climax(recent_data):
                events.append(WyckoffEvent.BC)
            if self._detect_automatic_reaction(recent_data):
                events.append(WyckoffEvent.AD)
            if self._detect_upthrust(recent_data):
                events.append(WyckoffEvent.UT)
            if self._detect_sign_of_weakness(recent_data):
                events.append(WyckoffEvent.SOW)
            if self._detect_last_point_supply(recent_data):
                events.append(WyckoffEvent.LPSY)
                
        elif phase == WyckoffPhase.MARKUP:
            # Markup phase signals
            if self._detect_sign_of_strength(recent_data):
                events.append(WyckoffEvent.SOS)
                
        elif phase == WyckoffPhase.MARKDOWN:
            # Markdown phase signals  
            if self._detect_sign_of_weakness(recent_data):
                events.append(WyckoffEvent.SOW)
        
        return events
    
    def _detect_selling_climax(self, df: pd.DataFrame) -> bool:
        """Detect selling climax pattern"""
        # High volume, sharp price drop
        max_volume_idx = df['volume'].idxmax()
        max_volume_row = df.loc[max_volume_idx]
        
        return (max_volume_row['volume'] > df['volume'].mean() * 2 and
                max_volume_row['close'] < max_volume_row['open'])
    
    def _detect_buying_climax(self, df: pd.DataFrame) -> bool:
        """Detect buying climax pattern"""
        # High volume, sharp price rise followed by weakness
        max_volume_idx = df['volume'].idxmax()
        max_volume_row = df.loc[max_volume_idx]
        
        return (max_volume_row['volume'] > df['volume'].mean() * 2 and
                max_volume_row['close'] > max_volume_row['open'])
    
    def _detect_secondary_test(self, df: pd.DataFrame) -> bool:
        """Detect secondary test pattern"""
        lows = df['low'].tail(10)
        return len(lows[lows <= lows.min() * 1.02]) >= 2  # Multiple tests of lows
    
    def _detect_upthrust(self, df: pd.DataFrame) -> bool:
        """Detect upthrust pattern"""
        recent = df.tail(5)
        return (recent['high'].iloc[-1] > recent['high'].iloc[:-1].max() and
                recent['close'].iloc[-1] < recent['close'].iloc[-2])
    
    # === NEW WYCKOFF SIGNAL DETECTION METHODS ===
    
    def _detect_preliminary_support(self, df: pd.DataFrame) -> bool:
        """Detect preliminary support - First signs of demand after decline"""
        if len(df) < 10:
            return False
        
        recent = df.tail(10)
        # Look for increased volume on bounce from low
        low_idx = recent['low'].idxmin()
        low_row = recent.loc[low_idx]
        
        # Check if volume is above average and price bounced
        avg_volume = df['volume'].tail(20).mean()
        return (low_row['volume'] > avg_volume * 1.3 and
                low_row['close'] > low_row['low'] and
                recent['close'].iloc[-1] > low_row['close'])
    
    def _detect_automatic_rally(self, df: pd.DataFrame) -> bool:
        """Detect automatic rally - Strong bounce after selling climax"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        # Find potential selling climax
        max_volume_idx = recent['volume'].idxmax()
        sc_row = recent.loc[max_volume_idx]
        
        # Check for strong rally after climax
        post_sc = recent.loc[max_volume_idx:].tail(5)
        if len(post_sc) < 3:
            return False
            
        rally_strength = (post_sc['close'].iloc[-1] - sc_row['low']) / sc_row['low']
        return (sc_row['volume'] > recent['volume'].mean() * 2 and
                rally_strength > 0.05 and  # At least 5% rally
                post_sc['volume'].mean() > recent['volume'].mean())
    
    def _detect_sign_of_strength(self, df: pd.DataFrame) -> bool:
        """Detect sign of strength - Price advance on increasing volume"""
        if len(df) < 10:
            return False
        
        recent = df.tail(10)
        mid_point = len(recent) // 2
        
        # Compare first half vs second half
        first_half = recent.iloc[:mid_point]
        second_half = recent.iloc[mid_point:]
        
        price_advance = (second_half['close'].mean() - first_half['close'].mean()) / first_half['close'].mean()
        volume_increase = second_half['volume'].mean() / first_half['volume'].mean()
        
        return (price_advance > 0.02 and  # At least 2% price advance
                volume_increase > 1.1)   # Volume increased
    
    def _detect_last_point_support(self, df: pd.DataFrame) -> bool:
        """Detect last point of support - Final test before markup"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        # Look for test of previous support with lighter volume
        support_level = recent['low'].rolling(window=5).min().min()
        current_low = recent['low'].iloc[-1]
        
        # Check if current low is near support with lighter volume
        near_support = abs(current_low - support_level) / support_level < 0.02
        avg_volume = recent['volume'].mean()
        recent_volume = recent['volume'].tail(3).mean()
        
        return (near_support and
                recent_volume < avg_volume * 0.8 and  # Lighter volume
                recent['close'].iloc[-1] > current_low)  # Closed above low
    
    def _detect_preliminary_supply(self, df: pd.DataFrame) -> bool:
        """Detect preliminary supply - First signs of selling after advance"""
        if len(df) < 10:
            return False
        
        recent = df.tail(10)
        # Look for increased volume on rejection from high
        high_idx = recent['high'].idxmax()
        high_row = recent.loc[high_idx]
        
        # Check if volume is above average and price rejected
        avg_volume = df['volume'].tail(20).mean()
        return (high_row['volume'] > avg_volume * 1.3 and
                high_row['close'] < high_row['high'] and
                recent['close'].iloc[-1] < high_row['close'])
    
    def _detect_automatic_reaction(self, df: pd.DataFrame) -> bool:
        """Detect automatic reaction - Sharp decline after buying climax"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        # Find potential buying climax
        max_volume_idx = recent['volume'].idxmax()
        bc_row = recent.loc[max_volume_idx]
        
        # Check for sharp reaction after climax
        post_bc = recent.loc[max_volume_idx:].tail(5)
        if len(post_bc) < 3:
            return False
            
        decline_strength = (bc_row['high'] - post_bc['close'].iloc[-1]) / bc_row['high']
        return (bc_row['volume'] > recent['volume'].mean() * 2 and
                decline_strength > 0.05 and  # At least 5% decline
                post_bc['volume'].mean() > recent['volume'].mean())
    
    def _detect_sign_of_weakness(self, df: pd.DataFrame) -> bool:
        """Detect sign of weakness - Price decline on increasing volume"""
        if len(df) < 10:
            return False
        
        recent = df.tail(10)
        mid_point = len(recent) // 2
        
        # Compare first half vs second half
        first_half = recent.iloc[:mid_point]
        second_half = recent.iloc[mid_point:]
        
        price_decline = (first_half['close'].mean() - second_half['close'].mean()) / first_half['close'].mean()
        volume_increase = second_half['volume'].mean() / first_half['volume'].mean()
        
        return (price_decline > 0.02 and  # At least 2% price decline
                volume_increase > 1.1)   # Volume increased
    
    def _detect_last_point_supply(self, df: pd.DataFrame) -> bool:
        """Detect last point of supply - Final rally before markdown"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        # Look for test of previous resistance with lighter volume
        resistance_level = recent['high'].rolling(window=5).max().max()
        current_high = recent['high'].iloc[-1]
        
        # Check if current high is near resistance with lighter volume
        near_resistance = abs(current_high - resistance_level) / resistance_level < 0.02
        avg_volume = recent['volume'].mean()
        recent_volume = recent['volume'].tail(3).mean()
        
        return (near_resistance and
                recent_volume < avg_volume * 0.8 and  # Lighter volume
                recent['close'].iloc[-1] < current_high)  # Closed below high
    
    def _find_support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Find key support and resistance levels"""
        recent_data = df.tail(50)
        
        # Find pivot points
        highs = recent_data['high']
        lows = recent_data['low']
        
        resistance = highs.rolling(window=5).max().max()
        support = lows.rolling(window=5).min().min()
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': df['close'].iloc[-1]
        }
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend (positive = increasing)"""
        recent_volume = df['volume'].tail(10).mean()
        older_volume = df['volume'].iloc[-20:-10].mean()
        return (recent_volume - older_volume) / older_volume
    
    def _calculate_price_trend(self, df: pd.DataFrame) -> float:
        """Calculate price trend (positive = uptrend)"""
        return (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        price_trend = self._calculate_price_trend(df)
        volume_trend = self._calculate_volume_trend(df)
        
        # Combine price and volume trends
        return min(abs(price_trend) + abs(volume_trend * 0.5), 1.0)
    
    def _create_unknown_analysis(self, symbol: str, timeframe: str) -> WyckoffAnalysis:
        """Create analysis for unknown/insufficient data"""
        return WyckoffAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            phase=WyckoffPhase.UNKNOWN,
            confidence=0.0,
            volume_confirmation=False,
            price_action_strength=0.0,
            key_events=[],
            support_resistance_levels={},
            trend_strength=0.0
        )