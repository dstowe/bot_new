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
import yfinance as yf
from datetime import datetime, timedelta
import logging

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
    point_figure_signals: Dict[str, any] = None
    multi_timeframe_confirmation: Dict[str, bool] = None
    institutional_flow: Dict[str, float] = None

class WyckoffAnalyzer:
    """
    Enhanced Wyckoff analysis engine with multi-timeframe and institutional features
    Analyzes price and volume data to identify Wyckoff structures across timeframes
    """
    
    def __init__(self, lookback_periods: int = 50, logger: logging.Logger = None):
        self.lookback_periods = lookback_periods
        self.volume_analyzer = None  # Will be set by import
        self.price_analyzer = None   # Will be set by import
        self.logger = logger or logging.getLogger(__name__)
        
        # Multi-timeframe configuration
        self.timeframes = {
            'daily': '1D',
            'hourly_4': '4H', 
            'hourly_1': '1H',
            'minutes_15': '15M'
        }
        
        # Point & Figure settings
        self.pf_box_size = 0.02  # 2% box size
        self.pf_reversal = 3     # 3-box reversal
        
        # Multi-timeframe analysis weights
        self.timeframe_weights = {
            '1D': 1.0,   # Primary trend
            '4H': 0.7,   # Intermediate trend  
            '1H': 0.5,   # Short-term trend
            '15M': 0.3   # Entry timing
        }
        
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
        
        # Enhanced analysis with institutional features
        pf_signals = self._analyze_point_figure(df)
        institutional_flow = self._analyze_institutional_flow(df)
        
        return WyckoffAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            phase=phase,
            confidence=confidence,
            volume_confirmation=volume_confirmation,
            price_action_strength=price_strength,
            key_events=key_events,
            support_resistance_levels=sr_levels,
            trend_strength=trend_strength,
            point_figure_signals=pf_signals,
            institutional_flow=institutional_flow
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
    
    def analyze_multi_timeframe_legacy(self, symbol: str) -> Dict[str, WyckoffAnalysis]:
        """
        Perform multi-timeframe Wyckoff analysis
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dict[str, WyckoffAnalysis]: Analysis for each timeframe
        """
        analyses = {}
        
        try:
            ticker = yf.Ticker(symbol)
            
            for tf_name, tf_period in self.timeframes.items():
                try:
                    # Get appropriate period for timeframe
                    period = self._get_period_for_timeframe(tf_name)
                    interval = self._get_interval_for_timeframe(tf_name)
                    
                    # Download data
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty and len(df) >= 20:
                        # Standardize column names
                        df = df.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })
                        
                        # Perform analysis
                        analysis = self.analyze(df, symbol, tf_name)
                        analyses[tf_name] = analysis
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol} on {tf_name}: {e}")
                    
            # Add multi-timeframe confirmation to each analysis
            for tf_name, analysis in analyses.items():
                analysis.multi_timeframe_confirmation = self._calculate_mtf_confirmation(analyses, tf_name)
                
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            
        return analyses
    
    def _get_period_for_timeframe(self, timeframe: str) -> str:
        """Get appropriate data period for timeframe"""
        periods = {
            'weekly': '2y',
            'daily': '1y', 
            'hourly_4': '6mo',
            'hourly_1': '3mo'
        }
        return periods.get(timeframe, '1y')
    
    def _get_interval_for_timeframe(self, timeframe: str) -> str:
        """Get yfinance interval for timeframe"""
        intervals = {
            'weekly': '1wk',
            'daily': '1d',
            'hourly_4': '4h', 
            'hourly_1': '1h'
        }
        return intervals.get(timeframe, '1d')
    
    def _calculate_mtf_confirmation(self, analyses: Dict[str, WyckoffAnalysis], current_tf: str) -> Dict[str, bool]:
        """Calculate multi-timeframe confirmation signals"""
        confirmation = {}
        
        if current_tf not in analyses:
            return confirmation
            
        current_phase = analyses[current_tf].phase
        
        # Check confirmation from higher timeframes
        timeframe_hierarchy = ['weekly', 'daily', 'hourly_4', 'hourly_1']
        current_idx = timeframe_hierarchy.index(current_tf) if current_tf in timeframe_hierarchy else -1
        
        if current_idx >= 0:
            # Check higher timeframes
            for i in range(current_idx):
                higher_tf = timeframe_hierarchy[i]
                if higher_tf in analyses:
                    higher_phase = analyses[higher_tf].phase
                    confirmation[f'{higher_tf}_confirms'] = self._phases_align(current_phase, higher_phase)
            
            # Check lower timeframes
            for i in range(current_idx + 1, len(timeframe_hierarchy)):
                lower_tf = timeframe_hierarchy[i]
                if lower_tf in analyses:
                    lower_phase = analyses[lower_tf].phase
                    confirmation[f'{lower_tf}_confirms'] = self._phases_align(current_phase, lower_phase)
                    
        return confirmation
    
    def _phases_align(self, phase1: WyckoffPhase, phase2: WyckoffPhase) -> bool:
        """Check if two phases align/support each other"""
        # Define compatible phases
        compatible_phases = {
            WyckoffPhase.ACCUMULATION: [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP],
            WyckoffPhase.MARKUP: [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP],
            WyckoffPhase.DISTRIBUTION: [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN],
            WyckoffPhase.MARKDOWN: [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]
        }
        
        return phase2 in compatible_phases.get(phase1, [])
    
    def _analyze_point_figure(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze Point & Figure chart patterns
        
        Returns:
            Dict: Point & Figure signals and patterns
        """
        if len(df) < 20:
            return {}
            
        try:
            # Create Point & Figure chart data
            pf_data = self._create_point_figure_chart(df)
            
            if not pf_data:
                return {}
                
            # Analyze patterns
            patterns = self._detect_pf_patterns(pf_data)
            signals = self._generate_pf_signals(pf_data, patterns)
            
            return {
                'chart_data': pf_data[-20:],  # Last 20 columns
                'patterns': patterns,
                'signals': signals,
                'trend_direction': self._get_pf_trend(pf_data),
                'support_resistance': self._get_pf_support_resistance(pf_data)
            }
            
        except Exception as e:
            self.logger.warning(f"Error in Point & Figure analysis: {e}")
            return {}
    
    def _create_point_figure_chart(self, df: pd.DataFrame) -> List[Dict]:
        """Create Point & Figure chart from OHLC data"""
        pf_chart = []
        current_column = None
        
        for _, row in df.iterrows():
            high, low = row['high'], row['low']
            
            if current_column is None:
                # Start first column
                current_column = {
                    'type': 'X',  # Start with X (bullish)
                    'boxes': [self._price_to_box(row['close'])],
                    'start_price': row['close'],
                    'date': row.name
                }
                continue
            
            # Check for continuation or reversal
            if current_column['type'] == 'X':
                # In bullish column
                high_box = self._price_to_box(high)
                low_box = self._price_to_box(low)
                
                if high_box > current_column['boxes'][-1]:
                    # Continue up
                    current_column['boxes'].extend(range(current_column['boxes'][-1] + 1, high_box + 1))
                
                # Check for reversal
                if low_box <= current_column['boxes'][-1] - self.pf_reversal:
                    # Reversal to O column
                    pf_chart.append(current_column)
                    current_column = {
                        'type': 'O',
                        'boxes': list(range(low_box, current_column['boxes'][-1] - self.pf_reversal + 1)),
                        'start_price': self._box_to_price(current_column['boxes'][-1] - self.pf_reversal),
                        'date': row.name
                    }
            else:
                # In bearish column
                high_box = self._price_to_box(high)
                low_box = self._price_to_box(low)
                
                if low_box < current_column['boxes'][0]:
                    # Continue down
                    new_boxes = list(range(low_box, current_column['boxes'][0]))
                    current_column['boxes'] = new_boxes + current_column['boxes']
                
                # Check for reversal
                if high_box >= current_column['boxes'][0] + self.pf_reversal:
                    # Reversal to X column
                    pf_chart.append(current_column)
                    current_column = {
                        'type': 'X',
                        'boxes': list(range(current_column['boxes'][0] + self.pf_reversal, high_box + 1)),
                        'start_price': self._box_to_price(current_column['boxes'][0] + self.pf_reversal),
                        'date': row.name
                    }
        
        # Add final column
        if current_column:
            pf_chart.append(current_column)
            
        return pf_chart
    
    def _price_to_box(self, price: float) -> int:
        """Convert price to box number"""
        return int(np.log(price) / self.pf_box_size)
    
    def _box_to_price(self, box: int) -> float:
        """Convert box number to price"""
        return np.exp(box * self.pf_box_size)
    
    def _detect_pf_patterns(self, pf_data: List[Dict]) -> List[str]:
        """Detect Point & Figure patterns"""
        patterns = []
        
        if len(pf_data) < 3:
            return patterns
            
        # Simple pattern detection
        recent_columns = pf_data[-5:]  # Last 5 columns
        
        # Double top/bottom patterns
        if len(recent_columns) >= 3:
            if (recent_columns[-3]['type'] == 'X' and 
                recent_columns[-1]['type'] == 'X' and
                max(recent_columns[-3]['boxes']) == max(recent_columns[-1]['boxes'])):
                patterns.append('double_top')
                
            if (recent_columns[-3]['type'] == 'O' and 
                recent_columns[-1]['type'] == 'O' and
                min(recent_columns[-3]['boxes']) == min(recent_columns[-1]['boxes'])):
                patterns.append('double_bottom')
        
        # Triple top/bottom patterns  
        if len(recent_columns) >= 5:
            x_columns = [col for col in recent_columns if col['type'] == 'X']
            o_columns = [col for col in recent_columns if col['type'] == 'O']
            
            if len(x_columns) >= 3:
                highs = [max(col['boxes']) for col in x_columns[-3:]]
                if len(set(highs)) == 1:  # All same height
                    patterns.append('triple_top')
                    
            if len(o_columns) >= 3:
                lows = [min(col['boxes']) for col in o_columns[-3:]]
                if len(set(lows)) == 1:  # All same depth
                    patterns.append('triple_bottom')
        
        return patterns
    
    def _generate_pf_signals(self, pf_data: List[Dict], patterns: List[str]) -> Dict[str, any]:
        """Generate trading signals from Point & Figure analysis"""
        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'strength': 0.0
        }
        
        if not pf_data:
            return signals
            
        # Pattern-based signals
        if 'double_bottom' in patterns or 'triple_bottom' in patterns:
            signals['bullish_signals'].append('accumulation_pattern')
            signals['strength'] += 0.3
            
        if 'double_top' in patterns or 'triple_top' in patterns:
            signals['bearish_signals'].append('distribution_pattern')
            signals['strength'] += 0.3
        
        # Trend signals
        if len(pf_data) >= 2:
            last_column = pf_data[-1]
            if last_column['type'] == 'X':
                # Check for bullish breakout
                if len(pf_data) >= 3:
                    prev_x_columns = [col for col in pf_data[-5:] if col['type'] == 'X'][:-1]
                    if prev_x_columns:
                        max_prev_high = max(max(col['boxes']) for col in prev_x_columns)
                        if max(last_column['boxes']) > max_prev_high:
                            signals['bullish_signals'].append('breakout')
                            signals['strength'] += 0.4
            
            elif last_column['type'] == 'O':
                # Check for bearish breakdown
                if len(pf_data) >= 3:
                    prev_o_columns = [col for col in pf_data[-5:] if col['type'] == 'O'][:-1]
                    if prev_o_columns:
                        min_prev_low = min(min(col['boxes']) for col in prev_o_columns)
                        if min(last_column['boxes']) < min_prev_low:
                            signals['bearish_signals'].append('breakdown')
                            signals['strength'] += 0.4
        
        return signals
    
    def _get_pf_trend(self, pf_data: List[Dict]) -> str:
        """Get Point & Figure trend direction"""
        if len(pf_data) < 2:
            return 'neutral'
            
        # Simple trend based on last few columns
        recent = pf_data[-3:] if len(pf_data) >= 3 else pf_data
        
        bullish_count = sum(1 for col in recent if col['type'] == 'X')
        bearish_count = sum(1 for col in recent if col['type'] == 'O')
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_pf_support_resistance(self, pf_data: List[Dict]) -> Dict[str, float]:
        """Get Point & Figure support and resistance levels"""
        if not pf_data:
            return {}
            
        all_highs = []
        all_lows = []
        
        for col in pf_data[-10:]:  # Last 10 columns
            if col['type'] == 'X':
                all_highs.extend(col['boxes'])
            else:
                all_lows.extend(col['boxes'])
        
        resistance = self._box_to_price(max(all_highs)) if all_highs else None
        support = self._box_to_price(min(all_lows)) if all_lows else None
        
        return {
            'resistance': resistance,
            'support': support
        }
    
    def _analyze_institutional_flow(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze institutional money flow indicators
        
        Returns:
            Dict: Institutional flow metrics
        """
        if len(df) < 20:
            return {}
            
        try:
            # Calculate various institutional flow indicators
            flow_metrics = {}
            
            # 1. Large Volume Days (potential institutional activity)
            avg_volume = df['volume'].tail(50).mean()
            large_volume_threshold = avg_volume * 2
            recent_large_volume_days = df.tail(20)['volume'] > large_volume_threshold
            flow_metrics['large_volume_frequency'] = recent_large_volume_days.sum() / len(recent_large_volume_days)
            
            # 2. Volume-Price Relationship (VPR)
            recent_data = df.tail(20)
            price_changes = recent_data['close'].pct_change()
            volume_changes = recent_data['volume'].pct_change()
            
            # Correlation between volume and price changes
            if len(price_changes) > 1 and price_changes.std() > 0 and volume_changes.std() > 0:
                vpr_correlation = price_changes.corr(volume_changes)
                flow_metrics['volume_price_correlation'] = vpr_correlation if not np.isnan(vpr_correlation) else 0
            else:
                flow_metrics['volume_price_correlation'] = 0
            
            # 3. Accumulation/Distribution Line (A/D Line)
            money_flow_multiplier = ((recent_data['close'] - recent_data['low']) - 
                                   (recent_data['high'] - recent_data['close'])) / (recent_data['high'] - recent_data['low'])
            money_flow_volume = money_flow_multiplier * recent_data['volume']
            ad_line = money_flow_volume.cumsum()
            flow_metrics['ad_line_trend'] = (ad_line.iloc[-1] - ad_line.iloc[0]) / abs(ad_line.iloc[0]) if ad_line.iloc[0] != 0 else 0
            
            # 4. On-Balance Volume (OBV) trend
            obv = []
            obv_value = 0
            for i in range(1, len(recent_data)):
                if recent_data['close'].iloc[i] > recent_data['close'].iloc[i-1]:
                    obv_value += recent_data['volume'].iloc[i]
                elif recent_data['close'].iloc[i] < recent_data['close'].iloc[i-1]:
                    obv_value -= recent_data['volume'].iloc[i]
                obv.append(obv_value)
            
            if len(obv) > 10:
                obv_series = pd.Series(obv)
                obv_trend = (obv_series.iloc[-1] - obv_series.iloc[0]) / abs(obv_series.iloc[0]) if obv_series.iloc[0] != 0 else 0
                flow_metrics['obv_trend'] = obv_trend
            else:
                flow_metrics['obv_trend'] = 0
            
            # 5. Institutional Activity Score (composite)
            institutional_score = (
                flow_metrics['large_volume_frequency'] * 0.3 +
                abs(flow_metrics['volume_price_correlation']) * 0.2 +
                abs(flow_metrics['ad_line_trend']) * 0.3 +
                abs(flow_metrics['obv_trend']) * 0.2
            )
            flow_metrics['institutional_activity_score'] = min(institutional_score, 1.0)
            
            return flow_metrics
            
        except Exception as e:
            self.logger.warning(f"Error in institutional flow analysis: {e}")
            return {}
    
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
            trend_strength=0.0,
            point_figure_signals={},
            institutional_flow={}
        )
    
    # Multi-timeframe analysis methods
    
    def analyze_multi_timeframe(self, data_by_timeframe: Dict[str, pd.DataFrame], 
                               symbol: str) -> Dict[str, WyckoffAnalysis]:
        """
        Perform Wyckoff analysis across multiple timeframes
        
        Args:
            data_by_timeframe: Dict mapping timeframe to DataFrame
            symbol: Stock symbol
            
        Returns:
            Dict mapping timeframe to WyckoffAnalysis
        """
        results = {}
        
        for timeframe, df in data_by_timeframe.items():
            try:
                if df is not None and not df.empty:
                    analysis = self.analyze(df, symbol, timeframe)
                    results[timeframe] = analysis
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                results[timeframe] = self._create_unknown_analysis(symbol, timeframe)
        
        # Add multi-timeframe confirmation to each analysis
        for timeframe, analysis in results.items():
            analysis.multi_timeframe_confirmation = self._calculate_multi_timeframe_confirmation(
                results, timeframe
            )
        
        return results
    
    def _calculate_multi_timeframe_confirmation(self, analyses: Dict[str, WyckoffAnalysis], 
                                             current_timeframe: str) -> Dict[str, bool]:
        """Calculate multi-timeframe confirmation signals"""
        confirmations = {}
        current_analysis = analyses.get(current_timeframe)
        
        if not current_analysis:
            return confirmations
        
        current_phase = current_analysis.phase
        
        # Check phase alignment across timeframes
        for tf, analysis in analyses.items():
            if tf == current_timeframe:
                continue
                
            # Higher timeframe should provide direction context
            if self._is_higher_timeframe(tf, current_timeframe):
                confirmations[f'{tf}_direction_alignment'] = self._phases_aligned(
                    current_phase, analysis.phase, direction='higher'
                )
            
            # Lower timeframe should provide entry timing confirmation
            elif self._is_lower_timeframe(tf, current_timeframe):
                confirmations[f'{tf}_entry_confirmation'] = self._phases_aligned(
                    current_phase, analysis.phase, direction='lower'
                )
            
            # Same level timeframes should show similar patterns
            else:
                confirmations[f'{tf}_pattern_confirmation'] = (
                    current_phase == analysis.phase and 
                    abs(current_analysis.confidence - analysis.confidence) < 0.3
                )
        
        return confirmations
    
    def _is_higher_timeframe(self, tf1: str, tf2: str) -> bool:
        """Check if tf1 is higher timeframe than tf2"""
        timeframe_order = ['15M', '1H', '4H', '1D']
        try:
            return timeframe_order.index(tf1) > timeframe_order.index(tf2)
        except ValueError:
            return False
    
    def _is_lower_timeframe(self, tf1: str, tf2: str) -> bool:
        """Check if tf1 is lower timeframe than tf2"""
        timeframe_order = ['15M', '1H', '4H', '1D']
        try:
            return timeframe_order.index(tf1) < timeframe_order.index(tf2)
        except ValueError:
            return False
    
    def _phases_aligned(self, phase1: WyckoffPhase, phase2: WyckoffPhase, 
                       direction: str) -> bool:
        """Check if phases are aligned based on timeframe relationship"""
        
        if direction == 'higher':
            # Higher timeframe should provide directional bias
            if phase1 in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]:
                return phase2 in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]
            elif phase1 in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]:
                return phase2 in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]
        
        elif direction == 'lower':
            # Lower timeframe can show more granular phases within higher TF phase
            if phase1 == WyckoffPhase.ACCUMULATION:
                return phase2 in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]
            elif phase1 == WyckoffPhase.MARKUP:
                return phase2 in [WyckoffPhase.MARKUP, WyckoffPhase.DISTRIBUTION]
            elif phase1 == WyckoffPhase.DISTRIBUTION:
                return phase2 in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]
            elif phase1 == WyckoffPhase.MARKDOWN:
                return phase2 in [WyckoffPhase.MARKDOWN, WyckoffPhase.ACCUMULATION]
        
        return phase1 == phase2
    
    def get_multi_timeframe_signal(self, analyses: Dict[str, WyckoffAnalysis]) -> Dict:
        """
        Generate trading signal based on multi-timeframe analysis
        
        Returns:
            Dict with signal strength, direction, and confidence
        """
        if not analyses:
            return {'signal': 'NONE', 'strength': 0.0, 'confidence': 0.0}
        
        # Weight analyses by timeframe importance
        weighted_signals = {}
        total_weight = 0
        
        for timeframe, analysis in analyses.items():
            weight = self.timeframe_weights.get(timeframe, 0.5)
            total_weight += weight
            
            # Convert phase to signal strength
            if analysis.phase == WyckoffPhase.ACCUMULATION:
                signal_strength = 0.6  # Moderate bullish
            elif analysis.phase == WyckoffPhase.MARKUP:
                signal_strength = 1.0  # Strong bullish
            elif analysis.phase == WyckoffPhase.DISTRIBUTION:
                signal_strength = -0.6  # Moderate bearish
            elif analysis.phase == WyckoffPhase.MARKDOWN:
                signal_strength = -1.0  # Strong bearish
            else:
                signal_strength = 0.0  # Neutral
            
            # Adjust by confidence
            adjusted_strength = signal_strength * analysis.confidence
            weighted_signals[timeframe] = adjusted_strength * weight
        
        # Calculate overall signal
        if total_weight > 0:
            overall_strength = sum(weighted_signals.values()) / total_weight
        else:
            overall_strength = 0.0
        
        # Determine signal direction and strength
        if overall_strength > 0.3:
            signal_direction = 'BUY'
        elif overall_strength < -0.3:
            signal_direction = 'SELL' 
        else:
            signal_direction = 'HOLD'
        
        # Calculate confidence from timeframe agreement
        confirmations = []
        for analysis in analyses.values():
            if analysis.multi_timeframe_confirmation:
                agreement = sum(1 for conf in analysis.multi_timeframe_confirmation.values() if conf)
                total_checks = len(analysis.multi_timeframe_confirmation)
                if total_checks > 0:
                    confirmations.append(agreement / total_checks)
        
        overall_confidence = np.mean(confirmations) if confirmations else 0.5
        
        return {
            'signal': signal_direction,
            'strength': abs(overall_strength),
            'confidence': overall_confidence,
            'timeframe_signals': {tf: weighted_signals.get(tf, 0) for tf in analyses.keys()},
            'primary_phase': max(analyses.items(), key=lambda x: self.timeframe_weights.get(x[0], 0))[1].phase.value
        }
    
    def get_entry_timing_signal(self, analyses: Dict[str, WyckoffAnalysis], 
                               primary_signal: str) -> Dict:
        """
        Get precise entry timing based on lower timeframe analysis
        
        Args:
            analyses: Multi-timeframe analyses
            primary_signal: Primary signal direction (BUY/SELL)
            
        Returns:
            Dict with entry timing information
        """
        # Use 1H and 15M for entry timing
        timing_timeframes = ['1H', '15M']
        timing_analyses = {tf: analysis for tf, analysis in analyses.items() 
                         if tf in timing_timeframes}
        
        if not timing_analyses:
            return {'timing': 'WAIT', 'confidence': 0.0}
        
        # Check for pullback completion or breakout confirmation
        entry_signals = []
        
        for tf, analysis in timing_analyses.items():
            if primary_signal == 'BUY':
                # Look for pullback completion in bullish trend
                if analysis.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]:
                    # Check if we're near support and showing strength
                    sr_levels = analysis.support_resistance_levels
                    if 'support' in sr_levels and analysis.price_action_strength > 0.5:
                        entry_signals.append(('PULLBACK_COMPLETE', analysis.confidence))
                    elif analysis.key_events and WyckoffEvent.SOS in analysis.key_events:
                        entry_signals.append(('BREAKOUT_CONFIRMED', analysis.confidence))
                        
            elif primary_signal == 'SELL':
                # Look for rally completion in bearish trend
                if analysis.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]:
                    sr_levels = analysis.support_resistance_levels
                    if 'resistance' in sr_levels and analysis.price_action_strength > 0.5:
                        entry_signals.append(('RALLY_COMPLETE', analysis.confidence))
                    elif analysis.key_events and WyckoffEvent.SOW in analysis.key_events:
                        entry_signals.append(('BREAKDOWN_CONFIRMED', analysis.confidence))
        
        if entry_signals:
            # Take highest confidence signal
            best_signal, confidence = max(entry_signals, key=lambda x: x[1])
            return {
                'timing': 'ENTER',
                'signal_type': best_signal,
                'confidence': confidence
            }
        
        return {'timing': 'WAIT', 'confidence': 0.0}