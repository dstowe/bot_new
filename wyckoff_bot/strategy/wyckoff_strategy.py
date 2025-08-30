# wyckoff_bot/strategy/wyckoff_strategy.py
"""
Wyckoff Trading Strategy Implementation
======================================
Main strategy logic based on Wyckoff methodology
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..analysis.wyckoff_analyzer import WyckoffAnalyzer, WyckoffAnalysis, WyckoffPhase, WyckoffEvent
from ..analysis.volume_analysis import VolumeAnalyzer
from ..analysis.price_action import PriceActionAnalyzer

class TradeAction(Enum):
    """Trade actions - Long only strategy"""
    BUY = "buy"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out" 
    CLOSE_LONG = "close_long"
    HOLD = "hold"

@dataclass
class TradeSignal:
    """Complete trade signal"""
    symbol: str
    action: TradeAction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timeframe: str

class WyckoffStrategy:
    """
    Main Wyckoff trading strategy
    Generates trade signals based on Wyckoff analysis
    """
    
    def __init__(self, min_confidence: float = 0.6, risk_reward_ratio: float = 2.0):
        self.min_confidence = min_confidence
        self.risk_reward_ratio = risk_reward_ratio
        
        # Initialize analyzers
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.price_analyzer = PriceActionAnalyzer()
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, 
                       current_positions: Dict = None) -> Optional[TradeSignal]:
        """
        Generate trading signal based on Wyckoff analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            current_positions: Current positions for this symbol
            
        Returns:
            TradeSignal: Trade signal or None if no signal
        """
        # Perform comprehensive Wyckoff analysis
        analysis = self.wyckoff_analyzer.analyze(df, symbol)
        
        # Skip if confidence too low
        if analysis.confidence < self.min_confidence:
            return None
        
        # Determine trade action based on phase and events
        trade_action = self._determine_trade_action(analysis, current_positions)
        
        if trade_action == TradeAction.HOLD:
            return None
        
        # Calculate entry, stop loss, and take profit
        entry_price = round(df['close'].iloc[-1], 2)  # Round to 2 decimal places for Webull API
        stop_loss, take_profit = self._calculate_levels(
            df, analysis, trade_action, entry_price
        )
        
        # Calculate position size (will be refined by risk manager)
        position_size = self._calculate_base_position_size(analysis)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(analysis, trade_action)
        
        return TradeSignal(
            symbol=symbol,
            action=trade_action,
            confidence=analysis.confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=reasoning,
            timeframe="1D"
        )
    
    def _determine_trade_action(self, analysis: WyckoffAnalysis, 
                               current_positions: Dict = None) -> TradeAction:
        """Determine appropriate trade action based on analysis - Long only strategy"""
        phase = analysis.phase
        key_events = analysis.key_events
        
        # Handle existing positions first
        if current_positions:
            return self._handle_existing_positions(analysis, current_positions)
        
        # New position logic based on phase and events - LONG ONLY
        if phase == WyckoffPhase.ACCUMULATION:
            # All accumulation signals for buying
            if (WyckoffEvent.PS in key_events or   # Preliminary support
                WyckoffEvent.SC in key_events or   # Selling climax
                WyckoffEvent.AR in key_events or   # Automatic rally
                WyckoffEvent.ST in key_events or   # Secondary test
                WyckoffEvent.SOS in key_events or  # Sign of strength
                WyckoffEvent.LPS in key_events):   # Last point of support
                return TradeAction.BUY
                
        elif phase == WyckoffPhase.MARKUP:
            # Continue with uptrend
            if (analysis.trend_strength > 0.7 and
                analysis.volume_confirmation):
                return TradeAction.BUY
                
        elif phase == WyckoffPhase.DISTRIBUTION:
            # No new long positions in distribution
            return TradeAction.HOLD
                
        elif phase == WyckoffPhase.MARKDOWN:
            # No new long positions in markdown
            return TradeAction.HOLD
        
        return TradeAction.HOLD
    
    def _handle_existing_positions(self, analysis: WyckoffAnalysis, 
                                  positions: Dict) -> TradeAction:
        """Handle existing positions based on current analysis - Long only strategy"""
        phase = analysis.phase
        key_events = analysis.key_events
        
        if positions.get('long_position'):
            position_size = positions['long_position'].get('size', 0)
            
            # Distribution phase - Scale out or close
            if phase == WyckoffPhase.DISTRIBUTION:
                if (WyckoffEvent.PSY in key_events or   # Preliminary supply
                    WyckoffEvent.BC in key_events or    # Buying climax
                    WyckoffEvent.UT in key_events):     # Upthrust
                    return TradeAction.SCALE_OUT
                elif WyckoffEvent.LPSY in key_events:   # Last point of supply
                    return TradeAction.CLOSE_LONG
                    
            # Markdown phase - Close position
            elif phase == WyckoffPhase.MARKDOWN:
                if (WyckoffEvent.AD in key_events or    # Automatic reaction
                    WyckoffEvent.SOW in key_events):    # Sign of weakness
                    return TradeAction.CLOSE_LONG
                    
            # Accumulation phase with existing position - Consider scaling in
            elif phase == WyckoffPhase.ACCUMULATION:
                if (position_size < 1.0 and  # Not at full position
                    (WyckoffEvent.ST in key_events or   # Secondary test
                     WyckoffEvent.LPS in key_events)):  # Last point of support
                    return TradeAction.SCALE_IN
                    
            # Markup phase - Hold or scale in on pullbacks
            elif phase == WyckoffPhase.MARKUP:
                if (position_size < 1.0 and
                    analysis.trend_strength > 0.6 and
                    analysis.volume_confirmation):
                    return TradeAction.SCALE_IN
        
        return TradeAction.HOLD
    
    def _calculate_levels(self, df: pd.DataFrame, analysis: WyckoffAnalysis, 
                         action: TradeAction, entry_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels - Long only strategy"""
        sr_levels = analysis.support_resistance_levels
        
        if action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            # Long positions only
            support = sr_levels.get('support', entry_price * 0.95)
            stop_loss = support * 0.99  # Slightly below support
            take_profit = entry_price + (entry_price - stop_loss) * self.risk_reward_ratio
            
        elif action == TradeAction.SCALE_OUT:
            # Partial profit taking
            resistance = sr_levels.get('resistance', entry_price * 1.05)
            stop_loss = entry_price * 0.98  # Tight stop for remaining position
            take_profit = resistance * 0.99  # Just below resistance
            
        else:
            # For close positions or hold, use current price
            stop_loss = entry_price
            take_profit = entry_price
        
        return round(stop_loss, 2), round(take_profit, 2)  # Round to 2 decimal places for Webull API
    
    def _calculate_base_position_size(self, analysis: WyckoffAnalysis) -> float:
        """Calculate base position size based on confidence"""
        # Base size scales with confidence
        base_size = 0.1 + (analysis.confidence - 0.6) * 0.2  # 0.1 to 0.18
        
        # Adjust for volume confirmation
        if analysis.volume_confirmation:
            base_size *= 1.2
        
        # Adjust for trend strength
        base_size *= (0.8 + analysis.trend_strength * 0.4)
        
        return min(base_size, 0.25)  # Cap at 25%
    
    def _generate_reasoning(self, analysis: WyckoffAnalysis, action: TradeAction) -> str:
        """Generate human-readable reasoning for the trade"""
        phase_desc = analysis.phase.value.replace('_', ' ').title()
        confidence_pct = int(analysis.confidence * 100)
        
        reasoning = f"{phase_desc} phase detected with {confidence_pct}% confidence. "
        
        if analysis.key_events:
            events_desc = [event.value.replace('_', ' ').title() for event in analysis.key_events]
            reasoning += f"Key events: {', '.join(events_desc)}. "
        
        if analysis.volume_confirmation:
            reasoning += "Volume confirms the move. "
        
        if action == TradeAction.BUY:
            reasoning += "Entering long position based on bullish Wyckoff signals."
        elif action == TradeAction.SCALE_IN:
            reasoning += "Scaling into existing long position on Wyckoff confirmation."
        elif action == TradeAction.SCALE_OUT:
            reasoning += "Scaling out of long position based on distribution signals."
        elif action == TradeAction.CLOSE_LONG:
            reasoning += "Closing long position based on bearish Wyckoff signals."
        
        return reasoning
    
    def validate_signal(self, signal: TradeSignal, account_balance: float, 
                       risk_per_trade: float = 0.02) -> bool:
        """
        Validate signal against risk management rules
        
        Args:
            signal: Trade signal to validate
            account_balance: Current account balance
            risk_per_trade: Maximum risk per trade as fraction
            
        Returns:
            bool: True if signal is valid
        """
        # Calculate risk amount - Long only trades
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
            max_risk = account_balance * risk_per_trade
            
            # Check if risk is acceptable
            if risk_amount > max_risk:
                return False
        
        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            return False
        
        # Check risk-reward ratio for entry trades
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            profit_potential = abs(signal.take_profit - signal.entry_price)
            risk_amount = abs(signal.entry_price - signal.stop_loss)
            
            if risk_amount > 0:
                actual_rr_ratio = profit_potential / risk_amount
                if actual_rr_ratio < self.risk_reward_ratio * 0.8:  # 20% tolerance
                    return False
        
        return True
    
    def get_strategy_status(self) -> Dict[str, any]:
        """Get current strategy status and parameters"""
        return {
            'strategy_name': 'Wyckoff Method',
            'min_confidence': self.min_confidence,
            'risk_reward_ratio': self.risk_reward_ratio,
            'version': '1.0'
        }