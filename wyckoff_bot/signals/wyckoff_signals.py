# wyckoff_bot/signals/wyckoff_signals.py
"""
Wyckoff Signal Generator
========================
Main signal generation engine for Wyckoff trading
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from ..strategy.wyckoff_strategy import WyckoffStrategy, TradeSignal
from ..strategy.risk_management import RiskManager
from ..strategy.position_sizing import PositionSizer

@dataclass
class MarketSignal:
    """Complete market signal with all analysis"""
    symbol: str
    timestamp: datetime
    trade_signal: TradeSignal
    risk_metrics: Dict
    position_sizing: Dict
    market_data: Dict
    strength_score: float

class WyckoffSignalGenerator:
    """
    Main signal generation engine
    Orchestrates analysis, strategy, and risk management
    """
    
    def __init__(self, min_confidence: float = 0.6, 
                 risk_per_trade: float = 0.02,
                 min_trade_amount: float = 6.0,
                 logger: logging.Logger = None):
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.strategy = WyckoffStrategy(min_confidence=min_confidence)
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(
            base_risk_per_trade=risk_per_trade,
            min_trade_amount=min_trade_amount
        )
        
        # Track generated signals
        self.recent_signals = {}
        self.signal_history = []
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        account_info: Dict,
                        current_positions: Dict = None) -> List[MarketSignal]:
        """
        Generate trading signals for multiple symbols
        
        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            account_info: Account balance and information
            current_positions: Current positions by symbol
            
        Returns:
            List[MarketSignal]: Generated market signals
        """
        signals = []
        current_positions = current_positions or {}
        
        self.logger.info(f"Generating signals for {len(market_data)} symbols")
        
        for symbol, df in market_data.items():
            try:
                signal = self._generate_single_signal(
                    symbol, df, account_info, 
                    current_positions.get(symbol)
                )
                
                if signal:
                    signals.append(signal)
                    self.recent_signals[symbol] = signal
                    self.signal_history.append(signal)
                    
                    self.logger.info(f"Generated {signal.trade_signal.action.value} signal "
                                   f"for {symbol} with {signal.trade_signal.confidence:.1%} confidence")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Sort by strength score
        signals.sort(key=lambda x: x.strength_score, reverse=True)
        
        self.logger.info(f"Generated {len(signals)} total signals")
        return signals
    
    def _generate_single_signal(self, symbol: str, df: pd.DataFrame,
                               account_info: Dict, current_position: Dict = None) -> Optional[MarketSignal]:
        """Generate signal for single symbol"""
        
        # Generate base trade signal
        trade_signal = self.strategy.generate_signal(df, symbol, 
                                                   {symbol: current_position} if current_position else None)
        
        if not trade_signal:
            return None
        
        # Apply risk management
        adjusted_signal = self.risk_manager.validate_and_adjust_signal(
            trade_signal, account_info, {symbol: current_position} if current_position else None
        )
        
        if not adjusted_signal:
            self.logger.debug(f"Signal for {symbol} rejected by risk management")
            return None
        
        # Calculate position sizing
        position_sizing = self.position_sizer.calculate_position_size(
            adjusted_signal.entry_price,
            adjusted_signal.stop_loss,
            adjusted_signal.confidence,
            account_info.get('balance', 0),
            df
        )
        
        # Update signal with refined position size
        adjusted_signal.position_size = position_sizing.recommended_size
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_position_risk(
            adjusted_signal, account_info.get('balance', 0)
        )
        
        # Calculate strength score
        strength_score = self._calculate_strength_score(adjusted_signal, risk_metrics, df)
        
        # Create market data summary
        market_data_summary = self._create_market_summary(df)
        
        return MarketSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            trade_signal=adjusted_signal,
            risk_metrics=risk_metrics,
            position_sizing=position_sizing.__dict__,
            market_data=market_data_summary,
            strength_score=strength_score
        )
    
    def _calculate_strength_score(self, signal: TradeSignal, risk_metrics: Dict, 
                                 df: pd.DataFrame) -> float:
        """
        Calculate overall signal strength score (0-100)
        Combines multiple factors into single score
        """
        score = 0.0
        
        # Base score from confidence (0-40 points)
        score += signal.confidence * 40
        
        # Risk-reward ratio bonus (0-25 points)
        rr_ratio = risk_metrics.get('risk_reward_ratio', 0)
        score += min(rr_ratio * 10, 25)
        
        # Volume confirmation (0-15 points)
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume'].tail(20).mean()
        if recent_volume > avg_volume * 1.2:
            score += 15
        elif recent_volume > avg_volume:
            score += 10
        
        # Price momentum (0-10 points) - Updated for long-only strategy
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        if signal.action.value in ['buy', 'scale_in'] and price_change > 0.02:
            score += 10
        elif signal.action.value in ['scale_out', 'close_long'] and price_change < -0.02:
            score += 10  # Scaling out on weakness is good
        elif abs(price_change) < 0.01:  # Consolidation
            score += 5
        
        # Position size efficiency (0-10 points)
        if signal.position_size > 0.15:  # Good size
            score += 10
        elif signal.position_size > 0.10:
            score += 7
        elif signal.position_size > 0.05:
            score += 5
        
        return min(score, 100.0)
    
    def _create_market_summary(self, df: pd.DataFrame) -> Dict:
        """Create market data summary for signal"""
        recent_data = df.tail(5)
        
        return {
            'current_price': df['close'].iloc[-1],
            'price_change_5d': (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5],
            'volume_ratio': df['volume'].tail(5).mean() / df['volume'].tail(20).mean(),
            'volatility': df['close'].pct_change().tail(20).std(),
            'range_position': (df['close'].iloc[-1] - df['low'].tail(20).min()) / 
                            (df['high'].tail(20).max() - df['low'].tail(20).min())
        }
    
    def filter_signals_by_criteria(self, signals: List[MarketSignal], 
                                  criteria: Dict) -> List[MarketSignal]:
        """
        Filter signals based on various criteria
        
        Args:
            signals: List of market signals
            criteria: Filtering criteria dict
            
        Returns:
            List[MarketSignal]: Filtered signals
        """
        filtered = signals.copy()
        
        # Filter by minimum confidence
        if 'min_confidence' in criteria:
            filtered = [s for s in filtered if s.trade_signal.confidence >= criteria['min_confidence']]
        
        # Filter by minimum strength score
        if 'min_strength' in criteria:
            filtered = [s for s in filtered if s.strength_score >= criteria['min_strength']]
        
        # Filter by action type - Updated for new long-only actions
        if 'action_types' in criteria:
            allowed_actions = criteria['action_types']
            filtered = [s for s in filtered if s.trade_signal.action.value in allowed_actions]
        
        # Filter by risk-reward ratio
        if 'min_risk_reward' in criteria:
            min_rr = criteria['min_risk_reward']
            filtered = [s for s in filtered if s.risk_metrics.get('risk_reward_ratio', 0) >= min_rr]
        
        # Filter by maximum position size
        if 'max_position_size' in criteria:
            max_size = criteria['max_position_size']
            filtered = [s for s in filtered if s.trade_signal.position_size <= max_size]
        
        # Limit number of signals
        if 'max_signals' in criteria:
            filtered = filtered[:criteria['max_signals']]
        
        return filtered
    
    def get_signal_summary(self, signals: List[MarketSignal]) -> Dict:
        """Get summary statistics for signals"""
        if not signals:
            return {'total_signals': 0}
        
        actions = [s.trade_signal.action.value for s in signals]
        confidences = [s.trade_signal.confidence for s in signals]
        strengths = [s.strength_score for s in signals]
        
        return {
            'total_signals': len(signals),
            'buy_signals': actions.count('buy'),
            'scale_in_signals': actions.count('scale_in'),
            'scale_out_signals': actions.count('scale_out'),
            'close_long_signals': actions.count('close_long'),
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_strength': sum(strengths) / len(strengths),
            'top_signal': signals[0].symbol if signals else None
        }
    
    def clear_old_signals(self, hours_to_keep: int = 24):
        """Clear old signals from history"""
        cutoff_time = datetime.now().timestamp() - (hours_to_keep * 3600)
        
        # Clear recent signals
        self.recent_signals = {
            symbol: signal for symbol, signal in self.recent_signals.items()
            if signal.timestamp.timestamp() > cutoff_time
        }
        
        # Clear signal history
        self.signal_history = [
            signal for signal in self.signal_history
            if signal.timestamp.timestamp() > cutoff_time
        ]
        
        self.logger.info(f"Cleared signals older than {hours_to_keep} hours")
    
    def get_signal_for_symbol(self, symbol: str) -> Optional[MarketSignal]:
        """Get most recent signal for specific symbol"""
        return self.recent_signals.get(symbol)
    
    def update_strategy_parameters(self, params: Dict):
        """Update strategy parameters"""
        if 'min_confidence' in params:
            self.strategy.min_confidence = params['min_confidence']
        
        if 'risk_reward_ratio' in params:
            self.strategy.risk_reward_ratio = params['risk_reward_ratio']
        
        if 'risk_per_trade' in params:
            self.position_sizer.base_risk_per_trade = params['risk_per_trade']
        
        self.logger.info(f"Updated strategy parameters: {params}")