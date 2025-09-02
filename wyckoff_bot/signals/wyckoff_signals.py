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
import numpy as np
from scipy import stats

from ..strategy.wyckoff_strategy import WyckoffStrategy, TradeSignal
from ..strategy.risk_management import RiskManager
from ..strategy.position_sizing import PositionSizer
from ..analysis.wyckoff_analyzer import WyckoffAnalyzer
from ..analysis.market_regime import MarketRegimeAnalyzer, MarketRegime

@dataclass
class MarketSignal:
    """Enhanced market signal with institutional analysis"""
    symbol: str
    timestamp: datetime
    trade_signal: TradeSignal
    risk_metrics: Dict
    position_sizing: Dict
    market_data: Dict
    strength_score: float
    wyckoff_analysis: Optional[Dict] = None
    market_regime_context: Optional[Dict] = None
    multi_timeframe_confirmation: Optional[Dict] = None
    institutional_flow: Optional[Dict] = None
    sentiment_indicators: Optional[Dict] = None
    liquidity_analysis: Optional[Dict] = None

class WyckoffSignalGenerator:
    """
    Enhanced Signal Generation Engine with Institutional Features
    Multi-layer validation with market regime analysis, sentiment, and liquidity
    """
    
    def __init__(self, min_confidence: float = 0.6, 
                 risk_per_trade: float = 0.02,
                 min_trade_amount: float = 6.0,
                 multi_tf_data_manager = None,
                 logger: logging.Logger = None):
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize core components
        self.strategy = WyckoffStrategy(min_confidence=min_confidence)
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(
            base_risk_per_trade=risk_per_trade,
            min_trade_amount=min_trade_amount
        )
        
        # Enhanced analysis components
        self.wyckoff_analyzer = WyckoffAnalyzer(logger=self.logger)
        self.multi_tf_data_manager = multi_tf_data_manager
        self.regime_analyzer = MarketRegimeAnalyzer(logger=self.logger)
        
        # Multi-layer validation settings
        self.validation_layers = {
            'wyckoff_confirmation': True,
            'regime_alignment': True,
            'sentiment_check': True,
            'liquidity_validation': True,
            'multi_timeframe': True
        }
        
        # Signal filtering thresholds
        self.institutional_thresholds = {
            'min_daily_volume': 1000000,  # $1M daily volume
            'min_market_cap': 100000000,  # $100M market cap
            'max_spread': 0.02,           # 2% max bid-ask spread
            'min_float': 0.20,            # 20% minimum float
            'liquidity_score': 0.6        # Minimum liquidity score
        }
        
        # Track generated signals
        self.recent_signals = {}
        self.signal_history = []
        
        # Market regime cache
        self.current_regime = None
        self.regime_cache_time = None
    
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
        
        # Enhanced analysis layers
        wyckoff_analysis = self._perform_wyckoff_analysis(symbol, df)
        regime_context = self._get_market_regime_context()
        mtf_confirmation = self._get_multi_timeframe_confirmation(symbol)
        institutional_flow = self._analyze_institutional_flow(df)
        sentiment = self._analyze_sentiment_indicators(symbol, df)
        liquidity = self._analyze_liquidity(symbol, df)
        
        # Apply multi-layer validation
        validation_result = self._validate_signal_layers(
            adjusted_signal, wyckoff_analysis, regime_context, 
            mtf_confirmation, institutional_flow, sentiment, liquidity
        )
        
        if not validation_result['passes_validation']:
            self.logger.debug(f"Signal for {symbol} failed multi-layer validation: {validation_result['reason']}")
            return None
        
        # Adjust strength score based on institutional factors
        enhanced_strength = self._calculate_enhanced_strength_score(
            adjusted_signal, risk_metrics, df, wyckoff_analysis, 
            regime_context, institutional_flow, sentiment, liquidity
        )
        
        return MarketSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            trade_signal=adjusted_signal,
            risk_metrics=risk_metrics,
            position_sizing=position_sizing.__dict__,
            market_data=market_data_summary,
            strength_score=enhanced_strength,
            wyckoff_analysis=wyckoff_analysis,
            market_regime_context=regime_context,
            multi_timeframe_confirmation=mtf_confirmation,
            institutional_flow=institutional_flow,
            sentiment_indicators=sentiment,
            liquidity_analysis=liquidity
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
            'top_signal': signals[0].symbol if signals else None,
            'institutional_signals': sum(1 for s in signals 
                                       if s.institutional_flow and 
                                       s.institutional_flow.get('institutional_activity_score', 0) > 0.6),
            'high_conviction_signals': sum(1 for s in signals if s.strength_score > 80),
            'multi_timeframe_confirmed': sum(1 for s in signals 
                                          if s.multi_timeframe_confirmation and 
                                          s.multi_timeframe_confirmation.get('strong_alignment', False))
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
    
    def _perform_wyckoff_analysis(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive Wyckoff analysis
        
        Args:
            symbol: Stock symbol
            df: Price data
            
        Returns:
            Dict: Wyckoff analysis results
        """
        try:
            # Single timeframe analysis
            analysis = self.wyckoff_analyzer.analyze(df, symbol)
            
            result = {
                'phase': analysis.phase.value,
                'confidence': analysis.confidence,
                'volume_confirmation': analysis.volume_confirmation,
                'price_action_strength': analysis.price_action_strength,
                'key_events': [event.value for event in analysis.key_events],
                'support_resistance': analysis.support_resistance_levels,
                'trend_strength': analysis.trend_strength,
                'point_figure_signals': analysis.point_figure_signals or {},
                'institutional_flow': analysis.institutional_flow or {}
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error in Wyckoff analysis for {symbol}: {e}")
            return {}
    
    def _get_market_regime_context(self) -> Dict:
        """
        Get current market regime context with caching
        
        Returns:
            Dict: Market regime analysis
        """
        try:
            # Check cache (refresh every hour)
            current_time = datetime.now()
            if (self.current_regime is None or 
                self.regime_cache_time is None or 
                (current_time - self.regime_cache_time).total_seconds() > 3600):
                
                # Refresh market regime analysis
                regime_analysis = self.regime_analyzer.analyze_market_regime()
                self.current_regime = regime_analysis
                self.regime_cache_time = current_time
                
                self.logger.info(f"Market regime updated: {regime_analysis.regime.value} "
                               f"(confidence: {regime_analysis.confidence:.1%})")
            
            regime = self.current_regime
            return {
                'regime': regime.regime.value,
                'confidence': regime.confidence,
                'trend_strength': regime.trend_strength.value,
                'volatility_regime': regime.volatility_regime,
                'cash_allocation_rec': regime.cash_allocation_recommendation,
                'sector_rotation': regime.sector_rotation_signal,
                'vix_analysis': regime.vix_analysis,
                'regime_duration_days': regime.regime_duration_days
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting market regime context: {e}")
            return {}
    
    def _get_multi_timeframe_confirmation(self, symbol: str) -> Dict:
        """
        Get multi-timeframe confirmation for signal
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Multi-timeframe analysis
        """
        try:
            # Perform multi-timeframe analysis using cached data if available
            if self.multi_tf_data_manager:
                # Get cached multi-timeframe data
                timeframes = ['1D', '4H', '1H']  # Standard timeframes for confirmation
                mtf_data = {}
                
                for tf in timeframes:
                    df = self.multi_tf_data_manager.get_cached_data(symbol, tf, bars=50)
                    if df is not None and len(df) >= 20:
                        mtf_data[tf] = df
                
                if mtf_data:
                    # Use the enhanced multi-timeframe analyzer
                    mtf_analyses = self.wyckoff_analyzer.analyze_multi_timeframe(mtf_data, symbol)
                else:
                    # Fallback to legacy method
                    mtf_analyses = self.wyckoff_analyzer.analyze_multi_timeframe_legacy(symbol)
            else:
                # Use legacy method
                mtf_analyses = self.wyckoff_analyzer.analyze_multi_timeframe_legacy(symbol)
            
            if not mtf_analyses:
                return {}
                
            # Calculate timeframe alignment
            confirmations = {}
            phases = {}
            
            for timeframe, analysis in mtf_analyses.items():
                phases[timeframe] = analysis.phase.value
                if analysis.multi_timeframe_confirmation:
                    confirmations.update(analysis.multi_timeframe_confirmation)
            
            # Calculate overall alignment score
            alignment_score = 0.0
            total_checks = 0
            
            for confirmation_key, confirms in confirmations.items():
                if confirms:
                    alignment_score += 1.0
                total_checks += 1
            
            overall_alignment = alignment_score / total_checks if total_checks > 0 else 0.0
            
            return {
                'timeframe_phases': phases,
                'confirmations': confirmations,
                'alignment_score': overall_alignment,
                'strong_alignment': overall_alignment > 0.7
            }
            
        except Exception as e:
            self.logger.warning(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {}
    
    def _analyze_institutional_flow(self, df: pd.DataFrame) -> Dict:
        """
        Analyze institutional money flow patterns
        
        Args:
            df: Price data
            
        Returns:
            Dict: Institutional flow indicators
        """
        try:
            if len(df) < 20:
                return {}
            
            recent_data = df.tail(20)
            
            # Block trading detection (large volume bars)
            avg_volume = df['volume'].tail(50).mean()
            block_trades = recent_data[recent_data['volume'] > avg_volume * 2]
            
            # Volume-weighted average price analysis
            vwap_20 = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
            current_price = df['close'].iloc[-1]
            
            # Dark pool indicator (simplified using volume patterns)
            volume_spikes = (recent_data['volume'] > avg_volume * 1.5).sum()
            price_efficiency = recent_data['close'].pct_change().abs().mean()
            
            # Smart money indicator
            # High volume + small price change = institutional accumulation/distribution
            smart_money_days = 0
            for i in range(len(recent_data)):
                vol_ratio = recent_data['volume'].iloc[i] / avg_volume if avg_volume > 0 else 0
                price_change = abs(recent_data['close'].pct_change().iloc[i])
                
                if vol_ratio > 1.5 and price_change < 0.02:  # High vol, low price change
                    smart_money_days += 1
            
            smart_money_ratio = smart_money_days / len(recent_data)
            
            return {
                'block_trade_count': len(block_trades),
                'block_trade_ratio': len(block_trades) / len(recent_data),
                'vwap_position': (current_price - vwap_20) / vwap_20,
                'volume_spikes': volume_spikes,
                'price_efficiency': price_efficiency,
                'smart_money_ratio': smart_money_ratio,
                'institutional_activity_score': min(smart_money_ratio * 2 + 
                                                   (len(block_trades) / len(recent_data)) * 1.5, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing institutional flow: {e}")
            return {}
    
    def _analyze_sentiment_indicators(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Analyze market sentiment indicators
        
        Args:
            symbol: Stock symbol
            df: Price data
            
        Returns:
            Dict: Sentiment analysis
        """
        try:
            if len(df) < 30:
                return {}
            
            recent_data = df.tail(20)
            
            # Price momentum sentiment
            momentum_5d = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            momentum_20d = (df['close'].iloc[-1] - df['close'].iloc[-21]) / df['close'].iloc[-21]
            
            # Volume sentiment (increasing volume on up days)
            up_days = recent_data[recent_data['close'] > recent_data['open']]
            down_days = recent_data[recent_data['close'] < recent_data['open']]
            
            avg_up_volume = up_days['volume'].mean() if len(up_days) > 0 else 0
            avg_down_volume = down_days['volume'].mean() if len(down_days) > 0 else 0
            
            volume_sentiment = (avg_up_volume - avg_down_volume) / (avg_up_volume + avg_down_volume) \
                             if (avg_up_volume + avg_down_volume) > 0 else 0
            
            # Volatility sentiment (fear/greed proxy)
            volatility = recent_data['close'].pct_change().std()
            vol_percentile = (df['close'].pct_change().tail(252).std() <= volatility).mean()
            
            # Price action sentiment (higher highs/lower lows)
            highs = recent_data['high']
            lows = recent_data['low']
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs.iloc[i] > highs.iloc[i-1])
            higher_lows = sum(1 for i in range(1, len(lows)) if lows.iloc[i] > lows.iloc[i-1])
            
            trend_sentiment = (higher_highs + higher_lows) / (2 * (len(recent_data) - 1))
            
            # Composite sentiment score
            sentiment_components = [
                max(-1, min(1, momentum_5d * 10)),  # Normalize to -1 to 1
                max(-1, min(1, momentum_20d * 5)),
                max(-1, min(1, volume_sentiment)),
                max(-1, min(1, trend_sentiment * 2 - 1))  # Convert 0-1 to -1-1
            ]
            
            composite_sentiment = sum(sentiment_components) / len(sentiment_components)
            
            return {
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'volume_sentiment': volume_sentiment,
                'volatility_percentile': vol_percentile,
                'trend_sentiment': trend_sentiment,
                'composite_sentiment': composite_sentiment,
                'sentiment_strength': abs(composite_sentiment),
                'sentiment_direction': 'bullish' if composite_sentiment > 0.1 else 'bearish' if composite_sentiment < -0.1 else 'neutral'
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment for {symbol}: {e}")
            return {}
    
    def _analyze_liquidity(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Analyze liquidity characteristics
        
        Args:
            symbol: Stock symbol
            df: Price data
            
        Returns:
            Dict: Liquidity analysis
        """
        try:
            if len(df) < 20:
                return {}
            
            recent_data = df.tail(20)
            
            # Volume-based liquidity
            avg_daily_volume = recent_data['volume'].mean()
            avg_daily_value = (recent_data['close'] * recent_data['volume']).mean()
            
            # Price impact measure (volatility per unit volume)
            volume_normalized_volatility = recent_data['close'].pct_change().std() / \
                                         (recent_data['volume'].mean() / 1000000) if recent_data['volume'].mean() > 0 else float('inf')
            
            # Spread estimation (using high-low as proxy)
            avg_spread_pct = ((recent_data['high'] - recent_data['low']) / recent_data['close']).mean()
            
            # Liquidity score calculation
            # Higher volume and lower spread = better liquidity
            volume_score = min(avg_daily_value / 1000000, 10) / 10  # Cap at $10M, normalize to 0-1
            spread_score = max(0, (0.05 - avg_spread_pct) / 0.05)  # Better score for smaller spreads
            
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            # Market impact estimation
            # Simplified model: impact = k * (order_size / avg_volume)^0.6
            def estimate_impact(order_size_pct):
                if avg_daily_volume == 0:
                    return float('inf')
                return 0.1 * (order_size_pct * 0.2) ** 0.6  # Assume 20% of daily volume participation
            
            return {
                'avg_daily_volume': avg_daily_volume,
                'avg_daily_value': avg_daily_value,
                'avg_spread_pct': avg_spread_pct,
                'volume_normalized_volatility': volume_normalized_volatility,
                'liquidity_score': liquidity_score,
                'estimated_impact_1pct': estimate_impact(0.01),
                'estimated_impact_5pct': estimate_impact(0.05),
                'liquidity_tier': self._classify_liquidity_tier(avg_daily_value, avg_spread_pct)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing liquidity for {symbol}: {e}")
            return {}
    
    def _classify_liquidity_tier(self, avg_daily_value: float, avg_spread_pct: float) -> str:
        """
        Classify liquidity into tiers
        
        Args:
            avg_daily_value: Average daily trading value
            avg_spread_pct: Average spread percentage
            
        Returns:
            str: Liquidity tier (TIER_1, TIER_2, TIER_3, ILLIQUID)
        """
        if avg_daily_value > 10000000 and avg_spread_pct < 0.01:  # >$10M, <1% spread
            return 'TIER_1'
        elif avg_daily_value > 1000000 and avg_spread_pct < 0.02:  # >$1M, <2% spread
            return 'TIER_2'
        elif avg_daily_value > 100000 and avg_spread_pct < 0.05:  # >$100K, <5% spread
            return 'TIER_3'
        else:
            return 'ILLIQUID'
    
    def _validate_signal_layers(self, signal: TradeSignal, wyckoff_analysis: Dict,
                               regime_context: Dict, mtf_confirmation: Dict,
                               institutional_flow: Dict, sentiment: Dict,
                               liquidity: Dict) -> Dict:
        """
        Apply multi-layer validation to signal
        
        Returns:
            Dict: Validation result with pass/fail and reasons
        """
        validation_result = {
            'passes_validation': True,
            'reason': '',
            'layer_results': {}
        }
        
        try:
            # Layer 1: Wyckoff Confirmation
            if self.validation_layers.get('wyckoff_confirmation', True):
                wyckoff_confidence = wyckoff_analysis.get('confidence', 0)
                volume_confirmation = wyckoff_analysis.get('volume_confirmation', False)
                
                if wyckoff_confidence < 0.4 or not volume_confirmation:
                    validation_result['passes_validation'] = False
                    validation_result['reason'] = f'Wyckoff confirmation failed (conf: {wyckoff_confidence:.1%})'
                    return validation_result
                
                validation_result['layer_results']['wyckoff'] = 'PASS'
            
            # Layer 2: Market Regime Alignment
            if self.validation_layers.get('regime_alignment', True):
                regime = regime_context.get('regime', 'unknown')
                regime_confidence = regime_context.get('confidence', 0)
                
                # Only trade in favorable regimes with sufficient confidence
                favorable_regimes = ['bull', 'range']
                if signal.action.value in ['buy', 'scale_in']:
                    if regime not in favorable_regimes or regime_confidence < 0.5:
                        validation_result['passes_validation'] = False
                        validation_result['reason'] = f'Unfavorable market regime: {regime} (conf: {regime_confidence:.1%})'
                        return validation_result
                
                validation_result['layer_results']['regime'] = 'PASS'
            
            # Layer 3: Sentiment Check
            if self.validation_layers.get('sentiment_check', True):
                composite_sentiment = sentiment.get('composite_sentiment', 0)
                sentiment_strength = sentiment.get('sentiment_strength', 0)
                
                # For buy signals, avoid strong bearish sentiment
                if signal.action.value in ['buy', 'scale_in']:
                    if composite_sentiment < -0.3 and sentiment_strength > 0.5:
                        validation_result['passes_validation'] = False
                        validation_result['reason'] = f'Strong bearish sentiment detected ({composite_sentiment:.2f})'
                        return validation_result
                
                validation_result['layer_results']['sentiment'] = 'PASS'
            
            # Layer 4: Liquidity Validation
            if self.validation_layers.get('liquidity_validation', True):
                liquidity_score = liquidity.get('liquidity_score', 0)
                liquidity_tier = liquidity.get('liquidity_tier', 'ILLIQUID')
                avg_daily_value = liquidity.get('avg_daily_value', 0)
                
                # Ensure minimum liquidity standards
                if (liquidity_score < self.institutional_thresholds['liquidity_score'] or 
                    avg_daily_value < self.institutional_thresholds['min_daily_volume'] or
                    liquidity_tier == 'ILLIQUID'):
                    
                    validation_result['passes_validation'] = False
                    validation_result['reason'] = f'Insufficient liquidity (score: {liquidity_score:.2f}, tier: {liquidity_tier})'
                    return validation_result
                
                validation_result['layer_results']['liquidity'] = 'PASS'
            
            # Layer 5: Multi-timeframe Confirmation
            if self.validation_layers.get('multi_timeframe', True):
                alignment_score = mtf_confirmation.get('alignment_score', 0)
                strong_alignment = mtf_confirmation.get('strong_alignment', False)
                
                # Require some degree of multi-timeframe alignment
                if alignment_score < 0.3:
                    validation_result['passes_validation'] = False
                    validation_result['reason'] = f'Poor multi-timeframe alignment ({alignment_score:.1%})'
                    return validation_result
                
                validation_result['layer_results']['multi_timeframe'] = 'PASS'
            
            validation_result['reason'] = 'All validation layers passed'
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"Error in multi-layer validation: {e}")
            validation_result['passes_validation'] = False
            validation_result['reason'] = f'Validation error: {str(e)}'
            return validation_result
    
    def _calculate_enhanced_strength_score(self, signal: TradeSignal, risk_metrics: Dict,
                                         df: pd.DataFrame, wyckoff_analysis: Dict,
                                         regime_context: Dict, institutional_flow: Dict,
                                         sentiment: Dict, liquidity: Dict) -> float:
        """
        Calculate enhanced strength score incorporating institutional factors
        
        Returns:
            float: Enhanced strength score (0-100)
        """
        try:
            # Start with base score
            base_score = self._calculate_strength_score(signal, risk_metrics, df)
            
            # Institutional enhancement factors
            enhancement_factors = []
            
            # Wyckoff analysis enhancement
            if wyckoff_analysis:
                wyckoff_conf = wyckoff_analysis.get('confidence', 0)
                price_action_strength = wyckoff_analysis.get('price_action_strength', 0)
                wyckoff_bonus = (wyckoff_conf + price_action_strength) / 2 * 10
                enhancement_factors.append(wyckoff_bonus)
            
            # Market regime enhancement
            if regime_context:
                regime = regime_context.get('regime', 'unknown')
                regime_conf = regime_context.get('confidence', 0)
                
                if regime in ['bull', 'range'] and signal.action.value in ['buy', 'scale_in']:
                    regime_bonus = regime_conf * 8
                    enhancement_factors.append(regime_bonus)
            
            # Institutional flow enhancement
            if institutional_flow:
                activity_score = institutional_flow.get('institutional_activity_score', 0)
                smart_money_ratio = institutional_flow.get('smart_money_ratio', 0)
                
                institutional_bonus = (activity_score + smart_money_ratio) / 2 * 6
                enhancement_factors.append(institutional_bonus)
            
            # Sentiment enhancement
            if sentiment:
                sentiment_direction = sentiment.get('sentiment_direction', 'neutral')
                sentiment_strength = sentiment.get('sentiment_strength', 0)
                
                if ((signal.action.value in ['buy', 'scale_in'] and sentiment_direction == 'bullish') or
                    (signal.action.value in ['scale_out', 'close_long'] and sentiment_direction == 'bearish')):
                    sentiment_bonus = sentiment_strength * 5
                    enhancement_factors.append(sentiment_bonus)
            
            # Liquidity enhancement
            if liquidity:
                liquidity_score = liquidity.get('liquidity_score', 0)
                liquidity_tier = liquidity.get('liquidity_tier', 'ILLIQUID')
                
                tier_bonuses = {'TIER_1': 8, 'TIER_2': 5, 'TIER_3': 2, 'ILLIQUID': 0}
                liquidity_bonus = tier_bonuses.get(liquidity_tier, 0) + liquidity_score * 3
                enhancement_factors.append(liquidity_bonus)
            
            # Apply enhancements
            total_enhancement = sum(enhancement_factors)
            enhanced_score = base_score + total_enhancement
            
            # Cap at 100
            return min(enhanced_score, 100.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating enhanced strength score: {e}")
            return self._calculate_strength_score(signal, risk_metrics, df)