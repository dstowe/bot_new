# wyckoff_bot/signals/signal_validator.py
"""
Signal Validation System
========================
Validates trading signals against multiple criteria
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import logging
from datetime import datetime, timedelta

from ..strategy.wyckoff_strategy import TradeSignal, TradeAction

@dataclass
class ValidationResult:
    """Signal validation result"""
    is_valid: bool
    confidence_adjustment: float
    warnings: List[str]
    blockers: List[str]
    validation_score: float

class SignalValidator:
    """
    Comprehensive signal validation system
    Checks signals against market conditions and risk rules
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation thresholds
        self.min_liquidity_volume = 500000  # Min daily volume
        self.max_volatility = 0.50  # Max 50% annualized volatility
        self.min_price = 5.00  # Minimum stock price
        self.max_spread_pct = 0.005  # Max 0.5% spread
        
    def validate_signal(self, signal: TradeSignal, market_data: pd.DataFrame,
                       market_conditions: Dict = None) -> ValidationResult:
        """
        Comprehensive signal validation
        
        Args:
            signal: Trade signal to validate
            market_data: Price/volume data for the symbol
            market_conditions: Overall market conditions
            
        Returns:
            ValidationResult: Validation result with recommendations
        """
        warnings = []
        blockers = []
        confidence_adjustment = 1.0
        validation_score = 100.0
        
        # Basic data validation
        if not self._validate_data_quality(market_data):
            blockers.append("Insufficient or poor quality data")
            validation_score -= 50
        
        # Liquidity validation
        liquidity_result = self._validate_liquidity(market_data)
        if not liquidity_result[0]:
            if liquidity_result[1] == "blocker":
                blockers.append("Insufficient liquidity")
                validation_score -= 30
            else:
                warnings.append("Low liquidity detected")
                validation_score -= 10
        
        # Volatility validation
        volatility_result = self._validate_volatility(market_data)
        if not volatility_result[0]:
            warnings.append(f"High volatility: {volatility_result[1]:.1%}")
            confidence_adjustment *= 0.9
            validation_score -= 15
        
        # Price level validation
        current_price = market_data['close'].iloc[-1]
        if current_price < self.min_price:
            blockers.append(f"Price too low: ${current_price:.2f}")
            validation_score -= 25
        
        # Market conditions validation
        if market_conditions:
            market_result = self._validate_market_conditions(signal, market_conditions)
            if not market_result[0]:
                warnings.append(market_result[1])
                confidence_adjustment *= 0.95
                validation_score -= 10
        
        # Technical validation
        tech_result = self._validate_technical_setup(signal, market_data)
        confidence_adjustment *= tech_result[1]
        validation_score += tech_result[2] - 20  # Adjust around baseline
        
        # Risk validation
        risk_result = self._validate_risk_parameters(signal)
        if not risk_result[0]:
            warnings.append(risk_result[1])
            validation_score -= 15
        
        # Time-based validation
        time_result = self._validate_timing(signal, market_data)
        if not time_result[0]:
            warnings.append(time_result[1])
            validation_score -= 5
        
        # Determine overall validity
        is_valid = len(blockers) == 0 and validation_score >= 40
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            warnings=warnings,
            blockers=blockers,
            validation_score=max(0, validation_score)
        )
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        if df.empty or len(df) < 20:
            return False
        
        # Check for missing data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for null values in recent data
        recent_data = df.tail(10)
        if recent_data[required_columns].isnull().any().any():
            return False
        
        # Check for zero volume days (suspicious)
        if (recent_data['volume'] == 0).sum() > 2:
            return False
        
        return True
    
    def _validate_liquidity(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate liquidity levels"""
        avg_volume = df['volume'].tail(20).mean()
        recent_volume = df['volume'].tail(5).mean()
        
        if avg_volume < self.min_liquidity_volume / 2:
            return False, "blocker"
        elif avg_volume < self.min_liquidity_volume:
            return False, "warning"
        elif recent_volume < avg_volume * 0.3:
            return False, "warning"
        
        return True, "ok"
    
    def _validate_volatility(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Validate volatility levels"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(20).std() * (252 ** 0.5)  # Annualized
        
        return volatility <= self.max_volatility, volatility
    
    def _validate_market_conditions(self, signal: TradeSignal, 
                                   conditions: Dict) -> Tuple[bool, str]:
        """Validate against overall market conditions"""
        market_trend = conditions.get('trend', 'neutral')
        vix_level = conditions.get('vix', 20)
        
        # Check for conflicting signals - Long only strategy
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN] and market_trend == 'bearish':
            return False, "Bullish signal in bearish market"
        
        # High volatility environment
        if vix_level > 30:
            return False, f"High market volatility (VIX: {vix_level})"
        
        return True, "Market conditions favorable"
    
    def _validate_technical_setup(self, signal: TradeSignal, 
                                 df: pd.DataFrame) -> Tuple[bool, float, float]:
        """Validate technical setup quality"""
        confidence_mult = 1.0
        score = 20  # Baseline score
        
        # Check support/resistance levels
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        
        # Position relative to range
        range_position = (current_price - recent_low) / (recent_high - recent_low)
        
        if signal.action == TradeAction.BUY:
            # Buying near support is good
            if range_position < 0.3:
                confidence_mult *= 1.1
                score += 15
            # Buying near resistance is bad
            elif range_position > 0.8:
                confidence_mult *= 0.9
                score -= 10
        
        elif signal.action in [TradeAction.SCALE_OUT, TradeAction.CLOSE_LONG]:
            # Exiting near resistance is good
            if range_position > 0.7:
                confidence_mult *= 1.1
                score += 15
            # Exiting near support may be panic selling
            elif range_position < 0.2:
                confidence_mult *= 0.9
                score -= 5
        
        # Volume confirmation
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume'].tail(20).mean()
        
        if recent_volume > avg_volume * 1.5:
            confidence_mult *= 1.1
            score += 10
        elif recent_volume < avg_volume * 0.7:
            confidence_mult *= 0.9
            score -= 5
        
        return True, confidence_mult, score
    
    def _validate_risk_parameters(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Validate risk parameters of the signal"""
        # Check stop loss distance
        risk_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        
        if risk_distance > 0.10:  # More than 10% risk
            return False, f"Stop loss too wide: {risk_distance:.1%}"
        elif risk_distance < 0.005:  # Less than 0.5% risk
            return False, f"Stop loss too tight: {risk_distance:.1%}"
        
        # Check risk-reward ratio
        profit_distance = abs(signal.take_profit - signal.entry_price)
        risk_distance_abs = abs(signal.entry_price - signal.stop_loss)
        
        if risk_distance_abs > 0:
            rr_ratio = profit_distance / risk_distance_abs
            if rr_ratio < 1.5:
                return False, f"Poor risk-reward ratio: {rr_ratio:.1f}"
        
        # Check position size
        if signal.position_size > 0.30:  # More than 30%
            return False, f"Position size too large: {signal.position_size:.1%}"
        elif signal.position_size < 0.01:  # Less than 1%
            return False, f"Position size too small: {signal.position_size:.1%}"
        
        return True, "Risk parameters acceptable"
    
    def _validate_timing(self, signal: TradeSignal, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate signal timing"""
        # Check if market just made large move (avoid chasing)
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        if abs(recent_change) > 0.05:  # 5% single-day move
            if ((signal.action in [TradeAction.BUY, TradeAction.SCALE_IN] and recent_change > 0) or
                (signal.action in [TradeAction.SCALE_OUT, TradeAction.CLOSE_LONG] and recent_change < 0)):
                return False, "Avoid chasing large moves"
        
        # Check for gap
        gap_size = abs(df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        if gap_size > 0.03:  # 3% gap
            return False, f"Large gap detected: {gap_size:.1%}"
        
        return True, "Timing acceptable"
    
    def batch_validate_signals(self, signals: List[TradeSignal], 
                              market_data: Dict[str, pd.DataFrame],
                              market_conditions: Dict = None) -> Dict[str, ValidationResult]:
        """
        Validate multiple signals in batch
        
        Returns:
            Dict[str, ValidationResult]: Symbol -> ValidationResult mapping
        """
        results = {}
        
        for signal in signals:
            if signal.symbol in market_data:
                results[signal.symbol] = self.validate_signal(
                    signal, market_data[signal.symbol], market_conditions
                )
            else:
                # Create failed validation for missing data
                results[signal.symbol] = ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    warnings=[],
                    blockers=["No market data available"],
                    validation_score=0.0
                )
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict:
        """Get summary of validation results"""
        total = len(results)
        valid = sum(1 for r in results.values() if r.is_valid)
        avg_score = sum(r.validation_score for r in results.values()) / total if total > 0 else 0
        
        common_warnings = {}
        common_blockers = {}
        
        for result in results.values():
            for warning in result.warnings:
                common_warnings[warning] = common_warnings.get(warning, 0) + 1
            for blocker in result.blockers:
                common_blockers[blocker] = common_blockers.get(blocker, 0) + 1
        
        return {
            'total_signals': total,
            'valid_signals': valid,
            'validation_rate': valid / total if total > 0 else 0,
            'average_score': avg_score,
            'common_warnings': common_warnings,
            'common_blockers': common_blockers
        }