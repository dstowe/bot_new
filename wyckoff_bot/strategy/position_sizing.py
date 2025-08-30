# wyckoff_bot/strategy/position_sizing.py
"""
Position Sizing for Wyckoff Trading
===================================
Advanced position sizing based on volatility and confidence
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    recommended_size: float
    max_shares: int
    fractional_shares: float
    dollar_amount: float
    is_fractional: bool
    risk_amount: float
    confidence_adjustment: float
    volatility_adjustment: float
    reasoning: str

class PositionSizer:
    """
    Advanced position sizing system
    Calculates optimal position sizes based on multiple factors with fractional share support
    """
    
    def __init__(self, base_risk_per_trade: float = 0.02, 
                 volatility_lookback: int = 20, min_trade_amount: float = 6.0):
        self.base_risk_per_trade = base_risk_per_trade
        self.volatility_lookback = volatility_lookback
        self.min_trade_amount = min_trade_amount  # Minimum $6 trade
    
    def calculate_position_size(self, entry_price: float, stop_loss: float,
                               confidence: float, account_balance: float,
                               price_data: pd.DataFrame = None) -> PositionSizeResult:
        """
        Calculate optimal position size with fractional share support
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss level
            confidence: Signal confidence (0-1)
            account_balance: Account balance
            price_data: Historical price data for volatility calc
            
        Returns:
            PositionSizeResult: Complete sizing calculation with fractional support
        """
        # Base risk amount
        base_risk_amount = account_balance * self.base_risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return self._create_zero_position("Zero risk per share")
        
        # Confidence adjustment
        confidence_adj = self._calculate_confidence_adjustment(confidence)
        
        # Volatility adjustment
        volatility_adj = self._calculate_volatility_adjustment(price_data, entry_price)
        
        # Adjusted risk amount
        adjusted_risk_amount = base_risk_amount * confidence_adj * volatility_adj
        
        # Calculate fractional shares based on risk
        fractional_shares = adjusted_risk_amount / risk_per_share
        
        # Calculate dollar amount
        dollar_amount = fractional_shares * entry_price
        
        # Ensure minimum trade amount
        if dollar_amount < self.min_trade_amount:
            fractional_shares = self.min_trade_amount / entry_price
            dollar_amount = self.min_trade_amount
        
        # Cap position at 25% of account
        max_dollar_amount = account_balance * 0.25
        if dollar_amount > max_dollar_amount:
            dollar_amount = max_dollar_amount
            fractional_shares = dollar_amount / entry_price
        
        # Check if we can afford a full share and have enough for multiple shares
        full_shares = int(fractional_shares)
        is_fractional = full_shares == 0 or (dollar_amount < entry_price * 2 and account_balance < 500)
        
        # For very small accounts, always use fractional
        if account_balance < 200:
            is_fractional = True
        
        # Calculate final risk
        final_risk = fractional_shares * risk_per_share
        position_size_pct = dollar_amount / account_balance
        
        reasoning = self._generate_fractional_sizing_reasoning(
            confidence, confidence_adj, volatility_adj, 
            dollar_amount, fractional_shares, full_shares, 
            is_fractional, account_balance
        )
        
        return PositionSizeResult(
            recommended_size=position_size_pct,
            max_shares=full_shares,
            fractional_shares=round(fractional_shares, 4),  # Round to 4 decimal places for Webull API
            dollar_amount=dollar_amount,
            is_fractional=is_fractional,
            risk_amount=final_risk,
            confidence_adjustment=confidence_adj,
            volatility_adjustment=volatility_adj,
            reasoning=reasoning
        )
    
    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Calculate position size adjustment based on signal confidence
        Higher confidence = larger position (up to a limit)
        """
        if confidence < 0.6:
            return 0.5  # Reduce size for low confidence
        elif confidence < 0.75:
            return 0.8  # Slightly reduce size
        elif confidence < 0.85:
            return 1.0  # Normal size
        else:
            return 1.2  # Increase size for high confidence (max 20% increase)
    
    def _calculate_volatility_adjustment(self, price_data: pd.DataFrame, 
                                       current_price: float) -> float:
        """
        Calculate position size adjustment based on volatility
        Higher volatility = smaller position for risk management
        """
        if price_data is None or len(price_data) < self.volatility_lookback:
            return 1.0  # No adjustment if no data
        
        # Calculate historical volatility
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.tail(self.volatility_lookback).std() * np.sqrt(252)  # Annualized
        
        # Average market volatility is around 0.20 (20%)
        # Adjust position size inversely to volatility
        if volatility < 0.15:  # Low volatility
            return 1.1  # Slightly increase position
        elif volatility < 0.25:  # Normal volatility  
            return 1.0  # No adjustment
        elif volatility < 0.40:  # High volatility
            return 0.8  # Reduce position
        else:  # Very high volatility
            return 0.6  # Significantly reduce position
    
    def _generate_fractional_sizing_reasoning(self, confidence: float, confidence_adj: float,
                                             volatility_adj: float, dollar_amount: float,
                                             fractional_shares: float, full_shares: int,
                                             is_fractional: bool, account_balance: float) -> str:
        """Generate explanation for fractional position sizing decision"""
        reasoning_parts = []
        
        # Trade type
        if is_fractional:
            reasoning_parts.append(f"Fractional trade: ${dollar_amount:.2f} ({fractional_shares:.3f} shares)")
        else:
            reasoning_parts.append(f"Full share trade: {full_shares} shares (${dollar_amount:.2f})")
        
        # Account size context
        reasoning_parts.append(f"Account: ${account_balance:.2f}")
        
        # Minimum trade enforcement
        if dollar_amount == self.min_trade_amount:
            reasoning_parts.append(f"Minimum ${self.min_trade_amount:.0f} trade enforced")
        
        # Confidence adjustment
        if confidence_adj > 1.0:
            reasoning_parts.append(f"Size increased {(confidence_adj-1)*100:.0f}% (high confidence)")
        elif confidence_adj < 1.0:
            reasoning_parts.append(f"Size reduced {(1-confidence_adj)*100:.0f}% (low confidence)")
        
        # Volatility adjustment
        if volatility_adj != 1.0:
            adj_pct = abs((volatility_adj - 1) * 100)
            direction = "increased" if volatility_adj > 1.0 else "reduced"
            reasoning_parts.append(f"Volatility {direction} size {adj_pct:.0f}%")
        
        return ". ".join(reasoning_parts) + "."
    
    def _create_zero_position(self, reason: str) -> PositionSizeResult:
        """Create zero position result"""
        return PositionSizeResult(
            recommended_size=0.0,
            max_shares=0,
            fractional_shares=0.0,
            dollar_amount=0.0,
            is_fractional=False,
            risk_amount=0.0,
            confidence_adjustment=0.0,
            volatility_adjustment=0.0,
            reasoning=f"No position: {reason}"
        )
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                 avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            float: Kelly criterion percentage
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_f = (b * p - q) / b
        
        # Cap Kelly at 25% and ensure it's positive
        return max(0, min(kelly_f, 0.25))
    
    def calculate_fixed_fractional(self, confidence: float, 
                                  base_fraction: float = 0.02) -> float:
        """
        Calculate fixed fractional position size
        
        Args:
            confidence: Signal confidence (0-1)
            base_fraction: Base fraction of capital to risk
            
        Returns:
            float: Position size as fraction of account
        """
        # Scale base fraction by confidence
        adjusted_fraction = base_fraction * (0.5 + confidence)
        return min(adjusted_fraction, 0.25)  # Cap at 25%
    
    def get_sizing_recommendations(self, signals: list, account_balance: float) -> Dict:
        """
        Get position sizing recommendations for multiple signals
        
        Args:
            signals: List of trade signals
            account_balance: Current account balance
            
        Returns:
            Dict: Sizing analysis and recommendations
        """
        if not signals:
            return {'total_allocation': 0.0, 'recommendations': []}
        
        total_allocation = 0.0
        recommendations = []
        
        for signal in signals:
            sizing = self.calculate_position_size(
                signal.entry_price,
                signal.stop_loss,
                signal.confidence,
                account_balance
            )
            
            total_allocation += sizing.recommended_size
            recommendations.append({
                'symbol': signal.symbol,
                'recommended_size': sizing.recommended_size,
                'shares': sizing.max_shares,
                'risk_amount': sizing.risk_amount
            })
        
        return {
            'total_allocation': total_allocation,
            'recommendations': recommendations,
            'allocation_warning': total_allocation > 0.75,  # Warning if >75% allocated
            'diversification_score': len(signals) / max(total_allocation, 0.01)
        }