# wyckoff_bot/strategy/risk_management.py
"""
Risk Management for Wyckoff Trading
===================================
Comprehensive risk management following trading best practices
"""

from typing import Dict, Optional
from dataclasses import dataclass
from ..strategy.wyckoff_strategy import TradeSignal, TradeAction

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_portfolio_risk: float = 0.10   # 10% total portfolio
    max_position_size: float = 0.25    # 25% per position
    max_correlation_risk: float = 0.50  # 50% in correlated assets
    stop_loss_buffer: float = 0.01     # 1% buffer on stops

class RiskManager:
    """
    Comprehensive risk management system
    Validates and adjusts trade signals based on risk rules
    """
    
    def __init__(self, risk_params: RiskParameters = None):
        self.risk_params = risk_params or RiskParameters()
        self.current_portfolio_risk = 0.0
    
    def validate_and_adjust_signal(self, signal: TradeSignal, account_info: Dict, 
                                  current_positions: Dict = None) -> Optional[TradeSignal]:
        """
        Validate trade signal against risk rules and adjust if needed
        
        Args:
            signal: Original trade signal
            account_info: Account balance and info
            current_positions: Current positions
            
        Returns:
            TradeSignal: Adjusted signal or None if rejected
        """
        if not self._validate_basic_risk(signal, account_info):
            return None
        
        # Adjust position size based on risk
        adjusted_signal = self._adjust_position_size(signal, account_info)
        
        # Apply stop loss buffer
        adjusted_signal = self._apply_stop_loss_buffer(adjusted_signal)
        
        # Validate portfolio risk
        if not self._validate_portfolio_risk(adjusted_signal, account_info, current_positions):
            return None
        
        return adjusted_signal
    
    def _validate_basic_risk(self, signal: TradeSignal, account_info: Dict) -> bool:
        """Validate basic risk parameters"""
        account_balance = account_info.get('balance', 0)
        
        if account_balance <= 0:
            return False
        
        # Calculate risk amount for this trade - Long only
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            total_shares = (signal.position_size * account_balance) / signal.entry_price
            total_risk = risk_per_share * total_shares
            risk_percentage = total_risk / account_balance
            
            # Check if risk exceeds maximum
            if risk_percentage > self.risk_params.max_risk_per_trade:
                return False
        
        # Check position size limits
        if signal.position_size > self.risk_params.max_position_size:
            return False
        
        return True
    
    def _adjust_position_size(self, signal: TradeSignal, account_info: Dict) -> TradeSignal:
        """Adjust position size to meet risk parameters"""
        account_balance = account_info.get('balance', 0)
        
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            # Calculate maximum position size based on risk
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            
            if risk_per_share > 0:
                max_risk_amount = account_balance * self.risk_params.max_risk_per_trade
                max_shares = max_risk_amount / risk_per_share
                max_position_value = max_shares * signal.entry_price
                max_position_size = max_position_value / account_balance
                
                # Use the more conservative of the two limits
                adjusted_size = min(
                    signal.position_size,
                    max_position_size,
                    self.risk_params.max_position_size
                )
                
                # Create adjusted signal
                adjusted_signal = TradeSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=adjusted_size,
                    reasoning=signal.reasoning + " (Position size risk-adjusted)",
                    timeframe=signal.timeframe
                )
                
                return adjusted_signal
        
        return signal
    
    def _apply_stop_loss_buffer(self, signal: TradeSignal) -> TradeSignal:
        """Apply buffer to stop loss to account for slippage"""
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            # For long positions, lower the stop loss slightly
            buffer_amount = signal.entry_price * self.risk_params.stop_loss_buffer
            adjusted_stop = signal.stop_loss - buffer_amount
        else:
            return signal
        
        return TradeSignal(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=adjusted_stop,
            take_profit=signal.take_profit,
            position_size=signal.position_size,
            reasoning=signal.reasoning,
            timeframe=signal.timeframe
        )
    
    def _validate_portfolio_risk(self, signal: TradeSignal, account_info: Dict, 
                                current_positions: Dict = None) -> bool:
        """Validate total portfolio risk"""
        current_positions = current_positions or {}
        
        # Calculate current portfolio risk
        current_risk = self._calculate_current_portfolio_risk(current_positions, account_info)
        
        # Calculate additional risk from new signal - Long only
        if signal.action in [TradeAction.BUY, TradeAction.SCALE_IN]:
            additional_risk = signal.position_size  # Simplified risk calculation
            total_risk = current_risk + additional_risk
            
            if total_risk > self.risk_params.max_portfolio_risk:
                return False
        
        return True
    
    def _calculate_current_portfolio_risk(self, positions: Dict, account_info: Dict) -> float:
        """Calculate current portfolio risk exposure"""
        # Simplified calculation - in reality would be more complex
        total_position_value = 0
        account_balance = account_info.get('balance', 1)
        
        for position in positions.values():
            position_value = position.get('market_value', 0)
            total_position_value += abs(position_value)
        
        return total_position_value / account_balance
    
    def calculate_position_risk(self, signal: TradeSignal, account_balance: float) -> Dict[str, float]:
        """
        Calculate detailed risk metrics for a position
        
        Returns:
            Dict with risk metrics
        """
        if signal.action not in [TradeAction.BUY, TradeAction.SCALE_IN, TradeAction.SCALE_OUT, TradeAction.CLOSE_LONG]:
            return {}
        
        # Position value
        position_value = signal.position_size * account_balance
        shares = position_value / signal.entry_price
        
        # Risk per share
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        
        # Total risk amount
        total_risk = risk_per_share * shares
        
        # Risk percentage
        risk_percentage = total_risk / account_balance
        
        # Potential profit
        profit_per_share = abs(signal.take_profit - signal.entry_price)
        total_profit_potential = profit_per_share * shares
        profit_percentage = total_profit_potential / account_balance
        
        # Risk-reward ratio
        risk_reward_ratio = profit_per_share / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'position_value': position_value,
            'shares': shares,
            'risk_amount': total_risk,
            'risk_percentage': risk_percentage,
            'profit_potential': total_profit_potential,
            'profit_percentage': profit_percentage,
            'risk_reward_ratio': risk_reward_ratio,
            'max_loss_per_trade': self.risk_params.max_risk_per_trade,
            'within_risk_limits': risk_percentage <= self.risk_params.max_risk_per_trade
        }
    
    def update_risk_parameters(self, new_params: Dict):
        """Update risk parameters"""
        for key, value in new_params.items():
            if hasattr(self.risk_params, key):
                setattr(self.risk_params, key, value)
    
    def get_risk_status(self) -> Dict[str, any]:
        """Get current risk management status"""
        return {
            'max_risk_per_trade': self.risk_params.max_risk_per_trade,
            'max_portfolio_risk': self.risk_params.max_portfolio_risk,
            'max_position_size': self.risk_params.max_position_size,
            'current_portfolio_risk': self.current_portfolio_risk,
            'risk_buffer_remaining': self.risk_params.max_portfolio_risk - self.current_portfolio_risk
        }