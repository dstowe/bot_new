# risk/portfolio_risk_monitor.py
"""
Portfolio Risk Monitor
======================
Monitors portfolio-wide risk metrics and correlations
"""

import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Import stock universe for sector mapping
from config.stock_universe import StockUniverse

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    unrealized_pnl: float
    risk_amount: float  # Amount at risk (entry to stop loss)
    position_value: float
    sector: str = "Unknown"

class PortfolioRiskMonitor:
    """
    Monitors portfolio-wide risk exposure
    Tracks correlations, sector concentration, and overall risk
    """
    
    def __init__(self, config, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Current positions
        self.positions: Dict[str, PositionRisk] = {}
        
        # Use comprehensive sector mapping from stock universe
        self.sector_map = StockUniverse.get_sector_mapping()
    
    def update_position(self, symbol: str, position_size: float, entry_price: float,
                       current_price: float, stop_loss: float) -> None:
        """Update or add position to risk monitoring"""
        
        unrealized_pnl = (current_price - entry_price) * position_size
        risk_amount = abs(entry_price - stop_loss) * position_size
        position_value = current_price * position_size
        sector = self.sector_map.get(symbol, "Unknown")
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=stop_loss,
            unrealized_pnl=unrealized_pnl,
            risk_amount=risk_amount,
            position_value=position_value,
            sector=sector
        )
        
        self.logger.debug(f"Updated position risk for {symbol}: ${risk_amount:.2f} at risk")
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from risk monitoring"""
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger.debug(f"Removed position {symbol} from risk monitoring")
    
    def check_portfolio_risk_limits(self, portfolio_value: float) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits
        
        Returns:
            Tuple[bool, List[str]]: (within_limits, violation_messages)
        """
        violations = []
        
        # Check total portfolio risk
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        portfolio_risk_ratio = total_risk / portfolio_value if portfolio_value > 0 else 0
        
        if portfolio_risk_ratio > self.config.MAX_PORTFOLIO_RISK:
            violations.append(f"Portfolio risk too high: {portfolio_risk_ratio:.1%} > {self.config.MAX_PORTFOLIO_RISK:.1%}")
        
        # Check maximum concurrent positions
        if len(self.positions) > self.config.MAX_CONCURRENT_POSITIONS:
            violations.append(f"Too many positions: {len(self.positions)} > {self.config.MAX_CONCURRENT_POSITIONS}")
        
        # Check sector concentration
        sector_violations = self._check_sector_concentration(portfolio_value)
        violations.extend(sector_violations)
        
        # Check correlation limits
        correlation_violations = self._check_correlation_limits()
        violations.extend(correlation_violations)
        
        return len(violations) == 0, violations
    
    def _check_sector_concentration(self, portfolio_value: float) -> List[str]:
        """Check for excessive sector concentration"""
        violations = []
        
        # Group positions by sector
        sector_exposure = defaultdict(float)
        for position in self.positions.values():
            sector_exposure[position.sector] += position.position_value
        
        # Check each sector
        for sector, exposure in sector_exposure.items():
            sector_ratio = exposure / portfolio_value if portfolio_value > 0 else 0
            if sector_ratio > self.config.MAX_SECTOR_EXPOSURE:
                violations.append(f"Excessive {sector} exposure: {sector_ratio:.1%} > {self.config.MAX_SECTOR_EXPOSURE:.1%}")
        
        return violations
    
    def _check_correlation_limits(self) -> List[str]:
        """Check for excessive correlated positions (simplified)"""
        violations = []
        
        # Group by sector (simplified correlation check)
        sector_counts = defaultdict(int)
        for position in self.positions.values():
            sector_counts[position.sector] += 1
        
        # Check if too many positions in same sector
        for sector, count in sector_counts.items():
            if count > self.config.MAX_CORRELATED_POSITIONS:
                violations.append(f"Too many correlated positions in {sector}: {count} > {self.config.MAX_CORRELATED_POSITIONS}")
        
        return violations
    
    def get_portfolio_metrics(self, portfolio_value: float) -> Dict:
        """Get comprehensive portfolio risk metrics"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'total_risk': 0.0,
                'portfolio_risk_ratio': 0.0,
                'unrealized_pnl': 0.0,
                'sector_exposure': {},
                'largest_position': None
            }
        
        total_value = sum(pos.position_value for pos in self.positions.values())
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Sector breakdown
        sector_exposure = defaultdict(float)
        for position in self.positions.values():
            sector_exposure[position.sector] += position.position_value
        
        # Convert to percentages
        sector_percentages = {
            sector: (value / portfolio_value * 100) if portfolio_value > 0 else 0
            for sector, value in sector_exposure.items()
        }
        
        # Largest position
        largest_position = max(self.positions.values(), key=lambda p: p.position_value, default=None)
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_risk': total_risk,
            'portfolio_risk_ratio': (total_risk / portfolio_value) if portfolio_value > 0 else 0,
            'unrealized_pnl': total_pnl,
            'sector_exposure': sector_percentages,
            'largest_position': {
                'symbol': largest_position.symbol,
                'value': largest_position.position_value,
                'percentage': (largest_position.position_value / portfolio_value * 100) if portfolio_value > 0 else 0
            } if largest_position else None
        }
    
    def get_positions_exceeding_risk(self, max_risk_per_position: float) -> List[PositionRisk]:
        """Get positions that exceed individual risk limits"""
        return [
            pos for pos in self.positions.values() 
            if pos.risk_amount > max_risk_per_position
        ]
    
    def suggest_risk_reduction_actions(self, portfolio_value: float) -> List[str]:
        """Suggest actions to reduce portfolio risk"""
        suggestions = []
        
        within_limits, violations = self.check_portfolio_risk_limits(portfolio_value)
        if within_limits:
            return ["Portfolio risk is within acceptable limits"]
        
        # Analyze violations and suggest specific actions
        for violation in violations:
            if "Portfolio risk too high" in violation:
                # Suggest reducing position sizes
                largest_risks = sorted(self.positions.values(), 
                                     key=lambda p: p.risk_amount, reverse=True)
                suggestions.append(f"Consider reducing position size in {largest_risks[0].symbol} (highest risk)")
            
            elif "Too many positions" in violation:
                suggestions.append("Consider closing some smaller or underperforming positions")
            
            elif "Excessive" in violation and "exposure" in violation:
                # Sector concentration issue
                sector = violation.split()[1]  # Extract sector name
                sector_positions = [p for p in self.positions.values() if p.sector == sector]
                suggestions.append(f"Reduce {sector} exposure by closing or trimming positions")
            
            elif "correlated positions" in violation:
                sector = violation.split()[-1]  # Extract sector name  
                suggestions.append(f"Reduce correlation risk in {sector} sector")
        
        return suggestions
    
    def get_risk_alert_level(self, portfolio_value: float) -> str:
        """Get current risk alert level"""
        within_limits, violations = self.check_portfolio_risk_limits(portfolio_value)
        
        if not violations:
            return "GREEN"  # All good
        
        # Count severity of violations
        severe_violations = sum(1 for v in violations if any(word in v.lower() 
                               for word in ["too high", "excessive", "too many"]))
        
        if severe_violations >= 2:
            return "RED"    # Multiple severe violations
        elif severe_violations >= 1:
            return "YELLOW" # Single severe violation
        else:
            return "GREEN"  # Minor violations only
    
    def export_risk_report(self) -> Dict:
        """Export comprehensive risk report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'positions': {symbol: {
                'symbol': pos.symbol,
                'size': pos.position_size,
                'value': pos.position_value,
                'risk': pos.risk_amount,
                'pnl': pos.unrealized_pnl,
                'sector': pos.sector
            } for symbol, pos in self.positions.items()},
            'risk_limits': {
                'max_portfolio_risk': self.config.MAX_PORTFOLIO_RISK,
                'max_concurrent_positions': self.config.MAX_CONCURRENT_POSITIONS,
                'max_sector_exposure': self.config.MAX_SECTOR_EXPOSURE,
                'max_correlated_positions': self.config.MAX_CORRELATED_POSITIONS
            }
        }