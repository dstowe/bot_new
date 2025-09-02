# risk/account_risk_manager.py
"""
Account-Level Risk Manager
=========================
Monitors and enforces account-specific risk limits
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from collections import defaultdict

@dataclass
class DailyRiskMetrics:
    """Daily risk tracking for an account"""
    account_id: str
    date: str
    starting_balance: float
    current_balance: float
    daily_pnl: float
    max_daily_loss_hit: bool
    trades_halted: bool
    risk_percentage: float
    var_1d_95: float = 0.0
    var_1d_99: float = 0.0
    expected_shortfall: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    beta_to_market: float = 1.0
    correlation_to_market: float = 0.0

class AccountRiskManager:
    """
    Enhanced Account Risk Manager with VaR and Advanced Risk Analytics
    Manages risk at the account level with institutional-grade risk metrics
    """
    
    def __init__(self, config, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Track daily metrics per account
        self.daily_metrics: Dict[str, DailyRiskMetrics] = {}
        
        # Emergency mode tracking
        self.emergency_mode = False
        self.emergency_mode_start = None
        
        # VaR calculation parameters
        self.var_lookback_days = 252  # 1 year of trading days
        self.var_confidence_levels = [0.95, 0.99]
        
        # Historical data cache for VaR calculations
        self.historical_returns_cache: Dict[str, pd.Series] = {}
        self.cache_expiry = datetime.now()
        
        # Market benchmark for beta calculations
        self.benchmark_symbol = 'SPY'
        self.benchmark_returns = None
        
    def check_daily_risk_limits(self, account_id: str, account_balance: float, 
                              starting_balance: float) -> Tuple[bool, str]:
        """
        Check if account has hit daily risk limits
        
        Returns:
            Tuple[bool, str]: (can_trade, reason)
        """
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = account_balance - starting_balance
        
        # Update daily metrics
        self.daily_metrics[account_id] = DailyRiskMetrics(
            account_id=account_id,
            date=today,
            starting_balance=starting_balance,
            current_balance=account_balance,
            daily_pnl=daily_pnl,
            max_daily_loss_hit=False,
            trades_halted=False,
            risk_percentage=abs(daily_pnl) / starting_balance if starting_balance > 0 else 0
        )
        
        # Check absolute daily loss limit
        if daily_pnl <= -self.config.MAX_DAILY_LOSS:
            self.daily_metrics[account_id].max_daily_loss_hit = True
            self.daily_metrics[account_id].trades_halted = True
            self.logger.error(f"🚨 Daily loss limit hit for {account_id}: ${daily_pnl:.2f}")
            return False, f"Daily loss limit exceeded: ${daily_pnl:.2f}"
        
        # Check percentage daily loss limit
        loss_percentage = abs(daily_pnl) / starting_balance if starting_balance > 0 else 0
        if daily_pnl < 0 and loss_percentage >= self.config.MAX_DAILY_LOSS_PERCENT:
            self.daily_metrics[account_id].max_daily_loss_hit = True
            self.daily_metrics[account_id].trades_halted = True
            self.logger.error(f"🚨 Daily loss percentage limit hit for {account_id}: {loss_percentage:.1%}")
            return False, f"Daily loss percentage exceeded: {loss_percentage:.1%}"
        
        # Calculate VaR if we have sufficient historical data
        var_metrics = self._calculate_var_metrics(account_id, account_balance)
        if var_metrics:
            self.daily_metrics[account_id].var_1d_95 = var_metrics.get('var_1d_95', 0.0)
            self.daily_metrics[account_id].var_1d_99 = var_metrics.get('var_1d_99', 0.0)
            self.daily_metrics[account_id].expected_shortfall = var_metrics.get('expected_shortfall', 0.0)
            self.daily_metrics[account_id].volatility = var_metrics.get('volatility', 0.0)
            self.daily_metrics[account_id].sharpe_ratio = var_metrics.get('sharpe_ratio', 0.0)
            self.daily_metrics[account_id].max_drawdown = var_metrics.get('max_drawdown', 0.0)
            self.daily_metrics[account_id].beta_to_market = var_metrics.get('beta_to_market', 1.0)
            self.daily_metrics[account_id].correlation_to_market = var_metrics.get('correlation_to_market', 0.0)
        
        return True, "Risk limits OK"
    
    def calculate_max_position_size(self, account_id: str, account_balance: float, 
                                  entry_price: float, stop_loss: float) -> float:
        """
        Calculate maximum position size based on risk limits
        
        Args:
            account_id: Account identifier
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss: Stop loss price
            
        Returns:
            float: Maximum position size in shares/units
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
        
        # Calculate maximum risk amount
        max_risk_amount = account_balance * self.config.MAX_SINGLE_POSITION_RISK
        
        # Calculate position size
        max_position_size = max_risk_amount / risk_per_share
        
        # Apply account-specific limits
        account_config = self._get_account_config(account_id)
        if account_config:
            max_position_value = account_balance * account_config.get('max_position_size', 0.25)
            max_shares_by_value = max_position_value / entry_price
            max_position_size = min(max_position_size, max_shares_by_value)
        
        self.logger.debug(f"Max position size for {account_id}: {max_position_size:.2f} shares")
        return max_position_size
    
    def check_emergency_mode(self, portfolio_value: float, 
                           portfolio_high_water_mark: float) -> bool:
        """
        Check if emergency mode should be activated
        
        Args:
            portfolio_value: Current total portfolio value
            portfolio_high_water_mark: Highest portfolio value reached
            
        Returns:
            bool: True if emergency mode should be active
        """
        # Check if we're in cooldown period
        if self.emergency_mode_start:
            cooldown_end = self.emergency_mode_start + timedelta(
                seconds=self.config.EMERGENCY_MODE_COOLDOWN
            )
            if datetime.now() < cooldown_end:
                return True  # Stay in emergency mode during cooldown
        
        # Calculate drawdown
        if portfolio_high_water_mark > 0:
            drawdown = (portfolio_high_water_mark - portfolio_value) / portfolio_high_water_mark
            
            if drawdown >= self.config.MAX_DRAWDOWN_PERCENT:
                if not self.emergency_mode:
                    self.emergency_mode = True
                    self.emergency_mode_start = datetime.now()
                    self.logger.error(f"🚨 EMERGENCY MODE ACTIVATED - Drawdown: {drawdown:.1%}")
                return True
        
        # Exit emergency mode if conditions improve and cooldown passed
        if self.emergency_mode and self.emergency_mode_start:
            cooldown_end = self.emergency_mode_start + timedelta(
                seconds=self.config.EMERGENCY_MODE_COOLDOWN
            )
            if datetime.now() >= cooldown_end:
                self.emergency_mode = False
                self.emergency_mode_start = None
                self.logger.info("✅ Emergency mode deactivated - cooldown period completed")
        
        return False
    
    def get_account_risk_status(self, account_id: str) -> Dict:
        """Get current risk status for account"""
        metrics = self.daily_metrics.get(account_id)
        
        if not metrics:
            return {
                'account_id': account_id,
                'risk_status': 'unknown',
                'can_trade': True,
                'daily_pnl': 0.0,
                'risk_percentage': 0.0
            }
        
        risk_status = 'normal'
        if metrics.max_daily_loss_hit:
            risk_status = 'daily_limit_hit'
        elif metrics.risk_percentage > 0.01:  # 1% risk threshold
            risk_status = 'elevated'
        
        return {
            'account_id': account_id,
            'risk_status': risk_status,
            'can_trade': not metrics.trades_halted,
            'daily_pnl': metrics.daily_pnl,
            'risk_percentage': metrics.risk_percentage,
            'emergency_mode': self.emergency_mode,
            'var_1d_95': metrics.var_1d_95,
            'var_1d_99': metrics.var_1d_99,
            'expected_shortfall': metrics.expected_shortfall,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'beta_to_market': metrics.beta_to_market,
            'risk_assessment': self._assess_risk_level(metrics) if metrics else 'UNKNOWN'
        }
    
    def reset_daily_limits(self, account_id: str) -> None:
        """Reset daily limits (typically called at market open)"""
        if account_id in self.daily_metrics:
            metrics = self.daily_metrics[account_id]
            metrics.max_daily_loss_hit = False
            metrics.trades_halted = False
            self.logger.info(f"📊 Daily risk limits reset for {account_id}")
    
    def _get_account_config(self, account_id: str) -> Optional[Dict]:
        """Get account-specific configuration"""
        # This would map account_id to account type
        # For now, return default configuration
        return self.config.ACCOUNT_CONFIGURATIONS.get('CASH', {})
    
    def _calculate_var_metrics(self, account_id: str, account_balance: float) -> Optional[Dict]:
        """
        Calculate Value at Risk and other advanced risk metrics
        
        Args:
            account_id: Account identifier
            account_balance: Current account balance
            
        Returns:
            Dict: VaR and risk metrics, or None if insufficient data
        """
        try:
            # Get historical portfolio returns (simulated for now)
            returns = self._get_portfolio_returns_history(account_id)
            
            if returns is None or len(returns) < 30:
                self.logger.debug(f"Insufficient data for VaR calculation: {account_id}")
                return None
            
            metrics = {}
            
            # 1. Value at Risk (Historical Simulation Method)
            metrics['var_1d_95'] = self._calculate_historical_var(returns, 0.95) * account_balance
            metrics['var_1d_99'] = self._calculate_historical_var(returns, 0.99) * account_balance
            
            # 2. Expected Shortfall (Conditional VaR)
            metrics['expected_shortfall'] = self._calculate_expected_shortfall(returns, 0.95) * account_balance
            
            # 3. Volatility (annualized)
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            # 4. Sharpe Ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            metrics['sharpe_ratio'] = excess_returns / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
            # 5. Maximum Drawdown
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            
            # 6. Beta and Correlation to Market
            market_metrics = self._calculate_market_metrics(returns)
            metrics.update(market_metrics)
            
            # 7. VaR-based Position Sizing Recommendation
            metrics['var_position_limit'] = self._calculate_var_based_position_limit(metrics['var_1d_95'], account_balance)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating VaR metrics for {account_id}: {e}")
            return None
    
    def _get_portfolio_returns_history(self, account_id: str) -> Optional[pd.Series]:
        """
        Get historical portfolio returns for VaR calculation
        In a real implementation, this would fetch actual portfolio performance data
        """
        # For now, simulate portfolio returns based on market data
        # In production, this should fetch actual account performance history
        
        try:
            # Use SPY as proxy for portfolio returns (would be actual portfolio data in production)
            spy = yf.Ticker('SPY')
            data = spy.history(period='1y')
            
            if data.empty:
                return None
                
            returns = data['Close'].pct_change().dropna()
            
            # Add some portfolio-specific noise (in reality, this would be actual portfolio returns)
            portfolio_multiplier = 1.2  # Assume slightly higher volatility than market
            portfolio_returns = returns * portfolio_multiplier + np.random.normal(0, 0.002, len(returns))
            
            return pd.Series(portfolio_returns, index=returns.index)
            
        except Exception as e:
            self.logger.warning(f"Error fetching portfolio returns history: {e}")
            return None
    
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Value at Risk using historical simulation method
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            float: VaR as a positive number representing potential loss
        """
        if len(returns) == 0:
            return 0.0
            
        # Calculate percentile for loss (negative returns)
        var_percentile = 1 - confidence_level
        var_value = returns.quantile(var_percentile)
        
        # Return as positive number (potential loss)
        return abs(var_value) if var_value < 0 else 0.0
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level
            
        Returns:
            float: Expected shortfall as positive number
        """
        if len(returns) == 0:
            return 0.0
            
        var_threshold = returns.quantile(1 - confidence_level)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
            
        expected_shortfall = tail_losses.mean()
        return abs(expected_shortfall) if expected_shortfall < 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns series
        
        Args:
            returns: Historical returns series
            
        Returns:
            float: Maximum drawdown as positive percentage
        """
        if len(returns) == 0:
            return 0.0
            
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Return maximum drawdown as positive number
        return abs(drawdown.min())
    
    def _calculate_market_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate beta and correlation to market benchmark
        
        Args:
            portfolio_returns: Portfolio returns series
            
        Returns:
            Dict: Market-relative metrics
        """
        metrics = {'beta_to_market': 1.0, 'correlation_to_market': 0.0}
        
        try:
            # Get benchmark returns
            if self.benchmark_returns is None or len(self.benchmark_returns) < len(portfolio_returns):
                spy = yf.Ticker(self.benchmark_symbol)
                benchmark_data = spy.history(period='1y')
                if not benchmark_data.empty:
                    self.benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            if self.benchmark_returns is not None and len(self.benchmark_returns) > 20:
                # Align dates
                common_dates = portfolio_returns.index.intersection(self.benchmark_returns.index)
                if len(common_dates) > 20:
                    aligned_portfolio = portfolio_returns.loc[common_dates]
                    aligned_benchmark = self.benchmark_returns.loc[common_dates]
                    
                    # Calculate correlation
                    correlation = aligned_portfolio.corr(aligned_benchmark)
                    metrics['correlation_to_market'] = correlation if not np.isnan(correlation) else 0.0
                    
                    # Calculate beta
                    if aligned_benchmark.var() > 0:
                        covariance = aligned_portfolio.cov(aligned_benchmark)
                        beta = covariance / aligned_benchmark.var()
                        metrics['beta_to_market'] = beta if not np.isnan(beta) else 1.0
                        
        except Exception as e:
            self.logger.warning(f"Error calculating market metrics: {e}")
            
        return metrics
    
    def _calculate_var_based_position_limit(self, var_1d_95: float, account_balance: float) -> float:
        """
        Calculate position size limit based on VaR
        
        Args:
            var_1d_95: 1-day 95% VaR
            account_balance: Current account balance
            
        Returns:
            float: Maximum position size as percentage of account
        """
        if var_1d_95 <= 0 or account_balance <= 0:
            return 0.25  # Default to 25% position limit
            
        # Target maximum 1% portfolio risk from VaR
        target_var_percentage = 0.01
        current_var_percentage = var_1d_95 / account_balance
        
        if current_var_percentage <= 0:
            return 0.25
            
        # Scale position limit based on VaR
        var_based_limit = target_var_percentage / current_var_percentage
        
        # Cap at reasonable limits
        return min(max(var_based_limit, 0.05), 0.50)  # Between 5% and 50%
    
    def get_var_report(self, account_id: str) -> Dict:
        """
        Get comprehensive VaR report for account
        
        Args:
            account_id: Account identifier
            
        Returns:
            Dict: Complete VaR and risk analysis
        """
        metrics = self.daily_metrics.get(account_id)
        
        if not metrics:
            return {'error': f'No metrics available for account {account_id}'}
            
        return {
            'account_id': account_id,
            'date': metrics.date,
            'portfolio_value': metrics.current_balance,
            'daily_pnl': metrics.daily_pnl,
            'risk_metrics': {
                'var_1d_95_percent': (metrics.var_1d_95 / metrics.current_balance * 100) if metrics.current_balance > 0 else 0,
                'var_1d_95_dollar': metrics.var_1d_95,
                'var_1d_99_percent': (metrics.var_1d_99 / metrics.current_balance * 100) if metrics.current_balance > 0 else 0,
                'var_1d_99_dollar': metrics.var_1d_99,
                'expected_shortfall_dollar': metrics.expected_shortfall,
                'annualized_volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'beta_to_market': metrics.beta_to_market,
                'correlation_to_market': metrics.correlation_to_market
            },
            'risk_assessment': self._assess_risk_level(metrics),
            'recommendations': self._generate_risk_recommendations(metrics)
        }
    
    def _assess_risk_level(self, metrics: DailyRiskMetrics) -> str:
        """
        Assess overall risk level based on multiple metrics
        
        Args:
            metrics: Daily risk metrics
            
        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        risk_score = 0
        
        # VaR-based scoring
        var_percentage = (metrics.var_1d_95 / metrics.current_balance) if metrics.current_balance > 0 else 0
        if var_percentage > 0.05:  # >5% daily VaR
            risk_score += 3
        elif var_percentage > 0.03:  # >3% daily VaR
            risk_score += 2
        elif var_percentage > 0.015:  # >1.5% daily VaR
            risk_score += 1
            
        # Volatility scoring
        if metrics.volatility > 0.40:  # >40% annualized volatility
            risk_score += 2
        elif metrics.volatility > 0.25:  # >25% annualized volatility
            risk_score += 1
            
        # Drawdown scoring
        if metrics.max_drawdown > 0.20:  # >20% max drawdown
            risk_score += 2
        elif metrics.max_drawdown > 0.10:  # >10% max drawdown
            risk_score += 1
            
        # Sharpe ratio scoring (negative scoring for poor risk-adjusted returns)
        if metrics.sharpe_ratio < -0.5:
            risk_score += 2
        elif metrics.sharpe_ratio < 0:
            risk_score += 1
            
        # Risk level determination
        if risk_score >= 6:
            return 'CRITICAL'
        elif risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_risk_recommendations(self, metrics: DailyRiskMetrics) -> List[str]:
        """
        Generate risk management recommendations
        
        Args:
            metrics: Daily risk metrics
            
        Returns:
            List[str]: Risk management recommendations
        """
        recommendations = []
        
        var_percentage = (metrics.var_1d_95 / metrics.current_balance) if metrics.current_balance > 0 else 0
        
        # VaR-based recommendations
        if var_percentage > 0.05:
            recommendations.append("Reduce position sizes - daily VaR exceeds 5% of portfolio")
        elif var_percentage > 0.03:
            recommendations.append("Consider reducing position concentration - daily VaR is elevated")
            
        # Volatility recommendations
        if metrics.volatility > 0.40:
            recommendations.append("Portfolio volatility is very high - consider more defensive positions")
            
        # Drawdown recommendations
        if metrics.max_drawdown > 0.15:
            recommendations.append("Significant drawdown detected - review stop-loss strategies")
            
        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 0:
            recommendations.append("Poor risk-adjusted returns - review trading strategy")
        elif metrics.sharpe_ratio < 0.5:
            recommendations.append("Below-average risk-adjusted returns - consider strategy optimization")
            
        # Beta recommendations
        if metrics.beta_to_market > 1.5:
            recommendations.append("High market sensitivity - consider hedging during volatile periods")
        elif metrics.beta_to_market < 0.3:
            recommendations.append("Low market correlation - ensure adequate diversification")
            
        if not recommendations:
            recommendations.append("Risk metrics within acceptable ranges - continue monitoring")
            
        return recommendations
    
    def get_risk_summary(self) -> Dict:
        """Get overall risk summary"""
        total_accounts = len(self.daily_metrics)
        halted_accounts = sum(1 for m in self.daily_metrics.values() if m.trades_halted)
        
        # Calculate aggregate VaR metrics
        aggregate_var_95 = sum(m.var_1d_95 for m in self.daily_metrics.values() if m.var_1d_95 > 0)
        aggregate_var_99 = sum(m.var_1d_99 for m in self.daily_metrics.values() if m.var_1d_99 > 0)
        avg_sharpe = np.mean([m.sharpe_ratio for m in self.daily_metrics.values() if m.sharpe_ratio != 0]) if self.daily_metrics else 0
        
        return {
            'total_accounts': total_accounts,
            'halted_accounts': halted_accounts,
            'emergency_mode': self.emergency_mode,
            'emergency_mode_start': self.emergency_mode_start.isoformat() if self.emergency_mode_start else None,
            'aggregate_var_95': aggregate_var_95,
            'aggregate_var_99': aggregate_var_99,
            'average_sharpe_ratio': avg_sharpe,
            'risk_utilization': halted_accounts / total_accounts if total_accounts > 0 else 0
        }