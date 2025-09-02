# wyckoff_bot/analytics/performance_analytics.py
"""
Performance Analytics Module
============================
Comprehensive trade attribution analysis with risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy import stats
import sqlite3
from collections import defaultdict

@dataclass
class TradeMetrics:
    """Individual trade performance metrics"""
    trade_id: str
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl_gross: float
    pnl_net: float
    fees: float
    hold_time_days: float
    return_pct: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    wyckoff_phase: str
    confidence_score: float
    market_regime: str
    sector: str
    trade_type: str  # 'buy', 'scale_in', 'scale_out', 'close_long'

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return Metrics
    total_return: float
    annualized_return: float
    cumulative_pnl: float
    win_rate: float
    profit_factor: float
    
    # Risk Metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # Risk-Adjusted Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_hold_time: float
    
    # Advanced Metrics
    kelly_criterion: float
    recovery_factor: float
    expectancy: float
    trade_efficiency: float
    market_correlation: float

@dataclass
class AttributionAnalysis:
    """Trade attribution breakdown"""
    by_phase: Dict[str, PerformanceMetrics]
    by_regime: Dict[str, PerformanceMetrics]
    by_sector: Dict[str, PerformanceMetrics]
    by_month: Dict[str, PerformanceMetrics]
    by_confidence: Dict[str, PerformanceMetrics]
    by_hold_time: Dict[str, PerformanceMetrics]

class PerformanceAnalyzer:
    """
    Comprehensive performance analytics engine
    Provides institutional-grade performance attribution and risk metrics
    """
    
    def __init__(self, db_path: str = "data/trading_data.db", 
                 benchmark_symbol: str = "SPY",
                 risk_free_rate: float = 0.02,
                 logger: logging.Logger = None):
        
        self.db_path = db_path
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance tracking
        self.equity_curve = pd.Series()
        self.benchmark_returns = pd.Series()
        self.trade_history: List[TradeMetrics] = []
        
        # Attribution categories
        self.attribution_categories = {
            'phase': ['accumulation', 'markup', 'distribution', 'markdown', 'unknown'],
            'regime': ['bull', 'bear', 'range', 'transition', 'unknown'],
            'confidence': ['high', 'medium', 'low'],  # Based on confidence score ranges
            'hold_time': ['intraday', 'short_term', 'medium_term', 'long_term']  # <1, 1-7, 7-30, >30 days
        }
        
        # Benchmark data cache
        self.benchmark_cache = {}
        self.cache_expiry = datetime.now()
        
    def analyze_performance(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Tuple[PerformanceMetrics, AttributionAnalysis]:
        """
        Perform comprehensive performance analysis
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Tuple[PerformanceMetrics, AttributionAnalysis]: Complete performance analysis
        """
        try:
            self.logger.info("Starting comprehensive performance analysis...")
            
            # Load trade data
            trades = self._load_trade_data(start_date, end_date)
            
            if not trades:
                self.logger.warning("No trade data found for analysis period")
                return self._create_empty_metrics(), self._create_empty_attribution()
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(trades)
            self.equity_curve = equity_curve
            
            # Load benchmark data
            benchmark_returns = self._load_benchmark_data(start_date, end_date)
            self.benchmark_returns = benchmark_returns
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(trades, equity_curve, benchmark_returns)
            
            # Perform attribution analysis
            attribution_analysis = self._perform_attribution_analysis(trades)
            
            self.logger.info(f"Performance analysis completed: {len(trades)} trades analyzed")
            return performance_metrics, attribution_analysis
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            return self._create_empty_metrics(), self._create_empty_attribution()



    def _load_trade_data(self, start_date: Optional[datetime], 
                        end_date: Optional[datetime]) -> List[TradeMetrics]:
        """Load trade data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query with date filters
            query = """
            SELECT 
                id,
                symbol,
                date as entry_date,
                CASE WHEN action = 'SELL' THEN date ELSE NULL END as exit_date,
                price as entry_price,
                CASE WHEN action = 'SELL' THEN price ELSE NULL END as exit_price,
                quantity as position_size,
                CASE WHEN action = 'SELL' THEN total_value ELSE -total_value END as pnl_gross,
                CASE WHEN action = 'SELL' THEN total_value ELSE -total_value END as pnl_net,
                0 as fees,
                signal_phase as wyckoff_phase,
                signal_strength as confidence_score,
                'unknown' as market_regime,
                'Unknown' as sector,
                action as trade_type
            FROM trades 
            WHERE 1=1
            """
            
            params = []
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())
                
            query += " ORDER BY date"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                entry_date = datetime.fromisoformat(row[2])
                exit_date = datetime.fromisoformat(row[3]) if row[3] else None
                
                # Calculate derived metrics
                hold_time_days = 0
                return_pct = 0
                
                # Only calculate metrics for trades that have exit data
                if exit_date and entry_date and row[5] is not None:  # exit_price is not None
                    try:
                        time_diff = exit_date - entry_date
                        hold_time_days = max(1, time_diff.days if hasattr(time_diff, 'days') else 1)
                    except (AttributeError, TypeError):
                        # If dates are the same or there's an issue, default to 1 day
                        hold_time_days = 1
                    
                    if row[4] > 0:  # entry_price > 0
                        return_pct = ((row[5] or 0) - row[4]) / row[4]  # (exit_price - entry_price) / entry_price
                else:
                    # For open positions or buy-only trades, set reasonable defaults
                    hold_time_days = 1
                    return_pct = 0
                
                # These would ideally come from trade execution tracking
                max_adverse_excursion = abs(min(0, return_pct)) * row[4] * row[6]  # Simplified
                max_favorable_excursion = max(0, return_pct) * row[4] * row[6]  # Simplified
                
                trade = TradeMetrics(
                    trade_id=str(row[0]),
                    symbol=row[1],
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=row[4],
                    exit_price=row[5],
                    position_size=row[6],
                    pnl_gross=row[7] or 0,
                    pnl_net=row[8] or 0,
                    fees=row[9] or 0,
                    hold_time_days=float(hold_time_days),
                    return_pct=return_pct,
                    max_adverse_excursion=max_adverse_excursion,
                    max_favorable_excursion=max_favorable_excursion,
                    wyckoff_phase=row[10] or 'unknown',
                    confidence_score=row[11] or 0.0,
                    market_regime=row[12] or 'unknown',
                    sector=row[13] or 'Unknown',
                    trade_type=row[14] or 'buy'
                )
                trades.append(trade)
            
            conn.close()
            self.trade_history = trades
            self.logger.debug(f"Loaded {len(trades)} trades from database")
            return trades
            
        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")
            return []
    
    def _calculate_equity_curve(self, trades: List[TradeMetrics]) -> pd.Series:
        """Calculate equity curve from trades"""
        if not trades:
            return pd.Series()
            
        # Create daily equity curve
        start_date = min(trade.entry_date for trade in trades)
        end_date = max(trade.exit_date for trade in trades if trade.exit_date) or datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        equity_curve = pd.Series(0.0, index=date_range)
        
        # Assume starting capital (would come from account data in production)
        starting_capital = 10000.0
        equity_curve.iloc[0] = starting_capital
        
        # Add completed trades
        for trade in trades:
            if trade.exit_date and trade.pnl_net != 0:
                equity_curve[trade.exit_date] += trade.pnl_net
                
        # Calculate cumulative equity
        equity_curve = equity_curve.cumsum() + starting_capital
        
        return equity_curve
    
    def _load_benchmark_data(self, start_date: Optional[datetime], 
                           end_date: Optional[datetime]) -> pd.Series:
        """Load benchmark data for comparison"""
        try:
            # Use cached data if available and fresh
            cache_key = f"{start_date}_{end_date}_{self.benchmark_symbol}"
            if (cache_key in self.benchmark_cache and 
                (datetime.now() - self.cache_expiry).total_seconds() < 3600):  # 1 hour cache
                return self.benchmark_cache[cache_key]
            
            import yfinance as yf
            
            # Default date range if not specified
            if not start_date:
                start_date = datetime.now() - timedelta(days=365)
            if not end_date:
                end_date = datetime.now()
                
            ticker = yf.Ticker(self.benchmark_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No benchmark data available for {self.benchmark_symbol}")
                return pd.Series()
                
            returns = data['Close'].pct_change().fillna(0)
            
            # Cache the result
            self.benchmark_cache[cache_key] = returns
            self.cache_expiry = datetime.now()
            
            return returns
            
        except Exception as e:
            self.logger.warning(f"Error loading benchmark data: {e}")
            return pd.Series()
    
    def _calculate_performance_metrics(self, trades: List[TradeMetrics], 
                                     equity_curve: pd.Series,
                                     benchmark_returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._create_empty_metrics()
            
        try:
            # Basic trade statistics
            completed_trades = [t for t in trades if t.exit_date is not None]
            total_trades = len(completed_trades)
            
            if total_trades == 0:
                return self._create_empty_metrics()
            
            pnl_values = [t.pnl_net for t in completed_trades]
            returns = [t.return_pct for t in completed_trades]
            
            winning_trades = [t for t in completed_trades if t.pnl_net > 0]
            losing_trades = [t for t in completed_trades if t.pnl_net < 0]
            
            # Return metrics
            total_return = sum(pnl_values)
            cumulative_return = total_return / (equity_curve.iloc[0] if len(equity_curve) > 0 else 10000.0)
            
            # Calculate annualized return
            if len(equity_curve) > 1:
                time_diff = equity_curve.index[-1] - equity_curve.index[0]
                days = time_diff.days if hasattr(time_diff, 'days') else int(time_diff.total_seconds() / 86400) if hasattr(time_diff, 'total_seconds') else 1
                annualized_return = (1 + cumulative_return) ** (365.25 / max(days, 1)) - 1
            else:
                annualized_return = 0
            
            # Win rate and profit factor
            win_rate = len(winning_trades) / total_trades
            gross_profit = sum(t.pnl_net for t in winning_trades)
            gross_loss = abs(sum(t.pnl_net for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk metrics
            returns_series = pd.Series(returns)
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
            
            # Drawdown calculation
            max_drawdown, max_dd_duration = self._calculate_drawdown(equity_curve)
            
            # VaR calculation
            var_95 = np.percentile(pnl_values, 5) if pnl_values else 0
            var_99 = np.percentile(pnl_values, 1) if pnl_values else 0
            expected_shortfall = np.mean([x for x in pnl_values if x <= var_95]) if pnl_values else 0
            
            # Risk-adjusted metrics
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Information ratio (vs benchmark)
            information_ratio = self._calculate_information_ratio(equity_curve, benchmark_returns)
            
            # Trade statistics
            avg_win = np.mean([t.pnl_net for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl_net for t in losing_trades]) if losing_trades else 0
            largest_win = max([t.pnl_net for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.pnl_net for t in losing_trades]) if losing_trades else 0
            avg_hold_time = np.mean([float(t.hold_time_days) for t in completed_trades]) if completed_trades else 0
            
            # Advanced metrics
            kelly_criterion = self._calculate_kelly_criterion(winning_trades, losing_trades)
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            trade_efficiency = len(winning_trades) / total_trades if total_trades > 0 else 0
            market_correlation = self._calculate_market_correlation(equity_curve, benchmark_returns)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                cumulative_pnl=total_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                volatility=volatility,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_hold_time=avg_hold_time,
                kelly_criterion=kelly_criterion,
                recovery_factor=recovery_factor,
                expectancy=expectancy,
                trade_efficiency=trade_efficiency,
                market_correlation=market_correlation
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_metrics()
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if len(equity_curve) < 2:
            return 0.0, 0
            
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        max_duration = 0
        current_duration = 0
        
        for is_drawdown in drawdown_periods:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_drawdown, max_duration
    
    def _calculate_information_ratio(self, equity_curve: pd.Series, 
                                   benchmark_returns: pd.Series) -> float:
        """Calculate information ratio vs benchmark"""
        if len(equity_curve) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        try:
            # Calculate portfolio returns
            portfolio_returns = equity_curve.pct_change().fillna(0)
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 10:  # Need sufficient data
                return 0.0
                
            aligned_portfolio = portfolio_returns.loc[common_dates]
            aligned_benchmark = benchmark_returns.loc[common_dates]
            
            # Calculate tracking error and excess return
            excess_returns = aligned_portfolio - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            if tracking_error == 0:
                return 0.0
                
            excess_return = excess_returns.mean() * 252
            information_ratio = excess_return / tracking_error
            
            return information_ratio
            
        except Exception as e:
            self.logger.warning(f"Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_kelly_criterion(self, winning_trades: List[TradeMetrics], 
                                 losing_trades: List[TradeMetrics]) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not winning_trades or not losing_trades:
            return 0.0
            
        try:
            avg_win_pct = np.mean([t.return_pct for t in winning_trades])
            avg_loss_pct = abs(np.mean([t.return_pct for t in losing_trades]))
            win_probability = len(winning_trades) / (len(winning_trades) + len(losing_trades))
            
            if avg_loss_pct == 0:
                return 0.0
                
            kelly_pct = win_probability - ((1 - win_probability) / (avg_win_pct / avg_loss_pct))
            return max(0, min(kelly_pct, 0.25))  # Cap at 25% for safety
            
        except Exception as e:
            self.logger.warning(f"Error calculating Kelly criterion: {e}")
            return 0.0
    
    def _calculate_market_correlation(self, equity_curve: pd.Series, 
                                    benchmark_returns: pd.Series) -> float:
        """Calculate correlation with market benchmark"""
        if len(equity_curve) < 10 or len(benchmark_returns) < 10:
            return 0.0
            
        try:
            portfolio_returns = equity_curve.pct_change().fillna(0)
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) < 10:
                return 0.0
                
            aligned_portfolio = portfolio_returns.loc[common_dates]
            aligned_benchmark = benchmark_returns.loc[common_dates]
            
            correlation = aligned_portfolio.corr(aligned_benchmark)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating market correlation: {e}")
            return 0.0
    
    def _perform_attribution_analysis(self, trades: List[TradeMetrics]) -> AttributionAnalysis:
        """Perform comprehensive attribution analysis"""
        completed_trades = [t for t in trades if t.exit_date is not None]
        
        if not completed_trades:
            return self._create_empty_attribution()
            
        try:
            # Group trades by different attributes
            attribution = AttributionAnalysis(
                by_phase=self._group_by_phase(completed_trades),
                by_regime=self._group_by_regime(completed_trades),
                by_sector=self._group_by_sector(completed_trades),
                by_month=self._group_by_month(completed_trades),
                by_confidence=self._group_by_confidence(completed_trades),
                by_hold_time=self._group_by_hold_time(completed_trades)
            )
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error in attribution analysis: {e}")
            return self._create_empty_attribution()
    
    def _group_by_phase(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by Wyckoff phase"""
        phase_groups = defaultdict(list)
        for trade in trades:
            phase_groups[trade.wyckoff_phase].append(trade)
            
        return {phase: self._calculate_group_metrics(group_trades) 
                for phase, group_trades in phase_groups.items()}
    
    def _group_by_regime(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by market regime"""
        regime_groups = defaultdict(list)
        for trade in trades:
            regime_groups[trade.market_regime].append(trade)
            
        return {regime: self._calculate_group_metrics(group_trades) 
                for regime, group_trades in regime_groups.items()}
    
    def _group_by_sector(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by sector"""
        sector_groups = defaultdict(list)
        for trade in trades:
            sector_groups[trade.sector].append(trade)
            
        return {sector: self._calculate_group_metrics(group_trades) 
                for sector, group_trades in sector_groups.items()}
    
    def _group_by_month(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by month"""
        month_groups = defaultdict(list)
        for trade in trades:
            month_key = trade.entry_date.strftime('%Y-%m')
            month_groups[month_key].append(trade)
            
        return {month: self._calculate_group_metrics(group_trades) 
                for month, group_trades in month_groups.items()}
    
    def _group_by_confidence(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by confidence level"""
        confidence_groups = defaultdict(list)
        for trade in trades:
            if trade.confidence_score >= 0.8:
                confidence_groups['high'].append(trade)
            elif trade.confidence_score >= 0.6:
                confidence_groups['medium'].append(trade)
            else:
                confidence_groups['low'].append(trade)
                
        return {conf: self._calculate_group_metrics(group_trades) 
                for conf, group_trades in confidence_groups.items()}
    
    def _group_by_hold_time(self, trades: List[TradeMetrics]) -> Dict[str, PerformanceMetrics]:
        """Group performance by hold time"""
        hold_time_groups = defaultdict(list)
        for trade in trades:
            if trade.hold_time_days < 1:
                hold_time_groups['intraday'].append(trade)
            elif trade.hold_time_days <= 7:
                hold_time_groups['short_term'].append(trade)
            elif trade.hold_time_days <= 30:
                hold_time_groups['medium_term'].append(trade)
            else:
                hold_time_groups['long_term'].append(trade)
                
        return {period: self._calculate_group_metrics(group_trades) 
                for period, group_trades in hold_time_groups.items()}
    
    def _calculate_group_metrics(self, trades: List[TradeMetrics]) -> PerformanceMetrics:
        """Calculate performance metrics for a group of trades"""
        if not trades:
            return self._create_empty_metrics()
            
        # Create a mini equity curve for this group
        group_equity = pd.Series([10000.0])  # Starting value
        for trade in trades:
            group_equity = pd.concat([group_equity, pd.Series([group_equity.iloc[-1] + trade.pnl_net])])
            
        return self._calculate_performance_metrics(trades, group_equity, pd.Series())
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, cumulative_pnl=0.0,
            win_rate=0.0, profit_factor=0.0, volatility=0.0,
            max_drawdown=0.0, max_drawdown_duration=0,
            var_95=0.0, var_99=0.0, expected_shortfall=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, information_ratio=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0, avg_hold_time=0.0,
            kelly_criterion=0.0, recovery_factor=0.0, expectancy=0.0,
            trade_efficiency=0.0, market_correlation=0.0
        )
    
    def _create_empty_attribution(self) -> AttributionAnalysis:
        """Create empty attribution analysis"""
        empty_metrics = self._create_empty_metrics()
        return AttributionAnalysis(
            by_phase={}, by_regime={}, by_sector={},
            by_month={}, by_confidence={}, by_hold_time={}
        )
    
    def generate_performance_report(self, metrics: PerformanceMetrics, 
                                  attribution: AttributionAnalysis) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'summary': {
                'total_return': f"{metrics.total_return:.2f}",
                'annualized_return': f"{metrics.annualized_return:.1%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.1%}",
                'win_rate': f"{metrics.win_rate:.1%}",
                'total_trades': metrics.total_trades
            },
            'risk_metrics': {
                'volatility': f"{metrics.volatility:.1%}",
                'var_95': f"{metrics.var_95:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'information_ratio': f"{metrics.information_ratio:.2f}"
            },
            'trade_analysis': {
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'expectancy': f"{metrics.expectancy:.2f}",
                'avg_win': f"{metrics.avg_win:.2f}",
                'avg_loss': f"{metrics.avg_loss:.2f}",
                'avg_hold_time': f"{metrics.avg_hold_time:.1f} days",
                'kelly_criterion': f"{metrics.kelly_criterion:.1%}"
            },
            'attribution': {
                'best_phase': self._get_best_performing_category(attribution.by_phase),
                'best_regime': self._get_best_performing_category(attribution.by_regime),
                'best_sector': self._get_best_performing_category(attribution.by_sector),
                'best_confidence': self._get_best_performing_category(attribution.by_confidence)
            },
            'recommendations': self._generate_performance_recommendations(metrics, attribution)
        }
        
        return report
    
    def _get_best_performing_category(self, category_metrics: Dict[str, PerformanceMetrics]) -> str:
        """Get best performing category by total return"""
        if not category_metrics:
            return "N/A"
            
        best_category = max(category_metrics.keys(), 
                          key=lambda k: category_metrics[k].total_return)
        best_return = category_metrics[best_category].total_return
        
        return f"{best_category} ({best_return:.2f})"
    
    def _generate_performance_recommendations(self, metrics: PerformanceMetrics, 
                                            attribution: AttributionAnalysis) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk-adjusted returns - Sharpe ratio below 0.5")
        elif metrics.sharpe_ratio > 2.0:
            recommendations.append("Excellent risk-adjusted returns - consider increasing position sizes")
            
        # Win rate recommendations
        if metrics.win_rate < 0.4:
            recommendations.append("Low win rate detected - review entry criteria and signal quality")
        elif metrics.win_rate > 0.7:
            recommendations.append("High win rate - consider taking more selective, higher-conviction trades")
            
        # Drawdown recommendations
        if metrics.max_drawdown < -0.20:  # Less than -20%
            recommendations.append("Significant drawdown detected - review risk management and position sizing")
            
        # Profit factor recommendations
        if metrics.profit_factor < 1.2:
            recommendations.append("Low profit factor - focus on improving trade selection or risk/reward ratios")
            
        # Attribution-based recommendations
        if attribution.by_confidence:
            high_conf_metrics = attribution.by_confidence.get('high')
            low_conf_metrics = attribution.by_confidence.get('low')
            
            if (high_conf_metrics and low_conf_metrics and 
                high_conf_metrics.total_return > low_conf_metrics.total_return * 2):
                recommendations.append("High confidence trades significantly outperform - focus on quality over quantity")
                
        if not recommendations:
            recommendations.append("Performance metrics within acceptable ranges - continue current strategy")
            
        return recommendations
    
    def export_detailed_analysis(self, output_path: str, metrics: PerformanceMetrics, 
                               attribution: AttributionAnalysis) -> bool:
        """Export detailed analysis to CSV files"""
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export trade history
            if self.trade_history:
                trades_df = pd.DataFrame([asdict(trade) for trade in self.trade_history])
                trades_df.to_csv(f"{output_path}_trades.csv", index=False)
                
            # Export equity curve
            if not self.equity_curve.empty:
                equity_df = pd.DataFrame({'date': self.equity_curve.index, 'equity': self.equity_curve.values})
                equity_df.to_csv(f"{output_path}_equity_curve.csv", index=False)
                
            # Export attribution analysis
            attribution_data = []
            for category, results in attribution.by_phase.items():
                attribution_data.append({
                    'category': 'phase',
                    'subcategory': category,
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'win_rate': results.win_rate,
                    'total_trades': results.total_trades
                })
                
            if attribution_data:
                attribution_df = pd.DataFrame(attribution_data)
                attribution_df.to_csv(f"{output_path}_attribution.csv", index=False)
                
            self.logger.info(f"Detailed analysis exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
            return False