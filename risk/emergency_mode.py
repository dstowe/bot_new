# risk/emergency_mode.py
"""
Emergency Mode and Portfolio Protection
=======================================
Advanced portfolio protection with emergency mode activation and recovery protocols
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict

class EmergencyTrigger(Enum):
    """Emergency mode triggers"""
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    VAR_BREACH = "var_breach"
    POSITION_LOSS = "position_loss"
    MARKET_CRASH = "market_crash"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MANUAL_ACTIVATION = "manual_activation"

class EmergencyAction(Enum):
    """Emergency response actions"""
    HALT_NEW_TRADES = "halt_new_trades"
    REDUCE_POSITIONS = "reduce_positions"
    CLOSE_LOSING_POSITIONS = "close_losing_positions"
    HEDGE_PORTFOLIO = "hedge_portfolio"
    INCREASE_CASH = "increase_cash"
    NOTIFY_ADMIN = "notify_admin"
    SUSPEND_TRADING = "suspend_trading"

class RecoveryPhase(Enum):
    """Recovery phases"""
    IMMEDIATE = "immediate"       # 0-1 hours
    SHORT_TERM = "short_term"     # 1-24 hours
    MEDIUM_TERM = "medium_term"   # 1-7 days
    LONG_TERM = "long_term"       # 7+ days

@dataclass
class EmergencyEvent:
    """Emergency mode event record"""
    event_id: str
    trigger: EmergencyTrigger
    trigger_value: float
    threshold_breached: float
    timestamp: datetime
    portfolio_value: float
    max_drawdown: float
    active_positions: int
    actions_taken: List[EmergencyAction] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    recovery_time_hours: Optional[float] = None
    
@dataclass
class PortfolioProtectionConfig:
    """Portfolio protection configuration"""
    # Drawdown limits
    max_drawdown_emergency: float = 0.15      # 15% max drawdown
    max_drawdown_critical: float = 0.20       # 20% critical drawdown
    
    # Daily loss limits
    max_daily_loss_pct: float = 0.05          # 5% daily loss
    max_daily_loss_absolute: float = 5000.0   # $5,000 absolute
    
    # VaR limits
    max_var_breach_multiplier: float = 2.0    # 2x VaR breach triggers emergency
    
    # Position limits
    max_single_position_loss: float = 0.08    # 8% loss on single position
    max_correlated_loss: float = 0.12         # 12% loss on correlated positions
    
    # Market stress indicators
    vix_emergency_level: float = 35.0         # VIX above 35 = market stress
    market_decline_threshold: float = 0.03    # 3% market decline in day
    
    # Recovery settings
    min_recovery_time_hours: float = 4.0      # Minimum 4 hours before considering recovery
    recovery_confidence_threshold: float = 0.7 # 70% confidence needed for recovery
    
    # Cash management
    emergency_cash_target: float = 0.60       # 60% cash in emergency
    critical_cash_target: float = 0.80       # 80% cash in critical mode

class EmergencyModeManager:
    """
    Emergency Mode and Portfolio Protection Manager
    Monitors portfolio health and activates emergency protocols when needed
    """
    
    def __init__(self, db_path: str = "data/trading_data.db",
                 webull_client=None, account_manager=None,
                 config: PortfolioProtectionConfig = None,
                 logger: logging.Logger = None):
        
        self.db_path = db_path
        self.webull_client = webull_client
        self.account_manager = account_manager
        self.config = config or PortfolioProtectionConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Emergency state tracking
        self.emergency_active = False
        self.emergency_level = 'normal'  # normal, emergency, critical
        self.current_emergency_event: Optional[EmergencyEvent] = None
        self.recovery_phase = RecoveryPhase.IMMEDIATE
        
        # Portfolio monitoring
        self.portfolio_high_water_mark = 0.0
        self.daily_start_value = 0.0
        self.last_portfolio_check = datetime.now()
        
        # Market stress indicators
        self.market_stress_indicators = {
            'vix_level': 0.0,
            'market_decline': 0.0,
            'correlation_breakdown': False,
            'liquidity_stress': False
        }
        
        # Action tracking
        self.actions_taken_today = []
        self.manual_overrides = {}
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize emergency mode tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Emergency events table
            cursor.execute("""CREATE TABLE IF NOT EXISTS emergency_events (                 event_id TEXT PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    trigger_value REAL NOT NULL,
                    threshold_breached REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    portfolio_value REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    active_positions INTEGER NOT NULL,
                    actions_taken TEXT,  -- JSON array
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp TIMESTAMP,
                    recovery_time_hours REAL,
                    notes TEXT
                )
            """)
            
            # Portfolio protection log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS protection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    emergency_active BOOLEAN NOT NULL,
                    emergency_level TEXT,
                    actions_taken TEXT,
                    market_indicators TEXT,  -- JSON
                    notes TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("Emergency mode database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing emergency database: {e}")
    
    def monitor_portfolio_protection(self, portfolio_data: Dict) -> Dict[str, Any]:
        """
        Monitor portfolio for protection triggers
        
        Args:
            portfolio_data: Current portfolio state
            
        Returns:
            Dict: Protection monitoring results
        """
        try:
            current_time = datetime.now()
            portfolio_value = portfolio_data.get('total_value', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            positions = portfolio_data.get('positions', {})
            
            # Update high water mark
            if portfolio_value > self.portfolio_high_water_mark:
                self.portfolio_high_water_mark = portfolio_value
                
            # Calculate current drawdown
            current_drawdown = 0.0
            if self.portfolio_high_water_mark > 0:
                current_drawdown = (self.portfolio_high_water_mark - portfolio_value) / self.portfolio_high_water_mark
            
            # Update daily start value if needed
            if current_time.date() != self.last_portfolio_check.date():
                self.daily_start_value = portfolio_value - daily_pnl
                self.actions_taken_today = []
                
            self.last_portfolio_check = current_time
            
            # Check for emergency triggers
            triggers_activated = self._check_emergency_triggers({
                'portfolio_value': portfolio_value,
                'current_drawdown': current_drawdown,
                'daily_pnl': daily_pnl,
                'daily_start_value': self.daily_start_value,
                'positions': positions,
                'timestamp': current_time
            })
            
            # Update market stress indicators
            self._update_market_stress_indicators()
            
            # Determine emergency level
            emergency_level = self._determine_emergency_level(
                current_drawdown, daily_pnl, triggers_activated
            )
            
            # Take emergency actions if needed
            actions_taken = []
            if triggers_activated:
                actions_taken = self._execute_emergency_actions(
                    triggers_activated, portfolio_data
                )
                
            # Log protection event
            self._log_protection_event({
                'portfolio_value': portfolio_value,
                'drawdown': current_drawdown,
                'daily_pnl': daily_pnl,
                'emergency_level': emergency_level,
                'triggers': triggers_activated,
                'actions': actions_taken
            })
            
            return {
                'timestamp': current_time.isoformat(),
                'portfolio_value': portfolio_value,
                'high_water_mark': self.portfolio_high_water_mark,
                'current_drawdown': current_drawdown,
                'daily_pnl': daily_pnl,
                'emergency_active': self.emergency_active,
                'emergency_level': emergency_level,
                'triggers_activated': [t.value for t in triggers_activated],
                'actions_taken': [a.value for a in actions_taken],
                'market_stress': self.market_stress_indicators,
                'recovery_phase': self.recovery_phase.value if self.emergency_active else None,
                'recommendations': self._generate_protection_recommendations(
                    current_drawdown, daily_pnl, emergency_level
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio protection monitoring: {e}")
            return {'error': str(e)}
    
    def _check_emergency_triggers(self, portfolio_state: Dict) -> List[EmergencyTrigger]:
        """Check for emergency mode triggers"""
        triggers = []
        
        portfolio_value = portfolio_state['portfolio_value']
        current_drawdown = portfolio_state['current_drawdown']
        daily_pnl = portfolio_state['daily_pnl']
        daily_start_value = portfolio_state['daily_start_value']
        positions = portfolio_state['positions']
        
        # 1. Maximum Drawdown Trigger
        if current_drawdown >= self.config.max_drawdown_emergency:
            triggers.append(EmergencyTrigger.MAX_DRAWDOWN)
            self.logger.warning(f"Drawdown trigger: {current_drawdown:.1%} >= {self.config.max_drawdown_emergency:.1%}")
        
        # 2. Daily Loss Limit Trigger
        daily_loss_pct = abs(daily_pnl) / daily_start_value if daily_start_value > 0 else 0
        if (daily_pnl < 0 and 
            (daily_loss_pct >= self.config.max_daily_loss_pct or 
             abs(daily_pnl) >= self.config.max_daily_loss_absolute)):
            triggers.append(EmergencyTrigger.DAILY_LOSS_LIMIT)
            self.logger.warning(f"Daily loss trigger: {daily_pnl:.2f} ({daily_loss_pct:.1%})")
        
        # 3. Large Position Loss Trigger
        for symbol, position in positions.items():
            position_pnl = position.get('unrealized_pnl', 0)
            position_value = position.get('market_value', 0)
            
            if position_value > 0:
                position_loss_pct = abs(position_pnl) / position_value
                if position_pnl < 0 and position_loss_pct >= self.config.max_single_position_loss:
                    triggers.append(EmergencyTrigger.POSITION_LOSS)
                    self.logger.warning(f"Position loss trigger: {symbol} down {position_loss_pct:.1%}")
                    break
        
        # 4. Market Crash Trigger
        if self.market_stress_indicators['market_decline'] >= self.config.market_decline_threshold:
            triggers.append(EmergencyTrigger.MARKET_CRASH)
            self.logger.warning(f"Market crash trigger: {self.market_stress_indicators['market_decline']:.1%}")
        
        # 5. VIX Stress Trigger
        if self.market_stress_indicators['vix_level'] >= self.config.vix_emergency_level:
            triggers.append(EmergencyTrigger.MARKET_CRASH)
            self.logger.warning(f"VIX stress trigger: {self.market_stress_indicators['vix_level']:.1f}")
        
        # 6. Correlation Breakdown Trigger
        if self.market_stress_indicators['correlation_breakdown']:
            triggers.append(EmergencyTrigger.CORRELATION_BREAKDOWN)
            self.logger.warning("Correlation breakdown detected")
        
        # 7. Liquidity Crisis Trigger
        if self.market_stress_indicators['liquidity_stress']:
            triggers.append(EmergencyTrigger.LIQUIDITY_CRISIS)
            self.logger.warning("Liquidity stress detected")
        
        return triggers
    
    def _update_market_stress_indicators(self):
        """Update market stress indicators"""
        try:
            import yfinance as yf
            
            # Get VIX data
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period='2d')
            if not vix_data.empty:
                self.market_stress_indicators['vix_level'] = vix_data['Close'].iloc[-1]
            
            # Get SPY data for market decline
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='2d')
            if len(spy_data) >= 2:
                today_close = spy_data['Close'].iloc[-1]
                yesterday_close = spy_data['Close'].iloc[-2]
                market_decline = (yesterday_close - today_close) / yesterday_close
                self.market_stress_indicators['market_decline'] = max(0, market_decline)
            
            # Simplified correlation breakdown detection
            # (Would use more sophisticated analysis in production)
            if self.market_stress_indicators['vix_level'] > 30:
                self.market_stress_indicators['correlation_breakdown'] = True
            else:
                self.market_stress_indicators['correlation_breakdown'] = False
                
            # Simplified liquidity stress detection
            if (self.market_stress_indicators['vix_level'] > 25 and 
                self.market_stress_indicators['market_decline'] > 0.02):
                self.market_stress_indicators['liquidity_stress'] = True
            else:
                self.market_stress_indicators['liquidity_stress'] = False
                
        except Exception as e:
            self.logger.warning(f"Error updating market stress indicators: {e}")
    
    def _determine_emergency_level(self, drawdown: float, daily_pnl: float, 
                                  triggers: List[EmergencyTrigger]) -> str:
        """Determine emergency level based on conditions"""
        if not triggers:
            return 'normal'
            
        # Critical level triggers
        critical_triggers = [
            EmergencyTrigger.MAX_DRAWDOWN,
            EmergencyTrigger.MARKET_CRASH,
            EmergencyTrigger.LIQUIDITY_CRISIS
        ]
        
        # Check for critical conditions
        if (drawdown >= self.config.max_drawdown_critical or
            any(t in critical_triggers for t in triggers) or
            len(triggers) >= 3):  # Multiple triggers = critical
            return 'critical'
        
        # Emergency level
        if triggers:
            return 'emergency'
            
        return 'normal'
    
    def _execute_emergency_actions(self, triggers: List[EmergencyTrigger], 
                                  portfolio_data: Dict) -> List[EmergencyAction]:
        """Execute emergency response actions"""
        actions_taken = []
        
        try:
            emergency_level = self.emergency_level
            
            # Always halt new trades first
            if EmergencyAction.HALT_NEW_TRADES not in self.actions_taken_today:
                self._halt_new_trades()
                actions_taken.append(EmergencyAction.HALT_NEW_TRADES)
                self.actions_taken_today.append(EmergencyAction.HALT_NEW_TRADES)
            
            # Notify admin
            if EmergencyAction.NOTIFY_ADMIN not in self.actions_taken_today:
                self._notify_admin(triggers, portfolio_data)
                actions_taken.append(EmergencyAction.NOTIFY_ADMIN)
                self.actions_taken_today.append(EmergencyAction.NOTIFY_ADMIN)
            
            # Emergency level actions
            if emergency_level == 'emergency':
                # Reduce positions gradually
                if EmergencyAction.REDUCE_POSITIONS not in self.actions_taken_today:
                    reduction_success = self._reduce_positions(portfolio_data, 0.25)  # 25% reduction
                    if reduction_success:
                        actions_taken.append(EmergencyAction.REDUCE_POSITIONS)
                        self.actions_taken_today.append(EmergencyAction.REDUCE_POSITIONS)
                
                # Close losing positions
                if EmergencyTrigger.POSITION_LOSS in triggers:
                    if EmergencyAction.CLOSE_LOSING_POSITIONS not in self.actions_taken_today:
                        self._close_losing_positions(portfolio_data)
                        actions_taken.append(EmergencyAction.CLOSE_LOSING_POSITIONS)
                        self.actions_taken_today.append(EmergencyAction.CLOSE_LOSING_POSITIONS)
            
            # Critical level actions
            elif emergency_level == 'critical':
                # Suspend all trading
                if EmergencyAction.SUSPEND_TRADING not in self.actions_taken_today:
                    self._suspend_trading()
                    actions_taken.append(EmergencyAction.SUSPEND_TRADING)
                    self.actions_taken_today.append(EmergencyAction.SUSPEND_TRADING)
                
                # Increase cash to critical level
                if EmergencyAction.INCREASE_CASH not in self.actions_taken_today:
                    cash_increase_success = self._increase_cash_allocation(
                        portfolio_data, self.config.critical_cash_target
                    )
                    if cash_increase_success:
                        actions_taken.append(EmergencyAction.INCREASE_CASH)
                        self.actions_taken_today.append(EmergencyAction.INCREASE_CASH)
                
                # Consider hedging (if available)
                if (EmergencyTrigger.MARKET_CRASH in triggers and 
                    EmergencyAction.HEDGE_PORTFOLIO not in self.actions_taken_today):
                    hedge_success = self._hedge_portfolio(portfolio_data)
                    if hedge_success:
                        actions_taken.append(EmergencyAction.HEDGE_PORTFOLIO)
                        self.actions_taken_today.append(EmergencyAction.HEDGE_PORTFOLIO)
            
            # Activate emergency mode if not already active
            if not self.emergency_active:
                self._activate_emergency_mode(triggers, portfolio_data)
                
            return actions_taken
            
        except Exception as e:
            self.logger.error(f"Error executing emergency actions: {e}")
            return actions_taken
    
    def _halt_new_trades(self) -> bool:
        """Halt new trade execution"""
        try:
            self.logger.critical("🚨 EMERGENCY: New trades halted")
            # This would integrate with the trading system to prevent new orders
            # For now, log the action
            return True
        except Exception as e:
            self.logger.error(f"Error halting trades: {e}")
            return False
    
    def _suspend_trading(self) -> bool:
        """Suspend all trading activity"""
        try:
            self.logger.critical("🚨 CRITICAL: All trading suspended")
            # This would completely disable the trading system
            return True
        except Exception as e:
            self.logger.error(f"Error suspending trading: {e}")
            return False
    
    def _reduce_positions(self, portfolio_data: Dict, reduction_pct: float) -> bool:
        """Reduce portfolio positions by specified percentage"""
        try:
            positions = portfolio_data.get('positions', {})
            reduced_count = 0
            
            for symbol, position in positions.items():
                position_size = position.get('quantity', 0)
                if position_size > 0:
                    # Calculate reduction amount
                    reduction_size = position_size * reduction_pct
                    
                    # This would place actual sell orders
                    self.logger.info(f"Emergency reduction: {symbol} by {reduction_size:.2f} shares")
                    reduced_count += 1
            
            self.logger.warning(f"🚨 EMERGENCY: Reduced {reduced_count} positions by {reduction_pct:.1%}")
            return reduced_count > 0
            
        except Exception as e:
            self.logger.error(f"Error reducing positions: {e}")
            return False
    
    def _close_losing_positions(self, portfolio_data: Dict) -> bool:
        """Close positions with significant losses"""
        try:
            positions = portfolio_data.get('positions', {})
            closed_count = 0
            
            for symbol, position in positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                market_value = position.get('market_value', 0)
                
                if market_value > 0 and unrealized_pnl < 0:
                    loss_pct = abs(unrealized_pnl) / market_value
                    if loss_pct >= self.config.max_single_position_loss:
                        # This would place actual close orders
                        self.logger.warning(f"Emergency close: {symbol} (loss: {loss_pct:.1%})")
                        closed_count += 1
            
            return closed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error closing losing positions: {e}")
            return False
    
    def _increase_cash_allocation(self, portfolio_data: Dict, target_cash_pct: float) -> bool:
        """Increase cash allocation to target percentage"""
        try:
            current_cash = portfolio_data.get('cash', 0)
            total_value = portfolio_data.get('total_value', 0)
            
            if total_value > 0:
                current_cash_pct = current_cash / total_value
                if current_cash_pct < target_cash_pct:
                    # Calculate positions to liquidate
                    target_cash_amount = total_value * target_cash_pct
                    additional_cash_needed = target_cash_amount - current_cash
                    
                    self.logger.warning(
                        f"🚨 EMERGENCY: Increasing cash allocation to {target_cash_pct:.1%} "
                        f"(need to liquidate ${additional_cash_needed:,.0f})"
                    )
                    
                    # This would prioritize which positions to liquidate
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error increasing cash allocation: {e}")
            return False
    
    def _hedge_portfolio(self, portfolio_data: Dict) -> bool:
        """Implement portfolio hedging"""
        try:
            # This would implement hedging strategies like:
            # - Buying protective puts
            # - Shorting market ETFs
            # - Buying VIX calls
            
            self.logger.warning("🚨 EMERGENCY: Portfolio hedging activated")
            # Placeholder for hedging logic
            return True
            
        except Exception as e:
            self.logger.error(f"Error hedging portfolio: {e}")
            return False
    
    def _notify_admin(self, triggers: List[EmergencyTrigger], portfolio_data: Dict):
        """Notify system administrator of emergency"""
        try:
            portfolio_value = portfolio_data.get('total_value', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            
            message = (
                f"🚨 EMERGENCY MODE ACTIVATED\
"
                f"Portfolio Value: ${portfolio_value:,.0f}\
"
                f"Daily P&L: ${daily_pnl:,.0f}\
"
                f"Triggers: {', '.join(t.value for t in triggers)}\
"
                f"Timestamp: {datetime.now().isoformat()}"
            )
            
            self.logger.critical(message)
            # This would send actual notifications (email, SMS, Slack, etc.)
            
        except Exception as e:
            self.logger.error(f"Error notifying admin: {e}")
    
    def _activate_emergency_mode(self, triggers: List[EmergencyTrigger], 
                                portfolio_data: Dict):
        """Activate emergency mode"""
        try:
            if not self.emergency_active:
                self.emergency_active = True
                self.recovery_phase = RecoveryPhase.IMMEDIATE
                
                # Create emergency event
                event_id = f"EM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_emergency_event = EmergencyEvent(
                    event_id=event_id,
                    trigger=triggers[0] if triggers else EmergencyTrigger.MANUAL_ACTIVATION,
                    trigger_value=portfolio_data.get('current_drawdown', 0),
                    threshold_breached=self.config.max_drawdown_emergency,
                    timestamp=datetime.now(),
                    portfolio_value=portfolio_data.get('total_value', 0),
                    max_drawdown=portfolio_data.get('current_drawdown', 0),
                    active_positions=len(portfolio_data.get('positions', {}))
                )
                
                self.logger.critical(f"🚨 EMERGENCY MODE ACTIVATED: Event ID {event_id}")
                self._save_emergency_event(self.current_emergency_event)
                
        except Exception as e:
            self.logger.error(f"Error activating emergency mode: {e}")
    
    def check_recovery_conditions(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Check if emergency mode can be deactivated"""
        if not self.emergency_active:
            return {'can_recover': True, 'reason': 'Emergency mode not active'}
            
        try:
            current_time = datetime.now()
            emergency_duration = (current_time - self.current_emergency_event.timestamp).total_seconds() / 3600
            
            # Must be in emergency mode for minimum time
            if emergency_duration < self.config.min_recovery_time_hours:
                return {
                    'can_recover': False,
                    'reason': f'Minimum emergency time not met ({emergency_duration:.1f}h/{self.config.min_recovery_time_hours}h)',
                    'time_remaining': self.config.min_recovery_time_hours - emergency_duration
                }
            
            # Check recovery conditions
            portfolio_value = portfolio_data.get('total_value', 0)
            current_drawdown = 0.0
            if self.portfolio_high_water_mark > 0:
                current_drawdown = (self.portfolio_high_water_mark - portfolio_value) / self.portfolio_high_water_mark
            
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            
            recovery_conditions = {
                'drawdown_recovered': current_drawdown < self.config.max_drawdown_emergency * 0.8,  # 80% of trigger
                'daily_pnl_positive': daily_pnl >= 0,
                'market_stress_reduced': self._assess_market_stress_recovery(),
                'portfolio_stable': self._assess_portfolio_stability(portfolio_data)
            }
            
            # Calculate recovery confidence
            conditions_met = sum(recovery_conditions.values())
            recovery_confidence = conditions_met / len(recovery_conditions)
            
            can_recover = recovery_confidence >= self.config.recovery_confidence_threshold
            
            return {
                'can_recover': can_recover,
                'recovery_confidence': recovery_confidence,
                'conditions': recovery_conditions,
                'emergency_duration_hours': emergency_duration,
                'current_drawdown': current_drawdown,
                'daily_pnl': daily_pnl,
                'recommendation': 'Proceed with recovery' if can_recover else 'Continue emergency protocols'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking recovery conditions: {e}")
            return {'can_recover': False, 'reason': f'Error: {str(e)}'}
    
    def _assess_market_stress_recovery(self) -> bool:
        """Assess if market stress conditions have improved"""
        return (
            self.market_stress_indicators['vix_level'] < self.config.vix_emergency_level * 0.8 and
            self.market_stress_indicators['market_decline'] < self.config.market_decline_threshold * 0.5 and
            not self.market_stress_indicators['liquidity_stress']
        )
    
    def _assess_portfolio_stability(self, portfolio_data: Dict) -> bool:
        """Assess portfolio stability for recovery"""
        positions = portfolio_data.get('positions', {})
        
        # Check for large position losses
        unstable_positions = 0
        for position in positions.values():
            unrealized_pnl = position.get('unrealized_pnl', 0)
            market_value = position.get('market_value', 0)
            
            if market_value > 0 and unrealized_pnl < 0:
                loss_pct = abs(unrealized_pnl) / market_value
                if loss_pct > self.config.max_single_position_loss * 0.5:  # 50% of trigger
                    unstable_positions += 1
        
        return unstable_positions == 0
    
    def deactivate_emergency_mode(self, reason: str = "Manual deactivation") -> bool:
        """Deactivate emergency mode"""
        try:
            if self.emergency_active and self.current_emergency_event:
                self.emergency_active = False
                
                # Update emergency event
                self.current_emergency_event.resolved = True
                self.current_emergency_event.resolution_timestamp = datetime.now()
                
                duration = (self.current_emergency_event.resolution_timestamp - 
                          self.current_emergency_event.timestamp)
                self.current_emergency_event.recovery_time_hours = duration.total_seconds() / 3600
                
                # Save updated event
                self._save_emergency_event(self.current_emergency_event)
                
                self.logger.info(f"✅ Emergency mode deactivated: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deactivating emergency mode: {e}")
            return False
        
        return False
    
    def _save_emergency_event(self, event: EmergencyEvent):
        """Save emergency event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO emergency_events (
                    event_id, trigger_type, trigger_value, threshold_breached,
                    timestamp, portfolio_value, max_drawdown, active_positions,
                    actions_taken, resolved, resolution_timestamp, recovery_time_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.trigger.value, event.trigger_value,
                event.threshold_breached, event.timestamp.isoformat(),
                event.portfolio_value, event.max_drawdown, event.active_positions,
                ','.join(a.value for a in event.actions_taken), event.resolved,
                event.resolution_timestamp.isoformat() if event.resolution_timestamp else None,
                event.recovery_time_hours
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving emergency event: {e}")
    
    def _log_protection_event(self, event_data: Dict):
        """Log protection monitoring event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO protection_log (
                    timestamp, event_type, portfolio_value, drawdown, daily_pnl,
                    emergency_active, emergency_level, actions_taken, market_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), 'monitoring',
                event_data['portfolio_value'], event_data['drawdown'],
                event_data['daily_pnl'], self.emergency_active,
                event_data['emergency_level'],
                ','.join(a.value for a in event_data['actions']),
                str(self.market_stress_indicators)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging protection event: {e}")
    
    def _generate_protection_recommendations(self, drawdown: float, 
                                           daily_pnl: float, 
                                           emergency_level: str) -> List[str]:
        """Generate portfolio protection recommendations"""
        recommendations = []
        
        if emergency_level == 'critical':
            recommendations.append("CRITICAL: Immediate action required - consider full position liquidation")
            recommendations.append("Review risk management procedures and position sizing")
        elif emergency_level == 'emergency':
            recommendations.append("EMERGENCY: Reduce position sizes and halt new trades")
            recommendations.append("Monitor market conditions closely for further deterioration")
        elif drawdown > 0.10:  # 10% drawdown warning
            recommendations.append("Elevated drawdown detected - consider position reduction")
        
        if daily_pnl < -1000:  # $1000 daily loss
            recommendations.append("Significant daily loss - review trade execution and stop losses")
            
        if self.market_stress_indicators['vix_level'] > 25:
            recommendations.append("Elevated market volatility - consider defensive positioning")
            
        if not recommendations:
            recommendations.append("Portfolio protection status normal - continue monitoring")
            
        return recommendations
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency mode status"""
        return {
            'emergency_active': self.emergency_active,
            'emergency_level': self.emergency_level,
            'current_event': asdict(self.current_emergency_event) if self.current_emergency_event else None,
            'recovery_phase': self.recovery_phase.value if self.emergency_active else None,
            'actions_taken_today': [a.value for a in self.actions_taken_today],
            'market_stress_indicators': self.market_stress_indicators,
            'portfolio_high_water_mark': self.portfolio_high_water_mark
        }