# compliance/pdt_protection.py
"""
Pattern Day Trading (PDT) Protection Module
============================================
Comprehensive PDT compliance with real Webull account data integration
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import defaultdict

class AccountType(Enum):
    """Account types with different PDT rules"""
    CASH = "cash"
    MARGIN = "margin" 
    IRA = "ira"
    ROTH_IRA = "roth_ira"

class TradeDirection(Enum):
    """Trade direction for PDT calculations"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class DayTrade:
    """Individual day trade record"""
    symbol: str
    buy_time: datetime
    sell_time: datetime
    buy_price: float
    sell_price: float
    quantity: float
    account_id: str
    trade_date: datetime
    pnl: float

@dataclass
class PDTStatus:
    """PDT status for an account"""
    account_id: str
    account_type: AccountType
    account_value: float
    is_pdt_account: bool
    day_trades_used: int
    day_trades_remaining: int
    reset_date: datetime
    can_day_trade: bool
    restrictions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class TradeValidation:
    """Trade validation result"""
    is_valid: bool
    can_execute: bool
    reason: str
    warnings: List[str] = field(default_factory=list)
    alternative_suggestions: List[str] = field(default_factory=list)

class PDTProtectionManager:
    """
    Pattern Day Trading Protection Manager
    Prevents PDT violations using real Webull account data
    """
    
    def __init__(self, db_path: str = "data/trading_data.db", 
                 webull_client=None, account_manager=None,
                 logger: logging.Logger = None):
        
        self.db_path = db_path
        self.webull_client = webull_client
        self.account_manager = account_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # PDT Rules Configuration
        self.pdt_rules = {
            'min_account_value_pdt': 25000.0,  # Minimum for PDT account
            'max_day_trades_non_pdt': 3,       # Max day trades in 5 business days for non-PDT
            'day_trade_buying_power_ratio': 4,  # 4:1 leverage for PDT accounts
            'lookback_days': 5,                 # Rolling 5 business day period
            'settlement_days': 2                # T+2 settlement
        }

        
        # Account-specific PDT settings
        self.account_pdt_settings = {
            AccountType.CASH: {
                'day_trading_allowed': True,
                'pdt_rules_apply': False,
                'settlement_required': True,
                'max_daily_trades': None  # No limit for cash accounts
            },
            AccountType.MARGIN: {
                'day_trading_allowed': True,
                'pdt_rules_apply': True,
                'settlement_required': False,
                'max_daily_trades': None
            },
            AccountType.IRA: {
                'day_trading_allowed': False,  # Generally prohibited
                'pdt_rules_apply': False,
                'settlement_required': True,
                'max_daily_trades': 0
            },
            AccountType.ROTH_IRA: {
                'day_trading_allowed': False,  # Generally prohibited
                'pdt_rules_apply': False,
                'settlement_required': True,
                'max_daily_trades': 0
            }
        }
        
        # Day trade tracking
        self.day_trades_cache = {}
        self.cache_expiry = {}
        
        # Initialize database tables
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize PDT tracking database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Day trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS day_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    buy_time TIMESTAMP NOT NULL,
                    sell_time TIMESTAMP NOT NULL,
                    buy_price REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # PDT violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdt_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    violation_date DATE NOT NULL,
                    violation_type TEXT NOT NULL,
                    description TEXT,
                    day_trades_count INTEGER,
                    account_value REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Account PDT status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_pdt_status (
                    account_id TEXT PRIMARY KEY,
                    account_type TEXT NOT NULL,
                    is_pdt_account BOOLEAN DEFAULT FALSE,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    day_trades_reset_date DATE,
                    restrictions TEXT,  -- JSON string of current restrictions
                    notes TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("PDT protection database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing PDT database: {e}")
    
    def check_pdt_compliance(self, account_id: str, symbol: str, 
                           trade_action: str) -> TradeValidation:
        """
        Check if a trade would violate PDT rules
        
        Args:
            account_id: Account identifier
            symbol: Stock symbol
            trade_action: 'buy' or 'sell'
            
        Returns:
            TradeValidation: Detailed validation result
        """
        try:
            # Get current PDT status
            pdt_status = self.get_account_pdt_status(account_id)
            
            if not pdt_status:
                return TradeValidation(
                    is_valid=False,
                    can_execute=False,
                    reason="Could not determine account PDT status",
                    warnings=["Account information unavailable"]
                )
            
            # Check if day trading is allowed for this account type
            account_settings = self.account_pdt_settings.get(pdt_status.account_type)
            if not account_settings or not account_settings['day_trading_allowed']:
                return TradeValidation(
                    is_valid=False,
                    can_execute=False,
                    reason=f"Day trading not allowed for {pdt_status.account_type.value} accounts",
                    warnings=["Consider using a margin account for day trading"]
                )
            
            # For sell orders, check if this would create a day trade
            if trade_action.lower() == 'sell':
                validation = self._validate_potential_day_trade(
                    account_id, symbol, pdt_status
                )
                return validation
            
            # For buy orders, check buying power and existing positions
            elif trade_action.lower() == 'buy':
                validation = self._validate_buy_order_pdt(
                    account_id, symbol, pdt_status
                )
                return validation
            
            else:
                return TradeValidation(
                    is_valid=False,
                    can_execute=False,
                    reason=f"Unknown trade action: {trade_action}"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking PDT compliance: {e}")
            return TradeValidation(
                is_valid=False,
                can_execute=False,
                reason=f"PDT compliance check failed: {str(e)}"
            )
    
    def _validate_potential_day_trade(self, account_id: str, symbol: str, 
                                     pdt_status: PDTStatus) -> TradeValidation:
        """Validate if selling would create a day trade violation"""
        warnings = []
        
        # Check if we have same-day purchases that would create a day trade
        today_purchases = self._get_today_purchases(account_id, symbol)
        
        if not today_purchases:
            # No same-day purchases, so selling won't create a day trade
            return TradeValidation(
                is_valid=True,
                can_execute=True,
                reason="No same-day purchases found - not a day trade"
            )
        
        # This would be a day trade, check if allowed
        if pdt_status.day_trades_remaining <= 0 and not pdt_status.is_pdt_account:
            return TradeValidation(
                is_valid=False,
                can_execute=False,
                reason=f"Day trade limit exceeded ({pdt_status.day_trades_used}/3 used)",
                warnings=[
                    "Consider holding position overnight to avoid PDT violation",
                    f"Account needs ${self.pdt_rules['min_account_value_pdt']:,.0f} to become PDT account"
                ],
                alternative_suggestions=[
                    "Hold position until tomorrow",
                    "Deposit funds to reach $25,000 minimum",
                    "Use a different account if available"
                ]
            )
        
        # Day trade is allowed
        if pdt_status.day_trades_remaining > 0:
            warnings.append(f"This will use 1 day trade ({pdt_status.day_trades_remaining-1} remaining)")
        
        return TradeValidation(
            is_valid=True,
            can_execute=True,
            reason="Day trade allowed",
            warnings=warnings
        )
    
    def _validate_buy_order_pdt(self, account_id: str, symbol: str, 
                               pdt_status: PDTStatus) -> TradeValidation:
        """Validate buy order considering potential future day trading"""
        warnings = []
        
        # Check day trade buying power for PDT accounts
        if pdt_status.is_pdt_account:
            buying_power = self._get_day_trade_buying_power(account_id)
            if buying_power <= 0:
                return TradeValidation(
                    is_valid=False,
                    can_execute=False,
                    reason="Insufficient day trade buying power",
                    warnings=["Wait for buying power to reset or deposit additional funds"]
                )
        
        # Warn about potential day trade implications
        if not pdt_status.is_pdt_account and pdt_status.day_trades_remaining <= 1:
            warnings.append(
                f"Caution: Only {pdt_status.day_trades_remaining} day trade(s) remaining. "
                "Selling today would use your last day trade."
            )
        
        return TradeValidation(
            is_valid=True,
            can_execute=True,
            reason="Buy order allowed",
            warnings=warnings
        )
    
    def get_account_pdt_status(self, account_id: str) -> Optional[PDTStatus]:
        """
        Get comprehensive PDT status for account using real Webull data
        
        Args:
            account_id: Account identifier
            
        Returns:
            PDTStatus: Current PDT status or None if unavailable
        """
        try:
            # Try to get real account data from Webull
            real_account_data = self._get_real_account_data(account_id)
            
            if real_account_data:
                return self._create_pdt_status_from_real_data(account_id, real_account_data)
            
            # Fallback to cached/database data
            return self._get_cached_pdt_status(account_id)
            
        except Exception as e:
            self.logger.error(f"Error getting PDT status for {account_id}: {e}")
            return None
    
    def _get_real_account_data(self, account_id: str) -> Optional[Dict]:
        """Get real account data from Webull API"""
        try:
            if not self.webull_client or not self.account_manager:
                return None
            
            # Get account from account manager
            accounts = self.account_manager.accounts
            account = next((acc for acc in accounts if acc.account_id == account_id), None)
            
            if not account:
                return None
            
            # Get detailed account information
            account_details = self.webull_client.get_account()
            day_trades_info = self.webull_client.get_day_trades()
            
            return {
                'account_value': account.net_liquidation,
                'account_type': account.account_type,
                'day_trading_buying_power': getattr(account, 'day_trading_buying_power', 0),
                'is_pdt': account_details.get('is_pdt', False),
                'day_trades_count': day_trades_info.get('day_trades_count', 0),
                'day_trades_remaining': day_trades_info.get('day_trades_remaining', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Could not fetch real account data: {e}")
            return None
    
    def _create_pdt_status_from_real_data(self, account_id: str, 
                                         real_data: Dict) -> PDTStatus:
        """Create PDT status from real Webull account data"""
        account_value = real_data.get('account_value', 0)
        account_type_str = real_data.get('account_type', 'margin')
        
        # Map account type string to enum
        account_type_mapping = {
            'cash': AccountType.CASH,
            'margin': AccountType.MARGIN,
            'ira': AccountType.IRA,
            'roth': AccountType.ROTH_IRA,
            'roth_ira': AccountType.ROTH_IRA
        }
        
        account_type = account_type_mapping.get(account_type_str.lower(), AccountType.MARGIN)
        
        # Determine PDT status
        is_pdt_account = (
            real_data.get('is_pdt', False) or 
            account_value >= self.pdt_rules['min_account_value_pdt']
        )
        
        day_trades_used = real_data.get('day_trades_count', 0)
        day_trades_remaining = (
            float('inf') if is_pdt_account 
            else max(0, self.pdt_rules['max_day_trades_non_pdt'] - day_trades_used)
        )
        
        # Calculate reset date (next Monday if not PDT)
        reset_date = datetime.now()
        if not is_pdt_account:
            days_until_monday = (7 - reset_date.weekday()) % 7
            if days_until_monday == 0:  # If today is Monday
                days_until_monday = 7
            reset_date = reset_date + timedelta(days=days_until_monday)
        
        # Check restrictions
        restrictions = []
        warnings = []
        
        if account_type in [AccountType.IRA, AccountType.ROTH_IRA]:
            restrictions.append("Day trading prohibited in retirement accounts")
        
        if not is_pdt_account and day_trades_used >= self.pdt_rules['max_day_trades_non_pdt']:
            restrictions.append("Day trade limit reached - 90-day restriction may apply")
        
        if account_value < self.pdt_rules['min_account_value_pdt'] and day_trades_used >= 2:
            warnings.append(f"Close to day trade limit - need ${self.pdt_rules['min_account_value_pdt']:,.0f} for unlimited day trading")
        
        can_day_trade = (
            len(restrictions) == 0 and 
            (is_pdt_account or day_trades_remaining > 0)
        )
        
        return PDTStatus(
            account_id=account_id,
            account_type=account_type,
            account_value=account_value,
            is_pdt_account=is_pdt_account,
            day_trades_used=day_trades_used,
            day_trades_remaining=int(day_trades_remaining) if day_trades_remaining != float('inf') else 999,
            reset_date=reset_date,
            can_day_trade=can_day_trade,
            restrictions=restrictions,
            warnings=warnings
        )
    
    def _get_cached_pdt_status(self, account_id: str) -> Optional[PDTStatus]:
        """Get PDT status from cache/database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT account_type, is_pdt_account, day_trades_reset_date, restrictions
                FROM account_pdt_status 
                WHERE account_id = ?
            """, (account_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            # Get recent day trades count
            cursor.execute("""
                SELECT COUNT(*) FROM day_trades 
                WHERE account_id = ? AND trade_date >= date('now', '-5 days')
            """, (account_id,))
            
            day_trades_count = cursor.fetchone()[0]
            conn.close()
            
            # Create PDT status from cached data
            account_type = AccountType(row[0])
            is_pdt_account = bool(row[1])
            
            return PDTStatus(
                account_id=account_id,
                account_type=account_type,
                account_value=25000.0 if is_pdt_account else 10000.0,  # Estimated
                is_pdt_account=is_pdt_account,
                day_trades_used=day_trades_count,
                day_trades_remaining=(
                    999 if is_pdt_account 
                    else max(0, self.pdt_rules['max_day_trades_non_pdt'] - day_trades_count)
                ),
                reset_date=datetime.now() + timedelta(days=1),
                can_day_trade=is_pdt_account or day_trades_count < self.pdt_rules['max_day_trades_non_pdt'],
                restrictions=[],
                warnings=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cached PDT status: {e}")
            return None
    
    def _get_today_purchases(self, account_id: str, symbol: str) -> List[Dict]:
        """Get today's purchases for a symbol that could create day trades"""
        try:
            # This would integrate with the trade tracking system
            # For now, return mock data structure
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            cursor.execute("""
                SELECT buy_time, buy_price, quantity 
                FROM day_trades 
                WHERE account_id = ? AND symbol = ? AND trade_date = ?
                AND sell_time IS NULL  -- Only open positions
            """, (account_id, symbol, today))
            
            purchases = []
            for row in cursor.fetchall():
                purchases.append({
                    'time': datetime.fromisoformat(row[0]),
                    'price': row[1],
                    'quantity': row[2]
                })
                
            conn.close()
            return purchases
            
        except Exception as e:
            self.logger.error(f"Error getting today's purchases: {e}")
            return []
    
    def _get_day_trade_buying_power(self, account_id: str) -> float:
        """Get available day trade buying power"""
        try:
            if self.webull_client:
                account_info = self.webull_client.get_account()
                return account_info.get('day_trading_buying_power', 0)
            
            # Fallback calculation
            pdt_status = self.get_account_pdt_status(account_id)
            if pdt_status and pdt_status.is_pdt_account:
                return pdt_status.account_value * self.pdt_rules['day_trade_buying_power_ratio']
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting day trade buying power: {e}")
            return 0.0
    
    def record_day_trade(self, account_id: str, symbol: str, 
                        buy_time: datetime, sell_time: datetime,
                        buy_price: float, sell_price: float, 
                        quantity: float) -> bool:
        """
        Record a completed day trade
        
        Args:
            account_id: Account identifier
            symbol: Stock symbol
            buy_time: Purchase timestamp
            sell_time: Sale timestamp
            buy_price: Purchase price
            sell_price: Sale price
            quantity: Number of shares
            
        Returns:
            bool: Success status
        """
        try:
            trade_date = buy_time.date()
            pnl = (sell_price - buy_price) * quantity
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO day_trades (
                    account_id, symbol, trade_date, buy_time, sell_time,
                    buy_price, sell_price, quantity, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id, symbol, trade_date, buy_time.isoformat(),
                sell_time.isoformat(), buy_price, sell_price, quantity, pnl
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Recorded day trade: {symbol} for account {account_id}")
            
            # Clear cache to force refresh
            if account_id in self.day_trades_cache:
                del self.day_trades_cache[account_id]
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording day trade: {e}")
            return False
    
    def get_day_trades_summary(self, account_id: str, 
                              days_back: int = 30) -> Dict[str, Any]:
        """
        Get summary of day trading activity
        
        Args:
            account_id: Account identifier
            days_back: Number of days to look back
            
        Returns:
            Dict: Day trading summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days_back)).date()
            
            # Get day trades count and PnL
            cursor.execute("""
                SELECT COUNT(*), SUM(pnl), AVG(pnl)
                FROM day_trades 
                WHERE account_id = ? AND trade_date >= ?
            """, (account_id, start_date))
            
            count, total_pnl, avg_pnl = cursor.fetchone()
            
            # Get recent 5-day count for PDT tracking
            five_days_ago = (datetime.now() - timedelta(days=5)).date()
            cursor.execute("""
                SELECT COUNT(*) FROM day_trades 
                WHERE account_id = ? AND trade_date >= ?
            """, (account_id, five_days_ago))
            
            recent_count = cursor.fetchone()[0]
            
            # Get most traded symbols
            cursor.execute("""
                SELECT symbol, COUNT(*), SUM(pnl)
                FROM day_trades 
                WHERE account_id = ? AND trade_date >= ?
                GROUP BY symbol
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """, (account_id, start_date))
            
            top_symbols = []
            for row in cursor.fetchall():
                top_symbols.append({
                    'symbol': row[0],
                    'trades': row[1],
                    'pnl': row[2]
                })
            
            conn.close()
            
            # Get current PDT status
            pdt_status = self.get_account_pdt_status(account_id)
            
            return {
                'account_id': account_id,
                'period_days': days_back,
                'total_day_trades': count or 0,
                'total_pnl': total_pnl or 0.0,
                'average_pnl': avg_pnl or 0.0,
                'recent_5day_count': recent_count or 0,
                'day_trades_remaining': pdt_status.day_trades_remaining if pdt_status else 0,
                'is_pdt_account': pdt_status.is_pdt_account if pdt_status else False,
                'can_day_trade': pdt_status.can_day_trade if pdt_status else False,
                'top_symbols': top_symbols,
                'restrictions': pdt_status.restrictions if pdt_status else [],
                'warnings': pdt_status.warnings if pdt_status else []
            }
            
        except Exception as e:
            self.logger.error(f"Error getting day trades summary: {e}")
            return {
                'account_id': account_id,
                'error': str(e)
            }
    
    def update_account_pdt_status(self, account_id: str, 
                                 account_type: str, account_value: float) -> bool:
        """
        Update account PDT status in database
        
        Args:
            account_id: Account identifier
            account_type: Account type (cash, margin, ira, roth_ira)
            account_value: Current account value
            
        Returns:
            bool: Success status
        """
        try:
            is_pdt = account_value >= self.pdt_rules['min_account_value_pdt']
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO account_pdt_status 
                (account_id, account_type, is_pdt_account, last_update)
                VALUES (?, ?, ?, ?)
            """, (account_id, account_type, is_pdt, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated PDT status for {account_id}: PDT={is_pdt}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating PDT status: {e}")
            return False
    
    def get_compliance_report(self, account_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive PDT compliance report
        
        Args:
            account_id: Account identifier
            
        Returns:
            Dict: Compliance report
        """
        try:
            pdt_status = self.get_account_pdt_status(account_id)
            day_trades_summary = self.get_day_trades_summary(account_id, 30)
            
            if not pdt_status:
                return {'error': 'Could not determine PDT status'}
                
            # Calculate compliance score
            compliance_score = 100
            
            if pdt_status.restrictions:
                compliance_score -= len(pdt_status.restrictions) * 25
                
            if not pdt_status.is_pdt_account and pdt_status.day_trades_used >= 2:
                compliance_score -= 20
                
            compliance_score = max(0, compliance_score)
            
            return {
                'account_id': account_id,
                'compliance_score': compliance_score,
                'pdt_status': {
                    'is_pdt_account': pdt_status.is_pdt_account,
                    'account_value': pdt_status.account_value,
                    'account_type': pdt_status.account_type.value,
                    'can_day_trade': pdt_status.can_day_trade,
                    'day_trades_used': pdt_status.day_trades_used,
                    'day_trades_remaining': pdt_status.day_trades_remaining,
                    'reset_date': pdt_status.reset_date.isoformat()
                },
                'recent_activity': day_trades_summary,
                'restrictions': pdt_status.restrictions,
                'warnings': pdt_status.warnings,
                'recommendations': self._generate_compliance_recommendations(pdt_status, day_trades_summary)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {'error': str(e)}
    
    def _generate_compliance_recommendations(self, pdt_status: PDTStatus, 
                                           day_trades_summary: Dict) -> List[str]:
        """Generate PDT compliance recommendations"""
        recommendations = []
        
        if not pdt_status.is_pdt_account:
            if pdt_status.day_trades_remaining <= 1:
                recommendations.append("Caution: Approaching day trade limit. Consider depositing funds to reach $25,000.")
                
            if pdt_status.account_value < self.pdt_rules['min_account_value_pdt']:
                shortfall = self.pdt_rules['min_account_value_pdt'] - pdt_status.account_value
                recommendations.append(f"Deposit ${shortfall:,.0f} to become a Pattern Day Trader and remove day trading restrictions.")
        
        if pdt_status.account_type in [AccountType.IRA, AccountType.ROTH_IRA]:
            recommendations.append("Consider using a margin account for day trading as retirement accounts have restrictions.")
            
        recent_pnl = day_trades_summary.get('total_pnl', 0)
        if recent_pnl < 0:
            recommendations.append("Recent day trading has been unprofitable. Consider reviewing your strategy.")
            
        if not recommendations:
            recommendations.append("PDT compliance status is good. Continue monitoring day trade usage.")
            
        return recommendations