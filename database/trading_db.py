# database/trading_db.py
"""
Enhanced Trading Database Manager
================================
Extracted from fractional_position_system.py - Comprehensive tracking system
This is the single source of truth for all database operations
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StrategySignal:
    """Trading signal data structure"""
    symbol: str
    phase: str
    strength: float
    price: float
    volume_confirmation: bool
    sector: str
    combined_score: float


@dataclass
class DayTradeCheckResult:
    """Day trade compliance check result"""
    symbol: str
    action: str
    would_be_day_trade: bool
    db_trades_today: List[Dict]
    actual_trades_today: List[Dict]
    manual_trades_detected: bool
    recommendation: str  # 'ALLOW', 'BLOCK', 'EMERGENCY_OVERRIDE'
    details: str


class TradingDatabase:
    """
    Enhanced database manager with comprehensive tracking
    
    This class provides all database operations for the trading system including:
    - Signal logging
    - Trade execution tracking  
    - Position management
    - Day trade compliance tracking
    - Bot run statistics
    - Portfolio analytics
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        """
        Initialize the enhanced trading database
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.strategy_id = "enhanced_wyckoff_bot_v2"
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with enhanced capabilities"""
        with sqlite3.connect(self.db_path) as conn:
            # Multi-timeframe stock data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Technical indicators table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    indicator_type TEXT NOT NULL,
                    indicator_value REAL NOT NULL,
                    metadata TEXT,  -- JSON for complex indicator data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp, indicator_type)
                )
            ''')
            
            # Market regime cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regime_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL UNIQUE,
                    regime_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    vix_data TEXT,  -- JSON for VIX levels and metrics
                    sector_data TEXT,  -- JSON for sector rotation data
                    metadata TEXT,  -- JSON for additional market data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Current watchlist table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS current_watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    added_date DATETIME NOT NULL,
                    timeframe_analysis TEXT,  -- JSON for multi-timeframe analysis
                    phase TEXT NOT NULL,
                    strength REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading signals table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    strength REAL NOT NULL,
                    price REAL NOT NULL,
                    volume_confirmation BOOLEAN NOT NULL,
                    sector TEXT NOT NULL,
                    combined_score REAL NOT NULL,
                    action_taken TEXT,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced trades table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    signal_phase TEXT,
                    signal_strength REAL,
                    account_type TEXT,
                    order_id TEXT,
                    status TEXT,
                    day_trade_check TEXT,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    trade_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            
            # Enhanced positions table for detailed tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_enhanced (
                    symbol TEXT NOT NULL,
                    account_type TEXT NOT NULL, 
                    total_shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_purchase_date TEXT NOT NULL,
                    last_purchase_date TEXT NOT NULL,
                    entry_phase TEXT DEFAULT 'UNKNOWN',
                    entry_strength REAL DEFAULT 0.0,
                    position_size_pct REAL DEFAULT 0.1,
                    time_held_days INTEGER DEFAULT 0,
                    volatility_percentile REAL DEFAULT 0.5,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, account_type, strategy_id)
                )
            ''')
            
            # Day trade tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS day_trade_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    db_day_trade BOOLEAN NOT NULL,
                    actual_day_trade BOOLEAN NOT NULL,
                    manual_trades_detected BOOLEAN NOT NULL,
                    recommendation TEXT NOT NULL,
                    details TEXT,
                    emergency_override BOOLEAN DEFAULT FALSE,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced stop strategies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stop_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    initial_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    stop_percentage REAL NOT NULL,
                    trailing_high REAL,
                    key_support_level REAL,
                    key_resistance_level REAL,
                    breakout_level REAL,
                    pullback_low REAL,
                    time_entered TIMESTAMP,
                    context_data TEXT,
                    stop_reason TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Partial sales tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS partial_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sale_date TEXT NOT NULL,
                    shares_sold REAL NOT NULL,
                    sale_price REAL NOT NULL,
                    sale_reason TEXT NOT NULL,
                    remaining_shares REAL NOT NULL,
                    gain_pct REAL,
                    profit_amount REAL,
                    scaling_level TEXT,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Bot runs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    signals_found INTEGER NOT NULL,
                    trades_executed INTEGER NOT NULL,
                    wyckoff_sells INTEGER DEFAULT 0,
                    profit_scales INTEGER DEFAULT 0,
                    emergency_exits INTEGER DEFAULT 0,
                    day_trades_blocked INTEGER DEFAULT 0,
                    errors_encountered INTEGER NOT NULL,
                    total_portfolio_value REAL,
                    available_cash REAL,
                    emergency_mode BOOLEAN DEFAULT FALSE,
                    market_condition TEXT,
                    portfolio_drawdown_pct REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    log_details TEXT,
                    strategy_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_strategy_id ON positions_enhanced(strategy_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, sale_date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_day_trade_checks_date ON day_trade_checks(check_date, symbol)')
            
            # Multi-timeframe indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timeframe_timestamp ON stock_data(symbol, timeframe, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_timestamp ON stock_data(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timeframe_timestamp ON technical_indicators(symbol, timeframe, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timeframe_type ON technical_indicators(symbol, timeframe, indicator_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regime_cache_date ON market_regime_cache(date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_current_watchlist_symbol ON current_watchlist(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_current_watchlist_phase_strength ON current_watchlist(phase, strength)')
    
    def log_signal(self, signal: StrategySignal = None, action_taken: str = None):
        """Log a trading signal"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO signals (date, symbol, phase, strength, price, volume_confirmation, 
                                   sector, combined_score, action_taken, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                signal.symbol,
                signal.phase,
                signal.strength,
                signal.price,
                signal.volume_confirmation,
                signal.sector,
                signal.combined_score,
                action_taken,
                self.strategy_id
            ))
    
    def log_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  signal_phase: str, signal_strength: float, account_type: str, 
                  order_id: str = None, day_trade_check: str = None, status: str = 'PENDING', 
                  trade_date: datetime = None): 
        """
        Log a trade execution with day trade check info and status
        
        Args:
            trade_date: Actual trade date (defaults to current date if not provided)
        """
        # Use provided trade_date or default to current date
        date_to_use = trade_date.strftime('%Y-%m-%d') if trade_date else datetime.now().strftime('%Y-%m-%d')
        
        # For trade_datetime, use the full datetime if provided, otherwise current timestamp
        datetime_to_use = trade_date if trade_date else datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (date, symbol, action, quantity, price, total_value, 
                                  signal_phase, signal_strength, account_type, order_id, 
                                  day_trade_check, status, strategy_id, trade_datetime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_to_use,
                symbol,
                action,
                quantity,
                price,
                quantity * price,
                signal_phase,
                signal_strength,
                account_type,
                order_id,
                day_trade_check,
                status,
                self.strategy_id,
                datetime_to_use
            ))
    
    def update_position(self, symbol: str, shares: float, cost: float, 
                       account_type: str, entry_phase: str = None, entry_strength: float = None):
        """Update position in both regular and enhanced tables"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            self._update_positions_enhanced_table_fixed(conn, symbol, shares, cost, account_type, entry_phase, entry_strength, today)
            
            # After updating position, recalculate dates from actual trade history
            self._recalculate_position_dates(conn, symbol, account_type)

    def _recalculate_position_dates(self, conn, symbol: str, account_type: str):
        """
        Recalculate position first_purchase_date and last_purchase_date from actual trade history
        
        Args:
            conn: Database connection
            symbol: Stock symbol
            account_type: Account type
            
        Returns:
            bool: True if dates were actually updated, False otherwise
        """
        try:
            # Get current position dates
            current_position = conn.execute('''
                SELECT first_purchase_date, last_purchase_date FROM positions_enhanced
                WHERE symbol = ? AND account_type = ? AND strategy_id = ?
            ''', (symbol, account_type, self.strategy_id)).fetchone()
            
            if not current_position:
                return False
                
            # Get all BUY trades for this symbol and account, ordered by date
            trades = conn.execute('''
                SELECT date FROM trades 
                WHERE symbol = ? AND account_type = ? AND action = 'BUY' AND status = 'FILLED'
                ORDER BY date ASC
            ''', (symbol, account_type)).fetchall()
            
            if not trades:
                # No buy trades found, skip date recalculation
                return False
                
            # Extract dates
            trade_dates = [trade[0] for trade in trades]
            new_first_purchase_date = trade_dates[0]  # Earliest buy
            new_last_purchase_date = trade_dates[-1]  # Latest buy
            
            # Check if dates actually changed
            current_first, current_last = current_position
            if (current_first == new_first_purchase_date and 
                current_last == new_last_purchase_date):
                return False  # No change needed
            
            # Calculate time held days
            try:
                first_dt = datetime.strptime(new_first_purchase_date, '%Y-%m-%d')
                time_held_days = (datetime.now() - first_dt).days
            except:
                time_held_days = 0
            
            # Update the position with correct dates
            conn.execute('''
                UPDATE positions_enhanced 
                SET first_purchase_date = ?, last_purchase_date = ?, time_held_days = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND account_type = ? AND strategy_id = ?
            ''', (new_first_purchase_date, new_last_purchase_date, time_held_days, 
                  symbol, account_type, self.strategy_id))
                  
            return True  # Dates were updated
                  
        except Exception as e:
            # Don't fail the entire operation if date recalculation fails
            import logging
            logging.getLogger(__name__).debug(f"Error recalculating position dates for {symbol}: {e}")
            return False
    
    def rebuild_all_position_dates_from_trade_history(self):
        """
        Rebuild first_purchase_date and last_purchase_date for ALL positions based on actual trade history.
        
        This is a utility method to fix positions that have incorrect dates.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get all positions that exist
            positions = conn.execute('''
                SELECT DISTINCT symbol, account_type 
                FROM positions_enhanced 
                WHERE strategy_id = ?
            ''', (self.strategy_id,)).fetchall()
            
            updated_count = 0
            for symbol, account_type in positions:
                # Only recalculate if dates actually changed
                if self._recalculate_position_dates(conn, symbol, account_type):
                    updated_count += 1
            
            return updated_count
    
    def _update_positions_enhanced_table_fixed(self, conn, symbol: str, shares: float, cost: float, 
                                             account_type: str, entry_phase: str, entry_strength: float, today: str):
        """Enhanced positions table update with proper synchronization"""
        
        # Get current enhanced position if it exists
        existing_enhanced = conn.execute(
            '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, 
                    entry_phase, entry_strength, position_size_pct, time_held_days, volatility_percentile 
            FROM positions_enhanced 
            WHERE symbol = ? AND account_type = ? AND strategy_id = ?''',
            (symbol, account_type, self.strategy_id)
        ).fetchone()
        
        if existing_enhanced:
            old_shares, old_avg_cost, old_invested, first_date, old_phase, old_strength, old_size_pct, old_days, old_vol = existing_enhanced
            new_shares = old_shares + shares
            
            # Calculate time held
            try:
                first_purchase_dt = datetime.strptime(first_date, '%Y-%m-%d')
                time_held_days = (datetime.now() - first_purchase_dt).days
            except:
                time_held_days = old_days or 0
            
            if new_shares > 0:
                new_invested = old_invested + (shares * cost)
                new_avg_cost = new_invested / new_shares
                use_phase = entry_phase or old_phase or 'UNKNOWN'
                use_strength = entry_strength or old_strength or 0.0
            else:
                # Position closed
                new_invested = 0
                new_avg_cost = 0
                use_phase = old_phase
                use_strength = old_strength
            
            # Update enhanced position
            conn.execute('''
                UPDATE positions_enhanced 
                SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                    last_purchase_date = ?, entry_phase = ?, entry_strength = ?,
                    position_size_pct = ?, time_held_days = ?, volatility_percentile = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND account_type = ? AND strategy_id = ?
            ''', (new_shares, new_avg_cost, new_invested, today, use_phase, use_strength,
                  old_size_pct or 0.1, time_held_days, old_vol or 0.5,
                  symbol, account_type, self.strategy_id))
            
        else:
            # Insert new enhanced position - only if we're adding shares
            if shares > 0:
                conn.execute('''
                    INSERT INTO positions_enhanced (symbol, account_type, total_shares, avg_cost, total_invested, 
                                                  first_purchase_date, last_purchase_date, 
                                                  entry_phase, entry_strength, position_size_pct,
                                                  time_held_days, volatility_percentile, strategy_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, account_type, shares, cost, shares * cost, today, today,
                      entry_phase or 'UNKNOWN', entry_strength or 0.0, 0.1,
                      0, 0.5, self.strategy_id))
    
    def get_position(self, symbol: str, account_type: str = None) -> Optional[Dict]:
        """Get current position for a symbol in a specific account or all accounts"""
        with sqlite3.connect(self.db_path) as conn:
            if account_type:
                # Get position for specific account from enhanced table
                result = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                           first_purchase_date, last_purchase_date, entry_phase, 
                           entry_strength, position_size_pct, time_held_days, 
                           volatility_percentile, strategy_id, updated_at
                    FROM positions_enhanced 
                    WHERE symbol = ? AND account_type = ? AND strategy_id = ?
                ''', (symbol, account_type, self.strategy_id)).fetchone()
                
                if result:
                    columns = ['symbol', 'account_type', 'total_shares', 'avg_cost', 'total_invested', 
                              'first_purchase_date', 'last_purchase_date', 'entry_phase', 
                              'entry_strength', 'position_size_pct', 'time_held_days',
                              'volatility_percentile', 'strategy_id', 'updated_at']
                    return dict(zip(columns, result))
            else:
                # Get all positions for this symbol across accounts
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                           first_purchase_date, last_purchase_date, entry_phase, 
                           entry_strength, position_size_pct, time_held_days,
                           volatility_percentile, strategy_id, updated_at
                    FROM positions_enhanced 
                    WHERE symbol = ? AND strategy_id = ?
                ''', (symbol, self.strategy_id)).fetchall()
                
                if results:
                    columns = ['symbol', 'account_type', 'total_shares', 'avg_cost', 'total_invested', 
                              'first_purchase_date', 'last_purchase_date', 'entry_phase', 
                              'entry_strength', 'position_size_pct', 'time_held_days',
                              'volatility_percentile', 'strategy_id', 'updated_at']
                    return [dict(zip(columns, row)) for row in results]
        
        return None
    
    def get_todays_trades(self, symbol: str = None) -> List[Dict]:
        """Get today's trades from database for specific symbol or all"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = '''
                    SELECT * FROM trades 
                    WHERE date = ? AND symbol = ? AND strategy_id = ?
                    ORDER BY trade_datetime
                '''
                results = conn.execute(query, (today, symbol, self.strategy_id)).fetchall()
            else:
                query = '''
                    SELECT * FROM trades 
                    WHERE date = ? AND strategy_id = ?
                    ORDER BY trade_datetime
                '''
                results = conn.execute(query, (today, self.strategy_id)).fetchall()
            
            columns = ['id', 'date', 'symbol', 'action', 'quantity', 'price', 'total_value',
                      'signal_phase', 'signal_strength', 'account_type', 'order_id', 'status',
                      'day_trade_check', 'strategy_id', 'trade_datetime', 'created_at']
            
            return [dict(zip(columns, row)) for row in results]
    
    def would_create_day_trade(self, symbol: str, action: str) -> bool:
        """Check if this trade would create a day trade (DATABASE ONLY)"""
        today_trades = self.get_todays_trades(symbol)
        
        if not today_trades:
            return False
        
        # Count buys and sells today
        buys_today = sum(1 for trade in today_trades if trade['action'] == 'BUY')
        sells_today = sum(1 for trade in today_trades if trade['action'] == 'SELL')
        
        # Day trade occurs when we buy and sell the same security on the same day
        if action == 'SELL' and buys_today > 0:
            return True
        elif action == 'BUY' and sells_today > 0:
            return True
        
        return False
    
    def log_bot_run(self, signals_found: int, trades_executed: int, wyckoff_sells: int = 0,
                    profit_scales: int = 0, emergency_exits: int = 0, day_trades_blocked: int = 0,
                    errors: int = 0, portfolio_value: float = 0, available_cash: float = 0, 
                    emergency_mode: bool = False, market_condition: str = "NORMAL", 
                    portfolio_drawdown_pct: float = 0.0, status: str = "COMPLETED", log_details: str = ""):
        """Log enhanced bot run statistics with day trade blocking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bot_runs (run_date, signals_found, trades_executed, wyckoff_sells,
                                    profit_scales, emergency_exits, day_trades_blocked, errors_encountered, 
                                    total_portfolio_value, available_cash, emergency_mode,
                                    market_condition, portfolio_drawdown_pct, status, log_details, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                signals_found, trades_executed, wyckoff_sells, profit_scales, emergency_exits,
                day_trades_blocked, errors, portfolio_value, available_cash, emergency_mode,
                market_condition, portfolio_drawdown_pct, status, log_details, self.strategy_id
            ))
    
    def log_day_trade_check(self, check_result: DayTradeCheckResult, emergency_override: bool = False):
        """Log day trade compliance check"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO day_trade_checks (check_date, symbol, action, db_day_trade, 
                                            actual_day_trade, manual_trades_detected, 
                                            recommendation, details, emergency_override, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                check_result.symbol,
                check_result.action,
                check_result.would_be_day_trade,
                len(check_result.actual_trades_today) > 0,
                check_result.manual_trades_detected,
                check_result.recommendation,
                check_result.details,
                emergency_override,
                self.strategy_id
            ))
    
    def log_partial_sale(self, symbol: str, shares_sold: float, sale_price: float, 
                        sale_reason: str, remaining_shares: float, gain_pct: float, 
                        profit_amount: float, scaling_level: str):
        """Log partial sale for profit scaling tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO partial_sales (symbol, sale_date, shares_sold, sale_price, 
                                         sale_reason, remaining_shares, gain_pct, 
                                         profit_amount, scaling_level, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                shares_sold,
                sale_price,
                sale_reason,
                remaining_shares,
                gain_pct,
                profit_amount,
                scaling_level,
                self.strategy_id
            ))
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions grouped by account"""
        positions = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested,
                           entry_phase, entry_strength, first_purchase_date, last_purchase_date,
                           position_size_pct, time_held_days, volatility_percentile
                    FROM positions_enhanced 
                    WHERE total_shares > 0 AND strategy_id = ?
                    ORDER BY account_type, symbol
                ''', (self.strategy_id,)).fetchall()
                
                for row in results:
                    symbol, account_type, shares, avg_cost, invested, phase, strength, first_date, last_date, size_pct, days_held, volatility = row
                    
                    if account_type not in positions:
                        positions[account_type] = {}
                    
                    positions[account_type][symbol] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'entry_phase': phase,
                        'entry_strength': strength,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date,
                        'position_size_pct': size_pct,
                        'time_held_days': days_held,
                        'volatility_percentile': volatility
                    }
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return positions
    
    def get_current_portfolio(self) -> Dict:
        """Get current portfolio summary"""
        portfolio = {
            'total_positions': 0,
            'total_invested': 0.0,
            'positions_by_account': {},
            'symbols': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT account_type, symbol, total_shares, avg_cost, total_invested
                    FROM positions_enhanced 
                    WHERE total_shares > 0 AND strategy_id = ?
                    ORDER BY account_type, symbol
                ''', (self.strategy_id,)).fetchall()
                
                for row in results:
                    account_type, symbol, shares, avg_cost, invested = row
                    
                    if account_type not in portfolio['positions_by_account']:
                        portfolio['positions_by_account'][account_type] = {}
                    
                    portfolio['positions_by_account'][account_type][symbol] = {
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'invested': invested
                    }
                    
                    portfolio['total_positions'] += 1
                    portfolio['total_invested'] += invested
                    
                    if symbol not in portfolio['symbols']:
                        portfolio['symbols'].append(symbol)
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return portfolio

    def get_positions_summary(self) -> Dict:
        """Get summary of all positions"""
        summary = {
            'total_positions': 0,
            'total_value': 0.0,
            'by_account': {},
            'by_symbol': {}
        }
        
        try:
            positions = self.get_all_positions()
            
            for account_type, account_positions in positions.items():
                summary['by_account'][account_type] = {
                    'count': len(account_positions),
                    'total_invested': sum(pos['total_invested'] for pos in account_positions.values())
                }
                
                for symbol, position in account_positions.items():
                    summary['total_positions'] += 1
                    summary['total_value'] += position['total_invested']
                    
                    if symbol not in summary['by_symbol']:
                        summary['by_symbol'][symbol] = {
                            'total_shares': 0,
                            'total_invested': 0,
                            'accounts': []
                        }
                    
                    summary['by_symbol'][symbol]['total_shares'] += position['total_shares']
                    summary['by_symbol'][symbol]['total_invested'] += position['total_invested']
                    summary['by_symbol'][symbol]['accounts'].append(account_type)
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return summary

    def get_account_positions(self, account_type: str) -> Dict:
        """Get all positions for a specific account"""
        positions = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, total_shares, avg_cost, total_invested, 
                           entry_phase, entry_strength, first_purchase_date, 
                           last_purchase_date, position_size_pct, time_held_days, 
                           volatility_percentile
                    FROM positions_enhanced 
                    WHERE account_type = ? AND total_shares > 0 AND strategy_id = ?
                    ORDER BY symbol
                ''', (account_type, self.strategy_id)).fetchall()
                
                for row in results:
                    symbol, shares, avg_cost, invested, phase, strength, first_date, last_date, size_pct, days_held, volatility = row
                    
                    positions[symbol] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'entry_phase': phase,
                        'entry_strength': strength,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date,
                        'position_size_pct': size_pct,
                        'time_held_days': days_held,
                        'volatility_percentile': volatility
                    }
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return positions

    def get_position_count(self) -> int:
        """Get total number of positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT COUNT(*) FROM positions_enhanced 
                    WHERE total_shares > 0 AND strategy_id = ?
                ''', (self.strategy_id,)).fetchone()
                
                return result[0] if result else 0
        
        except Exception as e:
            return 0

    def has_position(self, symbol: str, account_type: str = None) -> bool:
        """Check if we have a position in a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if account_type:
                    result = conn.execute('''
                        SELECT total_shares FROM positions_enhanced 
                        WHERE symbol = ? AND account_type = ? AND strategy_id = ? AND total_shares > 0
                    ''', (symbol, account_type, self.strategy_id)).fetchone()
                else:
                    result = conn.execute('''
                        SELECT total_shares FROM positions_enhanced 
                        WHERE symbol = ? AND strategy_id = ? AND total_shares > 0
                    ''', (symbol, self.strategy_id)).fetchone()
                
                return result is not None
        
        except Exception as e:
            return False

    def clear_position(self, symbol: str, account_type: str):
        """Clear/close a position (set shares to 0)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE positions_enhanced 
                    SET total_shares = 0, total_invested = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND account_type = ? AND strategy_id = ?
                ''', (symbol, account_type, self.strategy_id))
                
                # Note: There's no 'positions' table in the schema, only 'positions_enhanced'
                # So we'll skip updating the non-existent table
                
        except Exception as e:
            # Log error but don't crash
            pass

    def get_symbols_list(self) -> List[str]:
        """Get list of all symbols we have positions in"""
        symbols = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT DISTINCT symbol FROM positions_enhanced 
                    WHERE total_shares > 0 AND strategy_id = ?
                    ORDER BY symbol
                ''', (self.strategy_id,)).fetchall()
                
                symbols = [row[0] for row in results]
                        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return symbols
    
    def already_scaled_at_level(self, symbol: str, gain_level: float, tolerance: float = 0.01) -> bool:
        """
        Check if we've already scaled out at a particular gain level
        
        Args:
            symbol: Stock symbol to check
            gain_level: Gain percentage level (e.g. 0.06 for 6%)
            tolerance: Tolerance for matching gain levels (default 1%)
        
        Returns:
            bool: True if already scaled at this level, False otherwise
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                # Look for partial sales at this gain level today
                results = conn.execute('''
                    SELECT gain_pct, scaling_level 
                    FROM partial_sales 
                    WHERE symbol = ? AND sale_date = ? AND strategy_id = ?
                    ORDER BY gain_pct DESC
                ''', (symbol, today, self.strategy_id)).fetchall()
                
                if not results:
                    return False
                
                # Check if any scaling was done at or near this gain level
                for gain_pct, scaling_level in results:
                    if gain_pct is not None:
                        # Check if within tolerance
                        if abs(gain_pct - gain_level) <= tolerance:
                            # Note: self.logger is not defined in this class
                            # We should just return True/False without logging
                            return True
                
                return False
                
        except Exception as e:
            # Conservative: assume not scaled to allow scaling
            return False

    def deactivate_stop_strategies(self, symbol: str):
        """
        Deactivate stop strategies for a symbol (called after exits)
        
        Args:
            symbol: Stock symbol to deactivate stops for
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE stop_strategies 
                    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND strategy_id = ? AND is_active = TRUE
                ''', (symbol, self.strategy_id))
                
                # Note: self.logger is not defined in this class
                # We should handle this silently
                
        except Exception as e:
            # Handle error silently
            pass

    # Multi-timeframe data methods
    
    def store_stock_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Store stock data for multi-timeframe analysis
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (1D, 4H, 1H, 15M)
            df: DataFrame with OHLCV data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for timestamp, row in df.iterrows():
                    # Skip future timestamps to prevent bad data
                    if hasattr(timestamp, 'date') and timestamp.date() > datetime.now().date():
                        continue
                        
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        symbol, timeframe, timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp), 
                        float(row['open']), float(row['high']), 
                        float(row['low']), float(row['close']), 
                        int(row['volume'])
                    ))
                conn.commit()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error storing stock data for {symbol} {timeframe}: {e}")
    
    def get_stock_data(self, symbol: str, timeframe: str, 
                       start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """
        Get stock data for analysis
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (1D, 4H, 1H, 15M)
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM stock_data 
                    WHERE symbol = ? AND timeframe = ?
                '''
                params = [symbol, timeframe]
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                    
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                    
                query += ' ORDER BY timestamp'
                
                df = pd.read_sql_query(query, conn, params=params, index_col='timestamp')
                return df if not df.empty else None
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting stock data for {symbol} {timeframe}: {e}")
            return None
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get latest timestamp for a symbol/timeframe"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT MAX(timestamp) FROM stock_data 
                    WHERE symbol = ? AND timeframe = ?
                ''', (symbol, timeframe)).fetchone()
                
                if result and result[0]:
                    return datetime.fromisoformat(result[0])
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting latest timestamp for {symbol} {timeframe}: {e}")
            
        return None
    
    def store_technical_indicator(self, symbol: str, timeframe: str, 
                                 timestamp: datetime, indicator_type: str, 
                                 value: float, metadata: Dict = None):
        """Store technical indicator data"""
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, timeframe, timestamp, indicator_type, indicator_value, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    symbol, timeframe, timestamp, indicator_type, value,
                    json.dumps(metadata) if metadata else None
                ))
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error storing indicator {indicator_type} for {symbol} {timeframe}: {e}")
    
    def get_technical_indicators(self, symbol: str, timeframe: str, 
                                indicator_type: str, days: int = 30) -> pd.DataFrame:
        """Get technical indicator data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, indicator_value, metadata
                    FROM technical_indicators 
                    WHERE symbol = ? AND timeframe = ? AND indicator_type = ?
                    AND timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp
                '''.format(days)
                
                return pd.read_sql_query(query, conn, params=(symbol, timeframe, indicator_type))
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting indicators for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def store_market_regime(self, date: datetime, regime_type: str, 
                           confidence: float, vix_data: Dict = None, 
                           sector_data: Dict = None, metadata: Dict = None):
        """Store market regime data"""
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO market_regime_cache 
                    (date, regime_type, confidence, vix_data, sector_data, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    date.date(), regime_type, confidence,
                    json.dumps(vix_data) if vix_data else None,
                    json.dumps(sector_data) if sector_data else None,
                    json.dumps(metadata) if metadata else None
                ))
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error storing market regime for {date}: {e}")
    
    def get_current_market_regime(self) -> Optional[Dict]:
        """Get current market regime"""
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT regime_type, confidence, vix_data, sector_data, metadata
                    FROM market_regime_cache 
                    ORDER BY date DESC LIMIT 1
                ''').fetchone()
                
                if result:
                    return {
                        'regime_type': result[0],
                        'confidence': result[1],
                        'vix_data': json.loads(result[2]) if result[2] else None,
                        'sector_data': json.loads(result[3]) if result[3] else None,
                        'metadata': json.loads(result[4]) if result[4] else None
                    }
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting market regime: {e}")
            
        return None
    
    def update_watchlist(self, symbols: List[str], analysis_data: Dict[str, Dict]):
        """Update current watchlist with analysis data"""
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing watchlist
                conn.execute('DELETE FROM current_watchlist')
                
                # Add new watchlist
                for symbol in symbols:
                    data = analysis_data.get(symbol, {})
                    conn.execute('''
                        INSERT INTO current_watchlist 
                        (symbol, added_date, timeframe_analysis, phase, strength, last_updated)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        symbol,
                        datetime.now(),
                        json.dumps(data.get('timeframe_analysis', {})),
                        data.get('phase', 'unknown'),
                        data.get('strength', 0.0)
                    ))
                    
                conn.commit()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating watchlist: {e}")
    
    def get_current_watchlist(self) -> List[Dict]:
        """Get current watchlist"""
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, added_date, timeframe_analysis, phase, strength, last_updated
                    FROM current_watchlist 
                    ORDER BY strength DESC
                ''').fetchall()
                
                watchlist = []
                for row in results:
                    watchlist.append({
                        'symbol': row[0],
                        'added_date': row[1],
                        'timeframe_analysis': json.loads(row[2]) if row[2] else {},
                        'phase': row[3],
                        'strength': row[4],
                        'last_updated': row[5]
                    })
                    
                return watchlist
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting watchlist: {e}")
            return []
    
    def data_exists(self, symbol: str, timeframe: str, days_back: int = 1) -> bool:
        """Check if recent data exists for symbol/timeframe"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                result = conn.execute('''
                    SELECT COUNT(*) FROM stock_data 
                    WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                ''', (symbol, timeframe, cutoff_date)).fetchone()
                
                return result[0] > 0 if result else False
                
        except Exception as e:
            return False