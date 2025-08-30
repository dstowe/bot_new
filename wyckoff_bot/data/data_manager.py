# wyckoff_bot/data/data_manager.py
"""
Wyckoff Data Manager
===================
Manages data storage and retrieval for Wyckoff analysis
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from database.trading_db import TradingDatabase
from ..strategy.wyckoff_strategy import TradeSignal

class WyckoffDataManager:
    """
    Data manager for Wyckoff bot
    Integrates with existing trading database
    """
    
    def __init__(self, db_path: str = "data/trading_data.db", 
                 logger: logging.Logger = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.trading_db = TradingDatabase(db_path=db_path)
        self._init_wyckoff_tables()
    
    def _init_wyckoff_tables(self):
        """Initialize Wyckoff-specific database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Wyckoff analysis results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wyckoff_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    phase TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    volume_confirmation BOOLEAN,
                    price_action_strength REAL,
                    trend_strength REAL,
                    key_events TEXT,  -- JSON array of events
                    support_level REAL,
                    resistance_level REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Wyckoff signals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wyckoff_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    position_size REAL NOT NULL,
                    strength_score REAL,
                    reasoning TEXT,
                    status TEXT DEFAULT 'active',  -- active, filled, cancelled, expired
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Wyckoff performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wyckoff_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    symbol TEXT NOT NULL,
                    entry_date DATETIME,
                    exit_date DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    max_favorable REAL,
                    max_adverse REAL,
                    hold_days INTEGER,
                    exit_reason TEXT,
                    FOREIGN KEY (signal_id) REFERENCES wyckoff_signals (id)
                )
            """)
            
            conn.commit()
    
    def save_wyckoff_analysis(self, symbol: str, analysis_data: Dict):
        """Save Wyckoff analysis to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO wyckoff_analysis 
                    (symbol, timestamp, phase, confidence, volume_confirmation, 
                     price_action_strength, trend_strength, key_events, 
                     support_level, resistance_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now(),
                    analysis_data.get('phase', ''),
                    analysis_data.get('confidence', 0),
                    analysis_data.get('volume_confirmation', False),
                    analysis_data.get('price_action_strength', 0),
                    analysis_data.get('trend_strength', 0),
                    str(analysis_data.get('key_events', [])),
                    analysis_data.get('support_level'),
                    analysis_data.get('resistance_level')
                ))
                
        except Exception as e:
            self.logger.error(f"Error saving Wyckoff analysis for {symbol}: {e}")
    
    def save_wyckoff_signal(self, signal: TradeSignal, strength_score: float = 0) -> int:
        """
        Save Wyckoff signal to database
        
        Returns:
            int: Signal ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO wyckoff_signals 
                    (symbol, timestamp, action, confidence, entry_price, 
                     stop_loss, take_profit, position_size, strength_score, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol,
                    datetime.now(),
                    signal.action.value,
                    signal.confidence,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.position_size,
                    strength_score,
                    signal.reasoning
                ))
                
                return cursor.lastrowid
                
        except Exception as e:
            self.logger.error(f"Error saving Wyckoff signal for {signal.symbol}: {e}")
            return 0
    
    def get_recent_wyckoff_analysis(self, symbol: str = None, 
                                   days: int = 30) -> pd.DataFrame:
        """Get recent Wyckoff analysis data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM wyckoff_analysis 
                    WHERE timestamp >= datetime('now', '-{} days')
                """.format(days)
                
                if symbol:
                    query += " AND symbol = ?"
                    params = (symbol,)
                else:
                    params = ()
                
                query += " ORDER BY timestamp DESC"
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error retrieving Wyckoff analysis: {e}")
            return pd.DataFrame()
    
    def get_active_wyckoff_signals(self, symbol: str = None) -> pd.DataFrame:
        """Get active Wyckoff signals"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM wyckoff_signals WHERE status = 'active'"
                
                if symbol:
                    query += " AND symbol = ?"
                    params = (symbol,)
                else:
                    params = ()
                
                query += " ORDER BY timestamp DESC"
                
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error retrieving active signals: {e}")
            return pd.DataFrame()
    
    def update_signal_status(self, signal_id: int, status: str):
        """Update signal status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE wyckoff_signals 
                    SET status = ? 
                    WHERE id = ?
                """, (status, signal_id))
                
        except Exception as e:
            self.logger.error(f"Error updating signal status: {e}")
    
    def record_trade_performance(self, signal_id: int, trade_data: Dict):
        """Record trade performance for a signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO wyckoff_performance 
                    (signal_id, symbol, entry_date, exit_date, entry_price, 
                     exit_price, position_size, pnl, pnl_pct, max_favorable, 
                     max_adverse, hold_days, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_id,
                    trade_data.get('symbol'),
                    trade_data.get('entry_date'),
                    trade_data.get('exit_date'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('position_size'),
                    trade_data.get('pnl'),
                    trade_data.get('pnl_pct'),
                    trade_data.get('max_favorable'),
                    trade_data.get('max_adverse'),
                    trade_data.get('hold_days'),
                    trade_data.get('exit_reason', 'unknown')
                ))
                
        except Exception as e:
            self.logger.error(f"Error recording trade performance: {e}")
    
    def get_wyckoff_performance_stats(self, days: int = 90) -> Dict:
        """Get Wyckoff strategy performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get performance data
                perf_df = pd.read_sql_query("""
                    SELECT * FROM wyckoff_performance 
                    WHERE entry_date >= datetime('now', '-{} days')
                """.format(days), conn)
                
                if perf_df.empty:
                    return {'total_trades': 0}
                
                # Calculate statistics
                total_trades = len(perf_df)
                winning_trades = len(perf_df[perf_df['pnl'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                total_pnl = perf_df['pnl'].sum()
                avg_win = perf_df[perf_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = perf_df[perf_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else 0,
                    'avg_hold_days': perf_df['hold_days'].mean()
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating performance stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 180):
        """Clean up old analysis and signal data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean up old analysis data
                conn.execute("""
                    DELETE FROM wyckoff_analysis 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                # Clean up old inactive signals
                conn.execute("""
                    DELETE FROM wyckoff_signals 
                    WHERE status != 'active' 
                    AND timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                conn.commit()
                self.logger.info(f"Cleaned up Wyckoff data older than {days_to_keep} days")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def export_wyckoff_data(self, output_path: str, days: int = 90):
        """Export Wyckoff data to CSV files"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)
            
            # Export analysis data
            analysis_df = self.get_recent_wyckoff_analysis(days=days)
            if not analysis_df.empty:
                analysis_df.to_csv(output_dir / 'wyckoff_analysis.csv', index=False)
            
            # Export signals data
            with sqlite3.connect(self.db_path) as conn:
                signals_df = pd.read_sql_query("""
                    SELECT * FROM wyckoff_signals 
                    WHERE timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days), conn)
                
                if not signals_df.empty:
                    signals_df.to_csv(output_dir / 'wyckoff_signals.csv', index=False)
                
                # Export performance data
                perf_df = pd.read_sql_query("""
                    SELECT * FROM wyckoff_performance 
                    WHERE entry_date >= datetime('now', '-{} days')
                    ORDER BY entry_date DESC
                """.format(days), conn)
                
                if not perf_df.empty:
                    perf_df.to_csv(output_dir / 'wyckoff_performance.csv', index=False)
            
            self.logger.info(f"Exported Wyckoff data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
    
    def get_symbol_analysis_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get analysis history for specific symbol"""
        try:
            df = self.get_recent_wyckoff_analysis(symbol=symbol, days=days)
            return df.to_dict('records') if not df.empty else []
            
        except Exception as e:
            self.logger.error(f"Error getting analysis history for {symbol}: {e}")
            return []