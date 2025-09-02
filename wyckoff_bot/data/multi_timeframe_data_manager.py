# wyckoff_bot/data/multi_timeframe_data_manager.py
"""
Multi-Timeframe Data Manager
===========================
Comprehensive data management system for multi-timeframe Wyckoff analysis
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from database.trading_db import TradingDatabase
from config.stock_universe import StockUniverse

class MultiTimeframeDataManager:
    """
    Advanced data management system for multi-timeframe analysis
    Handles smart data downloading, caching, and updates
    """
    
    # Timeframe configurations
    TIMEFRAMES = {
        '1D': {'yf_interval': '1d', 'update_freq_hours': 24, 'history_days': 252},  # 1 year
        '4H': {'yf_interval': '1h', 'update_freq_hours': 4, 'history_days': 60},    # 2 months  
        '1H': {'yf_interval': '1h', 'update_freq_hours': 1, 'history_days': 30},    # 1 month
        '15M': {'yf_interval': '15m', 'update_freq_hours': 0.25, 'history_days': 7} # 1 week
    }
    
    def __init__(self, db: TradingDatabase = None, logger: logging.Logger = None):
        self.db = db or TradingDatabase()
        self.logger = logger or logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self._request_lock = threading.Lock()
        
        # Cache for recent data
        self._data_cache = {}
        self._cache_lock = threading.Lock()
        
    def download_symbol_data(self, symbol: str, timeframes: List[str] = None, 
                           force_update: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download data for a symbol across multiple timeframes
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to download
            force_update: Force update even if recent data exists
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        timeframes = timeframes or list(self.TIMEFRAMES.keys())
        results = {}
        
        self.logger.info(f"Downloading data for {symbol}: {timeframes}")
        
        for timeframe in timeframes:
            try:
                # Check if update needed
                if not force_update and self._is_data_current(symbol, timeframe):
                    self.logger.debug(f"Skipping {symbol} {timeframe} - data is current")
                    continue
                
                # Download data
                df = self._download_timeframe_data(symbol, timeframe)
                if df is not None and not df.empty:
                    # Store in database
                    self.db.store_stock_data(symbol, timeframe, df)
                    results[timeframe] = df
                    
                    self.logger.debug(f"Downloaded {len(df)} bars for {symbol} {timeframe}")
                else:
                    self.logger.warning(f"No data returned for {symbol} {timeframe}")
                    
            except Exception as e:
                self.logger.error(f"Error downloading {symbol} {timeframe}: {e}")
                
        return results
    
    def _download_timeframe_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Download data for specific symbol/timeframe"""
        config = self.TIMEFRAMES.get(timeframe)
        if not config:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        current_date = datetime.now().date()
        is_holiday_today = self._is_weekend_or_holiday(current_date)
        
        # Rate limiting
        with self._request_lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        try:
            # Get existing data to determine download range
            latest_timestamp = self.db.get_latest_timestamp(symbol, timeframe)
            
            if latest_timestamp:
                # Check how old our data is
                hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
                update_threshold = config['update_freq_hours']
                
                # For intraday data, if we're within the update threshold and market hasn't been open long, skip
                if timeframe != '1D' and hours_old < update_threshold:
                    # If it's early in the day and we have recent data, don't update yet
                    now = datetime.now()
                    if now.hour < 11:  # Before 11 AM, let market settle
                        self.logger.debug(f"Too early for {symbol} {timeframe} update (data {hours_old:.1f}h old)")
                        return None
                
                # For date range downloads, they often fail - use period instead
                # Download recent data using period to ensure we get fresh data
                if timeframe == '15M':
                    period = "7d"  # Last 7 days for 15M
                elif timeframe in ['1H', '4H']:
                    period = "30d"  # Last 30 days for hourly data
                else:
                    period = f"{config['history_days']}d"
                
                start = None
                end = None
            else:
                # Download full history using period
                period = f"{config['history_days']}d"
                start = None
                end = None
            
            # Download from yfinance
            ticker = yf.Ticker(symbol)
            
            if period:
                df = ticker.history(period=period, interval=config['yf_interval'])
            else:
                df = ticker.history(start=start, end=end, interval=config['yf_interval'])
            
            if df.empty:
                self.logger.debug(f"No data available for {symbol} {timeframe} (start={start}, end={end})")
                return None
                
            # Process data for timeframe
            df = self._process_raw_data(df, timeframe)
            
            return df
            
        except Exception as e:
            self.logger.error(f"yfinance error for {symbol} {timeframe}: {e}")
            return None
    
    def _process_raw_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Process raw yfinance data for storage"""
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Handle different timeframes
        if timeframe == '4H':
            # Resample 1H data to 4H
            df = self._resample_to_4h(df)
        elif timeframe == '15M':
            # Keep 15M data as is
            pass
            
        # Clean data
        df = df.dropna()
        
        # Ensure proper types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(int)
        
        return df
    
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H data to 4H bars"""
        if df.empty:
            return df
            
        # Define 4H aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to 4H starting at market open (9:30 AM)
        df_4h = df.resample('4h', offset='0.5h').agg(agg_rules)
        
        # Remove bars with no data
        df_4h = df_4h.dropna()
        
        return df_4h
    
    def _is_data_current(self, symbol: str, timeframe: str) -> bool:
        """Check if data is current enough to skip download"""
        config = self.TIMEFRAMES.get(timeframe)
        if not config:
            return False
            
        latest_timestamp = self.db.get_latest_timestamp(symbol, timeframe)
        if not latest_timestamp:
            return False
        
        now = datetime.now()
        
        # Special handling for daily data - only download once per market day
        if timeframe == '1D':
            # Check if we already have today's data or if market isn't open yet
            latest_date = latest_timestamp.date()
            current_date = now.date()
            
            # If latest data is from today or future, no need to download
            if latest_date >= current_date:
                return True
                
            # If today is weekend/holiday, don't download
            if self._is_weekend_or_holiday(current_date):
                return True  # Don't download on weekends/holidays
                
            # If it's a weekday but before market open (9:30 AM ET), use yesterday's data
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                # Check if we have the previous market day's data
                last_market_day = self._get_last_market_day(current_date)
                return latest_date >= last_market_day
            
            # If we have Friday's data and today is Monday after a holiday weekend, it's current
            last_market_day = self._get_last_market_day(current_date)
            if latest_date >= last_market_day:
                return True
                
            # Otherwise check if we need to update
            return latest_date >= current_date
            
        # For intraday timeframes, consider market trading days
        # If today is weekend/holiday and we have recent data, don't download
        if self._is_weekend_or_holiday(now.date()):
            return True  # Don't download on weekends/holidays
        
        # If it's premarket (before 9:30 AM) and we have yesterday's data, consider it current
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            # Check if we have data from the last market day
            last_market_day = self._get_last_market_day(now.date())
            latest_date = latest_timestamp.date()
            if latest_date >= last_market_day:
                return True
        
        # Otherwise use time-based logic
        hours_old = (now - latest_timestamp).total_seconds() / 3600
        return hours_old < config['update_freq_hours']
    
    def _is_weekend_or_holiday(self, date) -> bool:
        """Check if date is weekend or market holiday"""
        # Weekend check
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return True
            
        # US market holidays for 2024-2025
        year = date.year
        month = date.month
        day = date.day
        
        # Fixed date holidays
        holidays_fixed = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day  
            (12, 25), # Christmas Day
        ]
        
        if (month, day) in holidays_fixed:
            return True
        
        # Labor Day - First Monday in September
        if year == 2024 and month == 9 and day == 2:
            return True
        if year == 2025 and month == 9 and day == 1:  # Labor Day 2025 is Sept 1st (Monday)
            return True
            
        # Martin Luther King Jr. Day - 3rd Monday in January
        if month == 1 and date.weekday() == 0:  # Monday
            # Calculate which Monday of January this is
            first_monday = 7 - (datetime(year, 1, 1).weekday() - 0) % 7
            if first_monday == 7:
                first_monday = 1
            third_monday = first_monday + 14
            if day == third_monday:
                return True
        
        # Presidents Day - 3rd Monday in February
        if month == 2 and date.weekday() == 0:  # Monday
            first_monday = 7 - (datetime(year, 2, 1).weekday() - 0) % 7
            if first_monday == 7:
                first_monday = 1
            third_monday = first_monday + 14
            if day == third_monday:
                return True
        
        # Memorial Day - Last Monday in May
        if month == 5 and date.weekday() == 0:  # Monday
            # Check if this is the last Monday of the month
            next_week = date + timedelta(days=7)
            if next_week.month != 5:  # Next Monday is in June
                return True
        
        # Thanksgiving - 4th Thursday in November
        if month == 11 and date.weekday() == 3:  # Thursday
            first_thursday = 7 - (datetime(year, 11, 1).weekday() - 3) % 7
            if first_thursday == 7:
                first_thursday = 1
            fourth_thursday = first_thursday + 21
            if day == fourth_thursday:
                return True
        
        # Day after Thanksgiving - 4th Friday in November
        if month == 11 and date.weekday() == 4:  # Friday
            first_thursday = 7 - (datetime(year, 11, 1).weekday() - 3) % 7
            if first_thursday == 7:
                first_thursday = 1
            fourth_friday = first_thursday + 22
            if day == fourth_friday:
                return True
        
        return False
    
    def _get_last_market_day(self, current_date) -> datetime.date:
        """Get the last trading day (excluding weekends and holidays)"""
        check_date = current_date - timedelta(days=1)
        
        # Go back until we find a trading day (not weekend or holiday)
        while self._is_weekend_or_holiday(check_date):
            check_date -= timedelta(days=1)
            
        return check_date
    
    def _get_next_trading_day(self, current_date) -> datetime.date:
        """Get the next trading day after current_date"""
        check_date = current_date + timedelta(days=1)
        
        # Go forward until we find a trading day
        while self._is_weekend_or_holiday(check_date):
            check_date += timedelta(days=1)
            
        return check_date
    
    def _is_market_open_today(self) -> bool:
        """Check if market is open today (considering trading hours)"""
        now = datetime.now()
        
        # Market is closed on weekends/holidays
        if self._is_weekend_or_holiday(now.date()):
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        # For simplicity, consider market "open" during and after market hours on trading days
        market_open_hour = 9
        market_open_minute = 30
        
        # If before market open, consider market not yet open for new data
        if now.hour < market_open_hour or (now.hour == market_open_hour and now.minute < market_open_minute):
            return False
        
        return True
    
    def bulk_download(self, symbols: List[str], timeframes: List[str] = None,
                     max_workers: int = 5) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download data for multiple symbols in parallel
        
        Args:
            symbols: List of symbols
            timeframes: List of timeframes  
            max_workers: Maximum concurrent downloads
            
        Returns:
            Dict[symbol][timeframe] -> DataFrame
        """
        timeframes = timeframes or list(self.TIMEFRAMES.keys())
        results = {}
        
        self.logger.info(f"Bulk downloading {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {}
            for symbol in symbols:
                future = executor.submit(self.download_symbol_data, symbol, timeframes)
                future_to_symbol[future] = symbol
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_data = future.result()
                    if symbol_data:
                        results[symbol] = symbol_data
                except Exception as e:
                    self.logger.error(f"Error downloading {symbol}: {e}")
        
        self.logger.info(f"Bulk download completed: {len(results)} symbols processed")
        return results
    
    def get_cached_data(self, symbol: str, timeframe: str, 
                       bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Get cached data from database
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            bars: Number of recent bars to get
            
        Returns:
            DataFrame with recent bars or None
        """
        try:
            # First try to get any available data for this symbol/timeframe
            df = self.db.get_stock_data(symbol, timeframe)
            
            if df is not None and not df.empty:
                # Return most recent bars
                return df.tail(bars)
            
            # If no data, return None
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached data for {symbol} {timeframe}: {e}")
            return None
    
    def update_watchlist_data(self, watchlist: List[str], 
                            priority_timeframes: List[str] = None) -> Dict[str, bool]:
        """
        Update data for watchlist symbols with priority timeframes first
        
        Args:
            watchlist: List of symbols to update
            priority_timeframes: Timeframes to update first
            
        Returns:
            Dict mapping symbol to success status
        """
        priority_timeframes = priority_timeframes or ['1D', '4H']
        all_timeframes = list(self.TIMEFRAMES.keys())
        results = {}
        
        self.logger.info(f"Updating watchlist data for {len(watchlist)} symbols")
        
        # Update priority timeframes first
        for timeframe in priority_timeframes:
            for symbol in watchlist:
                try:
                    symbol_results = self.download_symbol_data(symbol, [timeframe])
                    results[symbol] = len(symbol_results) > 0
                except Exception as e:
                    self.logger.error(f"Error updating {symbol} {timeframe}: {e}")
                    results[symbol] = False
        
        # Update remaining timeframes  
        remaining_timeframes = [tf for tf in all_timeframes if tf not in priority_timeframes]
        if remaining_timeframes:
            for symbol in watchlist:
                try:
                    symbol_results = self.download_symbol_data(symbol, remaining_timeframes)
                    # Update status - True if any timeframe succeeded
                    if symbol not in results or not results[symbol]:
                        results[symbol] = len(symbol_results) > 0
                except Exception as e:
                    self.logger.error(f"Error updating {symbol} remaining timeframes: {e}")
        
        success_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Watchlist update completed: {success_count}/{len(watchlist)} symbols updated")
        
        return results
    
    def get_data_status(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get data availability status for symbols"""
        if symbols is None:
            # Get symbols from watchlist
            watchlist = self.db.get_current_watchlist()
            symbols = [item['symbol'] for item in watchlist]
        
        status = {}
        
        for symbol in symbols:
            symbol_status = {}
            
            for timeframe in self.TIMEFRAMES.keys():
                latest = self.db.get_latest_timestamp(symbol, timeframe)
                if latest:
                    hours_old = (datetime.now() - latest).total_seconds() / 3600
                    symbol_status[timeframe] = {
                        'latest_timestamp': latest,
                        'hours_old': round(hours_old, 1),
                        'current': hours_old < self.TIMEFRAMES[timeframe]['update_freq_hours']
                    }
                else:
                    symbol_status[timeframe] = {
                        'latest_timestamp': None,
                        'hours_old': None,
                        'current': False
                    }
            
            status[symbol] = symbol_status
        
        return status
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old stock data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.db.db_path.parent:
                import sqlite3
                with sqlite3.connect(self.db.db_path) as conn:
                    # Clean up old stock data
                    result = conn.execute('''
                        DELETE FROM stock_data 
                        WHERE timestamp < ?
                    ''', (cutoff_date,))
                    
                    deleted_rows = result.rowcount
                    
                    # Clean up old technical indicators
                    conn.execute('''
                        DELETE FROM technical_indicators 
                        WHERE timestamp < ?
                    ''', (cutoff_date,))
                    
                    conn.commit()
                    
            self.logger.info(f"Cleaned up {deleted_rows} old data rows (older than {days_to_keep} days)")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def export_data(self, symbols: List[str], output_dir: str, 
                   timeframes: List[str] = None, days: int = 30):
        """Export data to CSV files"""
        timeframes = timeframes or list(self.TIMEFRAMES.keys())
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Exporting data for {len(symbols)} symbols to {output_dir}")
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    start_date = datetime.now() - timedelta(days=days)
                    df = self.db.get_stock_data(symbol, timeframe, start_date=start_date)
                    
                    if df is not None and not df.empty:
                        filename = f"{symbol}_{timeframe}_{days}d.csv"
                        df.to_csv(output_path / filename)
                        
                except Exception as e:
                    self.logger.error(f"Error exporting {symbol} {timeframe}: {e}")
        
        self.logger.info("Data export completed")
    
    def validate_data_integrity(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Validate data integrity and identify issues"""
        if symbols is None:
            watchlist = self.db.get_current_watchlist()
            symbols = [item['symbol'] for item in watchlist]
        
        issues = {}
        
        for symbol in symbols:
            symbol_issues = {}
            
            for timeframe in self.TIMEFRAMES.keys():
                try:
                    df = self.get_cached_data(symbol, timeframe, bars=100)
                    
                    if df is None or df.empty:
                        symbol_issues[timeframe] = ['No data available']
                        continue
                    
                    tf_issues = []
                    
                    # Check for missing data (gaps)
                    if len(df) < 50:  # Too little data
                        tf_issues.append('Insufficient data')
                    
                    # Check for invalid OHLC relationships
                    invalid_ohlc = ((df['high'] < df['low']) | 
                                   (df['high'] < df['open']) | 
                                   (df['high'] < df['close']) |
                                   (df['low'] > df['open']) | 
                                   (df['low'] > df['close']))
                    
                    if invalid_ohlc.any():
                        tf_issues.append('Invalid OHLC relationships')
                    
                    # Check for zero volume
                    if (df['volume'] == 0).any():
                        tf_issues.append('Zero volume bars detected')
                    
                    # Check for excessive gaps in timestamps
                    if len(df) > 1:
                        time_diffs = df.index.to_series().diff()
                        if timeframe == '1D':
                            max_gap = timedelta(days=5)  # Weekend + holiday
                        elif timeframe == '4H':
                            max_gap = timedelta(hours=72)  # Weekend
                        elif timeframe == '1H':
                            max_gap = timedelta(hours=72)  # Weekend  
                        else:  # 15M
                            max_gap = timedelta(hours=72)  # Weekend
                        
                        if (time_diffs > max_gap).any():
                            tf_issues.append('Large timestamp gaps detected')
                    
                    if tf_issues:
                        symbol_issues[timeframe] = tf_issues
                        
                except Exception as e:
                    symbol_issues[timeframe] = [f'Validation error: {e}']
            
            if symbol_issues:
                issues[symbol] = symbol_issues
        
        return issues