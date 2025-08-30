# wyckoff_bot/data/market_data.py
"""
Market Data Provider for Wyckoff Bot
====================================
Provides market data integration for Wyckoff analysis
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time

class MarketDataProvider:
    """
    Market data provider using yfinance
    Handles data collection and caching for Wyckoff analysis
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_price_data(self, symbol: str, period: str = '6mo',
                      interval: str = '1d') -> pd.DataFrame:
        """
        Get price data for symbol
        
        Args:
            symbol: Stock symbol
            period: Data period (1mo, 3mo, 6mo, 1y, etc.)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            pd.DataFrame: OHLCV data with standardized columns
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached data for {symbol}")
            return self.cache[cache_key]['data']
        
        try:
            self.logger.debug(f"Downloading data for {symbol} ({period})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Cache the data
            self.cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }
            
            self.logger.debug(f"Downloaded {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], period: str = '6mo',
                           interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols
        
        Args:
            symbols: List of symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> DataFrame mapping
        """
        self.logger.info(f"Downloading data for {len(symbols)} symbols")
        data = {}
        
        for symbol in symbols:
            df = self.get_price_data(symbol, period, interval)
            if not df.empty:
                data[symbol] = df
        
        self.logger.info(f"Successfully downloaded data for {len(data)} symbols")
        return data
    
    def get_intraday_data(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Get recent intraday data for detailed analysis"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get last 5 days of hourly data
            df = ticker.history(period=f'{days}d', interval='1h')
            
            if df.empty:
                return pd.DataFrame()
            
            return self._standardize_columns(df)
            
        except Exception as e:
            self.logger.error(f"Error getting intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_info(self, symbol: str) -> Dict:
        """Get additional market information for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            return {
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'float_shares': info.get('floatShares', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            self.logger.debug(f"Could not get market info for {symbol}: {e}")
            return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and has data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='5d')
            return not df.empty
            
        except Exception:
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get current price from info
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price:
                return float(price)
            
            # Fallback to recent history
            df = ticker.history(period='1d')
            if not df.empty:
                return float(df['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_premarket_data(self, symbol: str) -> Dict:
        """Get pre-market trading data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get pre-market data (if available)
            df = ticker.history(period='1d', prepost=True, interval='1m')
            
            if df.empty:
                return {}
            
            # Check for pre-market activity
            market_open = df.index.normalize() + pd.Timedelta(hours=9, minutes=30)
            premarket = df[df.index < market_open]
            
            if premarket.empty:
                return {}
            
            return {
                'premarket_volume': int(premarket['Volume'].sum()),
                'premarket_high': float(premarket['High'].max()),
                'premarket_low': float(premarket['Low'].min()),
                'premarket_last': float(premarket['Close'].iloc[-1])
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting pre-market data for {symbol}: {e}")
            return {}
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names"""
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.warning(f"Missing column: {col}")
        
        return df
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        age = time.time() - self.cache[cache_key]['timestamp']
        return age < self.cache_timeout
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.logger.info("Market data cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_info = {}
        current_time = time.time()
        
        for key, data in self.cache.items():
            age = current_time - data['timestamp']
            cache_info[key] = {
                'age_seconds': int(age),
                'rows': len(data['data']),
                'valid': age < self.cache_timeout
            }
        
        return cache_info
    
    def update_cache_timeout(self, seconds: int):
        """Update cache timeout period"""
        self.cache_timeout = seconds
        self.logger.info(f"Cache timeout updated to {seconds} seconds")
    
    def get_earnings_calendar(self, symbol: str) -> List[Dict]:
        """Get upcoming earnings dates (if available)"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return []
            
            # Convert to list of dicts
            earnings = []
            for date, row in calendar.iterrows():
                earnings.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'estimate': row.get('Earnings Estimate', 0)
                })
            
            return earnings
            
        except Exception as e:
            self.logger.debug(f"Could not get earnings calendar for {symbol}: {e}")
            return []