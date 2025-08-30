# wyckoff_bot/signals/market_scanner.py
"""
Market Scanner for Wyckoff Signals
==================================
Scans market for stocks meeting Wyckoff criteria
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import concurrent.futures
from dataclasses import dataclass

@dataclass
class ScanResult:
    """Market scan result"""
    symbol: str
    score: float
    phase: str
    confidence: float
    volume_spike: bool
    price_action_strength: float
    
class MarketScanner:
    """
    Market scanner for Wyckoff opportunities
    Screens stocks based on Wyckoff criteria
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Default stock universe (can be expanded)
        self.default_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'BA',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'MDT',
            'XOM', 'CVX', 'COP', 'SLB', 'MPC', 'VLO', 'PSX', 'KMI'
        ]
    
    def scan_market(self, symbols: List[str] = None, 
                   period: str = '3mo',
                   min_volume: int = 1000000) -> List[ScanResult]:
        """
        Scan market for Wyckoff opportunities
        
        Args:
            symbols: List of symbols to scan (uses default if None)
            period: Data period to download
            min_volume: Minimum average daily volume
            
        Returns:
            List[ScanResult]: Scan results sorted by score
        """
        symbols = symbols or self.default_universe
        self.logger.info(f"Scanning {len(symbols)} symbols for Wyckoff patterns")
        
        results = []
        
        # Use threading to speed up data collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._scan_single_symbol, symbol, period, min_volume): symbol 
                for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error scanning {symbol}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Scan completed: {len(results)} opportunities found")
        return results
    
    def _scan_single_symbol(self, symbol: str, period: str, 
                           min_volume: int) -> Optional[ScanResult]:
        """Scan single symbol for opportunities"""
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty or len(df) < 50:
                return None
            
            # Check volume requirement
            avg_volume = df['Volume'].tail(20).mean()
            if avg_volume < min_volume:
                return None
            
            # Perform Wyckoff analysis
            scan_result = self._analyze_for_scan(df, symbol)
            return scan_result
            
        except Exception as e:
            self.logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _analyze_for_scan(self, df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
        """Analyze single stock for Wyckoff patterns"""
        # Basic pattern recognition (simplified for scanning)
        
        # Calculate key metrics
        recent_data = df.tail(20)
        
        # Volume analysis
        volume_spike = self._detect_volume_spike(recent_data)
        volume_trend = self._calculate_volume_trend(recent_data)
        
        # Price action
        price_action_strength = self._calculate_price_action_strength(recent_data)
        
        # Phase identification (simplified)
        phase = self._identify_phase_simplified(recent_data)
        
        # Calculate overall score
        score = self._calculate_scan_score(
            recent_data, volume_spike, volume_trend, price_action_strength, phase
        )
        
        # Minimum score threshold
        if score < 30:
            return None
        
        # Estimate confidence
        confidence = min(score / 100, 0.95)
        
        return ScanResult(
            symbol=symbol,
            score=score,
            phase=phase,
            confidence=confidence,
            volume_spike=volume_spike,
            price_action_strength=price_action_strength
        )
    
    def _detect_volume_spike(self, df: pd.DataFrame) -> bool:
        """Detect unusual volume activity"""
        recent_volume = df['Volume'].tail(3).mean()
        avg_volume = df['Volume'].mean()
        
        return recent_volume > avg_volume * 1.5
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend (positive = increasing)"""
        recent_volume = df['Volume'].tail(5).mean()
        older_volume = df['Volume'].iloc[-10:-5].mean()
        
        if older_volume > 0:
            return (recent_volume - older_volume) / older_volume
        return 0
    
    def _calculate_price_action_strength(self, df: pd.DataFrame) -> float:
        """Calculate price action strength"""
        # Range analysis
        recent_range = (df['High'].max() - df['Low'].min()) / df['Close'].iloc[-1]
        
        # Trend strength
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        # Volume confirmation
        volume_strength = self._calculate_volume_trend(df)
        
        # Combine factors
        strength = abs(price_change) * 2 + abs(volume_strength) * 0.5 + recent_range * 0.3
        return min(strength, 1.0)
    
    def _identify_phase_simplified(self, df: pd.DataFrame) -> str:
        """Simplified phase identification for scanning"""
        price_volatility = df['Close'].pct_change().std()
        price_trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        volume_trend = self._calculate_volume_trend(df)
        
        # Trading range
        price_range = (df['High'].max() - df['Low'].min()) / df['Close'].iloc[-1]
        
        if price_range < 0.15 and price_volatility < 0.02:
            if volume_trend > 0:
                return 'accumulation' if price_trend >= 0 else 'distribution'
            else:
                return 'consolidation'
        elif price_trend > 0.05 and volume_trend > 0:
            return 'markup'
        elif price_trend < -0.05 and volume_trend > 0:
            return 'markdown'
        else:
            return 'unknown'
    
    def _calculate_scan_score(self, df: pd.DataFrame, volume_spike: bool, 
                             volume_trend: float, price_strength: float, phase: str) -> float:
        """Calculate overall scan score (0-100)"""
        score = 0.0
        
        # Phase scoring
        phase_scores = {
            'accumulation': 30,
            'distribution': 25,
            'markup': 20,
            'markdown': 15,
            'consolidation': 10,
            'unknown': 0
        }
        score += phase_scores.get(phase, 0)
        
        # Volume scoring
        if volume_spike:
            score += 25
        if volume_trend > 0.2:
            score += 15
        elif volume_trend > 0:
            score += 10
        
        # Price action scoring
        score += price_strength * 20
        
        # Recent price action bonus
        recent_change = abs((df['Close'].iloc[-1] - df['Close'].iloc[-3]) / df['Close'].iloc[-3])
        if recent_change > 0.03:  # 3% move
            score += 10
        
        return min(score, 100.0)
    
    def filter_scan_results(self, results: List[ScanResult], 
                           criteria: Dict) -> List[ScanResult]:
        """Filter scan results by criteria"""
        filtered = results.copy()
        
        if 'min_score' in criteria:
            filtered = [r for r in filtered if r.score >= criteria['min_score']]
        
        if 'min_confidence' in criteria:
            filtered = [r for r in filtered if r.confidence >= criteria['min_confidence']]
        
        if 'phases' in criteria:
            allowed_phases = criteria['phases']
            filtered = [r for r in filtered if r.phase in allowed_phases]
        
        if 'require_volume_spike' in criteria and criteria['require_volume_spike']:
            filtered = [r for r in filtered if r.volume_spike]
        
        if 'min_price_strength' in criteria:
            filtered = [r for r in filtered if r.price_action_strength >= criteria['min_price_strength']]
        
        if 'max_results' in criteria:
            filtered = filtered[:criteria['max_results']]
        
        return filtered
    
    def get_detailed_data(self, symbols: List[str], 
                         period: str = '6mo') -> Dict[str, pd.DataFrame]:
        """
        Get detailed price data for selected symbols
        
        Args:
            symbols: List of symbols to get data for
            period: Data period
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> DataFrame mapping
        """
        data = {}
        
        self.logger.info(f"Downloading detailed data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    # Standardize column names
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high', 
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    data[symbol] = df
                    
            except Exception as e:
                self.logger.warning(f"Error downloading data for {symbol}: {e}")
        
        self.logger.info(f"Downloaded data for {len(data)} symbols")
        return data
    
    def create_watchlist(self, scan_results: List[ScanResult], 
                        max_symbols: int = 20) -> List[str]:
        """Create watchlist from scan results"""
        # Take top results
        top_results = scan_results[:max_symbols]
        
        # Prefer certain phases
        priority_phases = ['accumulation', 'distribution', 'markup']
        
        watchlist = []
        
        # Add priority phase stocks first
        for phase in priority_phases:
            phase_stocks = [r.symbol for r in top_results if r.phase == phase]
            watchlist.extend(phase_stocks)
        
        # Add remaining stocks
        for result in top_results:
            if result.symbol not in watchlist:
                watchlist.append(result.symbol)
        
        return watchlist[:max_symbols]