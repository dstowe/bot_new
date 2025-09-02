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

# Import stock universe
from config.stock_universe import StockUniverse
from ..data.data_manager import WyckoffDataManager  
from ..analysis.wyckoff_analyzer import WyckoffAnalyzer
from ..analysis.market_regime import MarketRegimeAnalyzer

@dataclass
class ScanResult:
    """Enhanced market scan result with multi-timeframe data"""
    symbol: str
    score: float
    phase: str
    confidence: float
    volume_spike: bool
    price_action_strength: float
    timeframe_analysis: Dict[str, str] = None  # Multi-timeframe phase analysis
    entry_timing: str = 'WAIT'  # Entry timing signal
    market_regime_alignment: bool = False  # Aligns with current market regime
    
class MarketScanner:
    """
    Enhanced market scanner for Wyckoff opportunities with multi-timeframe analysis
    Screens stocks based on Wyckoff criteria across multiple timeframes
    """
    
    def __init__(self, data_manager: WyckoffDataManager = None, 
                 wyckoff_analyzer: WyckoffAnalyzer = None,
                 market_regime_analyzer: MarketRegimeAnalyzer = None,
                 multi_tf_data_manager = None,
                 logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize analyzers
        self.data_manager = data_manager or WyckoffDataManager(logger=self.logger)
        self.wyckoff_analyzer = wyckoff_analyzer or WyckoffAnalyzer(logger=self.logger)
        self.market_regime_analyzer = market_regime_analyzer or MarketRegimeAnalyzer(logger=self.logger)
        self.multi_tf_data_manager = multi_tf_data_manager
        
        # Use comprehensive stock universe optimized for Wyckoff trading
        self.default_universe = StockUniverse.get_wyckoff_optimized_list()
        self.high_volume_universe = StockUniverse.get_high_volume_stocks(min_volume=10000000)
        self.sector_map = StockUniverse.get_sector_mapping()
        
        # Multi-timeframe configuration
        self.scan_timeframes = ['1D', '4H']  # Primary timeframes for scanning
        self.confirmation_timeframes = ['1H']  # Confirmation timeframes
    
    def scan_market(self, symbols: List[str] = None, 
                   period: str = '3mo',
                   min_volume: int = 1000000,
                   use_cached_data: bool = True) -> List[ScanResult]:
        """
        Enhanced market scan with multi-timeframe analysis
        
        Args:
            symbols: List of symbols to scan (uses default if None)
            period: Data period to download (if not using cached data)
            min_volume: Minimum average daily volume
            use_cached_data: Use cached multi-timeframe data if available
            
        Returns:
            List[ScanResult]: Scan results sorted by score
        """
        symbols = symbols or self.default_universe
        self.logger.info(f"Enhanced scanning {len(symbols)} symbols for Wyckoff patterns")
        
        # Get current market regime for context
        market_regime = self.market_regime_analyzer.analyze_market_regime()
        
        results = []
        
        if use_cached_data:
            # Use cached multi-timeframe data
            results = self._scan_with_cached_data(symbols, market_regime, min_volume)
        else:
            # Download fresh data and scan
            results = self._scan_with_fresh_data(symbols, period, market_regime, min_volume)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Enhanced scan completed: {len(results)} opportunities found")
        return results
    
    def _scan_single_symbol(self, symbol: str, period: str, 
                           min_volume: int) -> Optional[ScanResult]:
        """Scan single symbol for opportunities"""
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty or len(df) < 20:  # Reduced from 50 to 20 for monthly scans
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
        
        # Minimum score threshold (lowered for better results)
        if score < 15:
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
        
        # Relaxed criteria for better detection
        if price_range < 0.25 and price_volatility < 0.035:  # Relaxed from 0.15 and 0.02
            if volume_trend > 0:
                return 'accumulation' if price_trend >= 0 else 'distribution'
            else:
                return 'consolidation'
        elif price_trend > 0.03 and volume_trend > -0.1:  # Relaxed trend requirements
            return 'markup'
        elif price_trend < -0.03 and volume_trend > -0.1:
            return 'markdown'
        elif abs(price_trend) < 0.02:  # Low volatility sideways movement
            return 'consolidation'
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
    
    # Enhanced multi-timeframe scanning methods
    
    def _scan_with_cached_data(self, symbols: List[str], market_regime: Dict, 
                              min_volume: int) -> List[ScanResult]:
        """Scan using cached multi-timeframe data"""
        results = []
        
        self.logger.info(f"Scanning {len(symbols)} symbols using cached data")
        
        # Use threading for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_symbol_cached, symbol, market_regime, min_volume): symbol
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
        
        return results
    
    def _scan_with_fresh_data(self, symbols: List[str], period: str, 
                             market_regime: Dict, min_volume: int) -> List[ScanResult]:
        """Scan with fresh data download"""
        results = []
        
        self.logger.info(f"Downloading fresh data for {len(symbols)} symbols")
        
        # Download multi-timeframe data
        download_results = self.data_manager.multi_tf_manager.bulk_download(
            symbols, self.scan_timeframes + self.confirmation_timeframes
        )
        
        # Analyze symbols with successful downloads
        successful_symbols = [symbol for symbol, data in download_results.items() if data]
        
        for symbol in successful_symbols:
            try:
                result = self._analyze_symbol_cached(symbol, market_regime, min_volume)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol}: {e}")
        
        return results
    
    def _analyze_symbol_cached(self, symbol: str, market_regime: Dict, 
                              min_volume: int) -> Optional[ScanResult]:
        """Analyze single symbol using cached multi-timeframe data"""
        try:
            # Get multi-timeframe data using the enhanced data manager if available
            all_timeframes = self.scan_timeframes + self.confirmation_timeframes
            
            if self.multi_tf_data_manager:
                # Try to get cached multi-timeframe data
                mtf_data = {}
                missing_timeframes = []
                
                for timeframe in all_timeframes:
                    df = self.multi_tf_data_manager.get_cached_data(
                        symbol, timeframe, bars=100
                    )
                    if df is not None and len(df) >= 50:
                        mtf_data[timeframe] = df
                    else:
                        missing_timeframes.append(timeframe)
                
                # If we're missing critical timeframes, try to download them
                if missing_timeframes and len(missing_timeframes) < len(all_timeframes) / 2:
                    try:
                        downloaded_data = self.multi_tf_data_manager.download_symbol_data(
                            symbol, missing_timeframes
                        )
                        mtf_data.update(downloaded_data)
                    except Exception as e:
                        self.logger.debug(f"Could not download missing data for {symbol}: {e}")
                
            else:
                # Fallback to original data manager
                mtf_data = self.data_manager.get_multi_timeframe_data(
                    symbol, all_timeframes, bars=100
                )
            
            if not mtf_data or len(mtf_data) < len(self.scan_timeframes):
                return None  # Insufficient data
            
            # Check volume requirement using daily data
            daily_data = mtf_data.get('1D')
            if daily_data is not None:
                avg_volume = daily_data['volume'].tail(20).mean()
                if avg_volume < min_volume:
                    return None
            
            # Perform multi-timeframe Wyckoff analysis
            mtf_analyses = self.wyckoff_analyzer.analyze_multi_timeframe(mtf_data, symbol)
            
            if not mtf_analyses:
                return None
            
            # Generate multi-timeframe signal
            mtf_signal = self.wyckoff_analyzer.get_multi_timeframe_signal(mtf_analyses)
            
            # Get primary analysis (highest timeframe)
            primary_tf = self.scan_timeframes[0]  # '1D'
            primary_analysis = mtf_analyses.get(primary_tf)
            
            if not primary_analysis:
                return None
            
            # Calculate enhanced score
            enhanced_score = self._calculate_enhanced_score(
                mtf_analyses, mtf_signal, market_regime
            )
            
            # Check entry timing
            entry_timing_signal = self.wyckoff_analyzer.get_entry_timing_signal(
                mtf_analyses, mtf_signal['signal']
            )
            
            # Create timeframe analysis summary
            timeframe_analysis = {
                tf: analysis.phase.value for tf, analysis in mtf_analyses.items()
            }
            
            # Check market regime alignment
            regime_alignment = self._check_regime_alignment(
                primary_analysis.phase, market_regime
            )
            
            return ScanResult(
                symbol=symbol,
                score=enhanced_score,
                phase=primary_analysis.phase.value,
                confidence=mtf_signal['confidence'],
                volume_spike=self._detect_volume_spike_mtf(mtf_data),
                price_action_strength=primary_analysis.price_action_strength,
                timeframe_analysis=timeframe_analysis,
                entry_timing=entry_timing_signal['timing'],
                market_regime_alignment=regime_alignment
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_enhanced_score(self, mtf_analyses: Dict, mtf_signal: Dict, 
                                 market_regime: Dict) -> float:
        """Calculate enhanced score based on multi-timeframe analysis"""
        base_score = 0.0
        
        # Multi-timeframe signal strength
        base_score += mtf_signal['strength'] * 40
        
        # Confidence factor
        base_score += mtf_signal['confidence'] * 25
        
        # Market regime alignment bonus
        regime_type = market_regime.get('regime_type', 'UNKNOWN')
        primary_phase = mtf_signal.get('primary_phase', 'unknown')
        
        if regime_type in ['BULL', 'bull'] and primary_phase in ['accumulation', 'markup']:
            base_score += 15
        elif regime_type in ['BEAR', 'bear'] and primary_phase in ['distribution', 'markdown']:
            base_score += 15
        elif regime_type in ['RANGE', 'ranging'] and primary_phase in ['accumulation', 'distribution']:
            base_score += 10
        
        # Multi-timeframe confirmation bonus
        confirmations = []
        for analysis in mtf_analyses.values():
            if analysis.multi_timeframe_confirmation:
                confirmation_pct = sum(1 for conf in analysis.multi_timeframe_confirmation.values() if conf) / len(analysis.multi_timeframe_confirmation)
                confirmations.append(confirmation_pct)
        
        if confirmations:
            avg_confirmation = sum(confirmations) / len(confirmations)
            base_score += avg_confirmation * 20
        
        return min(base_score, 100.0)
    
    def _detect_volume_spike_mtf(self, mtf_data: Dict[str, pd.DataFrame]) -> bool:
        """Detect volume spikes across timeframes"""
        # Check daily timeframe for volume spikes
        daily_data = mtf_data.get('1D')
        if daily_data is not None:
            recent_volume = daily_data['volume'].tail(3).mean()
            avg_volume = daily_data['volume'].tail(20).mean()
            if recent_volume > avg_volume * 1.5:
                return True
        
        # Check 4H timeframe for intraday spikes
        h4_data = mtf_data.get('4H')
        if h4_data is not None:
            recent_volume = h4_data['volume'].tail(2).mean()
            avg_volume = h4_data['volume'].tail(12).mean()  # 2 days worth
            if recent_volume > avg_volume * 2.0:
                return True
        
        return False
    
    def _check_regime_alignment(self, phase, market_regime: Dict) -> bool:
        """Check if phase aligns with current market regime"""
        regime_type = market_regime.get('regime_type', 'UNKNOWN')
        
        if regime_type in ['BULL', 'bull']:
            return phase.value in ['accumulation', 'markup']
        elif regime_type in ['BEAR', 'bear']:
            return phase.value in ['distribution', 'markdown']
        elif regime_type in ['RANGE', 'ranging']:
            return phase.value in ['accumulation', 'distribution']
        
        return False
    
    def create_enhanced_watchlist(self, scan_results: List[ScanResult], 
                                max_symbols: int = 20) -> Dict:
        """
        Create enhanced watchlist with entry timing and regime context
        
        Returns:
            Dict with watchlist and analysis context
        """
        # Filter for regime-aligned and high-scoring results
        filtered_results = [
            r for r in scan_results 
            if r.market_regime_alignment and r.score >= 30
        ]
        
        # Sort by score
        top_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)[:max_symbols]
        
        # Categorize by entry timing
        ready_to_enter = [r.symbol for r in top_results if r.entry_timing == 'ENTER']
        watch_for_entry = [r.symbol for r in top_results if r.entry_timing == 'WAIT']
        
        # Create phase distribution
        phase_distribution = {}
        for result in top_results:
            phase = result.phase
            if phase not in phase_distribution:
                phase_distribution[phase] = []
            phase_distribution[phase].append(result.symbol)
        
        return {
            'watchlist': [r.symbol for r in top_results],
            'ready_to_enter': ready_to_enter,
            'watch_for_entry': watch_for_entry,
            'phase_distribution': phase_distribution,
            'total_symbols': len(top_results),
            'regime_aligned_count': len([r for r in top_results if r.market_regime_alignment])
        }
    
    def update_watchlist_data(self, watchlist: List[str]) -> Dict[str, bool]:
        """Update multi-timeframe data for watchlist symbols"""
        return self.data_manager.download_watchlist_data(
            watchlist, self.scan_timeframes + self.confirmation_timeframes
        )