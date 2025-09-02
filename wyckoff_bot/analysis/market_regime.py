# wyckoff_bot/analysis/market_regime.py
"""
Market Regime Analysis
======================
Detects market regimes (Bull/Bear/Range) with VIX integration and breadth indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear" 
    RANGE = "range"
    TRANSITION = "transition"
    UNKNOWN = "unknown"

class TrendStrength(Enum):
    """Trend strength levels"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

@dataclass
class RegimeAnalysis:
    """Market regime analysis results"""
    regime: MarketRegime
    confidence: float
    trend_strength: TrendStrength
    volatility_regime: str  # "low", "normal", "high", "extreme"
    cash_allocation_recommendation: float  # 0.15 to 0.80
    sector_rotation_signal: Dict[str, float]
    vix_analysis: Dict[str, float]
    breadth_indicators: Dict[str, float]
    regime_duration_days: int
    next_regime_probability: Dict[str, float]

class MarketRegimeAnalyzer:
    """
    Analyzes market regimes using multiple indicators
    Provides dynamic cash allocation and sector rotation signals
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Key market indicators
        self.market_etfs = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'VIX': 'VIX'
        }
        
        # Sector ETFs for rotation analysis
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLC': 'Communication',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities'
        }
        
        # Regime detection parameters
        self.lookback_periods = {
            'short': 20,    # 1 month
            'medium': 63,   # 3 months  
            'long': 252     # 1 year
        }
        
        # Cash allocation ranges
        self.cash_allocation_ranges = {
            MarketRegime.BULL: (0.15, 0.30),
            MarketRegime.BEAR: (0.50, 0.80),
            MarketRegime.RANGE: (0.25, 0.45),
            MarketRegime.TRANSITION: (0.35, 0.60),
            MarketRegime.UNKNOWN: (0.40, 0.60)
        }
        
        # Data cache
        self.data_cache = {}
        self.cache_expiry = datetime.now()
        
    def analyze_market_regime(self, lookback_days: int = 252) -> RegimeAnalysis:
        """
        Perform comprehensive market regime analysis
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            RegimeAnalysis: Complete regime analysis
        """
        try:
            self.logger.info("Starting market regime analysis...")
            
            # Get market data
            market_data = self._fetch_market_data(lookback_days)
            if not market_data:
                return self._create_unknown_analysis()
            
            # 1. Trend Analysis
            trend_analysis = self._analyze_trends(market_data)
            
            # 2. VIX Analysis  
            vix_analysis = self._analyze_vix(market_data)
            
            # 3. Breadth Indicators
            breadth_analysis = self._analyze_market_breadth(market_data)
            
            # 4. Sector Rotation Analysis
            sector_analysis = self._analyze_sector_rotation()
            
            # 5. Determine Market Regime
            regime, confidence = self._determine_regime(
                trend_analysis, vix_analysis, breadth_analysis
            )
            
            # 6. Calculate trend strength
            trend_strength = self._calculate_trend_strength(trend_analysis, vix_analysis)
            
            # 7. Determine volatility regime
            volatility_regime = self._determine_volatility_regime(vix_analysis)
            
            # 8. Calculate cash allocation recommendation
            cash_allocation = self._calculate_cash_allocation(
                regime, volatility_regime, confidence
            )
            
            # 9. Estimate regime duration
            regime_duration = self._estimate_regime_duration(trend_analysis)
            
            # 10. Calculate regime transition probabilities
            transition_probs = self._calculate_transition_probabilities(
                trend_analysis, vix_analysis, breadth_analysis
            )
            
            result = RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                cash_allocation_recommendation=cash_allocation,
                sector_rotation_signal=sector_analysis,
                vix_analysis=vix_analysis,
                breadth_indicators=breadth_analysis,
                regime_duration_days=regime_duration,
                next_regime_probability=transition_probs
            )
            
            self.logger.info(f"Market regime analysis completed: {regime.value} (confidence: {confidence:.1%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {e}")
            return self._create_unknown_analysis()
    
    def _fetch_market_data(self, lookback_days: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch market data for all indicators"""
        # Check cache
        if (datetime.now() - self.cache_expiry).total_seconds() < 3600:  # 1 hour cache
            if self.data_cache:
                return self.data_cache
        
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
        
        try:
            # Fetch market indicators
            for symbol, name in self.market_etfs.items():
                try:
                    if symbol == 'VIX':
                        ticker = yf.Ticker('^VIX')
                    else:
                        ticker = yf.Ticker(symbol)
                    
                    df = ticker.history(start=start_date, end=end_date)
                    if not df.empty:
                        data[symbol] = df
                        self.logger.debug(f"Fetched {len(df)} days of data for {symbol}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            # Cache the data
            if data:
                self.data_cache = data
                self.cache_expiry = datetime.now()
                
            return data if data else None
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None
    
    def _analyze_trends(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market trends across multiple timeframes"""
        trend_metrics = {}
        
        try:
            spy_data = market_data.get('SPY')
            if spy_data is None or len(spy_data) < 50:
                return trend_metrics
            
            closes = spy_data['Close']
            
            # Moving averages for trend identification
            ma_20 = closes.rolling(window=20).mean()
            ma_50 = closes.rolling(window=50).mean() 
            ma_200 = closes.rolling(window=200).mean()
            
            current_price = closes.iloc[-1]
            
            # Trend direction indicators
            trend_metrics['price_vs_ma20'] = (current_price - ma_20.iloc[-1]) / ma_20.iloc[-1]
            trend_metrics['price_vs_ma50'] = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1]
            trend_metrics['price_vs_ma200'] = (current_price - ma_200.iloc[-1]) / ma_200.iloc[-1]
            
            # Moving average alignment
            trend_metrics['ma20_vs_ma50'] = (ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
            trend_metrics['ma50_vs_ma200'] = (ma_50.iloc[-1] - ma_200.iloc[-1]) / ma_200.iloc[-1]
            
            # Price momentum (various timeframes)
            trend_metrics['momentum_5d'] = (current_price - closes.iloc[-6]) / closes.iloc[-6]
            trend_metrics['momentum_20d'] = (current_price - closes.iloc[-21]) / closes.iloc[-21] 
            trend_metrics['momentum_63d'] = (current_price - closes.iloc[-64]) / closes.iloc[-64]
            
            # Trend consistency (percentage of days above MA)
            recent_closes = closes.tail(20)
            recent_ma20 = ma_20.tail(20)
            trend_metrics['consistency_20d'] = (recent_closes > recent_ma20).mean()
            
            # Rate of change in moving averages
            ma20_roc = (ma_20.iloc[-1] - ma_20.iloc[-21]) / ma_20.iloc[-21]
            ma50_roc = (ma_50.iloc[-1] - ma_50.iloc[-51]) / ma_50.iloc[-51]
            
            trend_metrics['ma20_roc'] = ma20_roc
            trend_metrics['ma50_roc'] = ma50_roc
            
            self.logger.debug(f"Trend analysis completed: {len(trend_metrics)} metrics")
            return trend_metrics
            
        except Exception as e:
            self.logger.warning(f"Error in trend analysis: {e}")
            return {}
    
    def _analyze_vix(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze VIX for volatility regime detection"""
        vix_metrics = {}
        
        try:
            vix_data = market_data.get('VIX')
            if vix_data is None or len(vix_data) < 20:
                return vix_metrics
            
            vix_closes = vix_data['Close']
            current_vix = vix_closes.iloc[-1]
            
            # VIX levels and percentiles
            vix_metrics['current_vix'] = current_vix
            vix_metrics['vix_percentile_252d'] = (vix_closes.tail(252) < current_vix).mean()
            vix_metrics['vix_percentile_63d'] = (vix_closes.tail(63) < current_vix).mean()
            
            # VIX moving averages
            vix_ma_20 = vix_closes.rolling(window=20).mean().iloc[-1]
            vix_ma_50 = vix_closes.rolling(window=50).mean().iloc[-1]
            
            vix_metrics['vix_vs_ma20'] = (current_vix - vix_ma_20) / vix_ma_20
            vix_metrics['vix_vs_ma50'] = (current_vix - vix_ma_50) / vix_ma_50
            
            # VIX term structure (using VIX vs VIX9D if available)
            vix_metrics['vix_mean_reversion'] = self._calculate_vix_mean_reversion(vix_closes)
            
            # Fear/Greed indicators based on VIX
            if current_vix > 30:
                fear_level = min((current_vix - 30) / 20, 1.0)  # Scale 30-50 to 0-1
            else:
                fear_level = 0.0
                
            vix_metrics['fear_level'] = fear_level
            vix_metrics['complacency_level'] = max(0, (20 - current_vix) / 10) if current_vix < 20 else 0
            
            # VIX spike detection
            vix_5d_avg = vix_closes.tail(5).mean()
            vix_20d_avg = vix_closes.tail(20).mean()
            vix_metrics['vix_spike'] = max(0, (current_vix - vix_20d_avg) / vix_20d_avg)
            
            self.logger.debug(f"VIX analysis: current={current_vix:.1f}, percentile={vix_metrics['vix_percentile_252d']:.1%}")
            return vix_metrics
            
        except Exception as e:
            self.logger.warning(f"Error in VIX analysis: {e}")
            return {}
    
    def _calculate_vix_mean_reversion(self, vix_closes: pd.Series) -> float:
        """Calculate VIX mean reversion tendency"""
        if len(vix_closes) < 50:
            return 0.0
            
        # Calculate how often VIX reverts to mean after spikes
        vix_mean = vix_closes.tail(252).mean()
        current_vix = vix_closes.iloc[-1]
        
        # Mean reversion score
        deviation_from_mean = abs(current_vix - vix_mean) / vix_mean
        return min(deviation_from_mean, 1.0)
    
    def _analyze_market_breadth(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market breadth indicators"""
        breadth_metrics = {}
        
        try:
            # Use multiple market indices as breadth proxies
            spy_data = market_data.get('SPY')
            qqq_data = market_data.get('QQQ')
            iwm_data = market_data.get('IWM')
            
            if not all([spy_data is not None, qqq_data is not None, iwm_data is not None]):
                return breadth_metrics
            
            # Index performance comparison (breadth proxy)
            spy_returns = spy_data['Close'].pct_change(20).iloc[-1]  # 20-day return
            qqq_returns = qqq_data['Close'].pct_change(20).iloc[-1] 
            iwm_returns = iwm_data['Close'].pct_change(20).iloc[-1]
            
            breadth_metrics['spy_20d_return'] = spy_returns
            breadth_metrics['qqq_20d_return'] = qqq_returns  
            breadth_metrics['iwm_20d_return'] = iwm_returns
            
            # Breadth divergence (small caps vs large caps)
            breadth_metrics['iwm_spy_divergence'] = iwm_returns - spy_returns
            breadth_metrics['qqq_spy_divergence'] = qqq_returns - spy_returns
            
            # Index correlation (measure of market cohesion)
            spy_returns_series = spy_data['Close'].pct_change().tail(63).dropna()
            iwm_returns_series = iwm_data['Close'].pct_change().tail(63).dropna()
            
            # Align series for correlation calculation
            common_index = spy_returns_series.index.intersection(iwm_returns_series.index)
            if len(common_index) > 20:
                correlation = spy_returns_series.loc[common_index].corr(iwm_returns_series.loc[common_index])
                breadth_metrics['spy_iwm_correlation'] = correlation if not np.isnan(correlation) else 0.5
            else:
                breadth_metrics['spy_iwm_correlation'] = 0.5
                
            # Market leadership (which index is outperforming)
            if spy_returns > qqq_returns and spy_returns > iwm_returns:
                breadth_metrics['market_leader'] = 1.0  # SPY leading
            elif qqq_returns > iwm_returns:
                breadth_metrics['market_leader'] = 0.5  # QQQ leading
            else:
                breadth_metrics['market_leader'] = 0.0  # IWM leading
                
            self.logger.debug(f"Breadth analysis: SPY={spy_returns:.1%}, IWM={iwm_returns:.1%}")
            return breadth_metrics
            
        except Exception as e:
            self.logger.warning(f"Error in breadth analysis: {e}")
            return {}
    
    def _analyze_sector_rotation(self) -> Dict[str, float]:
        """Analyze sector rotation signals"""
        sector_signals = {}
        
        try:
            # Fetch recent sector ETF data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            sector_performance = {}
            
            for etf, sector_name in self.sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if not data.empty and len(data) > 20:
                        # Calculate momentum scores
                        returns_20d = (data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21]
                        returns_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
                        
                        # Volume-weighted momentum
                        avg_volume = data['Volume'].tail(20).mean()
                        recent_volume = data['Volume'].tail(5).mean()
                        volume_factor = min(recent_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
                        
                        momentum_score = (returns_20d * 0.7 + returns_5d * 0.3) * volume_factor
                        sector_performance[sector_name] = momentum_score
                        
                except Exception as e:
                    self.logger.debug(f"Error fetching {etf} data: {e}")
            
            # Rank sectors and create rotation signals
            if sector_performance:
                sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
                
                # Create signals (1.0 = strong buy, 0.0 = neutral, -1.0 = strong sell)
                for i, (sector, performance) in enumerate(sorted_sectors):
                    rank_percentile = 1 - (i / len(sorted_sectors))
                    
                    if rank_percentile > 0.8 and performance > 0.02:  # Top 20% with >2% momentum
                        sector_signals[sector] = min(performance * 5, 1.0)
                    elif rank_percentile < 0.2 and performance < -0.02:  # Bottom 20% with <-2% momentum  
                        sector_signals[sector] = max(performance * 5, -1.0)
                    else:
                        sector_signals[sector] = performance * 2  # Neutral weighting
                        
            self.logger.debug(f"Sector rotation analysis: {len(sector_signals)} sectors analyzed")
            return sector_signals
            
        except Exception as e:
            self.logger.warning(f"Error in sector rotation analysis: {e}")
            return {}
    
    def _determine_regime(self, trend_analysis: Dict, vix_analysis: Dict, 
                         breadth_analysis: Dict) -> Tuple[MarketRegime, float]:
        """Determine market regime based on all indicators"""
        
        # Initialize regime scores
        regime_scores = {
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.RANGE: 0.0,
            MarketRegime.TRANSITION: 0.0
        }
        
        # Trend-based scoring
        if trend_analysis:
            # Price above moving averages = bullish
            if trend_analysis.get('price_vs_ma20', 0) > 0.02:
                regime_scores[MarketRegime.BULL] += 1.0
            elif trend_analysis.get('price_vs_ma20', 0) < -0.02:
                regime_scores[MarketRegime.BEAR] += 1.0
            else:
                regime_scores[MarketRegime.RANGE] += 0.5
                
            if trend_analysis.get('price_vs_ma200', 0) > 0.05:
                regime_scores[MarketRegime.BULL] += 1.5
            elif trend_analysis.get('price_vs_ma200', 0) < -0.05:
                regime_scores[MarketRegime.BEAR] += 1.5
                
            # Moving average alignment
            if trend_analysis.get('ma20_vs_ma50', 0) > 0 and trend_analysis.get('ma50_vs_ma200', 0) > 0:
                regime_scores[MarketRegime.BULL] += 1.0
            elif trend_analysis.get('ma20_vs_ma50', 0) < 0 and trend_analysis.get('ma50_vs_ma200', 0) < 0:
                regime_scores[MarketRegime.BEAR] += 1.0
            else:
                regime_scores[MarketRegime.TRANSITION] += 1.0
                
            # Momentum consistency
            momentum_20d = trend_analysis.get('momentum_20d', 0)
            if momentum_20d > 0.05:
                regime_scores[MarketRegime.BULL] += 0.5
            elif momentum_20d < -0.05:
                regime_scores[MarketRegime.BEAR] += 0.5
            else:
                regime_scores[MarketRegime.RANGE] += 0.5
        
        # VIX-based scoring
        if vix_analysis:
            current_vix = vix_analysis.get('current_vix', 20)
            fear_level = vix_analysis.get('fear_level', 0)
            
            if current_vix > 25:  # High fear
                regime_scores[MarketRegime.BEAR] += 1.0
                regime_scores[MarketRegime.TRANSITION] += 0.5
            elif current_vix < 15:  # Low fear/complacency  
                regime_scores[MarketRegime.BULL] += 0.5
                regime_scores[MarketRegime.RANGE] += 0.5
            else:
                regime_scores[MarketRegime.RANGE] += 0.5
        
        # Breadth-based scoring
        if breadth_analysis:
            iwm_spy_div = breadth_analysis.get('iwm_spy_divergence', 0)
            correlation = breadth_analysis.get('spy_iwm_correlation', 0.5)
            
            # Small cap outperformance = risk-on = bullish
            if iwm_spy_div > 0.02:
                regime_scores[MarketRegime.BULL] += 0.5
            elif iwm_spy_div < -0.02:
                regime_scores[MarketRegime.BEAR] += 0.5
                
            # High correlation = market cohesion
            if correlation > 0.8:
                # Determine if cohesive up or down
                spy_return = breadth_analysis.get('spy_20d_return', 0)
                if spy_return > 0.02:
                    regime_scores[MarketRegime.BULL] += 0.5
                elif spy_return < -0.02:
                    regime_scores[MarketRegime.BEAR] += 0.5
            elif correlation < 0.4:  # Low correlation = transition
                regime_scores[MarketRegime.TRANSITION] += 1.0
        
        # Find winning regime
        max_score = max(regime_scores.values())
        if max_score == 0:
            return MarketRegime.UNKNOWN, 0.0
            
        winning_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
        
        # Calculate confidence based on score separation
        total_score = sum(regime_scores.values())
        confidence = (max_score / total_score) if total_score > 0 else 0.0
        
        # Adjust confidence based on data quality
        data_quality_factor = 1.0
        if not trend_analysis:
            data_quality_factor *= 0.7
        if not vix_analysis:
            data_quality_factor *= 0.8
        if not breadth_analysis:
            data_quality_factor *= 0.9
            
        confidence *= data_quality_factor
        
        return winning_regime, min(confidence, 1.0)
    
    def _calculate_trend_strength(self, trend_analysis: Dict, vix_analysis: Dict) -> TrendStrength:
        """Calculate trend strength based on multiple factors"""
        
        if not trend_analysis:
            return TrendStrength.WEAK
            
        strength_score = 0.0
        
        # Price momentum strength
        momentum_20d = abs(trend_analysis.get('momentum_20d', 0))
        if momentum_20d > 0.10:
            strength_score += 2.0
        elif momentum_20d > 0.05:
            strength_score += 1.0
        elif momentum_20d > 0.02:
            strength_score += 0.5
            
        # Moving average alignment strength
        ma_alignment = (trend_analysis.get('ma20_vs_ma50', 0) > 0) == (trend_analysis.get('ma50_vs_ma200', 0) > 0)
        if ma_alignment:
            strength_score += 1.0
            
        # Trend consistency
        consistency = trend_analysis.get('consistency_20d', 0.5)
        if consistency > 0.8:
            strength_score += 1.0
        elif consistency > 0.6:
            strength_score += 0.5
            
        # VIX confirmation
        if vix_analysis:
            vix_level = vix_analysis.get('current_vix', 20)
            if vix_level < 15:  # Low volatility = strong trend continuation
                strength_score += 0.5
            elif vix_level > 30:  # High volatility = weak trend
                strength_score -= 1.0
                
        # Determine trend strength category
        if strength_score >= 3.5:
            return TrendStrength.VERY_STRONG
        elif strength_score >= 2.5:
            return TrendStrength.STRONG
        elif strength_score >= 1.5:
            return TrendStrength.MODERATE
        elif strength_score >= 0.5:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK
    
    def _determine_volatility_regime(self, vix_analysis: Dict) -> str:
        """Determine volatility regime"""
        if not vix_analysis:
            return "normal"
            
        current_vix = vix_analysis.get('current_vix', 20)
        vix_percentile = vix_analysis.get('vix_percentile_252d', 0.5)
        
        if current_vix > 35 or vix_percentile > 0.9:
            return "extreme"
        elif current_vix > 25 or vix_percentile > 0.75:
            return "high"
        elif current_vix < 12 or vix_percentile < 0.2:
            return "low"
        else:
            return "normal"
    
    def _calculate_cash_allocation(self, regime: MarketRegime, volatility_regime: str, 
                                 confidence: float) -> float:
        """Calculate recommended cash allocation"""
        
        # Base allocation from regime
        base_range = self.cash_allocation_ranges.get(regime, (0.40, 0.60))
        base_allocation = (base_range[0] + base_range[1]) / 2
        
        # Adjust for volatility
        volatility_adjustments = {
            "low": -0.05,
            "normal": 0.0,
            "high": 0.10,
            "extreme": 0.20
        }
        
        volatility_adj = volatility_adjustments.get(volatility_regime, 0.0)
        
        # Adjust for confidence (lower confidence = more cash)
        confidence_adj = (1.0 - confidence) * 0.15
        
        # Calculate final allocation
        final_allocation = base_allocation + volatility_adj + confidence_adj
        
        # Ensure within reasonable bounds
        return max(0.15, min(final_allocation, 0.80))
    
    def _estimate_regime_duration(self, trend_analysis: Dict) -> int:
        """Estimate how long current regime might last"""
        if not trend_analysis:
            return 30  # Default 30 days
            
        # Base duration on trend momentum and consistency
        momentum = abs(trend_analysis.get('momentum_63d', 0))
        consistency = trend_analysis.get('consistency_20d', 0.5)
        
        # Strong, consistent trends last longer
        if momentum > 0.15 and consistency > 0.8:
            return 90  # ~3 months
        elif momentum > 0.08 and consistency > 0.6:
            return 60  # ~2 months
        elif momentum > 0.03:
            return 45  # ~1.5 months
        else:
            return 20  # ~3 weeks
    
    def _calculate_transition_probabilities(self, trend_analysis: Dict, 
                                          vix_analysis: Dict, breadth_analysis: Dict) -> Dict[str, float]:
        """Calculate probabilities of transitioning to different regimes"""
        
        # Base probabilities (could be trained from historical data)
        base_probs = {
            'bull': 0.25,
            'bear': 0.25,
            'range': 0.25,
            'transition': 0.25
        }
        
        # Adjust based on current indicators
        if trend_analysis:
            momentum_20d = trend_analysis.get('momentum_20d', 0)
            if momentum_20d > 0.05:
                base_probs['bull'] += 0.2
                base_probs['bear'] -= 0.1
            elif momentum_20d < -0.05:
                base_probs['bear'] += 0.2
                base_probs['bull'] -= 0.1
        
        if vix_analysis:
            vix_spike = vix_analysis.get('vix_spike', 0)
            if vix_spike > 0.5:  # VIX spike = potential regime change
                base_probs['transition'] += 0.3
                base_probs['bear'] += 0.2
                base_probs['bull'] -= 0.2
                base_probs['range'] -= 0.3
        
        # Normalize probabilities
        total_prob = sum(base_probs.values())
        return {k: v/total_prob for k, v in base_probs.items()}
    
    def _create_unknown_analysis(self) -> RegimeAnalysis:
        """Create analysis for unknown regime"""
        return RegimeAnalysis(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            trend_strength=TrendStrength.WEAK,
            volatility_regime="normal",
            cash_allocation_recommendation=0.50,
            sector_rotation_signal={},
            vix_analysis={},
            breadth_indicators={},
            regime_duration_days=30,
            next_regime_probability={'bull': 0.25, 'bear': 0.25, 'range': 0.25, 'transition': 0.25}
        )
    
    def get_regime_summary(self, analysis: RegimeAnalysis) -> Dict:
        """Get human-readable regime summary"""
        return {
            'regime': analysis.regime.value.title(),
            'confidence': f"{analysis.confidence:.1%}",
            'trend_strength': analysis.trend_strength.value.replace('_', ' ').title(),
            'volatility': analysis.volatility_regime.title(),
            'recommended_cash_allocation': f"{analysis.cash_allocation_recommendation:.1%}",
            'estimated_duration_days': analysis.regime_duration_days,
            'key_vix_level': analysis.vix_analysis.get('current_vix', 'N/A'),
            'fear_greed_indicator': 'Fear' if analysis.vix_analysis.get('fear_level', 0) > 0.3 else 'Greed' if analysis.vix_analysis.get('complacency_level', 0) > 0.3 else 'Neutral',
            'top_sectors': [sector for sector, signal in analysis.sector_rotation_signal.items() if signal > 0.3][:3],
            'avoid_sectors': [sector for sector, signal in analysis.sector_rotation_signal.items() if signal < -0.3][:3]
        }