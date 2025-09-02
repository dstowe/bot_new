# config/multi_timeframe_config.py
"""
Multi-Timeframe Configuration
============================
Configuration settings for the multi-timeframe Wyckoff trading system
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import timedelta

@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    interval: str  # yfinance interval (1d, 1h, 15m, etc.)
    update_frequency_hours: float  # How often to update data
    history_days: int  # How much history to maintain
    analysis_bars: int  # How many bars to use for analysis
    weight: float  # Weight in multi-timeframe analysis
    description: str  # Human readable description

@dataclass  
class MultiTimeframeConfig:
    """
    Complete multi-timeframe system configuration
    Centralized settings for all timeframe-related parameters
    """
    
    # Timeframe definitions
    timeframes: Dict[str, TimeframeConfig]
    
    # Analysis settings
    primary_timeframes: List[str]  # Main timeframes for direction
    confirmation_timeframes: List[str]  # Timeframes for entry timing
    minimum_timeframes_required: int  # Minimum TFs needed for analysis
    
    # Data management
    max_cache_age_hours: float  # Max age before data refresh
    bulk_download_max_workers: int  # Parallel download workers
    rate_limit_delay_ms: int  # Delay between API calls
    
    # Scoring and filtering
    min_multi_timeframe_score: float  # Minimum score for signals
    regime_alignment_bonus: float  # Bonus for regime alignment
    confirmation_bonus: float  # Bonus for TF confirmation
    
    # Watchlist management
    max_watchlist_size: int  # Maximum symbols in watchlist
    min_volume_daily: int  # Minimum daily volume
    update_watchlist_hours: float  # How often to update watchlist
    
    # Performance settings
    enable_caching: bool  # Enable data caching
    cleanup_old_data_days: int  # Clean up data older than X days
    
    @classmethod
    def get_default_config(cls) -> 'MultiTimeframeConfig':
        """Get default multi-timeframe configuration"""
        
        timeframes = {
            '1D': TimeframeConfig(
                interval='1d',
                update_frequency_hours=24.0,
                history_days=252,  # 1 year
                analysis_bars=100,
                weight=1.0,
                description='Daily - Primary trend'
            ),
            '4H': TimeframeConfig(
                interval='1h',  # Downloaded as 1h then resampled
                update_frequency_hours=4.0,
                history_days=60,  # 2 months
                analysis_bars=120,  # ~20 days of 4H bars
                weight=0.7,
                description='4-Hour - Intermediate trend'
            ),
            '1H': TimeframeConfig(
                interval='1h',
                update_frequency_hours=1.0,
                history_days=30,  # 1 month
                analysis_bars=168,  # ~1 week of 1H bars
                weight=0.5,
                description='1-Hour - Short-term trend'
            ),
            '15M': TimeframeConfig(
                interval='15m',
                update_frequency_hours=0.25,
                history_days=7,  # 1 week
                analysis_bars=96,  # ~1 day of 15M bars
                weight=0.3,
                description='15-Minute - Entry timing'
            )
        }
        
        return cls(
            timeframes=timeframes,
            primary_timeframes=['1D', '4H'],
            confirmation_timeframes=['1H', '15M'],
            minimum_timeframes_required=2,
            
            max_cache_age_hours=1.0,
            bulk_download_max_workers=5,
            rate_limit_delay_ms=100,
            
            min_multi_timeframe_score=30.0,
            regime_alignment_bonus=15.0,
            confirmation_bonus=20.0,
            
            max_watchlist_size=20,
            min_volume_daily=1000000,
            update_watchlist_hours=24.0,
            
            enable_caching=True,
            cleanup_old_data_days=90
        )
    
    @classmethod
    def get_fast_config(cls) -> 'MultiTimeframeConfig':
        """Get configuration optimized for speed"""
        config = cls.get_default_config()
        
        # Reduce data requirements for speed
        for tf_config in config.timeframes.values():
            tf_config.history_days = min(tf_config.history_days, 60)
            tf_config.analysis_bars = min(tf_config.analysis_bars, 50)
        
        config.bulk_download_max_workers = 8
        config.rate_limit_delay_ms = 50
        config.max_cache_age_hours = 4.0  # Longer cache for speed
        
        return config
    
    @classmethod  
    def get_conservative_config(cls) -> 'MultiTimeframeConfig':
        """Get configuration for conservative/careful analysis"""
        config = cls.get_default_config()
        
        # More data for better analysis
        config.timeframes['1D'].history_days = 504  # 2 years
        config.timeframes['4H'].history_days = 120  # 4 months
        config.timeframes['1H'].history_days = 60   # 2 months
        
        config.minimum_timeframes_required = 3
        config.min_multi_timeframe_score = 50.0
        config.bulk_download_max_workers = 3  # More conservative
        config.rate_limit_delay_ms = 200
        
        return config
    
    def get_timeframe_list(self) -> List[str]:
        """Get list of all configured timeframes"""
        return list(self.timeframes.keys())
    
    def get_weighted_timeframes(self) -> Dict[str, float]:
        """Get timeframes with their analysis weights"""
        return {tf: config.weight for tf, config in self.timeframes.items()}
    
    def get_update_frequencies(self) -> Dict[str, float]:
        """Get update frequencies for each timeframe"""
        return {tf: config.update_frequency_hours for tf, config in self.timeframes.items()}
    
    def should_update_timeframe(self, timeframe: str, last_update_hours_ago: float) -> bool:
        """Check if timeframe needs updating"""
        tf_config = self.timeframes.get(timeframe)
        if not tf_config:
            return False
            
        return last_update_hours_ago >= tf_config.update_frequency_hours
    
    def get_analysis_bars(self, timeframe: str) -> int:
        """Get number of bars to use for analysis"""
        tf_config = self.timeframes.get(timeframe)
        return tf_config.analysis_bars if tf_config else 100
    
    def is_primary_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is primary for analysis"""
        return timeframe in self.primary_timeframes
    
    def is_confirmation_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is used for confirmation"""
        return timeframe in self.confirmation_timeframes
    
    def get_cache_duration(self, timeframe: str) -> timedelta:
        """Get cache duration for a timeframe"""
        tf_config = self.timeframes.get(timeframe)
        if tf_config:
            hours = min(tf_config.update_frequency_hours, self.max_cache_age_hours)
        else:
            hours = self.max_cache_age_hours
            
        return timedelta(hours=hours)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check timeframes exist
        if not self.timeframes:
            issues.append("No timeframes configured")
            
        # Check primary timeframes are valid
        for tf in self.primary_timeframes:
            if tf not in self.timeframes:
                issues.append(f"Primary timeframe {tf} not in timeframes")
        
        # Check confirmation timeframes are valid  
        for tf in self.confirmation_timeframes:
            if tf not in self.timeframes:
                issues.append(f"Confirmation timeframe {tf} not in timeframes")
        
        # Check weights are reasonable
        for tf, config in self.timeframes.items():
            if config.weight <= 0 or config.weight > 1:
                issues.append(f"Timeframe {tf} weight {config.weight} should be 0 < weight <= 1")
        
        # Check minimum requirements
        if self.minimum_timeframes_required > len(self.timeframes):
            issues.append("Minimum timeframes required exceeds available timeframes")
            
        return issues

# Global configuration instances
DEFAULT_CONFIG = MultiTimeframeConfig.get_default_config()
FAST_CONFIG = MultiTimeframeConfig.get_fast_config()
CONSERVATIVE_CONFIG = MultiTimeframeConfig.get_conservative_config()

def get_config(config_type: str = 'default') -> MultiTimeframeConfig:
    """
    Get configuration by type
    
    Args:
        config_type: 'default', 'fast', or 'conservative'
    
    Returns:
        MultiTimeframeConfig instance
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'fast': FAST_CONFIG,
        'conservative': CONSERVATIVE_CONFIG
    }
    
    return configs.get(config_type, DEFAULT_CONFIG)