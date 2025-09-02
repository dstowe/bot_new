# config/config.py - COMPLETE VERSION WITH ALL METHODS
"""
Personal Trading Configuration - SINGLE SOURCE OF TRUTH
This is the ONLY configuration class - completely self-contained
NO inheritance or dependencies on any other config files
ALL configuration parameters are defined here
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple

class PersonalTradingConfig:
    """
    STANDALONE COMPLETE TRADING CONFIGURATION
    This class contains ALL configuration parameters with no dependencies
    This is the ONLY configuration source for the entire trading system
    """

    # =================================================================
    # CORE TRADING PARAMETERS - ALL SELF-CONTAINED
    # =================================================================

    # Database
    DATABASE_PATH = "data/trading_data.db"
    
    # Account configurations
    ACCOUNT_CONFIGURATIONS = {
        'CASH': {
            'enabled': True,
            'day_trading_enabled': True,
            'options_enabled': True,
            'max_position_size': 0.25,  # 25% of account
            'min_trade_amount': 6.00,
            'max_trade_amount': 1000,
            'pdt_protection': True
        },
        'MARGIN': {
            'enabled': True,
            'day_trading_enabled': True,
            'options_enabled': True,
            'max_position_size': 0.25,  # 20% of account
            'min_trade_amount': 6.00,
            'max_trade_amount': 1000,
            'pdt_protection': True,
            'min_account_value_for_pdt': 25000
        },
        'IRA': {
            'enabled': False,  # Disabled by default
            'day_trading_enabled': False,
            'options_enabled': False,
            'max_position_size': 0.15,  # 15% of account
            'min_trade_amount': 6,
            'max_trade_amount': 1000,
            'pdt_protection': True
        },
        'ROTH': {
            'enabled': False,  # Disabled by default
            'day_trading_enabled': False,
            'options_enabled': False,
            'max_position_size': 0.15,  # 15% of account
            'min_trade_amount': 6,
            'max_trade_amount': 1000,
            'pdt_protection': True
        }
    }
    
    # Fractional Trading Parameters
    FRACTIONAL_TRADING_ENABLED = True
    MIN_FRACTIONAL_TRADE_AMOUNT = 6.00
    SMALL_ACCOUNT_THRESHOLD = 500.00  # Accounts below this prefer fractional
    VERY_SMALL_ACCOUNT_THRESHOLD = 200.00  # Accounts below this always use fractional
    FRACTIONAL_FULL_SHARE_MULTIPLE = 2.0  # Need 2x share price to consider full shares
    
    # Live Trading Safety
    LIVE_TRADING_ENABLED = True  # SET TO True TO ENABLE REAL TRADES
    REQUIRE_TRADE_CONFIRMATION = False  # Require user confirmation before placing trades
    
    # =================================================================
    # RISK MANAGEMENT PARAMETERS
    # =================================================================
    
    # Daily Risk Limits
    MAX_DAILY_LOSS = 500.0  # Maximum daily loss before system stops trading
    MAX_DAILY_LOSS_PERCENT = 0.02  # 2% of account value max daily loss
    
    # Portfolio Risk Limits
    MAX_PORTFOLIO_RISK = 0.10  # Maximum 10% of portfolio at risk at any time
    MAX_CONCURRENT_POSITIONS = 10  # Maximum number of open positions
    MAX_DRAWDOWN_PERCENT = 0.15  # 15% max drawdown before emergency mode
    
    # Position Risk Controls
    MAX_SINGLE_POSITION_RISK = 0.02  # 2% max risk per position
    EMERGENCY_STOP_LOSS_PERCENT = 0.05  # 5% emergency stop loss
    
    # Correlation Risk
    MAX_SECTOR_EXPOSURE = 0.30  # Maximum 30% exposure to single sector
    MAX_CORRELATED_POSITIONS = 3  # Maximum correlated positions
    
    # Risk Monitoring
    RISK_CHECK_INTERVAL = 60  # Check risk every 60 seconds
    EMERGENCY_MODE_COOLDOWN = 3600  # 1 hour cooldown after emergency mode
    
    