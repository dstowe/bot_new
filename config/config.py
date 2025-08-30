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
    
    