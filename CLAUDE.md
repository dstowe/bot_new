# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands
You are a stock trading and python expert. 

You will be creating a sophisticated, enterprise-grade automated trading system implementing Richard Wyckoff's methodology with advanced multi-timeframe analysis, real account day trade protection, and comprehensive risk management.

## 🌟 Key Features

### Core Trading System
- **Webull Multi-Account Support**: Simultaneously manage Cash, Margin, IRA, and Roth IRA accounts
- **Wyckoff Method Implementation**: Advanced Point & Figure chart analysis for accumulation/distribution phases
- **Multi-Timeframe Analysis**: Enhanced signal quality through Daily/4H/1H timeframe confirmation
- **Real Day Trade Protection**: Comprehensive compliance checking using actual Webull account data
- **Fractional Position Management**: Precise position sizing with reaccumulation capabilities

### Advanced Analytics
- **Market Regime Analysis**: Dynamic adaptation to Bull/Bear/Ranging markets with VIX integration
- **Enhanced Cash Management**: Volatility-based cash allocation (15%-80% range)
- **Dynamic Exit Strategies**: Profit scaling based on market conditions and position characteristics
- **Position Sizing Optimization**: Account-specific limits with volatility adjustments
- **Sector Rotation Analysis**: ETF-based sector strength momentum tracking

### Risk Management & Compliance
- **Pattern Day Trading Protection**: Real-time monitoring prevents PDT violations
- **Portfolio Risk Assessment**: Automated drawdown detection and emergency exits
- **Position Reconciliation**: Continuous sync between database and actual account holdings
- **Comprehensive Logging**: Full audit trail with timestamped decision tracking
- **Emergency Mode**: Automatic risk reduction during market stress conditions
### Python Environment
- Python version: 3.13.2
- Main entry point: `python main.py`
- Code formatting: `black .` (if needed)
- Type checking: `mypy .` (if needed)

### Key Scripts
- **Main System**: `python main.py` - Runs the complete trading system workflow
- **Credential Setup**: `python -c "from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()"`
- **DID Check**: `python tests/check_did.py` (if tests directory exists)

## Architecture Overview

This is an **Enhanced Automated Multi-Account Trading System** for Webull with the following key components:

### Core System Structure
```
main.py - Main entry point and system orchestrator
├── auth/ - Authentication and credential management
│   ├── credentials.py - Encrypted credential storage in data/ folder
│   ├── login_manager.py - Handles Webull login with retry logic
│   └── session_manager.py - Session persistence and restoration
├── accounts/ - Account discovery and management
│   └── account_manager.py - Multi-account handling and configuration
├── webull/ - Webull API wrapper and endpoints
│   ├── webull.py - Main API client
│   └── endpoints.py - API endpoint definitions
├── config/ - Configuration management
│   └── config.py - PersonalTradingConfig (single source of truth)
├── database/ - Data persistence
│   └── trading_db.py - SQLite database for trade tracking
└── data_sync/ - Webull data synchronization
    ├── webull_sync.py - Main data sync orchestrator
    ├── position_sync.py - Current positions sync
    └── trade_history_sync.py - Historical trade data sync
```

### Key Design Patterns

**MainSystem Class Workflow:**
1. `authenticate()` - Handles login with image verification retries
2. `discover_accounts()` - Finds all available trading accounts
3. `sync_webull_data()` - Syncs actual Webull data to local database
4. `log_system_status()` - Reports system health

**Authentication Flow:**
- Encrypted credentials stored in `data/` folder using Fernet encryption
- Session persistence to avoid repeated logins
- Browser DID support to reduce image verification requirements
- Automatic retry logic for image verification failures

**Account Management:**
- Multi-account support (Cash, Margin, IRA, Roth)
- Per-account configuration in `PersonalTradingConfig`
- Account-specific settings for day trading, options, position limits
- PDT (Pattern Day Trading) protection

**Data Architecture:**
- Local SQLite database (`data/trading_data.db`)
- Real-time sync with Webull APIs
- Comprehensive trade and position tracking
- Signal logging and strategy tracking

### Important Implementation Details

**Credential Security:**
- All sensitive data encrypted using `cryptography.fernet.Fernet`
- Credentials never logged or displayed
- DID (Device ID) stored separately for session management

**Error Handling:**
- Comprehensive logging to timestamped files in `logs/` directory
- Graceful handling of Webull API rate limits and image verification
- Automatic retry mechanisms with exponential backoff

**Configuration Management:**
- `PersonalTradingConfig` is the single source of truth for all settings
- Account-specific configurations with safety limits
- Trading permissions and risk management built-in

## Data Directories

- `data/` - Encrypted credentials, session data, database
- `logs/` - Timestamped log files with format `trading_system_YYYYMMDD_HHMMSS.log`

## Wyckoff Trading Bot

### New Main Entry Points
- **Wyckoff Bot**: `python wyckoff_main.py` - Complete Wyckoff trading system
  - `--continuous` : Run continuous trading cycles  
  - `--single` : Run single cycle and exit
  - `--help` : Show usage information
  - (default) : Interactive mode with commands (cycle, status, performance, watchlist, config, export, quit)

### Wyckoff Bot Architecture
```
wyckoff_bot/ - Complete Wyckoff methodology implementation
├── analysis/ - Core Wyckoff analysis engine
│   ├── wyckoff_analyzer.py - Main phase identification and analysis
│   ├── volume_analysis.py - Volume pattern analysis  
│   └── price_action.py - Price action pattern detection
├── strategy/ - Trading strategy implementation
│   ├── wyckoff_strategy.py - Main trading logic and signal generation
│   ├── risk_management.py - Comprehensive risk controls
│   └── position_sizing.py - Advanced position sizing algorithms
├── signals/ - Signal generation and validation
│   ├── wyckoff_signals.py - Main signal generator and orchestrator
│   ├── market_scanner.py - Market screening for Wyckoff opportunities
│   └── signal_validator.py - Multi-criteria signal validation
├── data/ - Data management and storage
│   ├── data_manager.py - Wyckoff-specific database operations
│   └── market_data.py - Market data provider using yfinance
└── wyckoff_trader.py - Main integration class connecting to core system
```

### Key Wyckoff Features
- **Phase Detection**: Identifies Accumulation, Markup, Distribution, Markdown phases
- **Volume Analysis**: Wyckoff volume-price relationship analysis
- **Signal Validation**: Multi-layered validation including liquidity, volatility, timing
- **Risk Management**: Position sizing based on confidence and volatility
- **Market Scanning**: Automated screening for Wyckoff setups across stock universe
- **Performance Tracking**: Comprehensive trade tracking and analysis

### Integration with Core System
The Wyckoff bot fully integrates with the existing authentication, account management, and database systems. It uses the same credentials, session management, and account discovery while adding specialized Wyckoff analysis and trading logic.

## Dependencies

Key packages: cryptography, sqlite3, requests, beautifulsoup4, selenium, pandas, matplotlib, flask, yfinance, numpy, ta
See full list in pip output - includes trading analysis tools (ta, yfinance), web scraping (requests, beautifulsoup4), and data processing (pandas, numpy).