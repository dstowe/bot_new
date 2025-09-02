# 🚀 Wyckoff Trading Bot

An **enterprise-grade automated trading system** implementing Richard Wyckoff's methodology with advanced multi-timeframe analysis, real account day trade protection, and comprehensive risk management for Webull trading accounts.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Trading](https://img.shields.io/badge/trading-automated-green.svg)
![Wyckoff](https://img.shields.io/badge/strategy-wyckoff-purple.svg)
![License](https://img.shields.io/badge/license-Private-red.svg)

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up credentials (interactive)
python -c "from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()"

# 3. Run the main system
python main.py

# 4. Run Wyckoff trading bot
python wyckoff_main.py --help
```

## 🌟 Key Features

### Core Trading System
- **🏦 Webull Multi-Account Support**: Simultaneously manage Cash, Margin, IRA, and Roth IRA accounts
- **📈 Wyckoff Method Implementation**: Advanced Point & Figure chart analysis for accumulation/distribution phases
- **⏱️ Multi-Timeframe Analysis**: Enhanced signal quality through Daily/4H/1H timeframe confirmation
- **🛡️ Real Day Trade Protection**: Comprehensive compliance checking using actual Webull account data
- **⚖️ Fractional Position Management**: Precise position sizing with reaccumulation capabilities

### Advanced Analytics
- **🌍 Market Regime Analysis**: Dynamic adaptation to Bull/Bear/Ranging markets with VIX integration
- **💰 Enhanced Cash Management**: Volatility-based cash allocation (15%-80% range)
- **🎯 Dynamic Exit Strategies**: Profit scaling based on market conditions and position characteristics
- **📊 Position Sizing Optimization**: Account-specific limits with volatility adjustments
- **🔄 Sector Rotation Analysis**: ETF-based sector strength momentum tracking
- **⚡ Multi-Timeframe Data Engine**: Lightning-fast cached data across 4 timeframes (1D/4H/1H/15M)

### Risk Management & Compliance
- **⚠️ Pattern Day Trading Protection**: Real-time monitoring prevents PDT violations
- **📉 Portfolio Risk Assessment**: Automated drawdown detection and emergency exits
- **🔄 Position Reconciliation**: Continuous sync between database and actual account holdings
- **📝 Comprehensive Logging**: Full audit trail with timestamped decision tracking
- **🚨 Emergency Mode**: Automatic risk reduction during market stress conditions

## 🏗️ Architecture Overview

```
├── main.py                    # Main system entry point
├── wyckoff_main.py           # Wyckoff bot entry point
├── auth/                     # Authentication & credentials
│   ├── credentials.py        # Encrypted credential storage
│   ├── login_manager.py      # Webull login with retry logic
│   └── session_manager.py    # Session persistence
├── accounts/                 # Multi-account management
│   ├── account_manager.py    # Account discovery & configuration
│   └── account_info.py       # Account data structures
├── webull/                   # Webull API wrapper
│   ├── webull.py            # Main API client
│   └── endpoints.py         # API endpoint definitions
├── config/                   # Configuration management
│   ├── config.py            # PersonalTradingConfig (single source of truth)
│   └── stock_universe.py    # Curated stock universe (70+ stocks)
├── risk/                     # Risk management system
│   ├── account_risk_manager.py # Account-level risk controls
│   └── portfolio_risk_monitor.py # Portfolio-wide risk monitoring
├── database/                 # Data persistence
│   └── trading_db.py        # SQLite database for trade tracking
├── data_sync/                # Webull data synchronization
│   ├── webull_sync.py       # Main data sync orchestrator
│   ├── position_sync.py     # Current positions sync
│   └── trade_history_sync.py # Historical trade data sync
└── wyckoff_bot/              # Complete Wyckoff methodology
    ├── analysis/             # Core Wyckoff analysis engine
    │   ├── wyckoff_analyzer.py # Phase identification & analysis
    │   ├── volume_analysis.py  # Volume pattern analysis
    │   └── price_action.py     # Price action pattern detection
    ├── strategy/             # Trading strategy implementation
    │   ├── wyckoff_strategy.py # Main trading logic & signals
    │   ├── risk_management.py  # Comprehensive risk controls
    │   └── position_sizing.py  # Advanced position sizing algorithms
    ├── signals/              # Signal generation & validation
    │   ├── wyckoff_signals.py  # Signal generator & orchestrator
    │   ├── market_scanner.py   # Market screening for opportunities
    │   └── signal_validator.py # Multi-criteria signal validation
    ├── data/                 # Enhanced multi-timeframe data management
    │   ├── data_manager.py    # Wyckoff-specific database operations
    │   ├── multi_timeframe_data_manager.py  # Smart cached data engine
    │   └── market_data.py     # Market data provider using yfinance
    ├── execution/            # Trade execution
    │   └── trade_executor.py  # Live order placement & management
    └── wyckoff_trader.py     # Main integration class
```

## 🎯 Wyckoff Trading Methodology

This system implements Richard Wyckoff's complete market cycle methodology with a curated universe of 70+ stocks:

## 📈 Stock Universe Features

### **Optimized Stock Selection** 
- **70+ Carefully Curated Stocks**: Focus on liquid, institutional-heavy stocks
- **Sector Diversification**: Technology, Financial, Healthcare, Energy, Industrial, Consumer, Communication, Utilities
- **Wyckoff-Friendly Filtering**: Stocks with clear institutional patterns and good volume/price relationships
- **Volume Requirements**: Minimum 1M+ daily average volume for liquidity
- **Institutional Activity Focus**: Prioritizes stocks with high smart money activity

### **Advanced Watchlist Generation**
- **Recommended Watchlists**: AI-scored based on Wyckoff criteria, volume, and institutional activity
- **Sector-Diversified Lists**: Automatic diversification across sectors
- **High-Volume Filtering**: Focus on most liquid opportunities
- **Dynamic Updates**: Watchlists refresh based on market scanning results

This system implements Richard Wyckoff's complete market cycle methodology:

### 📊 Market Phases
- **Accumulation**: Smart money accumulating positions → **BUY** signals
- **Markup**: Uptrend continuation → **BUY/SCALE_IN** signals
- **Distribution**: Smart money distributing → **SCALE_OUT** signals  
- **Markdown**: Downtrend → **CLOSE_LONG** signals

### 🔍 Key Wyckoff Events Detected
- **PS** (Preliminary Support), **SC** (Selling Climax), **AR** (Automatic Rally)
- **ST** (Secondary Test), **SOS** (Sign of Strength), **LPS** (Last Point Support)
- **PSY** (Preliminary Supply), **BC** (Buying Climax), **AD** (Automatic Reaction)
- **UT** (Upthrust), **SOW** (Sign of Weakness), **LPSY** (Last Point Supply)

### ⚡ Signal Validation
- Minimum 60% confidence threshold
- Volume confirmation analysis
- Risk-reward ratio ≥ 2:1
- Multi-timeframe confirmation
- Liquidity and volatility checks

## 🚀 Usage

### Quick Integration Test
```bash
# Test all new features (recommended first run)
python test_integration.py
```

### Main System
```bash
# Run complete system initialization
python main.py

# Available operations:
# - Authentication with Webull
# - Multi-account discovery  
# - Data synchronization
# - System health logging
```

### Wyckoff Trading Bot
```bash
# Interactive mode (default)
python wyckoff_main.py

# Continuous trading
python wyckoff_main.py --continuous

# Single cycle (recommended for Task Scheduler)
python wyckoff_main.py --single

# Help
python wyckoff_main.py --help
```

## ⏰ Windows Task Scheduler Setup (Recommended)

For optimal performance, run the system every **15 minutes** during market hours using Windows Task Scheduler:

### 📋 Step-by-Step Setup

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`
   - Or search "Task Scheduler" in Start Menu

2. **Create Basic Task**
   - Click "Create Basic Task" in Actions panel
   - Name: `Wyckoff Trading Bot`
   - Description: `Automated Wyckoff trading system - runs every 15 minutes`

3. **Set Trigger**
   - When: `Daily`
   - Start date: Today
   - Recur every: `1 day`
   - Advanced Settings → Repeat task every: `15 minutes`
   - For a duration of: `8 hours` (market hours)

4. **Configure Action**
   - Action: `Start a program`
   - Program: `python.exe` or full path (e.g., `C:\Python313\python.exe`)
   - Arguments: `wyckoff_main.py --single`
   - Start in: `C:\bot_new` (your project directory)

5. **Advanced Settings**
   - Check "Run whether user is logged on or not"
   - Check "Run with highest privileges" 
   - Check "Hidden" (optional - runs in background)
   - Stop task if runs longer than: `5 minutes`

### ⏰ Market Hours Schedule
```
Market Hours: 9:30 AM - 4:00 PM ET (Monday-Friday)
Recommended: 9:30 AM - 4:15 PM ET
Frequency: Every 15 minutes
Daily Schedule: 9:30, 9:45, 10:00, 10:15... 4:00, 4:15
```

### 🎯 Why Every 15 Minutes?
- **Optimal Balance**: Captures intraday opportunities without over-trading
- **System Performance**: Sub-5-second cycles leave plenty of buffer time
- **Market Efficiency**: Aligns with 15M timeframe analysis
- **Resource Friendly**: Minimal system load with smart caching
- **Holiday Aware**: Automatically skips weekends and market holidays

### Interactive Commands
When running in interactive mode:
- `cycle` - Run single trading cycle
- `status` - Show system status  
- `performance` - Show performance summary
- `watchlist` - Show current watchlist
- `config` - Update configuration
- `export` - Export trading data
- `quit` - Exit system

## ⚙️ Configuration

The system uses `PersonalTradingConfig` as the single source of truth for all settings:

```python
class PersonalTradingConfig:
    # Database
    DATABASE_PATH = "data/trading_data.db"
    
    # Account-specific configurations
    ACCOUNT_CONFIGURATIONS = {
        'CASH': {
            'max_position_size': 0.25,  # 25% of account
            'min_trade_amount': 6.00,
            'pdt_protection': True
        },
        'MARGIN': {
            'max_position_size': 0.25,  # 25% of account  
            'min_trade_amount': 6.00,
            'pdt_protection': True
        }
        # IRA and ROTH configurations available
    }
    
    # Risk Management Parameters
    MAX_DAILY_LOSS = 500.0  # Maximum daily loss
    MAX_DAILY_LOSS_PERCENT = 0.02  # 2% daily loss limit
    MAX_PORTFOLIO_RISK = 0.10  # 10% total portfolio risk
    MAX_CONCURRENT_POSITIONS = 10  # Position limit
    MAX_DRAWDOWN_PERCENT = 0.15  # 15% drawdown emergency mode
    MAX_SECTOR_EXPOSURE = 0.30  # 30% sector concentration limit
    
    # Fractional Trading
    MIN_FRACTIONAL_TRADE_AMOUNT = 6.00
    FRACTIONAL_TRADING_ENABLED = True
    
    # Live Trading Safety
    LIVE_TRADING_ENABLED = True
```

## 📁 Data Structure

```
data/
├── trading_data.db           # SQLite database
├── trading_credentials.enc   # Encrypted credentials
├── trading_key.key          # Encryption key
└── webull_session.json      # Session data

logs/
└── trading_system_YYYYMMDD_HHMMSS.log  # Timestamped logs

exports/                      # Data exports (optional)
```

## 🔐 Security Features

- **Fernet Encryption**: All sensitive data encrypted using `cryptography.fernet`
- **No Credential Logging**: Credentials never appear in logs or stdout
- **Session Management**: Automatic session restoration to minimize login frequency
- **Device ID Support**: Reduces image verification requirements

## 🛡️ Risk Management

### Account-Level Risk Controls (`AccountRiskManager`)
- **Daily Loss Limits**: $500 absolute or 2% of account value
- **Position Sizing**: Maximum 2% risk per position
- **Emergency Mode**: Activated on 15% drawdown with 1-hour cooldown
- **Real-time Monitoring**: Continuous risk assessment

### Portfolio-Level Risk Monitoring (`PortfolioRiskMonitor`)
- **Portfolio Risk**: Maximum 10% of total value at risk
- **Position Limits**: Maximum 10 concurrent positions
- **Sector Concentration**: Maximum 30% exposure per sector
- **Correlation Control**: Maximum 3 correlated positions
- **Risk Alerts**: GREEN/YELLOW/RED alert system

### Day Trading Protection
- **PDT Compliance**: Real-time pattern day trading rule monitoring
- **Account-specific Tracking**: Separate day trade counting per account
- **Automatic Prevention**: Blocks trades that would cause PDT violations
- **Multi-account Support**: Compliance across all account types

### Emergency Controls & Circuit Breakers
- **Drawdown Protection**: Auto-halt trading on excessive losses
- **Daily Limits**: Hard stops on daily loss thresholds
- **Portfolio Risk**: Automatic position reduction suggestions
- **Market Stress Response**: Emergency liquidation protocols

## 📈 Performance Tracking

- Win/loss ratios
- Profit factor calculation
- Sharpe ratio analysis
- Maximum drawdown monitoring
- Trade execution statistics
- Signal accuracy metrics

## 🔧 Development

### Prerequisites
- Python 3.13+
- Webull account with API access
- SQLite3 support

### Setup Development Environment
```bash
# Clone and enter directory
cd bot_new

# Install requirements
pip install -r requirements.txt

# Set up credentials
python -c "from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()"

# Run tests (if available)
python tests/check_did.py
```

### Key Components

**MainSystem Class Workflow:**
1. `authenticate()` - Handles login with image verification retries
2. `discover_accounts()` - Finds all available trading accounts  
3. `sync_webull_data()` - Syncs actual Webull data to local database
4. `log_system_status()` - Reports system health

**WyckoffTrader Integration:**
- Full integration with existing authentication and account systems
- Specialized Wyckoff analysis and trading logic
- Real-time signal generation and validation
- Live order placement with comprehensive error handling

## ⚠️ Important Notes

- **Paper Trading**: Test thoroughly before live trading
- **Risk Warning**: Automated trading involves substantial risk
- **Compliance**: Ensure compliance with all applicable regulations
- **Monitoring**: Always monitor system performance and positions
- **Data Privacy**: All data stored locally with encryption

## 📞 Support

- Check logs in `logs/` directory for troubleshooting
- Review `CLAUDE.md` for detailed implementation guidance
- Ensure all dependencies are properly installed
- Verify Webull API credentials and permissions

## 📄 License

Private - For authorized use only

---

## 🎯 **Integration Summary**

✅ **Complete Risk Management System**
- Account-level daily loss limits ($500 or 2%)
- Portfolio-wide risk monitoring (10% max at risk)
- Sector concentration limits (30% max per sector)  
- Emergency drawdown protection (15% max drawdown)

✅ **Curated Stock Universe (70+ Stocks)**
- Wyckoff-optimized selection criteria
- Comprehensive sector mapping
- Institutional activity focus
- Recommended and diversified watchlists

✅ **Enterprise Integration**
- Risk managers integrated into trading cycle
- Real-time position sizing adjustments
- Automated watchlist management
- Comprehensive testing framework

**⚡ Built with modern Python practices and enterprise-grade architecture for professional algorithmic trading.**

*Disclaimer: This software is for educational and research purposes. Trading involves substantial risk. Past performance does not guarantee future results.*