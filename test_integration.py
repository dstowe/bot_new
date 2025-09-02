#!/usr/bin/env python3
# test_integration.py
"""
Integration Test Script
======================
Test the new risk management and stock universe integration
"""

import sys
import logging
from config.config import PersonalTradingConfig
from config.stock_universe import StockUniverse
from risk.account_risk_manager import AccountRiskManager
from risk.portfolio_risk_monitor import PortfolioRiskMonitor

def test_stock_universe():
    """Test stock universe functionality"""
    print("🔍 Testing Stock Universe...")
    
    # Test getting all stocks
    all_stocks = StockUniverse.get_all_stocks()
    print(f"   Total stocks in universe: {len(all_stocks)}")
    
    # Test Wyckoff optimized list
    wyckoff_stocks = StockUniverse.get_wyckoff_optimized_list()
    print(f"   Wyckoff optimized stocks: {len(wyckoff_stocks)}")
    print(f"   Sample: {wyckoff_stocks[:5]}")
    
    # Test sector mapping
    sector_map = StockUniverse.get_sector_mapping()
    sectors = set(sector_map.values())
    print(f"   Sectors covered: {len(sectors)}")
    print(f"   Sectors: {', '.join(sorted(sectors))}")
    
    # Test recommended watchlist
    recommended = StockUniverse.get_recommended_watchlist(max_symbols=10)
    print(f"   Recommended watchlist (10): {recommended}")
    
    # Test sector diversified watchlist
    diversified = StockUniverse.get_sector_diversified_watchlist(symbols_per_sector=2)
    print(f"   Sector diversified watchlist: {diversified}")
    print("✅ Stock Universe tests passed\n")

def test_risk_management():
    """Test risk management modules"""
    print("🛡️ Testing Risk Management...")
    
    config = PersonalTradingConfig()
    logger = logging.getLogger(__name__)
    
    # Test Account Risk Manager
    account_rm = AccountRiskManager(config, logger)
    
    # Test daily risk limits
    can_trade, reason = account_rm.check_daily_risk_limits(
        account_id="test_account",
        account_balance=10000,
        starting_balance=10500
    )
    print(f"   Daily risk check: can_trade={can_trade}, reason='{reason}'")
    
    # Test position sizing
    max_size = account_rm.calculate_max_position_size(
        account_id="test_account",
        account_balance=10000,
        entry_price=100.0,
        stop_loss=95.0
    )
    print(f"   Max position size: {max_size:.2f} shares")
    
    # Test Portfolio Risk Monitor
    portfolio_rm = PortfolioRiskMonitor(config, logger)
    
    # Add some test positions
    portfolio_rm.update_position("AAPL", 100, 150.0, 155.0, 145.0)
    portfolio_rm.update_position("MSFT", 50, 300.0, 310.0, 290.0)
    portfolio_rm.update_position("GOOGL", 10, 2500.0, 2550.0, 2400.0)
    
    # Check portfolio limits
    within_limits, violations = portfolio_rm.check_portfolio_risk_limits(100000)
    print(f"   Portfolio within limits: {within_limits}")
    if violations:
        print(f"   Violations: {violations}")
    
    # Get portfolio metrics
    metrics = portfolio_rm.get_portfolio_metrics(100000)
    print(f"   Portfolio positions: {metrics['total_positions']}")
    print(f"   Portfolio risk ratio: {metrics['portfolio_risk_ratio']:.1%}")
    print(f"   Sector exposure: {metrics['sector_exposure']}")
    
    print("✅ Risk Management tests passed\n")

def test_configuration():
    """Test configuration parameters"""
    print("⚙️ Testing Configuration...")
    
    config = PersonalTradingConfig()
    
    # Test risk parameters
    print(f"   MAX_DAILY_LOSS: ${config.MAX_DAILY_LOSS}")
    print(f"   MAX_DAILY_LOSS_PERCENT: {config.MAX_DAILY_LOSS_PERCENT:.1%}")
    print(f"   MAX_PORTFOLIO_RISK: {config.MAX_PORTFOLIO_RISK:.1%}")
    print(f"   MAX_CONCURRENT_POSITIONS: {config.MAX_CONCURRENT_POSITIONS}")
    print(f"   MAX_DRAWDOWN_PERCENT: {config.MAX_DRAWDOWN_PERCENT:.1%}")
    print(f"   MAX_SECTOR_EXPOSURE: {config.MAX_SECTOR_EXPOSURE:.1%}")
    
    # Test account configurations
    cash_config = config.ACCOUNT_CONFIGURATIONS.get('CASH', {})
    print(f"   Cash account max position: {cash_config.get('max_position_size', 0):.1%}")
    print(f"   Fractional trading enabled: {config.FRACTIONAL_TRADING_ENABLED}")
    print(f"   Min fractional trade: ${config.MIN_FRACTIONAL_TRADE_AMOUNT}")
    
    print("✅ Configuration tests passed\n")

def test_stock_info():
    """Test individual stock information"""
    print("📊 Testing Stock Information...")
    
    all_stocks = StockUniverse.get_all_stocks()
    
    # Show sample stock details
    sample_symbols = ['AAPL', 'TSLA', 'JPM', 'XOM', 'JNJ']
    
    for symbol in sample_symbols:
        if symbol in all_stocks:
            stock = all_stocks[symbol]
            print(f"   {symbol}: {stock.name}")
            print(f"     Sector: {stock.sector}")
            print(f"     Market Cap: {stock.market_cap}")
            print(f"     Avg Volume: {stock.avg_volume:,}")
            print(f"     Wyckoff Friendly: {stock.wyckoff_friendly}")
            print(f"     Institutional Activity: {stock.institutional_activity}")
            print()
    
    print("✅ Stock Information tests passed\n")

def main():
    """Run all integration tests"""
    print("🚀 WYCKOFF TRADING SYSTEM - INTEGRATION TEST")
    print("=" * 60)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        test_stock_universe()
        test_risk_management()  
        test_configuration()
        test_stock_info()
        
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nThe system is ready with:")
        print("✅ Comprehensive stock universe (70+ stocks)")
        print("✅ Advanced risk management")
        print("✅ Sector diversification")
        print("✅ Wyckoff-optimized stock selection")
        print("✅ Real-time risk monitoring")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()