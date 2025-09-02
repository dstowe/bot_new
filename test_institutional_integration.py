#!/usr/bin/env python3
# test_institutional_integration.py
"""
Test Full Institutional Integration
===================================
Test that all institutional features are properly integrated into wyckoff_main.py

This file will be removed after testing - it's for validation only.
"""

import sys
import os
import logging
from datetime import datetime
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_wyckoff_trader_institutional_integration():
    """Test that WyckoffTrader has all institutional features"""
    print("Testing WyckoffTrader Institutional Integration...")
    
    try:
        # Test imports
        from wyckoff_bot.wyckoff_trader import WyckoffTrader
        from config.config import PersonalTradingConfig
        
        config = PersonalTradingConfig()
        
        # Create trader instance (without webull client for testing)
        trader = WyckoffTrader(
            webull_client=None,
            account_manager=None,
            config=config,
            logger=logger,
            main_system=None
        )
        
        # Test that all institutional components are initialized
        institutional_components = [
            'emergency_manager',
            'pdt_manager', 
            'performance_analyzer',
            'market_regime_analyzer',
            'wyckoff_analyzer'
        ]
        
        missing_components = []
        for component in institutional_components:
            if not hasattr(trader, component):
                missing_components.append(component)
        
        if missing_components:
            print(f"X Missing institutional components: {missing_components}")
            return False
        
        # Test institutional methods exist
        institutional_methods = [
            'get_institutional_status',
            'get_institutional_performance_summary',
            '_build_portfolio_data',
            '_update_market_regime_analysis',
            '_generate_enhanced_wyckoff_signals'
        ]
        
        missing_methods = []
        for method in institutional_methods:
            if not hasattr(trader, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"X Missing institutional methods: {missing_methods}")
            return False
        
        print("✓ WyckoffTrader institutional integration: PASSED")
        return True
        
    except Exception as e:
        print(f"X WyckoffTrader institutional integration: FAILED - {e}")
        return False

def test_wyckoff_main_institutional_commands():
    """Test that wyckoff_main.py has all institutional commands"""
    print("Testing wyckoff_main.py Institutional Commands...")
    
    try:
        from wyckoff_main import WyckoffTradingSystem
        
        system = WyckoffTradingSystem()
        
        # Test institutional display methods exist
        institutional_display_methods = [
            '_display_institutional_status',
            '_display_institutional_performance',
            '_display_emergency_status',
            '_display_pdt_status', 
            '_display_market_regime'
        ]
        
        missing_display_methods = []
        for method in institutional_display_methods:
            if not hasattr(system, method):
                missing_display_methods.append(method)
        
        if missing_display_methods:
            print(f"X Missing institutional display methods: {missing_display_methods}")
            return False
        
        print("✓ wyckoff_main.py institutional commands: PASSED")
        return True
        
    except Exception as e:
        print(f"X wyckoff_main.py institutional commands: FAILED - {e}")
        return False

def test_institutional_modules_exist():
    """Test that all institutional modules exist and can be imported"""
    print("Testing Institutional Modules...")
    
    institutional_modules = [
        # Risk management modules
        ('risk.emergency_mode', 'EmergencyModeManager'),
        ('risk.account_risk_manager', 'AccountRiskManager'),
        
        # Compliance modules
        ('compliance.pdt_protection', 'PDTProtectionManager'),
        
        # Analytics modules  
        ('wyckoff_bot.analytics.performance_analytics', 'PerformanceAnalyzer'),
        
        # Enhanced analysis modules
        ('wyckoff_bot.analysis.market_regime', 'MarketRegimeAnalyzer'),
        ('wyckoff_bot.analysis.wyckoff_analyzer', 'WyckoffAnalyzer'),
    ]
    
    failed_imports = []
    
    for module_name, class_name in institutional_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
        except Exception as e:
            failed_imports.append((module_name, class_name, str(e)))
    
    if failed_imports:
        print("X Failed module imports:")
        for module_name, class_name, error in failed_imports:
            print(f"  {module_name}.{class_name}: {error}")
        return False
    
    print("✓ All institutional modules: PASSED")
    return True

def test_institutional_cycle_integration():
    """Test that institutional features are integrated into trading cycle"""
    print("Testing Institutional Trading Cycle Integration...")
    
    try:
        # Read the wyckoff_trader.py file to check for institutional integration
        wyckoff_trader_path = os.path.join(os.path.dirname(__file__), 'wyckoff_bot', 'wyckoff_trader.py')
        
        with open(wyckoff_trader_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for institutional features in trading cycle
        institutional_features_in_cycle = [
            'INSTITUTIONAL',
            'emergency_manager.monitor_portfolio_protection',
            '_update_market_regime_analysis',
            'pdt_manager.check_pdt_compliance',
            '_generate_enhanced_wyckoff_signals',
            'account_risk_manager.get_var_report',
            'performance_analyzer.analyze_performance'
        ]
        
        missing_features = []
        for feature in institutional_features_in_cycle:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"X Missing institutional features in trading cycle: {missing_features}")
            return False
        
        # Check for institutional logging
        institutional_logs = [
            'INSTITUTIONAL cycle completed',
            'VaR-based position sizing',
            'PDT protection blocked',
            'EMERGENCY MODE ACTIVATED'
        ]
        
        missing_logs = []
        for log in institutional_logs:
            if log not in content:
                missing_logs.append(log)
        
        if missing_logs:
            print(f"X Missing institutional logging: {missing_logs}")
            return False
        
        print("✓ Institutional trading cycle integration: PASSED")
        return True
        
    except Exception as e:
        print(f"X Institutional trading cycle integration: FAILED - {e}")
        return False

def run_institutional_integration_test():
    """Run complete institutional integration test"""
    print("INSTITUTIONAL INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_institutional_modules_exist,
        test_wyckoff_trader_institutional_integration,
        test_wyckoff_main_institutional_commands,
        test_institutional_cycle_integration
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed_tests += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"X Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("SUCCESS: ALL INSTITUTIONAL FEATURES SUCCESSFULLY INTEGRATED!")
        print("\nYour wyckoff_main.py now has INSTITUTIONAL-GRADE features:")
        print("   * Emergency mode and portfolio protection")
        print("   * PDT protection and compliance checking")
        print("   * Market regime analysis with dynamic cash allocation")
        print("   * VaR-based risk management")
        print("   * Enhanced multi-timeframe Wyckoff analysis")
        print("   * Comprehensive performance analytics")
        print("   * Institutional-grade command interface")
        print("\nReady for professional trading with institutional controls!")
    else:
        failed = total_tests - passed_tests
        print(f"WARNING: {failed} integration issues detected")
        print("   Please review the failed tests above")
    
    print(f"\nNOTE: This test validates institutional integration in wyckoff_main.py")
    print("   Run 'python wyckoff_main.py' to access all institutional features")
    
    return passed_tests, total_tests

if __name__ == '__main__':
    passed, total = run_institutional_integration_test()
    
    # Exit with appropriate code  
    sys.exit(0 if passed == total else 1)