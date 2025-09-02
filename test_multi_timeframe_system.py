# test_multi_timeframe_system.py
"""
Test Script for Multi-Timeframe Wyckoff System
==============================================
Demonstrates the new multi-timeframe data storage and analysis system
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from database.trading_db import TradingDatabase
from wyckoff_bot.data.data_manager import WyckoffDataManager
from wyckoff_bot.data.multi_timeframe_data_manager import MultiTimeframeDataManager
from wyckoff_bot.analysis.wyckoff_analyzer import WyckoffAnalyzer
from wyckoff_bot.analysis.market_regime import MarketRegimeAnalyzer
from wyckoff_bot.signals.market_scanner import MarketScanner
from config.multi_timeframe_config import get_config

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/test_mtf_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_database_creation():
    """Test multi-timeframe database table creation"""
    print("\n=== Testing Database Creation ===")
    
    try:
        db = TradingDatabase(db_path="data/test_mtf_trading.db")
        print("✓ Database and multi-timeframe tables created successfully")
        
        # Test basic operations
        test_symbols = ['AAPL', 'MSFT']
        watchlist_data = {
            'AAPL': {'phase': 'accumulation', 'strength': 0.8, 'timeframe_analysis': {'1D': 'accumulation', '4H': 'markup'}},
            'MSFT': {'phase': 'markup', 'strength': 0.9, 'timeframe_analysis': {'1D': 'markup', '4H': 'markup'}}
        }
        
        db.update_watchlist(test_symbols, watchlist_data)
        print("✓ Watchlist update successful")
        
        watchlist = db.get_current_watchlist()
        print(f"✓ Retrieved watchlist: {[item['symbol'] for item in watchlist]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_configuration():
    """Test multi-timeframe configuration"""
    print("\n=== Testing Configuration System ===")
    
    try:
        # Test default config
        config = get_config('default')
        print(f"✓ Default config loaded: {len(config.timeframes)} timeframes")
        
        # Test config validation
        issues = config.validate_config()
        if not issues:
            print("✓ Configuration validation passed")
        else:
            print(f"⚠ Configuration issues: {issues}")
        
        # Test different config types
        fast_config = get_config('fast')
        conservative_config = get_config('conservative')
        
        print(f"✓ Fast config: {fast_config.bulk_download_max_workers} workers")
        print(f"✓ Conservative config: min score {conservative_config.min_multi_timeframe_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_download():
    """Test multi-timeframe data download"""
    print("\n=== Testing Data Download System ===")
    
    try:
        logger = logging.getLogger("test_data")
        db = TradingDatabase(db_path="data/test_mtf_trading.db")
        mtf_manager = MultiTimeframeDataManager(db=db, logger=logger)
        
        # Test single symbol download
        test_symbol = 'AAPL'
        print(f"Downloading data for {test_symbol}...")
        
        results = mtf_manager.download_symbol_data(test_symbol, ['1D', '4H'], force_update=True)
        if results:
            print(f"✓ Downloaded {len(results)} timeframes for {test_symbol}")
            for tf, df in results.items():
                print(f"  - {tf}: {len(df)} bars")
        else:
            print("⚠ No data downloaded")
        
        # Test cached data retrieval
        cached_data = mtf_manager.get_cached_data(test_symbol, '1D', bars=50)
        if cached_data is not None:
            print(f"✓ Retrieved cached data: {len(cached_data)} bars")
        else:
            print("⚠ No cached data available")
        
        return True
        
    except Exception as e:
        print(f"✗ Data download test failed: {e}")
        return False

def test_market_regime_analysis():
    """Test market regime analysis"""
    print("\n=== Testing Market Regime Analysis ===")
    
    try:
        logger = logging.getLogger("test_regime")
        regime_analyzer = MarketRegimeAnalyzer(logger=logger)
        
        print("Analyzing current market regime...")
        regime_analysis = regime_analyzer.analyze_market_regime(days_lookback=60)
        
        if regime_analysis and 'regime_type' in regime_analysis:
            print(f"✓ Market regime: {regime_analysis['regime_type']}")
            print(f"  - Confidence: {regime_analysis.get('confidence', 0):.1%}")
            
            # Get trading context
            context = regime_analyzer.get_trading_context()
            print(f"  - Recommended cash allocation: {context.get('cash_allocation', 0):.1%}")
            print(f"  - Preferred phases: {context.get('preferred_phases', [])}")
        else:
            print("⚠ Could not determine market regime")
        
        return True
        
    except Exception as e:
        print(f"✗ Market regime test failed: {e}")
        return False

def test_wyckoff_analysis():
    """Test multi-timeframe Wyckoff analysis"""
    print("\n=== Testing Multi-Timeframe Wyckoff Analysis ===")
    
    try:
        logger = logging.getLogger("test_wyckoff")
        db = TradingDatabase(db_path="data/test_mtf_trading.db")
        data_manager = WyckoffDataManager(db_path="data/test_mtf_trading.db", logger=logger)
        wyckoff_analyzer = WyckoffAnalyzer(logger=logger)
        
        test_symbol = 'AAPL'
        print(f"Performing multi-timeframe analysis for {test_symbol}...")
        
        # Get multi-timeframe data
        mtf_data = data_manager.get_multi_timeframe_data(test_symbol, ['1D', '4H', '1H'])
        
        if mtf_data and len(mtf_data) >= 2:
            print(f"✓ Retrieved data for {len(mtf_data)} timeframes")
            
            # Perform multi-timeframe analysis
            mtf_analyses = wyckoff_analyzer.analyze_multi_timeframe(mtf_data, test_symbol)
            
            if mtf_analyses:
                print(f"✓ Analysis completed for {len(mtf_analyses)} timeframes:")
                for tf, analysis in mtf_analyses.items():
                    print(f"  - {tf}: {analysis.phase.value} (confidence: {analysis.confidence:.2f})")
                
                # Generate multi-timeframe signal
                signal = wyckoff_analyzer.get_multi_timeframe_signal(mtf_analyses)
                print(f"✓ Multi-timeframe signal: {signal['signal']} (strength: {signal['strength']:.2f})")
                
                # Check entry timing
                entry_timing = wyckoff_analyzer.get_entry_timing_signal(mtf_analyses, signal['signal'])
                print(f"  - Entry timing: {entry_timing['timing']}")
                
            else:
                print("⚠ No analysis results")
        else:
            print("⚠ Insufficient multi-timeframe data")
        
        return True
        
    except Exception as e:
        print(f"✗ Wyckoff analysis test failed: {e}")
        return False

def test_market_scanning():
    """Test enhanced market scanning"""
    print("\n=== Testing Enhanced Market Scanning ===")
    
    try:
        logger = logging.getLogger("test_scanner")
        
        # Initialize components
        data_manager = WyckoffDataManager(db_path="data/test_mtf_trading.db", logger=logger)
        wyckoff_analyzer = WyckoffAnalyzer(logger=logger)
        regime_analyzer = MarketRegimeAnalyzer(logger=logger)
        
        scanner = MarketScanner(
            data_manager=data_manager,
            wyckoff_analyzer=wyckoff_analyzer,
            market_regime_analyzer=regime_analyzer,
            logger=logger
        )
        
        # Test with a small universe
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        print(f"Scanning {len(test_symbols)} symbols...")
        
        # This will use cached data if available, otherwise download fresh
        scan_results = scanner.scan_market(test_symbols, use_cached_data=True)
        
        if scan_results:
            print(f"✓ Scan completed: {len(scan_results)} opportunities found")
            
            # Show top results
            for i, result in enumerate(scan_results[:3]):
                print(f"  {i+1}. {result.symbol}: {result.phase} (score: {result.score:.1f})")
                print(f"     - Entry timing: {result.entry_timing}")
                print(f"     - Regime aligned: {result.market_regime_alignment}")
                if result.timeframe_analysis:
                    tf_summary = ', '.join([f"{tf}:{phase}" for tf, phase in result.timeframe_analysis.items()])
                    print(f"     - Timeframes: {tf_summary}")
            
            # Create enhanced watchlist
            watchlist_info = scanner.create_enhanced_watchlist(scan_results, max_symbols=10)
            print(f"✓ Enhanced watchlist created: {len(watchlist_info['watchlist'])} symbols")
            print(f"  - Ready to enter: {len(watchlist_info['ready_to_enter'])}")
            print(f"  - Watch for entry: {len(watchlist_info['watch_for_entry'])}")
            
        else:
            print("⚠ No scan results")
        
        return True
        
    except Exception as e:
        print(f"✗ Market scanning test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Multi-Timeframe Wyckoff System Test")
    print("=" * 50)
    
    # Setup
    logger = setup_logging()
    logger.info("Starting multi-timeframe system test")
    
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run tests
    tests = [
        ("Database Creation", test_database_creation),
        ("Configuration", test_configuration),
        ("Data Download", test_data_download),
        ("Market Regime Analysis", test_market_regime_analysis),
        ("Wyckoff Analysis", test_wyckoff_analysis),
        ("Market Scanning", test_market_scanning)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-timeframe system is ready.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check logs for details.")
    
    logger.info(f"Test completed: {passed}/{total} passed")

if __name__ == "__main__":
    main()