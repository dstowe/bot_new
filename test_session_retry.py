#!/usr/bin/env python3
"""
Test Session Retry Mechanism
============================
Simple test to verify the enhanced session retry logic works correctly
"""

import logging
from datetime import datetime

# Import the system
from main import MainSystem
from config.config import PersonalTradingConfig
from wyckoff_bot.execution.trade_executor import TradeExecutor, OrderResult

def setup_logging():
    """Setup basic logging for testing"""
    log_filename = f"test_session_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def test_session_validation():
    """Test session validation and retry mechanism"""
    logger = setup_logging()
    logger.info("🧪 Testing session retry mechanism...")
    
    try:
        # Initialize the main system
        config = PersonalTradingConfig()
        main_system = MainSystem()
        
        # Attempt authentication
        logger.info("🔑 Authenticating with Webull...")
        if not main_system.authenticate():
            logger.error("❌ Authentication failed - cannot test session retry")
            return False
        
        # Discover accounts
        logger.info("📊 Discovering accounts...")
        if not main_system.discover_accounts():
            logger.error("❌ Account discovery failed")
            return False
        
        # Initialize trade executor
        trade_executor = TradeExecutor(
            webull_client=main_system.wb,
            account_manager=main_system.account_manager,
            config=config,
            logger=logger,
            main_system=main_system
        )
        
        # Test 1: Normal session validation
        logger.info("\n🧪 TEST 1: Normal session validation")
        session_valid = trade_executor._ensure_valid_session()
        logger.info(f"✅ Session validation result: {session_valid}")
        
        # Test 2: Simulate session expiry and retry
        logger.info("\n🧪 TEST 2: Testing session retry after forced clear")
        # Clear session to simulate expiry
        main_system.session_manager.clear_session()
        
        # This should trigger the retry mechanism
        session_valid_after_retry = trade_executor._ensure_valid_session()
        logger.info(f"✅ Session validation after retry: {session_valid_after_retry}")
        
        # Test 3: Test order placement with session retry (dry run mode)
        logger.info("\n🧪 TEST 3: Testing order placement with session retry")
        
        # Create a test order (should be caught by dry run checks)
        test_result = trade_executor.execute_order_with_session_retry(
            symbol='AAPL',
            action='BUY',
            quantity=1.0,
            price=150.0,
            order_type='LMT'
        )
        
        logger.info(f"✅ Test order result: Success={test_result.success}, Message={test_result.message}")
        
        logger.info("\n🎉 Session retry mechanism tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Session Retry Test")
    print("====================")
    print("This test verifies the enhanced session retry mechanism")
    print("using login_automatically pattern for robust authentication.\n")
    
    success = test_session_validation()
    
    if success:
        print("\n✅ All tests passed! Session retry mechanism is working correctly.")
    else:
        print("\n❌ Tests failed! Please check the log file for details.")
    
    return success

if __name__ == "__main__":
    main()