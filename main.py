# main.py
#!/usr/bin/env python3
"""
Enhanced Automated Multi-Account Trading System - FIXED MAIN.PY
Handles authentication, account discovery, and comprehensive logging
"""

import logging
import sys
import traceback
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

# Import our modules
from webull.webull import webull
from auth.credentials import CredentialManager
from auth.login_manager import LoginManager
from auth.session_manager import SessionManager
from accounts.account_manager import AccountManager
from config.config import PersonalTradingConfig

class MainSystem:
    """
    Enhanced Automated Multi-Account Trading System - COMPLETE IMPLEMENTATION
    Handles authentication, account discovery, and logging to /logs directory
    """
    
    def __init__(self):
        # Initialize all attributes first
        self.logger = None
        self.is_logged_in = False
        self.trading_pin = None  # Store trading PIN for token refresh
        
        # Set up logging first
        self.setup_logging()
        
        # Initialize the system
        self._initialize_system()
    
    def setup_logging(self):
        """Set up comprehensive logging to /logs directory"""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Create timestamped log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = logs_dir / f"trading_system_{timestamp}.log"
            
            # Configure root logger
            logging.basicConfig(
                level=logging.INFO,  # Changed from DEBUG to INFO for cleaner output
                format='%(asctime)s - %(levelname)s - %(message)s',  # Simplified format
                datefmt='%H:%M:%S',  # Shorter time format
                handlers=[
                    logging.FileHandler(log_filename, encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)  # Also log to console
                ],
                force=True  # Force reconfiguration
            )
            
            # Create logger for this module
            self.logger = logging.getLogger(__name__)
            self.logger.info("🚀 ENHANCED MULTI-ACCOUNT TRADING SYSTEM")
            self.logger.info(f"📝 Log: {log_filename.name}")
            print()  # Add spacing for readability
            
        except Exception as e:
            print(f"❌ CRITICAL: Failed to setup logging: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("🔧 Initializing system components...")
            
            # Initialize all components
            self.config = PersonalTradingConfig()
            
            # Initialize webull with data folder support
            self.wb = webull()
            # Webull now automatically uses data folder for DID storage
            
            self.credential_manager = CredentialManager(logger=self.logger)
            self.login_manager = LoginManager(self.wb, self.credential_manager, logger=self.logger)
            self.session_manager = SessionManager(logger=self.logger)
            self.account_manager = AccountManager(self.wb, self.config, logger=self.logger)
            
            self.logger.info("✅ All components initialized")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ CRITICAL: Failed to initialize system: {e}")
            else:
                print(f"❌ CRITICAL: Failed to initialize system: {e}")
            raise
        
    def authenticate(self) -> bool:
        """Handle authentication with enhanced image verification support"""
        try:
            self.logger.info("🔐 Starting authentication process...")
            
            # Check if credentials exist
            if not self.credential_manager.credentials_exist():
                self.logger.error("❌ No encrypted credentials found!")
                self.logger.info("💡 Setup: python -c \"from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()\"")
                return False
            
            # Load credentials info for display
            cred_info = self.credential_manager.get_credential_info()
            self.logger.info(f"📋 Using credentials for: {cred_info.get('username', 'Unknown')}")
            
            if cred_info.get('has_did'):
                self.logger.info("✅ Browser DID configured (helps prevent image verification)")
            else:
                self.logger.warning("⚠️  No Browser DID - may get image verification errors")
                self.logger.info("💡 To add DID: python tests/check_did.py")
            
            # Load and set DID
            credentials = self.credential_manager.load_credentials()
            stored_did = credentials.get('did')
            
            if stored_did:
                self.wb._set_did(stored_did, data_folder='data')
                self.wb._did = stored_did
                self.logger.debug("✅ DID applied to Webull client")
            
            # Try existing session first (but don't spend too much time on it)
            self.logger.info("🔍 Checking for existing session...")
            session_loaded = self.session_manager.auto_manage_session(self.wb)
            
            if session_loaded:
                self.logger.info("📦 Session loaded, verifying with server...")
                if self.login_manager.check_login_status():
                    self.logger.info("✅ Existing session verified and active!")
                    self.is_logged_in = True
                    return True
                else:
                    self.logger.info("⚠️  Session expired on server")
                    self.session_manager.clear_session()
            
            # Perform fresh login with enhanced retry logic
            self.logger.info("🔑 Starting fresh login process...")
            self.logger.info("ℹ️  Note: If you see 'Image verification failed', this is normal")
            self.logger.info("     The system will automatically retry until it succeeds")
            
            if self.login_manager.login_automatically():
                self.logger.info("🎉 Authentication successful!")
                self.is_logged_in = True
                
                # Store trading PIN for token refresh
                credentials = self.credential_manager.load_credentials()
                self.trading_pin = credentials.get('trading_pin')
                if self.trading_pin:
                    self.wb.trade_pin = self.trading_pin  # Store in webull client too
                
                # Save successful session
                self.session_manager.save_session(self.wb)
                self.logger.info("💾 Session saved for future use")
                
                return True
            else:
                self.logger.error("❌ Authentication failed after all retries")
                
                # Provide helpful suggestions
                if not stored_did:
                    self.logger.error("💡 Consider adding a Browser DID to reduce image verification:")
                    self.logger.error("    python tests/check_did.py")
                else:
                    self.logger.error("💡 Troubleshooting suggestions:")
                    self.logger.error("    1. Wait 10 minutes and try again")
                    self.logger.error("    2. Check your credentials are correct")
                    self.logger.error("    3. Try getting a fresh Browser DID")
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Authentication error: {e}")
            self.logger.error("💡 Try running again - temporary issues often resolve")
            return False
    
    def discover_accounts(self) -> bool:
        """Discover and load all available trading accounts"""
        try:
            self.logger.info("🔍 Discovering accounts...")
            
            if not self.is_logged_in:
                self.logger.error("❌ Not authenticated")
                return False
            
            if self.account_manager.discover_accounts():
                summary = self.account_manager.get_account_summary()
                
                print()  # Add spacing
                self.logger.info("📊 ACCOUNT SUMMARY")
                self.logger.info(f"   Total Accounts: {summary['total_accounts']}")
                self.logger.info(f"   Total Value: ${summary['total_value']:,.2f}")
                self.logger.info(f"   Available Cash: ${summary['total_cash']:,.2f}")
                print()
                
                # Show each account
                for i, account in enumerate(summary['accounts'], 1):
                    status = "✅ ENABLED" if account['enabled'] else "❌ DISABLED"
                    self.logger.info(f"Account {i}: {account['account_type']} - {status}")
                    self.logger.info(f"   Balance: ${account['net_liquidation']:,.2f}")
                    self.logger.info(f"   Available: ${account['settled_funds']:,.2f}")
                    self.logger.info(f"   Positions: {account['positions_count']}")
                    if account['enabled']:
                        self.logger.info(f"   Day Trading: {'✅' if account['day_trading_enabled'] else '❌'} | Options: {'✅' if account['options_enabled'] else '❌'}")
                    print()
                
                return True
            else:
                self.logger.error("❌ Account discovery failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Account discovery error: {e}")
            return False
        
    def sync_webull_data(self, days_back: int = 30, include_positions: bool = True, 
                        include_trade_history: bool = True) -> bool:
        """
        Sync actual Webull data to local database
        
        Args:
            days_back: Number of days of trade history to sync
            include_positions: Whether to sync current positions
            include_trade_history: Whether to sync trade history
            
        Returns:
            bool: True if sync was successful
        """
        try:
            # Import the data sync module
            from data_sync import WebullDataSync
            
            # Check prerequisites
            if not self.is_logged_in:
                self.logger.error("❌ Not authenticated - cannot sync data")
                return False
            
            if not self.account_manager or not self.account_manager.accounts:
                self.logger.error("❌ No accounts discovered - cannot sync data")
                return False
            
            # Initialize data synchronization
            self.logger.info("🔄 Starting Webull data synchronization...")
            
            data_sync = WebullDataSync(
                wb=self.wb,
                account_manager=self.account_manager,
                config=self.config,
                logger=self.logger
            )
            
            # Perform the sync
            sync_results = data_sync.sync_all_data(
                days_back=days_back,
                include_positions=include_positions,
                include_trade_history=include_trade_history
            )
            
            # Log results
            if sync_results['success']:
                self.logger.info("✅ Data synchronization completed successfully!")
            else:
                self.logger.warning("⚠️  Data synchronization completed with errors")
            
            return sync_results['success']
            
        except Exception as e:
            self.logger.error(f"❌ Data synchronization error: {e}")
            return False       
    
    def log_system_status(self):
        """Log essential system status"""
        try:
            login_info = self.login_manager.get_login_info()
            self.logger.info("📊 SYSTEM STATUS")
            self.logger.info(f"   Authentication: {'✅' if login_info['is_logged_in'] else '❌'}")
            self.logger.info(f"   Accounts Loaded: {len(self.account_manager.accounts) if self.account_manager.accounts else 0}")
            enabled_count = len(self.account_manager.get_enabled_accounts()) if self.account_manager else 0
            self.logger.info(f"   Enabled for Trading: {enabled_count}")
            print()
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {e}")
    
    def run(self) -> bool:
        """Run the complete system workflow"""
        try:
            # Step 1: Authentication
            if not self.authenticate():
                self.logger.error("❌ Authentication failed")
                return False
            
            # Step 2: Account Discovery
            if not self.discover_accounts():
                self.logger.error("❌ Account discovery failed")
                return False
            
            # Step 3: Data Synchronization (NEW)
            self.logger.info("🔄 Syncing actual Webull data to database...")
            if not self.sync_webull_data(days_back=30):
                self.logger.warning("⚠️  Data sync had issues but continuing...")
            
            # Step 4: System Status
            self.log_system_status()
            
            # Success
            self.logger.info("🎉 SYSTEM READY FOR TRADING")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System workflow failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Restore original account context if needed
            if self.account_manager:
                self.account_manager.restore_original_account()
            
            # # Logout if needed
            # if self.login_manager and self.is_logged_in:
            #     self.login_manager.logout()
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Cleanup warning: {e}")
            else:
                print(f"⚠️ Cleanup warning: {e}")


def main():
    """Main entry point for enhanced automated system"""
    system = None
    success = False
    
    try:
        # Initialize and run the system
        system = MainSystem()
        success = system.run()
        
        # Exit with appropriate code
        if success:
            print("✅ System completed successfully!")
            sys.exit(0)
        else:
            print("❌ System failed! Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        if system and system.logger:
            system.logger.info("🛑 Interrupted by user")
        else:
            print("🛑 Interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        if system and system.logger:
            system.logger.error(f"❌ Unexpected error: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Always attempt cleanup
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()