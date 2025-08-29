# data_sync/webull_sync.py
"""
Main Webull Data Synchronization Manager
========================================
Orchestrates syncing of trade history and positions from Webull to local database
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from database.trading_db import TradingDatabase
from .trade_history_sync import TradeHistorySync
from .position_sync import PositionSync


class WebullDataSync:
    """
    Main class for syncing actual Webull data to local database
    
    This class coordinates:
    - Trade history synchronization
    - Position synchronization  
    - Account data sync across multiple accounts
    - Error handling and retry logic
    """
    
    def __init__(self, wb, account_manager, config, database=None, logger=None):
        """
        Initialize the Webull data synchronization system
        
        Args:
            wb: Authenticated Webull client instance
            account_manager: AccountManager instance with discovered accounts
            config: PersonalTradingConfig instance
            database: TradingDatabase instance (optional, will create if None)
            logger: Logger instance (optional)
        """
        self.wb = wb
        self.account_manager = account_manager
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize database
        self.db = database or TradingDatabase(config.DATABASE_PATH)
        
        # Initialize sync components
        self.trade_history_sync = TradeHistorySync(wb, self.db, logger=self.logger)
        self.position_sync = PositionSync(wb, self.db, logger=self.logger)
        
        # Sync statistics
        self.sync_stats = {
            'accounts_synced': 0,
            'trades_synced': 0,
            'positions_synced': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def sync_all_data(self, days_back: int = 30, include_positions: bool = True, 
                     include_trade_history: bool = True) -> Dict:
        """
        Sync all data for all enabled accounts
        
        Args:
            days_back: Number of days of trade history to sync
            include_positions: Whether to sync current positions
            include_trade_history: Whether to sync trade history
            
        Returns:
            Dict containing sync statistics and results
        """
        self.sync_stats['start_time'] = datetime.now()
        self.logger.info("🔄 Starting Webull data synchronization...")
        
        try:
            # Get enabled accounts
            enabled_accounts = self.account_manager.get_enabled_accounts()
            
            if not enabled_accounts:
                self.logger.warning("⚠️  No enabled accounts found for data sync")
                return self._get_sync_results()
            
            self.logger.info(f"📊 Syncing data for {len(enabled_accounts)} accounts")
            print()
            
            # Sync each account
            for i, account in enumerate(enabled_accounts, 1):
                self.logger.info(f"🔄 Syncing Account {i}/{len(enabled_accounts)}: {account.account_type}")
                
                try:
                    # Switch to this account
                    if not self.account_manager.switch_to_account(account):
                        self.logger.error(f"❌ Failed to switch to account {account.account_id}")
                        self.sync_stats['errors'] += 1
                        continue
                    
                    account_stats = {
                        'trades_synced': 0,
                        'positions_synced': 0,
                        'errors': 0
                    }
                    
                    # Sync trade history
                    if include_trade_history:
                        self.logger.info(f"   📈 Syncing trade history ({days_back} days)...")
                        trade_stats = self.trade_history_sync.sync_trade_history(
                            account, days_back=days_back
                        )
                        account_stats['trades_synced'] = trade_stats.get('trades_synced', 0)
                        account_stats['errors'] += trade_stats.get('errors', 0)
                        
                        self.logger.info(f"   ✅ Synced {trade_stats.get('trades_synced', 0)} trades")
                    
                    # Sync positions
                    if include_positions:
                        self.logger.info(f"   💼 Syncing current positions...")
                        position_stats = self.position_sync.sync_positions(account)
                        account_stats['positions_synced'] = position_stats.get('positions_synced', 0)
                        account_stats['errors'] += position_stats.get('errors', 0)
                        
                        self.logger.info(f"   ✅ Synced {position_stats.get('positions_synced', 0)} positions")
                    
                    # Update overall stats
                    self.sync_stats['accounts_synced'] += 1
                    self.sync_stats['trades_synced'] += account_stats['trades_synced']
                    self.sync_stats['positions_synced'] += account_stats['positions_synced']
                    self.sync_stats['errors'] += account_stats['errors']
                    
                    self.logger.info(f"   ✅ Account sync complete")
                    print()
                    
                    # Small delay between accounts
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error syncing account {account.account_type}: {e}")
                    self.sync_stats['errors'] += 1
                    continue
            
            # Restore original account context
            self.account_manager.restore_original_account()
            
            # Log final results
            self._log_sync_summary()
            
            return self._get_sync_results()
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in data sync: {e}")
            self.sync_stats['errors'] += 1
            return self._get_sync_results()
        
        finally:
            self.sync_stats['end_time'] = datetime.now()
    
    def sync_account_data(self, account, days_back: int = 30, 
                         include_positions: bool = True, 
                         include_trade_history: bool = True) -> Dict:
        """
        Sync data for a specific account
        
        Args:
            account: AccountInfo instance to sync
            days_back: Number of days of trade history to sync
            include_positions: Whether to sync current positions
            include_trade_history: Whether to sync trade history
            
        Returns:
            Dict containing sync results for this account
        """
        self.logger.info(f"🔄 Syncing data for {account.account_type} account...")
        
        account_stats = {
            'account_id': account.account_id,
            'account_type': account.account_type,
            'trades_synced': 0,
            'positions_synced': 0,
            'errors': 0,
            'success': False
        }
        
        try:
            # Switch to this account
            if not self.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account {account.account_id}")
                account_stats['errors'] += 1
                return account_stats
            
            # Sync trade history
            if include_trade_history:
                trade_stats = self.trade_history_sync.sync_trade_history(
                    account, days_back=days_back
                )
                account_stats['trades_synced'] = trade_stats.get('trades_synced', 0)
                account_stats['errors'] += trade_stats.get('errors', 0)
            
            # Sync positions
            if include_positions:
                position_stats = self.position_sync.sync_positions(account)
                account_stats['positions_synced'] = position_stats.get('positions_synced', 0) 
                account_stats['errors'] += position_stats.get('errors', 0)
            
            account_stats['success'] = account_stats['errors'] == 0
            
            self.logger.info(f"✅ Account sync complete: "
                           f"{account_stats['trades_synced']} trades, "
                           f"{account_stats['positions_synced']} positions")
            
            return account_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing account {account.account_type}: {e}")
            account_stats['errors'] += 1
            return account_stats
    
    def get_sync_status(self) -> Dict:
        """Get current sync status and statistics"""
        return {
            'last_sync': self.sync_stats.get('end_time'),
            'accounts_synced': self.sync_stats.get('accounts_synced', 0),
            'trades_synced': self.sync_stats.get('trades_synced', 0),
            'positions_synced': self.sync_stats.get('positions_synced', 0),
            'errors': self.sync_stats.get('errors', 0),
            'duration': self._get_sync_duration()
        }
    
    def _get_sync_results(self) -> Dict:
        """Get formatted sync results"""
        return {
            'success': self.sync_stats['errors'] == 0,
            'accounts_synced': self.sync_stats['accounts_synced'],
            'trades_synced': self.sync_stats['trades_synced'],
            'positions_synced': self.sync_stats['positions_synced'],
            'errors': self.sync_stats['errors'],
            'duration': self._get_sync_duration(),
            'start_time': self.sync_stats['start_time'],
            'end_time': self.sync_stats['end_time']
        }
    
    def _get_sync_duration(self) -> Optional[str]:
        """Get formatted sync duration"""
        if self.sync_stats['start_time'] and self.sync_stats['end_time']:
            duration = self.sync_stats['end_time'] - self.sync_stats['start_time']
            return str(duration).split('.')[0]  # Remove microseconds
        return None
    
    def _log_sync_summary(self):
        """Log sync summary"""
        duration = self._get_sync_duration()
        
        self.logger.info("📊 SYNC SUMMARY")
        self.logger.info(f"   Accounts: {self.sync_stats['accounts_synced']}")
        self.logger.info(f"   Trades: {self.sync_stats['trades_synced']}")
        self.logger.info(f"   Positions: {self.sync_stats['positions_synced']}")
        self.logger.info(f"   Errors: {self.sync_stats['errors']}")
        if duration:
            self.logger.info(f"   Duration: {duration}")
        
        if self.sync_stats['errors'] == 0:
            self.logger.info("🎉 Data synchronization completed successfully!")
        else:
            self.logger.warning(f"⚠️  Data synchronization completed with {self.sync_stats['errors']} errors")
        
        print()