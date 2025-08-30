# data_sync/position_sync.py
"""
Position Synchronization
========================
Fetches current positions from Webull API and syncs to database
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

class PositionSync:
    """
    Synchronizes current positions from Webull API to local database
    
    Handles:
    - Fetching current positions from Webull
    - Converting Webull position format to database format
    - Updating database positions to match actual positions
    - Handling closed positions
    """
    
    def __init__(self, wb, database, logger=None):
        """
        Initialize position synchronization
        
        Args:
            wb: Authenticated Webull client instance
            database: TradingDatabase instance
            logger: Logger instance (optional)
        """
        self.wb = wb
        self.db = database
        self.logger = logger or logging.getLogger(__name__)
    
    def sync_positions(self, account) -> Dict:
        """
        Sync current positions for a specific account
        
        Args:
            account: AccountInfo instance
            
        Returns:
            Dict with sync statistics
        """
        sync_stats = {
            'positions_synced': 0,
            'positions_closed': 0,
            'positions_updated': 0,
            'positions_created': 0,
            'errors': 0
        }
        
        try:
            self.logger.debug(f"Fetching positions for {account.account_type} account...")
            
            # Get current positions from Webull
            webull_positions = self._fetch_positions_from_webull()
            
            if not webull_positions:
                self.logger.debug("No positions found in Webull API response")
                # Still need to check for closed positions
                self._handle_closed_positions(account, [], sync_stats)
                return sync_stats
            
            self.logger.debug(f"Retrieved {len(webull_positions)} positions from Webull")
            
            # Get existing positions from database
            db_positions = self.db.get_account_positions(account.account_type)
            
            # Track which symbols we've seen from Webull
            webull_symbols = set()
            
            # Process each Webull position
            for position in webull_positions:
                try:
                    symbol = self._process_single_position(position, account, db_positions, sync_stats)
                    if symbol:
                        webull_symbols.add(symbol)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing position: {e}")
                    sync_stats['errors'] += 1
                    continue
            
            # Handle positions that exist in DB but not in Webull (closed positions)
            self._handle_closed_positions(account, webull_symbols, sync_stats)
            
            self.logger.debug(f"Position sync complete: {sync_stats['positions_synced']} positions processed")
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"Error in position sync: {e}")
            sync_stats['errors'] += 1
            return sync_stats
    
    def _fetch_positions_from_webull(self) -> List[Dict]:
        """
        Fetch current positions from Webull API
        
        Returns:
            List of position dictionaries from Webull API
        """
        try:
            # Try getting positions from account data first
            account_data = self.wb.get_account()
            
            if isinstance(account_data, dict) and 'positions' in account_data:
                positions = account_data['positions']
                if positions:
                    self.logger.debug(f"Found {len(positions)} positions in account data")
                    return positions
            
            # Fallback: try direct get_positions method
            positions = self.wb.get_positions()
            
            if isinstance(positions, list):
                self.logger.debug(f"Found {len(positions)} positions from get_positions()")
                return positions
            elif isinstance(positions, dict) and 'data' in positions:
                return positions['data']
            
            self.logger.debug("No positions found or unexpected format")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching positions from Webull: {e}")
            return []
    
    def _process_single_position(self, position: Dict, account, db_positions: Dict, sync_stats: Dict) -> Optional[str]:
        """
        Process a single position from Webull
        
        Args:
            position: Position dictionary from Webull
            account: AccountInfo instance
            db_positions: Dict of existing DB positions
            sync_stats: Stats dictionary to update
            
        Returns:
            Symbol if processed successfully, None otherwise
        """
        try:
            # Extract position data
            symbol = self._extract_symbol(position)
            if not symbol:
                self.logger.warning("Position missing symbol information")
                return None
            
            shares = self._extract_shares(position)
            cost_price = self._extract_cost_price(position)
            current_price = self._extract_current_price(position)
            
            if shares == 0:
                self.logger.debug(f"Skipping {symbol} - zero shares")
                return symbol
            
            # Check if position exists in database
            existing_position = db_positions.get(symbol)
            
            if existing_position:
                # Update existing position
                self._update_existing_position(
                    symbol, shares, cost_price, account, existing_position, sync_stats
                )
                sync_stats['positions_updated'] += 1
            else:
                # Create new position
                self._create_new_position(
                    symbol, shares, cost_price, account, sync_stats
                )
                sync_stats['positions_created'] += 1
            
            sync_stats['positions_synced'] += 1
            self.logger.debug(f"Synced position: {shares} shares of {symbol} @ ${cost_price:.2f}")
            
            return symbol
            
        except Exception as e:
            self.logger.error(f"Error processing position: {e}")
            raise
    
    def _extract_symbol(self, position: Dict) -> Optional[str]:
        """Extract symbol from Webull position data"""
        # Try different possible field names
        if 'ticker' in position and isinstance(position['ticker'], dict):
            return position['ticker'].get('symbol')
        elif 'symbol' in position:
            return position['symbol']
        elif 'tickerSymbol' in position:
            return position['tickerSymbol']
        
        return None
    
    def _extract_shares(self, position: Dict) -> float:
        """Extract number of shares from Webull position data"""
        # Try different possible field names
        for field in ['position', 'quantity', 'shares', 'totalQuantity']:
            if field in position:
                try:
                    return float(position[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _extract_cost_price(self, position: Dict) -> float:
        """Extract cost price from Webull position data"""
        # Try different possible field names
        for field in ['costPrice', 'avgCost', 'averagePrice', 'avgPrice']:
            if field in position:
                try:
                    return float(position[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _extract_current_price(self, position: Dict) -> float:
        """Extract current price from Webull position data"""
        # Try different possible field names
        for field in ['lastPrice', 'currentPrice', 'marketPrice', 'price']:
            if field in position:
                try:
                    return float(position[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _update_existing_position(self, symbol: str, shares: float, cost_price: float, 
                                 account, existing_position: Dict, sync_stats: Dict):
        """
        Update an existing position in the database
        
        Args:
            symbol: Stock symbol
            shares: Current number of shares
            cost_price: Current average cost
            account: AccountInfo instance
            existing_position: Existing position from database
            sync_stats: Stats dictionary to update
        """
        try:
            # Calculate the difference
            existing_shares = existing_position.get('total_shares', 0)
            share_diff = shares - existing_shares
            
            if abs(share_diff) < 0.001:  # No meaningful change
                self.logger.debug(f"No change in position for {symbol}")
                return
            
            # Update the position
            # Note: This is a simplified approach. In reality, you might want to
            # calculate the exact trades needed to get from old position to new position
            self.db.update_position(
                symbol=symbol,
                shares=share_diff,  # This will be added to existing position
                cost=cost_price,
                account_type=account.account_type,
                entry_phase='POSITION_SYNC',
                entry_strength=0.0
            )
            
            self.logger.debug(f"Updated position for {symbol}: {existing_shares} -> {shares} shares")
            
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")
            sync_stats['errors'] += 1
    
    def _create_new_position(self, symbol: str, shares: float, cost_price: float, 
                            account, sync_stats: Dict):
        """
        Create a new position in the database
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            cost_price: Average cost per share
            account: AccountInfo instance
            sync_stats: Stats dictionary to update
        """
        try:
            self.db.update_position(
                symbol=symbol,
                shares=shares,
                cost=cost_price,
                account_type=account.account_type,
                entry_phase='POSITION_SYNC',
                entry_strength=0.0
            )
            
            self.logger.debug(f"Created new position: {shares} shares of {symbol} @ ${cost_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error creating position for {symbol}: {e}")
            sync_stats['errors'] += 1
    
    def _handle_closed_positions(self, account, webull_symbols: set, sync_stats: Dict):
        """
        Handle positions that exist in database but not in Webull (closed positions)
        
        Args:
            account: AccountInfo instance
            webull_symbols: Set of symbols found in Webull positions
            sync_stats: Stats dictionary to update
        """
        try:
            # Get all database positions for this account
            db_positions = self.db.get_account_positions(account.account_type)
            
            for symbol, position in db_positions.items():
                if symbol not in webull_symbols and position.get('total_shares', 0) > 0:
                    # This position exists in DB but not in Webull - it was closed
                    self.logger.debug(f"Closing position for {symbol} - not found in Webull")
                    
                    self.db.clear_position(symbol, account.account_type)
                    sync_stats['positions_closed'] += 1
            
            if sync_stats['positions_closed'] > 0:
                self.logger.info(f"Closed {sync_stats['positions_closed']} positions that no longer exist in Webull")
            
        except Exception as e:
            self.logger.error(f"Error handling closed positions: {e}")
            sync_stats['errors'] += 1
    
    def get_position_summary(self, account) -> Dict:
        """
        Get a summary of current positions for an account
        
        Args:
            account: AccountInfo instance
            
        Returns:
            Dict with position summary
        """
        try:
            # Get positions from both Webull and database
            webull_positions = self._fetch_positions_from_webull()
            db_positions = self.db.get_account_positions(account.account_type)
            
            summary = {
                'webull_positions': len(webull_positions),
                'database_positions': len(db_positions),
                'symbols_in_both': 0,
                'symbols_webull_only': 0,
                'symbols_database_only': 0,
                'total_market_value': 0.0
            }
            
            webull_symbols = set()
            for pos in webull_positions:
                symbol = self._extract_symbol(pos)
                if symbol:
                    webull_symbols.add(symbol)
                    # Add market value if available
                    market_value = pos.get('marketValue', 0)
                    if market_value:
                        summary['total_market_value'] += float(market_value)
            
            db_symbols = set(db_positions.keys())
            
            summary['symbols_in_both'] = len(webull_symbols.intersection(db_symbols))
            summary['symbols_webull_only'] = len(webull_symbols - db_symbols)
            summary['symbols_database_only'] = len(db_symbols - webull_symbols)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting position summary: {e}")
            return {'error': str(e)}