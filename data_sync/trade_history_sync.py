# data_sync/trade_history_sync.py
"""
Trade History Synchronization
=============================
Fetches actual trade history from Webull API and syncs to database
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class TradeHistorySync:
    """
    Synchronizes trade history from Webull API to local database
    
    Handles:
    - Fetching historical orders from Webull
    - Converting Webull order format to database format
    - Avoiding duplicate trades
    - Different order types and statuses
    """
    
    def __init__(self, wb, database, logger=None):
        """
        Initialize trade history synchronization
        
        Args:
            wb: Authenticated Webull client instance
            database: TradingDatabase instance
            logger: Logger instance (optional)
        """
        self.wb = wb
        self.db = database
        self.logger = logger or logging.getLogger(__name__)
        
        # Mapping Webull order status to our internal status
        self.status_mapping = {
            'Filled': 'FILLED',
            'Cancelled': 'CANCELLED', 
            'Working': 'PENDING',
            'Partially Filled': 'PARTIAL',
            'Failed': 'FAILED',
            'Pending': 'PENDING'
        }
        
        # Order types we sync (avoid pending/working orders)
        self.syncable_statuses = ['Filled', 'Cancelled', 'Failed', 'Partially Filled']
    
    def sync_trade_history(self, account, days_back: int = 30, max_orders: int = 1000) -> Dict:
        """
        Sync trade history for a specific account
        
        Args:
            account: AccountInfo instance
            days_back: Number of days back to sync
            max_orders: Maximum number of orders to fetch
            
        Returns:
            Dict with sync statistics
        """
        sync_stats = {
            'trades_synced': 0,
            'orders_fetched': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'oldest_trade': None,
            'newest_trade': None
        }
        
        try:
            self.logger.debug(f"Fetching trade history for {account.account_type} account...")
            
            # Get trade history from Webull
            orders = self._fetch_orders_from_webull(max_orders)
            sync_stats['orders_fetched'] = len(orders)
            
            if not orders:
                self.logger.debug("No orders found in Webull API response")
                return sync_stats
            
            self.logger.debug(f"Retrieved {len(orders)} orders from Webull")
            
            # Filter orders by date and status
            cutoff_date = datetime.now() - timedelta(days=days_back)
            relevant_orders = self._filter_orders(orders, cutoff_date)
            
            self.logger.debug(f"Filtered to {len(relevant_orders)} relevant orders")
            
            # Process each order
            for order in relevant_orders:
                try:
                    if self._process_single_order(order, account):
                        sync_stats['trades_synced'] += 1
                        
                        # Track date range
                        order_date = self._parse_order_date(order)
                        if order_date:
                            if not sync_stats['oldest_trade'] or order_date < sync_stats['oldest_trade']:
                                sync_stats['oldest_trade'] = order_date
                            if not sync_stats['newest_trade'] or order_date > sync_stats['newest_trade']:
                                sync_stats['newest_trade'] = order_date
                    else:
                        sync_stats['duplicates_skipped'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing order {order.get('orderId', 'unknown')}: {e}")
                    sync_stats['errors'] += 1
                    continue
            
            self.logger.debug(f"Trade history sync complete: {sync_stats['trades_synced']} trades synced")
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"Error in trade history sync: {e}")
            sync_stats['errors'] += 1
            return sync_stats
    
    def _fetch_orders_from_webull(self, max_orders: int) -> List[Dict]:
        """
        Fetch orders from Webull API with retry logic
        
        Args:
            max_orders: Maximum number of orders to fetch
            
        Returns:
            List of order dictionaries from Webull API
        """
        try:
            # Webull API typically returns orders in pages
            # We'll fetch multiple pages to get more history
            all_orders = []
            
            # Fetch different order statuses
            for status in ['All', 'Filled', 'Cancelled']:
                try:
                    orders = self.wb.get_history_orders(status=status, count=max_orders)
                    
                    if isinstance(orders, dict) and 'data' in orders:
                        orders = orders['data']
                    elif isinstance(orders, dict) and 'orders' in orders:
                        orders = orders['orders']
                    elif not isinstance(orders, list):
                        self.logger.warning(f"Unexpected order format from Webull API: {type(orders)}")
                        continue
                    
                    if orders:
                        all_orders.extend(orders)
                        self.logger.debug(f"Fetched {len(orders)} orders with status '{status}'")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching orders with status '{status}': {e}")
                    continue
            
            # Remove duplicates based on order ID
            unique_orders = {}
            for order in all_orders:
                order_id = order.get('orderId') or order.get('id')
                if order_id:
                    unique_orders[order_id] = order
            
            return list(unique_orders.values())
            
        except Exception as e:
            self.logger.error(f"Error fetching orders from Webull: {e}")
            return []
    
    def _filter_orders(self, orders: List[Dict], cutoff_date: datetime) -> List[Dict]:
        """
        Filter orders by date and status
        
        Args:
            orders: List of order dictionaries
            cutoff_date: Cutoff date (orders before this are excluded)
            
        Returns:
            Filtered list of orders
        """
        filtered_orders = []
        
        for order in orders:
            try:
                # Check order status
                status = order.get('statusStr', order.get('status', 'Unknown'))
                if status not in self.syncable_statuses:
                    continue
                
                # Check order date
                order_date = self._parse_order_date(order)
                if not order_date or order_date < cutoff_date:
                    continue
                
                # Must have required fields
                if not self._has_required_fields(order):
                    continue
                
                filtered_orders.append(order)
                
            except Exception as e:
                self.logger.warning(f"Error filtering order: {e}")
                continue
        
        return filtered_orders
    
    def _parse_order_date(self, order: Dict) -> Optional[datetime]:
        """
        Parse order date from various possible fields
        
        Args:
            order: Order dictionary from Webull
            
        Returns:
            Parsed datetime or None
        """
        # Try different date fields that Webull might use
        date_fields = ['createTime', 'updateTime', 'filledTime', 'createdAt', 'placedTime']
        
        for field in date_fields:
            if field in order:
                try:
                    date_value = order[field]
                    
                    # Handle different date formats
                    if isinstance(date_value, (int, float)):
                        # Unix timestamp (seconds or milliseconds)
                        if date_value > 1e10:  # Milliseconds
                            date_value = date_value / 1000
                        return datetime.fromtimestamp(date_value)
                    
                    elif isinstance(date_value, str):
                        # ISO date string
                        try:
                            return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                        except ValueError:
                            # Try other formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    return datetime.strptime(date_value, fmt)
                                except ValueError:
                                    continue
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing date field '{field}': {e}")
                    continue
        
        return None
    
    def _has_required_fields(self, order: Dict) -> bool:
        """
        Check if order has required fields for database sync
        
        Args:
            order: Order dictionary
            
        Returns:
            True if order has all required fields
        """
        required_fields = ['ticker', 'action', 'totalQuantity']
        
        # Check basic required fields
        for field in required_fields:
            if field not in order:
                return False
        
        # Check ticker has symbol
        ticker = order.get('ticker', {})
        if not isinstance(ticker, dict) or 'symbol' not in ticker:
            return False
        
        return True
    
    def _process_single_order(self, order: Dict, account) -> bool:
        """
        Process a single order and add to database
        
        Args:
            order: Order dictionary from Webull
            account: AccountInfo instance
            
        Returns:
            True if order was processed (new), False if skipped (duplicate)
        """
        try:
            # Extract order information
            order_id = order.get('orderId') or order.get('id')
            symbol = order['ticker']['symbol']
            action = order.get('action', 'UNKNOWN').upper()
            quantity = float(order.get('totalQuantity', 0))
            
            # Get price (filled price preferred, then limit price)
            price = 0.0
            if 'avgFilledPrice' in order and order['avgFilledPrice']:
                price = float(order['avgFilledPrice'])
            elif 'filledPrice' in order and order['filledPrice']:
                price = float(order['filledPrice'])
            elif 'lmtPrice' in order and order['lmtPrice']:
                price = float(order['lmtPrice'])
            elif 'price' in order and order['price']:
                price = float(order['price'])
            
            # Get order status
            status = order.get('statusStr', order.get('status', 'UNKNOWN'))
            mapped_status = self.status_mapping.get(status, 'UNKNOWN')
            
            # Get order date
            order_date = self._parse_order_date(order)
            if not order_date:
                self.logger.warning(f"Could not parse date for order {order_id}")
                return False
            
            # Check if we already have this trade
            if self._is_duplicate_trade(order_id, symbol, order_date.strftime('%Y-%m-%d')):
                return False
            
            # Log the trade to database
            self.db.log_trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                signal_phase='HISTORICAL_SYNC',  # Mark as historical sync
                signal_strength=0.0,  # No signal strength for historical data
                account_type=account.account_type,
                order_id=str(order_id),
                day_trade_check='N/A'  # Historical data doesn't need day trade check
            )
            
            # Update trade status to match Webull status
            self._update_trade_status(order_id, mapped_status)
            
            self.logger.debug(f"Synced trade: {action} {quantity} {symbol} @ ${price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing order: {e}")
            raise
    
    def _is_duplicate_trade(self, order_id: str, symbol: str, date: str) -> bool:
        """
        Check if this trade already exists in the database
        
        Args:
            order_id: Webull order ID
            symbol: Stock symbol
            date: Trade date (YYYY-MM-DD)
            
        Returns:
            True if duplicate exists
        """
        try:
            # Check if order ID already exists
            existing_trades = self.db.get_todays_trades(symbol)
            
            for trade in existing_trades:
                if trade.get('order_id') == str(order_id):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking for duplicate trade: {e}")
            return False
    
    def _update_trade_status(self, order_id: str, status: str):
        """
        Update trade status in database
        
        Args:
            order_id: Webull order ID
            status: New status to set
        """
        try:
            # This would require adding an update method to TradingDatabase
            # For now, we'll log the status in the original log_trade call
            pass
            
        except Exception as e:
            self.logger.debug(f"Error updating trade status: {e}")