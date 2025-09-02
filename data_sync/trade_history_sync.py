# data_sync/trade_history_sync.py
"""
Trade History Synchronization - FIXED STATUS HANDLING
====================================================
Fixed to properly handle order status and only sync completed trades
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class TradeHistorySync:
    """
    Synchronizes trade history from Webull API to local database
    FIXED: Properly handles order status to only sync completed trades
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
            'Canceled': 'CANCELLED',
            'Working': 'PENDING',
            'Pending': 'PENDING',
            'Queued': 'PENDING',
            'Partially Filled': 'PARTIAL',
            'PartiallyFilled': 'PARTIAL',
            'Failed': 'FAILED',
            'Expired': 'CANCELLED',
            'Rejected': 'FAILED'
        }
        
        # Only sync these statuses (completed trades only)
        self.syncable_statuses = ['Filled', 'Partially Filled', 'PartiallyFilled']
    
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
            'orders_skipped_pending': 0,
            'orders_skipped_cancelled': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'oldest_trade': None,
            'newest_trade': None
        }
        
        try:
            self.logger.info(f"📊 Fetching trade history for {account.account_type} account...")
            
            # Get trade history from Webull - focus on FILLED orders
            orders = self._fetch_and_extract_orders(max_orders)
            sync_stats['orders_fetched'] = len(orders)
            
            if not orders:
                self.logger.warning("⚠️  No orders found in Webull API response")
                return sync_stats
            
            self.logger.info(f"✅ Retrieved {len(orders)} orders from Webull")
            
            # Debug: Show status distribution
            status_counts = {}
            for order in orders:
                status = self._get_order_status(order)
                status_counts[status] = status_counts.get(status, 0) + 1
            
            self.logger.debug(f"Order status distribution: {status_counts}")
            
            # Filter orders by date and status
            cutoff_date = datetime.now() - timedelta(days=days_back)
            self.logger.debug(f"🗓️  Filtering orders after {cutoff_date.strftime('%Y-%m-%d')}")
            
            relevant_orders = self._filter_orders(orders, cutoff_date, sync_stats)
            
            self.logger.info(f"📋 Filtered to {len(relevant_orders)} FILLED orders (last {days_back} days)")
            
            if sync_stats['orders_skipped_pending'] > 0:
                self.logger.debug(f"   Skipped {sync_stats['orders_skipped_pending']} pending/working orders")
            if sync_stats['orders_skipped_cancelled'] > 0:
                self.logger.debug(f"   Skipped {sync_stats['orders_skipped_cancelled']} cancelled orders")
            
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
                    self.logger.debug(f"Error processing order: {e}")
                    sync_stats['errors'] += 1
                    continue
            
            # Log summary
            if sync_stats['trades_synced'] > 0:
                self.logger.info(f"✅ Synced {sync_stats['trades_synced']} completed trades")
                if sync_stats['oldest_trade'] and sync_stats['newest_trade']:
                    self.logger.info(f"   Date range: {sync_stats['oldest_trade'].strftime('%Y-%m-%d')} to {sync_stats['newest_trade'].strftime('%Y-%m-%d')}")
            else:
                self.logger.info("ℹ️  No new completed trades to sync")
            
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error in trade history sync: {e}")
            sync_stats['errors'] += 1
            return sync_stats
    
    def _fetch_and_extract_orders(self, max_orders: int) -> List[Dict]:
        """
        Fetch orders from Webull and extract from nested structure
        Focus on FILLED orders first
        
        Args:
            max_orders: Maximum number of orders to fetch
            
        Returns:
            List of individual order dictionaries
        """
        try:
            all_orders = []
            
            # Prioritize fetching filled orders first
            status_priority = ['Filled', 'All']  # Skip 'Cancelled' initially
            
            for status in status_priority:
                try:
                    self.logger.debug(f"Fetching orders with status='{status}'...")
                    
                    # Get the combo orders list
                    combo_orders = self.wb.get_history_orders(status=status, count=max_orders)
                    
                    if not combo_orders:
                        continue
                    
                    # Extract individual orders from the nested structure
                    extracted_count = 0
                    
                    if isinstance(combo_orders, list):
                        for combo_order in combo_orders:
                            # Each combo_order has an 'orders' field containing the actual orders
                            if isinstance(combo_order, dict) and 'orders' in combo_order:
                                nested_orders = combo_order['orders']
                                if isinstance(nested_orders, list):
                                    for order in nested_orders:
                                        # Add combo info to each order for context
                                        order['comboId'] = combo_order.get('comboId')
                                        order['comboType'] = combo_order.get('comboType')
                                        all_orders.append(order)
                                        extracted_count += 1
                    
                    self.logger.debug(f"   Extracted {extracted_count} orders from {len(combo_orders)} combo orders")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    self.logger.debug(f"Error fetching {status} orders: {e}")
                    continue
            
            # Remove duplicates based on order ID
            unique_orders = {}
            for order in all_orders:
                order_id = self._get_order_id(order)
                if order_id:
                    unique_orders[order_id] = order
            
            final_orders = list(unique_orders.values())
            self.logger.debug(f"Total unique orders extracted: {len(final_orders)}")
            
            return final_orders
            
        except Exception as e:
            self.logger.error(f"Error fetching orders: {e}")
            return []
    
    def _get_order_status(self, order: Dict) -> str:
        """Get order status from various possible fields"""
        # Check multiple possible status fields
        for field in ['statusStr', 'status', 'orderStatus']:
            if field in order and order[field]:
                return str(order[field])
        
        # Check if filled based on quantity
        if order.get('filledQuantity', 0) > 0:
            return 'Filled'
        
        return 'Unknown'
    
    def _get_order_id(self, order: Dict) -> Optional[str]:
        """Get order ID from various possible fields"""
        for field in ['orderId', 'id', 'orderNo', 'orderNumber']:
            if field in order and order[field]:
                return str(order[field])
        return None
    
    def _get_symbol(self, order: Dict) -> Optional[str]:
        """Get symbol from order data"""
        # Check ticker object first (most common)
        if 'ticker' in order and isinstance(order['ticker'], dict):
            for field in ['symbol', 'tickerSymbol', 'disSymbol']:
                if field in order['ticker']:
                    return order['ticker'][field]
        
        # Check top-level fields
        for field in ['symbol', 'tickerSymbol', 'disSymbol']:
            if field in order and order[field]:
                return order[field]
        
        return None
    
    def _filter_orders(self, orders: List[Dict], cutoff_date: datetime, sync_stats: Dict) -> List[Dict]:
        """
        Filter orders by date and status - ONLY include filled trades
        
        Args:
            orders: List of order dictionaries
            cutoff_date: Cutoff date (orders before this are excluded)
            sync_stats: Stats dictionary to update
            
        Returns:
            Filtered list of orders
        """
        filtered_orders = []
        
        for order in orders:
            try:
                # Get order status
                status = self._get_order_status(order)
                
                # Skip pending/working orders
                if status.lower() in ['pending', 'working', 'queued']:
                    sync_stats['orders_skipped_pending'] += 1
                    continue
                
                # Skip cancelled orders
                if status.lower() in ['cancelled', 'canceled', 'expired', 'rejected']:
                    sync_stats['orders_skipped_cancelled'] += 1
                    continue
                
                # Only process filled orders
                is_filled = (
                    status in self.syncable_statuses or
                    status.lower() == 'filled' or
                    'filled' in status.lower() or
                    (order.get('filledQuantity', 0) > 0 and 
                     order.get('filledQuantity', 0) == order.get('totalQuantity', 0))
                )
                
                if not is_filled:
                    continue
                
                # Check order date
                order_date = self._parse_order_date(order)
                if not order_date:
                    continue
                    
                if order_date < cutoff_date:
                    continue
                
                # Must have required fields
                if not self._has_required_fields(order):
                    continue
                
                filtered_orders.append(order)
                
            except Exception as e:
                self.logger.debug(f"Error filtering order: {e}")
                continue
        
        return filtered_orders
    
    def _parse_order_date(self, order: Dict) -> Optional[datetime]:
        """
        Parse order date from various possible fields
        Prefer filled time for completed trades
        
        Args:
            order: Order dictionary from Webull
            
        Returns:
            Parsed datetime or None
        """
        # Prefer filledTime for completed trades, then createTime
        date_fields = ['filledTime', 'filledTime0', 'createTime', 'createTime0', 
                       'placeTime', 'updateTime', 'createdAt', 'orderDate', 'date']
        
        order_id = self._get_order_id(order)
        self.logger.debug(f"🕐 Parsing date for order {order_id}:")
        
        for field in date_fields:
            if field in order and order[field]:
                try:
                    date_value = order[field]
                    self.logger.debug(f"   Checking field '{field}': {date_value} (type: {type(date_value)})")
                    
                    # Skip zero or invalid values
                    if date_value == 0 or date_value == '0':
                        self.logger.debug(f"   -> Skipping zero value")
                        continue
                    
                    # Handle different date formats
                    if isinstance(date_value, (int, float)):
                        # Unix timestamp
                        if date_value > 1e10:  # Milliseconds
                            date_value = date_value / 1000
                        if date_value > 0:  # Valid timestamp
                            parsed_date = datetime.fromtimestamp(date_value)
                            self.logger.debug(f"   -> ✅ Successfully parsed as timestamp: {parsed_date}")
                            return parsed_date
                    
                    elif isinstance(date_value, str):
                        # Try ISO format first
                        try:
                            parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                            self.logger.debug(f"   -> ✅ Successfully parsed as ISO: {parsed_date}")
                            return parsed_date
                        except ValueError:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y',
                                        '%Y-%m-%dT%H:%M:%S', '%Y/%m/%d %H:%M:%S']:
                                try:
                                    parsed_date = datetime.strptime(date_value, fmt)
                                    self.logger.debug(f"   -> ✅ Successfully parsed with format {fmt}: {parsed_date}")
                                    return parsed_date
                                except ValueError:
                                    continue
                            self.logger.debug(f"   -> ❌ Could not parse string date")
                    
                except Exception as e:
                    self.logger.debug(f"   -> ❌ Exception parsing: {e}")
                    continue
        
        self.logger.warning(f"⚠️  Could not parse any date field for order {order_id}")
        return None
    
    def _has_required_fields(self, order: Dict) -> bool:
        """
        Check if order has required fields for database sync
        
        Args:
            order: Order dictionary
            
        Returns:
            True if order has all required fields
        """
        # Get symbol
        symbol = self._get_symbol(order)
        if not symbol:
            return False
        
        # Check for action
        has_action = any(field in order for field in ['action', 'side'])
        if not has_action:
            return False
        
        # Check for quantity - prefer filledQuantity for completed trades
        has_quantity = False
        for field in ['filledQuantity', 'totalQuantity', 'quantity', 'qty']:
            if field in order and order[field] and float(order[field]) > 0:
                has_quantity = True
                break
        
        if not has_quantity:
            return False
        
        return True
    
    def _process_single_order(self, order: Dict, account) -> bool:
        """
        Process a single FILLED order and add to database
        
        Args:
            order: Order dictionary from Webull
            account: AccountInfo instance
            
        Returns:
            True if order was processed (new), False if skipped (duplicate)
        """
        try:
            # Extract order information
            order_id = self._get_order_id(order)
            symbol = self._get_symbol(order)
            
            if not symbol:
                return False
            
            # Get action (BUY/SELL)
            action = order.get('action', order.get('side', 'UNKNOWN')).upper()
            if action in ['B', 'BUY', 'LONG']:
                action = 'BUY'
            elif action in ['S', 'SELL', 'SHORT']:
                action = 'SELL'
            
            # Get quantity (prefer filled quantity for completed trades)
            quantity = 0
            # Prefer filledQuantity for accurate completed trade amounts
            if 'filledQuantity' in order and order['filledQuantity']:
                quantity = abs(float(order['filledQuantity']))
            else:
                # Fallback to other quantity fields
                for field in ['totalQuantity', 'quantity', 'qty']:
                    if field in order and order[field]:
                        quantity = abs(float(order[field]))
                        break
            
            if quantity == 0:
                return False
            
            # Get price (prefer average filled price for completed trades)
            price = 0.0
            # Prefer avgFilledPrice for accurate execution price
            for field in ['avgFilledPrice', 'filledPrice', 'avgPrice', 'executedPrice', 'price', 'lmtPrice']:
                if field in order and order[field] and float(order[field]) > 0:
                    price = float(order[field])
                    break
            
            # Get order date (prefer filled time)
            order_date = self._parse_order_date(order)
            if not order_date:
                self.logger.warning(f"❌ Could not parse date for order {order_id}, skipping")
                return False
            
            self.logger.debug(f"📅 Final parsed date for order {order_id}: {order_date} (symbol: {symbol})")
            
            # Get order status for database
            status = self._get_order_status(order)
            mapped_status = self.status_mapping.get(status, 'FILLED')  # Default to FILLED since we're only syncing filled orders
            
            # Check if duplicate
            date_str = order_date.strftime('%Y-%m-%d')
            if self._is_duplicate_trade(order_id, symbol, date_str):
                return False
            
            # Log the trade to database with actual trade date
            self.db.log_trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                signal_phase='HISTORICAL_SYNC',
                signal_strength=0.0,
                account_type=account.account_type,
                order_id=str(order_id) if order_id else None,
                day_trade_check='N/A',
                status=mapped_status,
                trade_date=order_date  # Pass the actual trade date from Webull
            )
            
            self.logger.info(f"   ✅ Synced: {action} {quantity} {symbol} @ ${price:.2f} ({order_date.strftime('%Y-%m-%d')}) - {mapped_status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing order: {e}")
            return False
    
    # <<< REMOVED METHOD >>>
    # The _update_trade_status method is no longer needed and has been removed.
    
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
            if not order_id:
                return False
            
            # Check all trades (not just today's) for this order ID
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                result = conn.execute('''
                    SELECT COUNT(*) FROM trades 
                    WHERE order_id = ?
                ''', (str(order_id),)).fetchone()
                
                return result[0] > 0 if result else False
            
        except Exception:
            return False