# wyckoff_bot/execution/trade_executor.py
"""
Trade Executor for Wyckoff Bot
==============================
Handles actual order placement through Webull API with fractional share support
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

@dataclass
class OrderResult:
    """Result of order placement attempt"""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None
    response_data: Optional[Dict] = None

class TradeExecutor:
    """
    Handles live trading execution through Webull API
    Supports both fractional and full share orders
    """
    
    def __init__(self, webull_client, account_manager, config=None, logger: logging.Logger = None, main_system=None):
        self.wb = webull_client
        self.account_manager = account_manager
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.main_system = main_system  # Reference to MainSystem for re-authentication
    
    def _ensure_valid_session(self) -> bool:
        """ENHANCED: Ensure we have a valid session using login_automatically retry pattern"""
        try:
            # STORE current account context before validation
            current_account_id = self.main_system.wb._account_id
            current_zone = self.main_system.wb.zone_var
            
            self.logger.debug(f"🔍 Validating session (preserving account context: {current_account_id})")
            
            # Test session with a simple API call that doesn't change account context
            try:
                # Use get_quote instead of account-specific calls to test session
                test_quote = self.main_system.wb.get_quote('SPY')
                if test_quote and 'close' in test_quote:
                    self.logger.debug(f"✅ Session validation passed (quote successful)")
                    
                    # RESTORE account context (in case it got changed)
                    self.main_system.wb._account_id = current_account_id
                    self.main_system.wb.zone_var = current_zone
                    
                    return True
                else:
                    self.logger.warning("⚠️ Session validation failed (quote failed)")
                    
            except Exception as test_error:
                self.logger.warning(f"⚠️ Session test failed: {test_error}")
                
            # If we reach here, session validation failed
            self.logger.info("🔄 Attempting fresh authentication using login_automatically...")
            
            # Clear old session completely to force fresh login
            self.main_system.session_manager.clear_session()
            
            # Use login_automatically for robust authentication with retries
            if self.main_system.login_manager.login_automatically():
                self.logger.info("✅ Fresh authentication successful")
                
                # CRITICAL: Restore the account context after fresh login
                self.main_system.wb._account_id = current_account_id
                self.main_system.wb.zone_var = current_zone
                
                # Save the refreshed session for future use
                self.main_system.session_manager.save_session(self.main_system.wb)
                
                # Verify the refreshed session works
                try:
                    verify_quote = self.main_system.wb.get_quote('SPY')
                    if verify_quote and 'close' in verify_quote:
                        self.logger.debug("✅ Fresh session verified with test quote")
                        return True
                    else:
                        self.logger.error("❌ Fresh session verification failed")
                        return False
                except Exception as verify_error:
                    self.logger.error(f"❌ Fresh session verification error: {verify_error}")
                    return False
            else:
                self.logger.error("❌ Fresh authentication failed")
                return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error in session validation: {e}")
            return False
    
    def _execute_with_retry(self, execute_func, *args, **kwargs) -> OrderResult:
        """
        Execute an order function with automatic session refresh retry
        
        Args:
            execute_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            OrderResult: Result of the execution
        """
        # First attempt
        try:
            response = execute_func(*args, **kwargs)
            
            # Check if it's a session expiration error
            if (response and isinstance(response, dict) and 
                'msg' in response and 
                ('session' in response['msg'].lower() or 'expired' in response['msg'].lower())):
                
                self.logger.warning("⚠️  Detected session expiration, attempting refresh...")
                
                # Try to ensure valid session
                if self._ensure_valid_session():
                    self.logger.info("🔄 Retrying order after session validation...")
                    # Retry the order
                    response = execute_func(*args, **kwargs)
                else:
                    return OrderResult(
                        success=False,
                        message="Session validation failed - please check credentials",
                        error_code="SESSION_VALIDATION_FAILED"
                    )
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Exception in execute with retry: {e}")
            return OrderResult(
                success=False,
                message=f"Exception: {str(e)}",
                error_code="EXECUTION_EXCEPTION"
            )
    
    def _place_order_with_retry(self, **order_params):
        """
        Place order with automatic session refresh on failure
        
        Args:
            **order_params: Parameters to pass to wb.place_order()
            
        Returns:
            dict: Order response from Webull API
        """
        try:
            # First attempt
            self.logger.debug("🔄 Attempting order placement...")
            response = self.wb.place_order(**order_params)
            
            # Check if session expired
            if (response and 'msg' in response and 
                ('session' in response.get('msg', '').lower() or 
                 'expired' in response.get('msg', '').lower() or
                 'token' in response.get('msg', '').lower())):
                
                self.logger.warning("⚠️  Session/token expired, re-authenticating and retrying...")
                
                # Clear session to force fresh login
                self.main_system.session_manager.clear_session()
                
                # Attempt fresh login using login_automatically
                if self.main_system.login_manager.login_automatically():
                    self.logger.info("✅ Re-login successful, saving session and retrying order...")
                    
                    # Save the refreshed session
                    self.main_system.session_manager.save_session(self.main_system.wb)
                    
                    # Retry the order
                    self.logger.info("🔄 Retrying order placement after fresh authentication...")
                    response = self.wb.place_order(**order_params)
                    
                    if response and response.get('success'):
                        self.logger.info("✅ Order succeeded after fresh authentication")
                    else:
                        self.logger.error(f"❌ Order still failed after fresh authentication: {response.get('msg', 'Unknown error')}")
                else:
                    self.logger.error("❌ Fresh login failed, cannot retry order")
                    response = {'success': False, 'msg': 'Session refresh failed - please check credentials'}
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Exception in place order with retry: {e}")
            return {'success': False, 'msg': f'Exception: {str(e)}'}
        
    def execute_fractional_buy(self, symbol: str, dollar_amount: float, 
                              entry_price: float, account_info: Dict) -> OrderResult:
        """
        Execute a fractional share buy order
        
        Args:
            symbol: Stock symbol to buy
            dollar_amount: Dollar amount to invest
            entry_price: Target entry price
            account_info: Account information
            
        Returns:
            OrderResult: Result of the order attempt
        """
        try:
            # Calculate fractional shares
            fractional_shares = dollar_amount / entry_price
            
            # Check if live trading is enabled
            if not getattr(self.config, 'LIVE_TRADING_ENABLED', False):
                self.logger.info(f"🔒 SIMULATED FRACTIONAL BUY for {symbol}: "
                               f"${dollar_amount:.2f} ({fractional_shares:.3f} shares) at ${entry_price:.2f}")
                self.logger.info("    (Live trading disabled in config - set LIVE_TRADING_ENABLED = True)")
                
                return OrderResult(
                    success=True,
                    order_id="SIMULATED_" + str(int(datetime.now().timestamp())),
                    message=f"Simulated fractional buy order for {symbol}",
                    response_data={'simulated': True}
                )
            
            # Ask for confirmation if required
            if getattr(self.config, 'REQUIRE_TRADE_CONFIRMATION', True):
                self.logger.warning(f"⚠️  LIVE TRADE CONFIRMATION REQUIRED")
                self.logger.warning(f"   Symbol: {symbol}")
                self.logger.warning(f"   Type: FRACTIONAL BUY")
                self.logger.warning(f"   Amount: ${dollar_amount:.2f} ({fractional_shares:.3f} shares)")
                self.logger.warning(f"   Price: ${entry_price:.2f}")
                
                try:
                    response = input("\n🔴 CONFIRM LIVE TRADE: Type 'YES' to proceed, anything else to cancel: ")
                    if response.upper() != 'YES':
                        return OrderResult(
                            success=False,
                            message="Trade cancelled by user",
                            error_code="USER_CANCELLED"
                        )
                except:
                    # If input fails (like in automated mode), don't execute
                    return OrderResult(
                        success=False,
                        message="Cannot confirm trade in automated mode",
                        error_code="NO_CONFIRMATION"
                    )
            
            self.logger.info(f"🔄 Placing LIVE FRACTIONAL BUY order for {symbol}: "
                           f"${dollar_amount:.2f} ({fractional_shares:.3f} shares) at ${entry_price:.2f}")
            
            # Ensure we have a valid session before placing the order
            if not self._ensure_valid_session():
                return OrderResult(
                    success=False,
                    message="Session validation failed before order placement",
                    error_code="SESSION_INVALID"
                )
            
            # Get ticker ID
            ticker_id = self.wb.get_ticker(symbol)
            if not ticker_id:
                return OrderResult(
                    success=False, 
                    message=f"Could not get ticker ID for {symbol}",
                    error_code="TICKER_NOT_FOUND"
                )
            
            # Place the order with retry logic
            response = self._place_order_with_retry(
                tId=ticker_id,
                price=entry_price,
                action='BUY',
                orderType='MKT',  # Limit order for better control
                enforce='DAY',    # Day order
                quant=fractional_shares,  # Fractional quantity
                outsideRegularTradingHour=False
            )
            
            # Parse response
            if response and 'success' in response and response['success']:
                order_id = response.get('orderId') or response.get('data', {}).get('orderId')
                
                self.logger.info(f"✅ FRACTIONAL BUY order placed successfully for {symbol}")
                self.logger.info(f"   Order ID: {order_id}")
                self.logger.info(f"   Amount: ${dollar_amount:.2f} ({fractional_shares:.3f} shares)")
                self.logger.info(f"   Price: ${entry_price:.2f}")
                
                return OrderResult(
                    success=True,
                    order_id=str(order_id) if order_id else None,
                    message=f"Fractional buy order placed for {symbol}",
                    response_data=response
                )
            else:
                error_msg = response.get('msg', 'Unknown error') if response else 'No response'
                self.logger.error(f"❌ Failed to place fractional buy order for {symbol}: {error_msg}")
                self.logger.error(f"🔍 DEBUG - Full Webull response: {response}")
                
                return OrderResult(
                    success=False,
                    message=f"Order placement failed: {error_msg}",
                    error_code=response.get('code') if response else 'NO_RESPONSE',
                    response_data=response
                )
                
        except Exception as e:
            self.logger.error(f"❌ Exception placing fractional buy order for {symbol}: {e}")
            return OrderResult(
                success=False,
                message=f"Exception during order placement: {str(e)}",
                error_code="EXCEPTION"
            )
    
    def execute_full_share_buy(self, symbol: str, shares: int, 
                              entry_price: float, account_info: Dict) -> OrderResult:
        """
        Execute a full share buy order
        
        Args:
            symbol: Stock symbol to buy
            shares: Number of shares to buy
            entry_price: Target entry price
            account_info: Account information
            
        Returns:
            OrderResult: Result of the order attempt
        """
        try:
            dollar_amount = shares * entry_price
            
            # Check if live trading is enabled
            if not getattr(self.config, 'LIVE_TRADING_ENABLED', False):
                self.logger.info(f"🔒 SIMULATED FULL SHARE BUY for {symbol}: "
                               f"{shares} shares (${dollar_amount:.2f}) at ${entry_price:.2f}")
                self.logger.info("    (Live trading disabled in config - set LIVE_TRADING_ENABLED = True)")
                
                return OrderResult(
                    success=True,
                    order_id="SIMULATED_" + str(int(datetime.now().timestamp())),
                    message=f"Simulated full share buy order for {symbol}",
                    response_data={'simulated': True}
                )
            
            # Ask for confirmation if required
            if getattr(self.config, 'REQUIRE_TRADE_CONFIRMATION', True):
                self.logger.warning(f"⚠️  LIVE TRADE CONFIRMATION REQUIRED")
                self.logger.warning(f"   Symbol: {symbol}")
                self.logger.warning(f"   Type: FULL SHARE BUY")
                self.logger.warning(f"   Shares: {shares} (${dollar_amount:.2f})")
                self.logger.warning(f"   Price: ${entry_price:.2f}")
                
                try:
                    response = input("\n🔴 CONFIRM LIVE TRADE: Type 'YES' to proceed, anything else to cancel: ")
                    if response.upper() != 'YES':
                        return OrderResult(
                            success=False,
                            message="Trade cancelled by user",
                            error_code="USER_CANCELLED"
                        )
                except:
                    return OrderResult(
                        success=False,
                        message="Cannot confirm trade in automated mode",
                        error_code="NO_CONFIRMATION"
                    )
            
            self.logger.info(f"🔄 Placing LIVE FULL SHARE BUY order for {symbol}: "
                           f"{shares} shares (${dollar_amount:.2f}) at ${entry_price:.2f}")
            
            # Ensure we have a valid session before placing the order
            if not self._ensure_valid_session():
                return OrderResult(
                    success=False,
                    message="Session validation failed before order placement",
                    error_code="SESSION_INVALID"
                )
            
            # Get ticker ID
            ticker_id = self.wb.get_ticker(symbol)
            if not ticker_id:
                return OrderResult(
                    success=False, 
                    message=f"Could not get ticker ID for {symbol}",
                    error_code="TICKER_NOT_FOUND"
                )
            
            # Place the order with retry logic
            response = self._place_order_with_retry(
                tId=ticker_id,
                price=entry_price,
                action='BUY',
                orderType='LMT',  # Limit order for better control
                enforce='DAY',    # Day order
                quant=float(shares),  # Full share quantity
                outsideRegularTradingHour=False
            )
            
            # Parse response
            if response and 'success' in response and response['success']:
                order_id = response.get('orderId') or response.get('data', {}).get('orderId')
                
                self.logger.info(f"✅ FULL SHARE BUY order placed successfully for {symbol}")
                self.logger.info(f"   Order ID: {order_id}")
                self.logger.info(f"   Shares: {shares} (${dollar_amount:.2f})")
                self.logger.info(f"   Price: ${entry_price:.2f}")
                
                return OrderResult(
                    success=True,
                    order_id=str(order_id) if order_id else None,
                    message=f"Full share buy order placed for {symbol}",
                    response_data=response
                )
            else:
                error_msg = response.get('msg', 'Unknown error') if response else 'No response'
                self.logger.error(f"❌ Failed to place full share buy order for {symbol}: {error_msg}")
                self.logger.error(f"🔍 DEBUG - Full Webull response: {response}")
                
                return OrderResult(
                    success=False,
                    message=f"Order placement failed: {error_msg}",
                    error_code=response.get('code') if response else 'NO_RESPONSE',
                    response_data=response
                )
                
        except Exception as e:
            self.logger.error(f"❌ Exception placing full share buy order for {symbol}: {e}")
            return OrderResult(
                success=False,
                message=f"Exception during order placement: {str(e)}",
                error_code="EXCEPTION"
            )
    
    def execute_order_with_session_retry(self, symbol: str, action: str, quantity: float, 
                                        price: float, order_type: str = 'LMT', 
                                        retry_attempt: bool = False) -> OrderResult:
        """
        ENHANCED: Execute order with automatic session retry using login_automatically pattern
        
        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Number of shares (can be fractional)
            price: Price per share
            order_type: Order type (LMT, MKT)
            retry_attempt: Whether this is a retry attempt
            
        Returns:
            OrderResult: Result of the order execution
        """
        try:
            # Ensure we have a valid session before attempting order
            if not self._ensure_valid_session():
                return OrderResult(
                    success=False,
                    message="Session validation failed before order placement",
                    error_code="SESSION_INVALID"
                )
            
            # Get ticker ID
            ticker_id = self.wb.get_ticker(symbol)
            if not ticker_id:
                return OrderResult(
                    success=False, 
                    message=f"Could not get ticker ID for {symbol}",
                    error_code="TICKER_NOT_FOUND"
                )
            
            retry_text = " (RETRY)" if retry_attempt else ""
            self.logger.info(f"🔄 Placing {action} order for {symbol}{retry_text}")
            self.logger.info(f"   Quantity: {quantity}, Price: ${price:.2f}, Type: {order_type}")
            
            # Place the order
            response = self.wb.place_order(
                tId=ticker_id,
                price=price,
                action=action,
                orderType=order_type,
                enforce='DAY',
                quant=quantity,
                outsideRegularTradingHour=False
            )
            
            # Check if order succeeded
            if response and response.get('success'):
                order_id = (response.get('orderId') or 
                           response.get('data', {}).get('orderId') if response.get('data') else None)
                
                success_text = " (RETRY SUCCESS)" if retry_attempt else ""
                self.logger.info(f"✅ {action} order placed successfully for {symbol}{success_text}")
                self.logger.info(f"   Order ID: {order_id}")
                
                return OrderResult(
                    success=True,
                    order_id=str(order_id) if order_id else None,
                    message=f"{action} order placed for {symbol}",
                    response_data=response
                )
            else:
                error_msg = response.get('msg', 'Unknown error') if response else 'No response'
                self.logger.error(f"❌ {action} order failed for {symbol}: {error_msg}")
                
                # Check if it's a session issue and we haven't already retried
                if (not retry_attempt and response and 
                    ('session' in error_msg.lower() or 'expired' in error_msg.lower() or 'token' in error_msg.lower())):
                    
                    self.logger.warning("⚠️ Session issue detected, attempting fresh login and retry...")
                    
                    # Clear session to force fresh login
                    self.main_system.session_manager.clear_session()
                    
                    # Attempt fresh login using login_automatically
                    if self.main_system.login_manager.login_automatically():
                        self.logger.info("✅ Fresh login successful, retrying order...")
                        
                        # Save the refreshed session
                        self.main_system.session_manager.save_session(self.main_system.wb)
                        
                        # RETRY THE ORDER - this is the key enhancement!
                        return self.execute_order_with_session_retry(
                            symbol, action, quantity, price, order_type, retry_attempt=True
                        )
                    else:
                        self.logger.error("❌ Fresh login failed, cannot retry order")
                        return OrderResult(
                            success=False,
                            message="Session refresh failed - please check credentials",
                            error_code="SESSION_REFRESH_FAILED"
                        )
                elif retry_attempt:
                    self.logger.error(f"❌ Retry also failed for {symbol}, giving up")
                    
                return OrderResult(
                    success=False,
                    message=f"Order placement failed: {error_msg}",
                    error_code=response.get('code') if response else 'NO_RESPONSE',
                    response_data=response
                )
                
        except Exception as e:
            self.logger.error(f"❌ Exception executing order for {symbol}: {e}")
            return OrderResult(
                success=False,
                message=f"Exception during order execution: {str(e)}",
                error_code="EXCEPTION"
            )

    def execute_signal(self, signal, account_info: Dict) -> OrderResult:
        """
        Execute a trading signal (buy or sell) using enhanced session retry logic
        
        Args:
            signal: MarketSignal object with position sizing info
            account_info: Account information
            
        Returns:
            OrderResult: Result of the order attempt
        """
        trade_signal = signal.trade_signal
        pos_sizing = signal.position_sizing
        
        # Only handle buy signals for now
        if trade_signal.action.value.upper() != 'BUY':
            return OrderResult(
                success=False,
                message=f"Order type {trade_signal.action.value} not supported yet",
                error_code="UNSUPPORTED_ACTION"
            )
        
        # Get quantity and price
        is_fractional = pos_sizing.get('is_fractional', False)
        dollar_amount = pos_sizing.get('dollar_amount', 0)
        
        if is_fractional:
            quantity = pos_sizing.get('fractional_shares', dollar_amount / trade_signal.entry_price)
        else:
            quantity = pos_sizing.get('max_shares', 0)
        
        # Use the enhanced order execution with session retry
        return self.execute_order_with_session_retry(
            symbol=trade_signal.symbol,
            action='BUY',
            quantity=quantity,
            price=trade_signal.entry_price,
            order_type='LMT'
        )
    
    def validate_order_preconditions(self, signal, account_info: Dict) -> Tuple[bool, str]:
        """
        Validate that we can place this order
        
        Args:
            signal: MarketSignal object
            account_info: Account information
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pos_sizing = signal.position_sizing
        dollar_amount = pos_sizing.get('dollar_amount', 0)
        available_cash = account_info.get('available_cash', 0)
        
        # Check if we have enough cash
        if dollar_amount > available_cash:
            return False, f"Insufficient funds: Need ${dollar_amount:.2f}, have ${available_cash:.2f}"
        
        # Check minimum trade amount
        if dollar_amount < 6.0:
            return False, f"Trade amount ${dollar_amount:.2f} below minimum $6.00"
        
        # Check if market is open (basic check)
        # You could add more sophisticated market hours checking here
        
        return True, "Order validation passed"
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get the status of a placed order
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Dict: Order status information
        """
        try:
            orders = self.wb.get_current_orders()
            if orders and 'data' in orders:
                for order in orders['data']:
                    if str(order.get('orderId')) == str(order_id):
                        return order
            return {}
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {}