# wyckoff_bot/wyckoff_trader.py
"""
Wyckoff Trader - Main Integration Class
=======================================
Integrates Wyckoff bot with existing trading system
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

# Import existing system components
from auth.credentials import CredentialManager
from auth.login_manager import LoginManager
from auth.session_manager import SessionManager
from accounts.account_manager import AccountManager
from config.config import PersonalTradingConfig

# Import Wyckoff components
from .signals.wyckoff_signals import WyckoffSignalGenerator, MarketSignal
from .signals.market_scanner import MarketScanner
from .signals.signal_validator import SignalValidator
from .data.data_manager import WyckoffDataManager
from .data.market_data import MarketDataProvider
from .execution.trade_executor import TradeExecutor

class WyckoffTrader:
    """
    Main Wyckoff trading system integration
    Connects Wyckoff analysis with existing trading infrastructure
    """
    
    def __init__(self, webull_client=None, account_manager=None, 
                 config=None, logger=None, main_system=None):
        
        self.logger = logger or logging.getLogger(__name__)
        self.wb = webull_client
        self.account_manager = account_manager
        self.config = config or PersonalTradingConfig()
        self.main_system = main_system
        
        # Initialize Wyckoff components with fractional trading support
        min_trade_amount = getattr(self.config, 'MIN_FRACTIONAL_TRADE_AMOUNT', 6.0)
        self.signal_generator = WyckoffSignalGenerator(
            logger=self.logger, 
            min_trade_amount=min_trade_amount
        )
        self.market_scanner = MarketScanner(logger=self.logger)
        self.signal_validator = SignalValidator(logger=self.logger)
        self.data_manager = WyckoffDataManager(logger=self.logger)
        self.market_data = MarketDataProvider(logger=self.logger)
        self.trade_executor = TradeExecutor(webull_client, account_manager, config=self.config, logger=self.logger, main_system=self.main_system)
        
        # Trading state
        self.is_active = False
        self.watchlist = []
        self.active_signals = {}
        self.last_scan_time = None
        
    def initialize(self) -> bool:
        """Initialize the Wyckoff trading system"""
        try:
            self.logger.info("Initializing Wyckoff Trading System...")
            
            # Validate required components
            if not self.wb:
                self.logger.error("Webull client not provided")
                return False
            
            if not self.account_manager:
                self.logger.error("Account manager not provided") 
                return False
            
            # Load initial watchlist from scan
            self.logger.info("Performing initial market scan...")
            scan_results = self.market_scanner.scan_market()
            
            if scan_results:
                # Create watchlist from top scan results
                self.watchlist = self.market_scanner.create_watchlist(scan_results, max_symbols=20)
                self.logger.info(f"Created watchlist with {len(self.watchlist)} symbols")
            else:
                # Use default watchlist if scan fails
                self.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                self.logger.warning("Using default watchlist due to scan failure")
            
            self.is_active = True
            self.logger.info("✅ Wyckoff Trading System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Wyckoff system: {e}")
            return False
    
    def run_trading_cycle(self) -> Dict:
        """
        Run one complete trading cycle
        
        Returns:
            Dict: Cycle results and statistics
        """
        if not self.is_active:
            return {'error': 'System not initialized'}
        
        try:
            cycle_start = time.time()
            self.logger.info("🔄 Starting Wyckoff trading cycle")
            
            # Step 1: Get current account info
            account_info = self._get_account_info()
            if not account_info:
                return {'error': 'Could not get account information'}
            
            # Step 2: Get current positions
            current_positions = self._get_current_positions()
            
            # Step 3: Fetch market data for watchlist
            market_data = self.market_data.get_multiple_symbols(self.watchlist)
            
            if not market_data:
                self.logger.warning("No market data available")
                return {'error': 'No market data available'}
            
            # Step 4: Generate signals
            signals = self.signal_generator.generate_signals(
                market_data, account_info, current_positions
            )
            
            # Step 5: Validate signals
            validation_results = self.signal_validator.batch_validate_signals(
                [s.trade_signal for s in signals], market_data
            )
            
            # Step 6: Filter valid signals
            valid_signals = [
                s for s in signals 
                if validation_results.get(s.symbol) and validation_results[s.symbol].is_valid
            ]
            
            # Step 7: Execute trades (if any)
            execution_results = self._execute_signals(valid_signals, account_info)
            
            # Step 8: Update database
            self._save_cycle_results(signals, validation_results, execution_results)
            
            # Step 9: Prepare results
            cycle_time = time.time() - cycle_start
            results = {
                'cycle_time': cycle_time,
                'watchlist_size': len(self.watchlist),
                'data_symbols': len(market_data),
                'total_signals': len(signals),
                'valid_signals': len(valid_signals),
                'executed_trades': len(execution_results.get('executed', [])),
                'account_balance': account_info.get('balance', 0),
                'active_positions': len(current_positions),
                'top_signals': [s.symbol for s in signals[:5]]
            }
            
            self.logger.info(f"✅ Cycle completed in {cycle_time:.1f}s - "
                           f"{results['valid_signals']} signals, "
                           f"{results['executed_trades']} trades")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return {'error': str(e)}
    
    def _get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            if not self.account_manager or not self.account_manager.accounts:
                return {}
            
            # Get primary trading account
            enabled_accounts = self.account_manager.get_enabled_accounts()
            if not enabled_accounts:
                return {}
            
            primary_account = enabled_accounts[0]
            
            return {
                'balance': primary_account.net_liquidation,
                'available_cash': primary_account.settled_funds,
                'account_type': primary_account.account_type,
                'day_trading_enabled': primary_account.can_day_trade(self.config)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def _get_current_positions(self) -> Dict:
        """Get current positions"""
        try:
            # This would integrate with the existing position tracking
            # For now, return empty dict as placeholder
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def _execute_signals(self, signals: List[MarketSignal], account_info: Dict) -> Dict:
        """
        Execute trading signals with REAL order placement
        """
        executed = []
        failed = []
        
        for signal in signals[:3]:  # Limit to top 3 signals
            try:
                # Get position sizing details
                pos_sizing = signal.position_sizing
                is_fractional = pos_sizing.get('is_fractional', False)
                dollar_amount = pos_sizing.get('dollar_amount', 0)
                fractional_shares = pos_sizing.get('fractional_shares', 0)
                
                # Validate order preconditions
                is_valid, validation_msg = self.trade_executor.validate_order_preconditions(signal, account_info)
                if not is_valid:
                    self.logger.warning(f"❌ Order validation failed for {signal.symbol}: {validation_msg}")
                    failed.append({
                        'symbol': signal.symbol,
                        'error': f"Validation failed: {validation_msg}"
                    })
                    continue
                
                # Execute the actual order
                self.logger.info(f"🔄 EXECUTING LIVE ORDER for {signal.symbol}")
                order_result = self.trade_executor.execute_signal(signal, account_info)
                
                # Save signal to database
                signal_id = self.data_manager.save_wyckoff_signal(
                    signal.trade_signal, signal.strength_score
                )
                
                if order_result.success:
                    self.logger.info(f"✅ LIVE ORDER PLACED SUCCESSFULLY for {signal.symbol}")
                    self.logger.info(f"   Order ID: {order_result.order_id}")
                    
                    executed.append({
                        'signal_id': signal_id,
                        'symbol': signal.symbol,
                        'action': signal.trade_signal.action.value,
                        'price': signal.trade_signal.entry_price,
                        'size': signal.trade_signal.position_size,
                        'dollar_amount': dollar_amount,
                        'fractional_shares': fractional_shares,
                        'is_fractional': is_fractional,
                        'order_id': order_result.order_id,
                        'order_status': 'PLACED'
                    })
                else:
                    # Trade executor already logged the detailed error, just track the failure
                    failed.append({
                        'symbol': signal.symbol,
                        'error': f"Order placement failed: {order_result.message}",
                        'error_code': order_result.error_code
                    })
                
            except Exception as e:
                self.logger.error(f"❌ Exception executing signal for {signal.symbol}: {e}")
                failed.append({
                    'symbol': signal.symbol,
                    'error': f"Exception: {str(e)}"
                })
        
        return {
            'executed': executed,
            'failed': failed
        }
    
    def _save_cycle_results(self, signals: List[MarketSignal], 
                           validation_results: Dict, execution_results: Dict):
        """Save cycle results to database"""
        try:
            # Save analysis for each signal
            for signal in signals:
                analysis_data = {
                    'phase': signal.trade_signal.reasoning.split()[0].lower(),
                    'confidence': signal.trade_signal.confidence,
                    'volume_confirmation': True,  # Simplified
                    'price_action_strength': signal.strength_score / 100,
                    'trend_strength': signal.strength_score / 100,
                    'key_events': [],
                    'support_level': signal.trade_signal.stop_loss,
                    'resistance_level': signal.trade_signal.take_profit
                }
                
                self.data_manager.save_wyckoff_analysis(signal.symbol, analysis_data)
                
        except Exception as e:
            self.logger.error(f"Error saving cycle results: {e}")
    
    def update_watchlist(self, rescan: bool = True) -> bool:
        """Update the watchlist with new scan results"""
        try:
            if rescan:
                self.logger.info("Rescanning market for new opportunities...")
                scan_results = self.market_scanner.scan_market()
                
                if scan_results:
                    new_watchlist = self.market_scanner.create_watchlist(
                        scan_results, max_symbols=25
                    )
                    
                    # Merge with existing watchlist, keeping best performers
                    combined = list(set(new_watchlist + self.watchlist))
                    self.watchlist = combined[:20]  # Keep top 20
                    
                    self.logger.info(f"Updated watchlist: {len(self.watchlist)} symbols")
                    self.last_scan_time = datetime.now()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating watchlist: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_active': self.is_active,
            'watchlist_size': len(self.watchlist),
            'active_signals': len(self.active_signals),
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'components': {
                'signal_generator': 'active',
                'market_scanner': 'active',
                'signal_validator': 'active',
                'data_manager': 'active',
                'market_data': 'active'
            }
        }
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for specified period"""
        try:
            return self.data_manager.get_wyckoff_performance_stats(days)
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def stop_system(self):
        """Stop the Wyckoff trading system"""
        self.is_active = False
        self.market_data.clear_cache()
        self.logger.info("🛑 Wyckoff Trading System stopped")
    
    def configure_strategy(self, config_updates: Dict):
        """Update strategy configuration"""
        try:
            self.signal_generator.update_strategy_parameters(config_updates)
            self.logger.info(f"Updated strategy configuration: {config_updates}")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    def export_data(self, output_path: str, days: int = 90):
        """Export trading data for analysis"""
        try:
            self.data_manager.export_wyckoff_data(output_path, days)
            self.logger.info(f"Data exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 180):
        """Clean up old data"""
        try:
            self.data_manager.cleanup_old_data(days_to_keep)
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")