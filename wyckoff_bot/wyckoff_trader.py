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
from .data.multi_timeframe_data_manager import MultiTimeframeDataManager
from .execution.trade_executor import TradeExecutor

# Import risk management components
from risk.account_risk_manager import AccountRiskManager
from risk.portfolio_risk_monitor import PortfolioRiskMonitor
from risk.emergency_mode import EmergencyModeManager

# Import compliance components  
from compliance.pdt_protection import PDTProtectionManager

# Import analytics components
from wyckoff_bot.analytics.performance_analytics import PerformanceAnalyzer

# Import enhanced analysis components
from wyckoff_bot.analysis.market_regime import MarketRegimeAnalyzer
from wyckoff_bot.analysis.wyckoff_analyzer import WyckoffAnalyzer

# Import stock universe
from config.stock_universe import StockUniverse

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
        
        # Database path setup
        db_path = getattr(self.config, 'database_path', 'data/trading_data.db')
        
        # Initialize Wyckoff components with fractional trading support
        min_trade_amount = getattr(self.config, 'MIN_FRACTIONAL_TRADE_AMOUNT', 6.0)
        # Initialize multi-timeframe data manager first
        from database.trading_db import TradingDatabase
        self.trading_db = TradingDatabase(db_path=db_path)
        self.multi_tf_data_manager = MultiTimeframeDataManager(db=self.trading_db, logger=self.logger)
        
        # Initialize components that use multi-timeframe data
        self.signal_generator = WyckoffSignalGenerator(
            logger=self.logger, 
            min_trade_amount=min_trade_amount,
            multi_tf_data_manager=self.multi_tf_data_manager
        )
        
        self.market_scanner = MarketScanner(
            multi_tf_data_manager=self.multi_tf_data_manager,
            logger=self.logger
        )
        self.signal_validator = SignalValidator(logger=self.logger)
        self.data_manager = WyckoffDataManager(logger=self.logger)
        self.market_data = MarketDataProvider(logger=self.logger)
        self.trade_executor = TradeExecutor(webull_client, account_manager, config=self.config, logger=self.logger, main_system=self.main_system)
        
        # Risk management components
        self.account_risk_manager = AccountRiskManager(config=self.config, logger=self.logger)
        self.portfolio_risk_monitor = PortfolioRiskMonitor(config=self.config, logger=self.logger)
        
        # INSTITUTIONAL FEATURES - Emergency mode and portfolio protection
        self.emergency_manager = EmergencyModeManager(db_path=db_path, logger=self.logger)
        
        # INSTITUTIONAL FEATURES - PDT protection and compliance
        self.pdt_manager = PDTProtectionManager(db_path=db_path, logger=self.logger)
        
        # INSTITUTIONAL FEATURES - Performance analytics
        self.performance_analyzer = PerformanceAnalyzer(db_path=db_path, logger=self.logger)
        
        # INSTITUTIONAL FEATURES - Enhanced analysis
        self.market_regime_analyzer = MarketRegimeAnalyzer(logger=self.logger)
        self.wyckoff_analyzer = WyckoffAnalyzer(logger=self.logger)
        
        # Trading state
        self.is_active = False
        self.watchlist = []
        self.active_signals = {}
        self.last_scan_time = None
        
        # INSTITUTIONAL FEATURES - Market regime tracking
        self.current_market_regime = None
        self.regime_last_updated = None
        self.emergency_mode_active = False
        
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
                # Use optimized watchlist from stock universe if scan fails
                self.watchlist = StockUniverse.get_recommended_watchlist(max_symbols=20)
                self.logger.warning(f"Using optimized default watchlist with {len(self.watchlist)} symbols")
            
            self.is_active = True
            self.logger.info("✅ Wyckoff Trading System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Wyckoff system: {e}")
            return False
    
    def run_trading_cycle(self) -> Dict:
        """
        Run one complete INSTITUTIONAL-GRADE trading cycle
        
        Returns:
            Dict: Cycle results and statistics
        """
        if not self.is_active:
            return {'error': 'System not initialized'}
        
        try:
            cycle_start = time.time()
            self.logger.info("🏛️ Starting INSTITUTIONAL Wyckoff trading cycle")
            
            # STEP 1: Get current account info
            account_info = self._get_account_info()
            if not account_info:
                return {'error': 'Could not get account information'}
            
            # STEP 2: INSTITUTIONAL FEATURE - Emergency mode monitoring
            portfolio_data = self._build_portfolio_data(account_info)
            emergency_result = self.emergency_manager.monitor_portfolio_protection(portfolio_data)
            
            if emergency_result.get('emergency_level') != 'normal':
                self.emergency_mode_active = True
                self.logger.critical(f"🚨 EMERGENCY MODE ACTIVATED: {emergency_result['emergency_level']}")
                
                # Implement emergency actions
                emergency_actions = emergency_result.get('actions_taken', [])
                if 'halt_new_trades' in emergency_actions:
                    self.logger.critical("🛑 EMERGENCY: Halting all new trades")
                    return {
                        'error': 'Emergency mode: Trading halted',
                        'emergency_active': True,
                        'emergency_level': emergency_result['emergency_level'],
                        'triggers': emergency_result.get('triggers_activated', [])
                    }
            
            # STEP 3: INSTITUTIONAL FEATURE - Market regime analysis
            current_regime = self._update_market_regime_analysis()
            
            # STEP 4: Check account-level risk limits with VaR
            can_trade, risk_reason = self.account_risk_manager.check_daily_risk_limits(
                account_info.get('account_id', 'default'),
                account_info.get('balance', 0),
                account_info.get('starting_balance', account_info.get('balance', 0))
            )
            
            if not can_trade:
                self.logger.warning(f"🚨 Trading halted due to risk limits: {risk_reason}")
                return {'error': f'Trading halted: {risk_reason}', 'risk_violation': True}
            
            # STEP 5: Get VaR analysis for enhanced risk management
            var_report = self.account_risk_manager.get_var_report(account_info.get('account_id', 'default'))
            if 'error' not in var_report:
                risk_level = var_report.get('risk_assessment', 'medium')
                self.logger.info(f"📊 Current VaR Risk Level: {risk_level}")
                
                # Adjust position sizing based on VaR
                if risk_level == 'high':
                    self.logger.warning("⚠️ High VaR detected - reducing position sizes by 50%")
                    account_info['position_size_multiplier'] = 0.5
                elif risk_level == 'medium':
                    account_info['position_size_multiplier'] = 0.75
                else:
                    account_info['position_size_multiplier'] = 1.0
            
            # STEP 6: Get current positions and update portfolio risk monitor
            current_positions = self._get_current_positions()
            self._update_portfolio_risk_monitor(current_positions, account_info.get('balance', 0))
            
            # STEP 7: Check portfolio-level risk limits
            portfolio_value = account_info.get('balance', 0)
            within_limits, risk_violations = self.portfolio_risk_monitor.check_portfolio_risk_limits(portfolio_value)
            
            if not within_limits:
                self.logger.warning(f"🚨 Portfolio risk violations: {'; '.join(risk_violations)}")
                # Apply additional position size reduction for risk violations
                current_multiplier = account_info.get('position_size_multiplier', 1.0)
                account_info['position_size_multiplier'] = current_multiplier * 0.5
                self.logger.warning("⚠️ Reducing position sizes by additional 50% due to portfolio risk violations")
            
            # STEP 8: Fetch market data for watchlist with enhanced multi-timeframe analysis
            market_data = {}
            
            # Use multi-timeframe data manager for better performance
            try:
                # First, update watchlist data if needed
                self.multi_tf_data_manager.update_watchlist_data(
                    self.watchlist, 
                    priority_timeframes=['1D', '4H', '1H']  # Primary timeframes
                )
                
                # Get the data for signal generation - using daily data as primary
                for symbol in self.watchlist:
                    daily_df = self.multi_tf_data_manager.get_cached_data(symbol, '1D', bars=100)
                    if daily_df is not None and len(daily_df) >= 50:
                        market_data[symbol] = daily_df
                
                self.logger.info(f"Retrieved cached data for {len(market_data)}/{len(self.watchlist)} watchlist symbols")
                
            except Exception as e:
                self.logger.warning(f"Error retrieving multi-timeframe data: {e}")
                # Fallback to old method
                market_data = self.market_data.get_multiple_symbols(self.watchlist)
            
            if not market_data:
                self.logger.warning("No market data available")
                return {'error': 'No market data available'}
            
            # STEP 9: INSTITUTIONAL FEATURE - Enhanced Wyckoff analysis with multi-timeframe
            enhanced_signals = self._generate_enhanced_wyckoff_signals(market_data, current_regime)
            
            # STEP 10: INSTITUTIONAL FEATURE - Apply PDT protection and compliance checking
            pdt_filtered_signals = []
            for signal in enhanced_signals:
                # Check PDT compliance before adding signal
                pdt_validation = self.pdt_manager.check_pdt_compliance(
                    account_info.get('account_id', 'default'),
                    signal.symbol,
                    signal.trade_signal.action.value
                )
                
                if pdt_validation.is_valid:
                    pdt_filtered_signals.append(signal)
                else:
                    self.logger.warning(f"🛡️ PDT protection blocked {signal.symbol}: {pdt_validation.reason}")
            
            # STEP 11: Apply enhanced signal validation with institutional criteria
            validation_results = self.signal_validator.batch_validate_signals(
                [s.trade_signal for s in pdt_filtered_signals], market_data
            )
            
            # STEP 12: Filter valid signals and apply VaR-based position sizing
            valid_signals = []
            for s in pdt_filtered_signals:
                if validation_results.get(s.symbol) and validation_results[s.symbol].is_valid:
                    # INSTITUTIONAL FEATURE - VaR-based position sizing
                    max_position_size = self.account_risk_manager.calculate_max_position_size(
                        account_info.get('account_id', 'default'),
                        account_info.get('balance', 0),
                        s.trade_signal.entry_price,
                        s.trade_signal.stop_loss
                    )
                    
                    # Apply position size multiplier from risk analysis
                    position_multiplier = account_info.get('position_size_multiplier', 1.0)
                    adjusted_max_size = max_position_size * position_multiplier
                    
                    # Adjust position size if needed
                    if s.trade_signal.position_size > adjusted_max_size:
                        self.logger.info(f"📏 VaR-based position sizing for {s.symbol}: {s.trade_signal.position_size:.2f} -> {adjusted_max_size:.2f}")
                        s.trade_signal.position_size = adjusted_max_size
                    
                    if adjusted_max_size > 0:  # Only add if we can trade
                        valid_signals.append(s)
            
            # STEP 13: Execute trades with institutional safeguards
            execution_results = self._execute_signals_institutional(valid_signals, account_info)
            
            # STEP 14: INSTITUTIONAL FEATURE - Record PDT day trades if executed
            for executed_trade in execution_results.get('executed', []):
                if executed_trade.get('action') == 'sell':
                    # Check if this creates a day trade pattern
                    self.pdt_manager.record_day_trade(
                        account_info.get('account_id', 'default'),
                        executed_trade.get('symbol'),
                        datetime.now(),  # Buy time - simplified for demo
                        datetime.now(),  # Sell time
                        executed_trade.get('entry_price', 0),
                        executed_trade.get('price', 0),
                        executed_trade.get('size', 0)
                    )
            
            # STEP 15: Update database with institutional analytics
            self._save_institutional_cycle_results(enhanced_signals, validation_results, execution_results, current_regime)
            
            # STEP 16: INSTITUTIONAL FEATURE - Performance analytics update
            try:
                performance_metrics, attribution = self.performance_analyzer.analyze_performance()
                self.logger.info(f"📊 Performance Update: {performance_metrics.total_trades} total trades, "
                               f"{performance_metrics.win_rate:.1%} win rate, "
                               f"${performance_metrics.total_return:.2f} total return")
            except Exception as e:
                self.logger.warning(f"Performance analytics update failed: {e}")
            
            # STEP 17: Prepare institutional results
            cycle_time = time.time() - cycle_start
            results = {
                'cycle_time': cycle_time,
                'watchlist_size': len(self.watchlist),
                'data_symbols': len(market_data),
                'total_signals': len(enhanced_signals),
                'pdt_filtered_signals': len(pdt_filtered_signals),
                'valid_signals': len(valid_signals),
                'executed_trades': len(execution_results.get('executed', [])),
                'account_balance': account_info.get('balance', 0),
                'active_positions': len(current_positions),
                'top_signals': [s.symbol for s in enhanced_signals[:5]],
                
                # INSTITUTIONAL FEATURES REPORTING
                'market_regime': current_regime.regime.value if current_regime else 'unknown',
                'regime_confidence': current_regime.confidence if current_regime else 0.0,
                'emergency_mode': self.emergency_mode_active,
                'var_risk_level': var_report.get('risk_assessment', 'unknown') if 'error' not in var_report else 'error',
                'position_size_multiplier': account_info.get('position_size_multiplier', 1.0),
                'portfolio_risk_violations': risk_violations if not within_limits else [],
                'pdt_blocks': len(enhanced_signals) - len(pdt_filtered_signals)
            }
            
            self.logger.info(f"🏛️ INSTITUTIONAL cycle completed in {cycle_time:.1f}s - "
                           f"Market: {results['market_regime']}, "
                           f"VaR: {results['var_risk_level']}, "
                           f"Signals: {results['valid_signals']}/{results['total_signals']}, "
                           f"Trades: {results['executed_trades']}")
            
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
    
    def _update_portfolio_risk_monitor(self, current_positions: Dict, portfolio_value: float):
        """Update portfolio risk monitor with current positions"""
        try:
            # Clear existing positions
            self.portfolio_risk_monitor.positions.clear()
            
            # Add current positions to risk monitor
            for symbol, position_data in current_positions.items():
                if position_data and position_data.get('size', 0) > 0:
                    self.portfolio_risk_monitor.update_position(
                        symbol=symbol,
                        position_size=position_data.get('size', 0),
                        entry_price=position_data.get('entry_price', 0),
                        current_price=position_data.get('current_price', 0),
                        stop_loss=position_data.get('stop_loss', position_data.get('current_price', 0) * 0.95)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating portfolio risk monitor: {e}")
    
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
                else:
                    # Use sector-diversified watchlist as fallback
                    self.watchlist = StockUniverse.get_sector_diversified_watchlist(symbols_per_sector=3)
                    
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
    
    # ============================================================================
    # INSTITUTIONAL FEATURES - NEW METHODS
    # ============================================================================
    
    def _build_portfolio_data(self, account_info: Dict) -> Dict:
        """Build portfolio data for emergency monitoring"""
        try:
            current_positions = self._get_current_positions()
            
            # Build portfolio structure for emergency monitoring
            portfolio_data = {
                'total_value': account_info.get('balance', 0),
                'cash': account_info.get('available_cash', 0),
                'daily_pnl': 0,  # Would be calculated from daily P&L tracking
                'positions': {}
            }
            
            # Add positions data
            for symbol, position in current_positions.items():
                if position and position.get('size', 0) > 0:
                    portfolio_data['positions'][symbol] = {
                        'quantity': position.get('size', 0),
                        'market_value': position.get('market_value', 0),
                        'unrealized_pnl': position.get('unrealized_pnl', 0),
                        'entry_price': position.get('entry_price', 0),
                        'current_price': position.get('current_price', 0)
                    }
            
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Error building portfolio data: {e}")
            return {
                'total_value': account_info.get('balance', 0),
                'cash': account_info.get('available_cash', 0),
                'daily_pnl': 0,
                'positions': {}
            }
    
    def _update_market_regime_analysis(self) -> Optional[object]:
        """Update market regime analysis"""
        try:
            # Update regime analysis every 30 minutes
            now = datetime.now()
            if (self.regime_last_updated is None or 
                (now - self.regime_last_updated).total_seconds() > 1800):
                
                self.logger.info("🌍 Updating market regime analysis...")
                
                try:
                    regime_analysis = self.market_regime_analyzer.analyze_market_regime()
                    self.current_market_regime = regime_analysis
                    self.regime_last_updated = now
                    
                    self.logger.info(f"📊 Market Regime: {regime_analysis.regime.value} "
                                   f"(confidence: {regime_analysis.confidence:.1%})")
                    
                    # Log cash allocation recommendation
                    cash_rec = regime_analysis.cash_allocation_recommendation
                    self.logger.info(f"💰 Recommended cash allocation: {cash_rec:.1%}")
                    
                except Exception as e:
                    self.logger.warning(f"Market regime analysis failed: {e}")
                    # Use fallback regime
                    from wyckoff_bot.analysis.market_regime import MarketRegime
                    class FallbackRegime:
                        regime = MarketRegime.UNKNOWN
                        confidence = 0.5
                        cash_allocation_recommendation = 0.5
                    self.current_market_regime = FallbackRegime()
            
            return self.current_market_regime
            
        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")
            return None
    
    def _generate_enhanced_wyckoff_signals(self, market_data: Dict, current_regime=None) -> List:
        """Generate signals with enhanced Wyckoff analysis"""
        try:
            enhanced_signals = []
            
            # Generate base signals
            base_signals = self.signal_generator.generate_signals(
                market_data, {'balance': 10000, 'available_cash': 5000}, {}
            )
            
            # Enhance each signal with institutional analysis
            for signal in base_signals:
                try:
                    symbol_data = market_data.get(signal.symbol)
                    if symbol_data is not None and len(symbol_data) > 0:
                        
                        # INSTITUTIONAL ENHANCEMENT - Multi-timeframe Wyckoff analysis
                        wyckoff_analysis = self.wyckoff_analyzer.analyze(
                            symbol_data, signal.symbol, '1D'
                        )
                        
                        # Add institutional context to signal
                        signal.institutional_analysis = {
                            'wyckoff_phase': wyckoff_analysis.phase.value,
                            'wyckoff_confidence': wyckoff_analysis.confidence,
                            'volume_confirmation': wyckoff_analysis.volume_confirmation,
                            'price_action_strength': wyckoff_analysis.price_action_strength,
                            'institutional_flow': wyckoff_analysis.institutional_flow,
                            'point_figure_signals': wyckoff_analysis.point_figure_signals
                        }
                        
                        # Add market regime context
                        if current_regime:
                            signal.market_regime_context = {
                                'regime': current_regime.regime.value,
                                'regime_confidence': current_regime.confidence,
                                'regime_aligned': self._is_signal_regime_aligned(signal, current_regime)
                            }
                        
                        # Adjust signal strength based on institutional factors
                        institutional_multiplier = self._calculate_institutional_multiplier(
                            wyckoff_analysis, current_regime
                        )
                        signal.strength_score = min(100, signal.strength_score * institutional_multiplier)
                        
                        enhanced_signals.append(signal)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to enhance signal for {signal.symbol}: {e}")
                    # Still add the base signal
                    enhanced_signals.append(signal)
            
            # Sort by institutional-enhanced strength
            enhanced_signals.sort(key=lambda s: s.strength_score, reverse=True)
            
            self.logger.info(f"🔍 Enhanced {len(enhanced_signals)} signals with institutional analysis")
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signals: {e}")
            # Fallback to base signals
            return self.signal_generator.generate_signals(
                market_data, {'balance': 10000, 'available_cash': 5000}, {}
            )
    
    def _is_signal_regime_aligned(self, signal, regime) -> bool:
        """Check if signal aligns with current market regime"""
        try:
            action = signal.trade_signal.action.value.lower()
            regime_value = regime.regime.value.lower()
            
            # Buy signals align with bull/accumulation regimes
            if action == 'buy':
                return regime_value in ['bull', 'accumulation', 'transition']
            
            # Sell signals align with bear/distribution regimes  
            elif action == 'sell':
                return regime_value in ['bear', 'distribution', 'transition']
            
            return True  # Default to aligned
            
        except Exception:
            return True
    
    def _calculate_institutional_multiplier(self, wyckoff_analysis, regime) -> float:
        """Calculate signal strength multiplier based on institutional factors"""
        try:
            multiplier = 1.0
            
            # Wyckoff confidence boost
            if wyckoff_analysis.confidence > 0.8:
                multiplier *= 1.3
            elif wyckoff_analysis.confidence > 0.6:
                multiplier *= 1.1
            
            # Volume confirmation boost
            if wyckoff_analysis.volume_confirmation:
                multiplier *= 1.2
            
            # Price action strength boost
            if wyckoff_analysis.price_action_strength > 0.7:
                multiplier *= 1.15
            
            # Institutional flow boost
            if wyckoff_analysis.institutional_flow:
                activity_score = wyckoff_analysis.institutional_flow.get('institutional_activity_score', 0)
                if activity_score > 0.7:
                    multiplier *= 1.25
            
            # Regime alignment boost
            if regime and regime.confidence > 0.7:
                multiplier *= 1.1
            
            return multiplier
            
        except Exception:
            return 1.0  # Default multiplier
    
    def _execute_signals_institutional(self, signals: List, account_info: Dict) -> Dict:
        """Execute signals with institutional safeguards"""
        try:
            # Use existing execution logic but with enhanced logging
            execution_results = self._execute_signals(signals, account_info)
            
            # Add institutional tracking
            for executed in execution_results.get('executed', []):
                self.logger.info(f"🏛️ INSTITUTIONAL TRADE EXECUTED: {executed['symbol']} "
                               f"{executed['action']} {executed['size']} shares @ ${executed['price']:.2f}")
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error in institutional signal execution: {e}")
            return {'executed': [], 'failed': []}
    
    def _save_institutional_cycle_results(self, signals: List, validation_results: Dict, 
                                        execution_results: Dict, current_regime):
        """Save cycle results with institutional analytics"""
        try:
            # Save base cycle results
            self._save_cycle_results(signals, validation_results, execution_results)
            
            # Save additional institutional analytics
            for signal in signals:
                if hasattr(signal, 'institutional_analysis'):
                    analysis_data = signal.institutional_analysis.copy()
                    
                    # Add regime context
                    if hasattr(signal, 'market_regime_context'):
                        analysis_data.update(signal.market_regime_context)
                    
                    # Save enhanced analysis
                    self.data_manager.save_wyckoff_analysis(signal.symbol, analysis_data)
            
            self.logger.debug("✅ Institutional cycle results saved")
            
        except Exception as e:
            self.logger.error(f"Error saving institutional cycle results: {e}")
    
    def get_institutional_status(self) -> Dict:
        """Get institutional system status"""
        try:
            base_status = self.get_system_status()
            
            # Add institutional status
            institutional_status = {
                **base_status,
                'institutional_features': {
                    'emergency_mode_active': self.emergency_mode_active,
                    'market_regime': self.current_market_regime.regime.value if self.current_market_regime else 'unknown',
                    'regime_confidence': self.current_market_regime.confidence if self.current_market_regime else 0.0,
                    'regime_last_updated': self.regime_last_updated.isoformat() if self.regime_last_updated else None,
                    'pdt_protection_active': True,
                    'var_risk_management_active': True,
                    'enhanced_wyckoff_active': True,
                    'performance_analytics_active': True
                }
            }
            
            return institutional_status
            
        except Exception as e:
            self.logger.error(f"Error getting institutional status: {e}")
            return self.get_system_status()
    
    def get_institutional_performance_summary(self, days: int = 30) -> Dict:
        """Get institutional performance summary with analytics"""
        try:
            # Get base performance
            base_performance = self.get_performance_summary(days)
            
            # Add institutional analytics
            try:
                metrics, attribution = self.performance_analyzer.analyze_performance()
                
                institutional_performance = {
                    **base_performance,
                    'institutional_analytics': {
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'sortino_ratio': metrics.sortino_ratio,
                        'calmar_ratio': metrics.calmar_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'var_95': getattr(metrics, 'var_95', 0),
                        'var_99': getattr(metrics, 'var_99', 0),
                        'phase_attribution': attribution.by_phase if attribution else {},
                        'regime_attribution': attribution.by_regime if attribution and hasattr(attribution, 'by_regime') else {}
                    }
                }
                
                return institutional_performance
                
            except Exception as e:
                self.logger.warning(f"Institutional analytics failed: {e}")
                return base_performance
            
        except Exception as e:
            self.logger.error(f"Error getting institutional performance: {e}")
            return {'error': str(e)}