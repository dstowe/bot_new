# wyckoff_main.py
"""
Wyckoff Trading Bot - Main Entry Point
======================================
Main executable for the Wyckoff trading bot using existing core system
"""

import logging
import sys
import time
import signal
import os
from datetime import datetime
from typing import Dict, Any

# Enable UTF-8 support for Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Import existing core system components
from main import MainSystem

# Import Wyckoff bot components
from wyckoff_bot.wyckoff_trader import WyckoffTrader

class WyckoffTradingSystem:
    """
    Complete Wyckoff trading system with existing infrastructure integration
    """
    
    def __init__(self):
        self.logger = None
        self.core_system = None
        self.wyckoff_trader = None
        self.is_running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self) -> bool:
        """Initialize the complete trading system"""
        try:
            print("🚀 WYCKOFF TRADING SYSTEM")
            print("=" * 50)
            
            # Step 1: Initialize core system (authentication, accounts, etc.)
            print("📋 Initializing core trading system...")
            self.core_system = MainSystem()
            
            if not self.core_system.run():
                print("❌ Core system initialization failed")
                return False
            
            self.logger = self.core_system.logger
            self.logger.info("✅ Core system initialized")
            
            # Step 2: Initialize Wyckoff trader
            print("🔍 Initializing Wyckoff trading system...")
            self.wyckoff_trader = WyckoffTrader(
                webull_client=self.core_system.wb,
                account_manager=self.core_system.account_manager,
                config=self.core_system.config,
                logger=self.logger,
                main_system=self.core_system
            )
            
            if not self.wyckoff_trader.initialize():
                print("❌ Wyckoff system initialization failed")
                return False
            
            self.logger.info("✅ Wyckoff trading system initialized")
            
            # Step 3: System ready
            print("\n🎉 SYSTEM READY FOR WYCKOFF TRADING")
            self._display_system_status()
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"System initialization failed: {e}")
            else:
                print(f"❌ System initialization failed: {e}")
            return False
    
    def run_continuous_trading(self, cycle_interval: int = 300) -> None:
        """
        Run continuous trading with specified cycle interval
        
        Args:
            cycle_interval: Seconds between trading cycles (default: 5 minutes)
        """
        if not self.wyckoff_trader:
            self.logger.error("System not initialized")
            return
        
        self.is_running = True
        cycle_count = 0
        
        self.logger.info(f"🔄 Starting continuous trading (cycle interval: {cycle_interval}s)")
        
        try:
            while self.is_running:
                cycle_count += 1
                cycle_start = time.time()
                
                self.logger.info(f"📊 Starting trading cycle #{cycle_count}")
                
                # Run trading cycle
                cycle_results = self.wyckoff_trader.run_trading_cycle()
                
                # Log results
                self._log_cycle_results(cycle_results, cycle_count)
                
                # Update watchlist periodically (every 10 cycles)
                if cycle_count % 10 == 0:
                    self.wyckoff_trader.update_watchlist(rescan=True)
                
                # Calculate sleep time
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, cycle_interval - cycle_time)
                
                if sleep_time > 0:
                    self.logger.info(f"⏱️  Sleeping for {sleep_time:.1f}s until next cycle")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Received interrupt signal")
        except Exception as e:
            self.logger.error(f"❌ Error in continuous trading: {e}")
        finally:
            self.is_running = False
            self.logger.info("🏁 Continuous trading stopped")
    
    def run_single_cycle(self) -> Dict[str, Any]:
        """Run a single trading cycle"""
        if not self.wyckoff_trader:
            return {'error': 'System not initialized'}
        
        self.logger.info("🔄 Running single Wyckoff trading cycle")
        
        results = self.wyckoff_trader.run_trading_cycle()
        self._log_cycle_results(results, 1)
        
        return results
    
    def interactive_mode(self) -> None:
        """Run in interactive mode with user commands"""
        if not self.wyckoff_trader:
            print("❌ System not initialized")
            return
        
        print("\n🎛️  INSTITUTIONAL INTERACTIVE MODE")
        print("Commands: cycle, status, performance, watchlist, config, export")  
        print("          institutional, emergency, pdt, regime, analytics, quit")
        
        while True:
            try:
                command = input("\nwyckoff> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'cycle':
                    results = self.run_single_cycle()
                    print(f"Cycle completed: {results.get('valid_signals', 0)} valid signals")
                elif command == 'status':
                    status = self.wyckoff_trader.get_system_status()
                    self._display_status(status)
                elif command == 'institutional':
                    # INSTITUTIONAL FEATURE - Enhanced status
                    institutional_status = self.wyckoff_trader.get_institutional_status()
                    self._display_institutional_status(institutional_status)
                elif command == 'performance':
                    perf = self.wyckoff_trader.get_performance_summary(days=30)
                    self._display_performance(perf)
                elif command == 'analytics':
                    # INSTITUTIONAL FEATURE - Full performance analytics
                    institutional_perf = self.wyckoff_trader.get_institutional_performance_summary(days=30)
                    self._display_institutional_performance(institutional_perf)
                elif command == 'watchlist':
                    print(f"Current watchlist ({len(self.wyckoff_trader.watchlist)} symbols):")
                    for i, symbol in enumerate(self.wyckoff_trader.watchlist, 1):
                        print(f"  {i:2d}. {symbol}")
                elif command == 'config':
                    self._interactive_config()
                elif command == 'export':
                    self.wyckoff_trader.export_data('exports', days=90)
                    print("📁 Data exported to 'exports' folder")
                elif command == 'emergency':
                    # INSTITUTIONAL FEATURE - Emergency mode status
                    self._display_emergency_status()
                elif command == 'pdt':
                    # INSTITUTIONAL FEATURE - PDT compliance status
                    self._display_pdt_status()
                elif command == 'regime':
                    # INSTITUTIONAL FEATURE - Market regime analysis
                    self._display_market_regime()
                elif command == 'help':
                    print("Available commands:")
                    print("  cycle       - Run single trading cycle")
                    print("  status      - Show basic system status")  
                    print("  institutional - Show institutional system status")
                    print("  performance - Show basic performance summary")
                    print("  analytics   - Show institutional performance analytics")
                    print("  watchlist   - Show current watchlist")
                    print("  config      - Update configuration")
                    print("  export      - Export trading data")
                    print("  emergency   - Show emergency mode status")
                    print("  pdt         - Show PDT compliance status")
                    print("  regime      - Show market regime analysis")
                    print("  quit        - Exit interactive mode")
                elif command:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Exiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.logger:
            self.logger.info(f"Received signal {signum}, shutting down...")
        else:
            print(f"\n🛑 Received signal {signum}, shutting down...")
        
        self.is_running = False
        
        if self.wyckoff_trader:
            self.wyckoff_trader.stop_system()
        
        if self.core_system:
            self.core_system.cleanup()
    
    def _display_system_status(self):
        """Display initial system status"""
        if self.wyckoff_trader:
            status = self.wyckoff_trader.get_system_status()
            print(f"📊 Watchlist: {status['watchlist_size']} symbols")
            print(f"🎯 Active signals: {status['active_signals']}")
            
        if self.core_system and self.core_system.account_manager:
            accounts = self.core_system.account_manager.get_enabled_accounts()
            print(f"💰 Trading accounts: {len(accounts)} enabled")
    
    def _log_cycle_results(self, results: Dict[str, Any], cycle_num: int):
        """Log trading cycle results"""
        if 'error' in results:
            self.logger.error(f"Cycle #{cycle_num} failed: {results['error']}")
            return
        
        self.logger.info(
            f"Cycle #{cycle_num} completed in {results.get('cycle_time', 0):.1f}s: "
            f"{results.get('valid_signals', 0)}/{results.get('total_signals', 0)} signals, "
            f"{results.get('executed_trades', 0)} trades executed"
        )
        
        if results.get('top_signals'):
            self.logger.info(f"Top signals: {', '.join(results['top_signals'])}")
    
    def _display_status(self, status: Dict):
        """Display system status"""
        print(f"System Active: {status['is_active']}")
        print(f"Watchlist Size: {status['watchlist_size']}")
        print(f"Active Signals: {status['active_signals']}")
        print(f"Last Scan: {status.get('last_scan', 'Never')}")
    
    def _display_performance(self, perf: Dict):
        """Display performance summary"""
        if 'error' in perf:
            print(f"Error getting performance: {perf['error']}")
            return
        
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Win Rate: {perf.get('win_rate', 0):.1%}")
        print(f"Total P&L: ${perf.get('total_pnl', 0):.2f}")
        if perf.get('profit_factor', 0) > 0:
            print(f"Profit Factor: {perf['profit_factor']:.2f}")
    
    def _interactive_config(self):
        """Interactive configuration update"""
        print("Configuration options:")
        print("1. Minimum confidence threshold")
        print("2. Risk per trade")
        print("3. Risk-reward ratio")
        
        try:
            choice = input("Select option (1-3): ").strip()
            
            if choice == '1':
                value = float(input("Enter minimum confidence (0.0-1.0): "))
                if 0 <= value <= 1:
                    self.wyckoff_trader.configure_strategy({'min_confidence': value})
                    print(f"✅ Updated minimum confidence to {value}")
            elif choice == '2':
                value = float(input("Enter risk per trade (0.01-0.10): "))
                if 0.01 <= value <= 0.10:
                    self.wyckoff_trader.configure_strategy({'risk_per_trade': value})
                    print(f"✅ Updated risk per trade to {value:.1%}")
            elif choice == '3':
                value = float(input("Enter risk-reward ratio (1.0-5.0): "))
                if 1.0 <= value <= 5.0:
                    self.wyckoff_trader.configure_strategy({'risk_reward_ratio': value})
                    print(f"✅ Updated risk-reward ratio to {value}")
            else:
                print("Invalid option")
                
        except ValueError:
            print("❌ Invalid value entered")
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
    
    # ============================================================================
    # INSTITUTIONAL FEATURES - DISPLAY METHODS
    # ============================================================================
    
    def _display_institutional_status(self, status: Dict):
        """Display comprehensive institutional system status"""
        print("🏛️ INSTITUTIONAL SYSTEM STATUS")
        print("=" * 50)
        
        # Basic status
        print(f"System Active: {status['is_active']}")
        print(f"Watchlist Size: {status['watchlist_size']}")
        print(f"Active Signals: {status['active_signals']}")
        print(f"Last Scan: {status.get('last_scan', 'Never')}")
        
        # Institutional features
        if 'institutional_features' in status:
            features = status['institutional_features']
            print("\n🏛️ INSTITUTIONAL FEATURES:")
            print(f"  Emergency Mode: {'🚨 ACTIVE' if features['emergency_mode_active'] else '✅ Normal'}")
            print(f"  Market Regime: {features['market_regime'].title()} ({features['regime_confidence']:.1%} confidence)")
            print(f"  Regime Updated: {features.get('regime_last_updated', 'Never')}")
            print(f"  PDT Protection: {'✅ Active' if features['pdt_protection_active'] else '❌ Inactive'}")
            print(f"  VaR Risk Mgmt: {'✅ Active' if features['var_risk_management_active'] else '❌ Inactive'}")
            print(f"  Enhanced Wyckoff: {'✅ Active' if features['enhanced_wyckoff_active'] else '❌ Inactive'}")
            print(f"  Performance Analytics: {'✅ Active' if features['performance_analytics_active'] else '❌ Inactive'}")
    
    def _display_institutional_performance(self, perf: Dict):
        """Display institutional performance analytics"""
        if 'error' in perf:
            print(f"Error getting institutional performance: {perf['error']}")
            return
        
        print("📊 INSTITUTIONAL PERFORMANCE ANALYTICS")
        print("=" * 50)
        
        # Basic performance
        print(f"Total Trades: {perf.get('total_trades', 0)}")
        print(f"Win Rate: {perf.get('win_rate', 0):.1%}")
        print(f"Total P&L: ${perf.get('total_pnl', 0):.2f}")
        if perf.get('profit_factor', 0) > 0:
            print(f"Profit Factor: {perf['profit_factor']:.2f}")
        
        # Institutional analytics
        if 'institutional_analytics' in perf:
            analytics = perf['institutional_analytics']
            print("\n📈 RISK-ADJUSTED METRICS:")
            print(f"  Sharpe Ratio: {analytics.get('sharpe_ratio', 0):.2f}")
            print(f"  Sortino Ratio: {analytics.get('sortino_ratio', 0):.2f}")
            print(f"  Calmar Ratio: {analytics.get('calmar_ratio', 0):.2f}")
            print(f"  Max Drawdown: {analytics.get('max_drawdown', 0):.1%}")
            
            if analytics.get('var_95', 0) > 0:
                print(f"  VaR (95%): ${analytics['var_95']:.2f}")
                print(f"  VaR (99%): ${analytics['var_99']:.2f}")
            
            # Phase attribution
            if analytics.get('phase_attribution'):
                print("\n🔍 WYCKOFF PHASE ATTRIBUTION:")
                for phase, metrics in analytics['phase_attribution'].items():
                    if hasattr(metrics, 'total_return'):
                        print(f"  {phase.title()}: ${metrics.total_return:.2f} ({metrics.total_trades} trades)")
    
    def _display_emergency_status(self):
        """Display emergency mode status"""
        try:
            # Get emergency status from wyckoff trader
            emergency_active = getattr(self.wyckoff_trader, 'emergency_mode_active', False)
            
            print("🚨 EMERGENCY MODE STATUS")
            print("=" * 30)
            print(f"Emergency Mode: {'🚨 ACTIVE' if emergency_active else '✅ Normal'}")
            
            if hasattr(self.wyckoff_trader, 'emergency_manager'):
                # Try to get more detailed emergency status
                try:
                    # This would show recent emergency events, thresholds, etc.
                    print("📊 Emergency Thresholds:")
                    print(f"  Max Drawdown Limit: 15%")
                    print(f"  Daily Loss Limit: ${self.wyckoff_trader.config.MAX_DAILY_LOSS}")
                    print(f"  Portfolio Risk Limit: {self.wyckoff_trader.config.MAX_PORTFOLIO_RISK:.1%}")
                    
                except Exception as e:
                    print(f"Could not get detailed emergency status: {e}")
            
        except Exception as e:
            print(f"Error getting emergency status: {e}")
    
    def _display_pdt_status(self):
        """Display PDT compliance status"""
        try:
            print("🛡️ PDT COMPLIANCE STATUS")
            print("=" * 30)
            
            if hasattr(self.wyckoff_trader, 'pdt_manager'):
                # Get account info for PDT status
                account_info = self.wyckoff_trader._get_account_info()
                account_id = account_info.get('account_id', 'default')
                
                # Get compliance report
                try:
                    report = self.wyckoff_trader.pdt_manager.get_compliance_report(account_id)
                    
                    if 'error' not in report:
                        print(f"Compliance Score: {report['compliance_score']}/100")
                        
                        pdt_status = report.get('pdt_status', {})
                        print(f"Is PDT Account: {pdt_status.get('is_pdt_account', 'Unknown')}")
                        print(f"Day Trades This Week: {pdt_status.get('day_trades_this_week', 0)}")
                        print(f"Day Trades Remaining: {pdt_status.get('day_trades_remaining', 'N/A')}")
                        
                        recommendations = report.get('recommendations', [])
                        if recommendations:
                            print("\n💡 Recommendations:")
                            for rec in recommendations[:3]:
                                print(f"  - {rec}")
                    else:
                        print(f"Error getting PDT report: {report['error']}")
                        
                except Exception as e:
                    print(f"Error getting PDT compliance report: {e}")
            else:
                print("PDT manager not available")
                
        except Exception as e:
            print(f"Error getting PDT status: {e}")
    
    def _display_market_regime(self):
        """Display market regime analysis"""
        try:
            print("🌍 MARKET REGIME ANALYSIS")
            print("=" * 30)
            
            if hasattr(self.wyckoff_trader, 'current_market_regime') and self.wyckoff_trader.current_market_regime:
                regime = self.wyckoff_trader.current_market_regime
                
                print(f"Current Regime: {regime.regime.value.title()}")
                print(f"Confidence: {regime.confidence:.1%}")
                
                if hasattr(regime, 'trend_strength'):
                    print(f"Trend Strength: {regime.trend_strength.value.replace('_', ' ').title()}")
                
                if hasattr(regime, 'volatility_regime'):
                    print(f"Volatility: {regime.volatility_regime.title()}")
                    
                print(f"Recommended Cash Allocation: {regime.cash_allocation_recommendation:.1%}")
                
                # Show when last updated
                if hasattr(self.wyckoff_trader, 'regime_last_updated') and self.wyckoff_trader.regime_last_updated:
                    print(f"Last Updated: {self.wyckoff_trader.regime_last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show sector rotation if available
                if hasattr(regime, 'sector_rotation_signal') and regime.sector_rotation_signal:
                    sorted_sectors = sorted(regime.sector_rotation_signal.items(), key=lambda x: x[1], reverse=True)
                    top_3 = sorted_sectors[:3]
                    print(f"\n🔄 Top Performing Sectors:")
                    for sector, score in top_3:
                        print(f"  {sector}: {score:.2f}")
                
            else:
                print("Market regime analysis not available or not yet updated")
                print("Run a trading cycle to update regime analysis")
                
        except Exception as e:
            print(f"Error displaying market regime: {e}")


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        mode = 'interactive'  # Default mode
        
        if len(sys.argv) > 1:
            if sys.argv[1] == '--continuous':
                mode = 'continuous'
            elif sys.argv[1] == '--single':
                mode = 'single'
            elif sys.argv[1] == '--help':
                print("Wyckoff Trading Bot")
                print("Usage:")
                print("  python wyckoff_main.py [--continuous|--single|--help]")
                print("")
                print("Modes:")
                print("  --continuous : Run continuous trading cycles")
                print("  --single     : Run single cycle and exit") 
                print("  (default)    : Interactive mode")
                return
        
        # Initialize system
        system = WyckoffTradingSystem()
        
        if not system.initialize():
            print("❌ System initialization failed")
            sys.exit(1)
        
        # Run in selected mode
        if mode == 'continuous':
            print("🔄 Running in continuous mode (Ctrl+C to stop)")
            system.run_continuous_trading(cycle_interval=300)  # 5 minutes
        elif mode == 'single':
            print("🔄 Running single cycle...")
            results = system.run_single_cycle()
            print(f"✅ Single cycle completed: {results}")
        else:
            system.interactive_mode()
        
        print("👋 Wyckoff Trading System shutdown complete")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()