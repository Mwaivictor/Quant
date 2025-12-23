"""
Demo: MT5 Account Integration

Shows how RPM can sync with live MT5 account balance, equity, and positions.
"""

from arbitrex.risk_portfolio_manager import (
    RiskPortfolioManager,
    RPMConfig,
    MT5AccountSync,
    create_mt5_synced_portfolio,
)


def demo_mt5_sync():
    """Demonstrate MT5 account synchronization"""
    
    print("\n" + "="*80)
    print("MT5 ACCOUNT SYNCHRONIZATION DEMO")
    print("="*80)
    
    print("\n1. Check if MT5 is available and initialize if needed...")
    syncer = MT5AccountSync(auto_initialize=True)  # Will auto-initialize from .env
    
    if not syncer.mt5_available:
        print("  ✗ MetaTrader5 library not installed")
        print("    Install with: pip install MetaTrader5")
        return
    
    print("  ✓ MetaTrader5 library available")
    
    if not syncer.is_mt5_initialized():
        print("  ✗ MT5 initialization failed")
        print("    Check your .env file has MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
        print("    Or make sure MT5 terminal is running and logged in")
        return
    
    print("  ✓ MT5 initialized and connected")
    
    print("\n2. Fetch live account information...")
    account_info = syncer.get_account_info()
    
    if account_info:
        print(f"  Account: {account_info['login']} @ {account_info['server']}")
        print(f"  Balance: ${account_info['balance']:,.2f}")
        print(f"  Equity: ${account_info['equity']:,.2f}")
        print(f"  Margin: ${account_info['margin']:,.2f}")
        print(f"  Free Margin: ${account_info['margin_free']:,.2f}")
        if account_info['margin_level']:
            print(f"  Margin Level: {account_info['margin_level']:.2f}%")
        print(f"  Unrealized P/L: ${account_info['profit']:,.2f}")
        print(f"  Currency: {account_info['currency']}")
        print(f"  Leverage: 1:{account_info['leverage']}")
    else:
        print("  ✗ Failed to get account info")
        return
    
    print("\n3. Fetch open positions from MT5...")
    positions = syncer.get_open_positions()
    
    if positions:
        print(f"  Found {len(positions)} open position(s):")
        for pos in positions:
            direction = "LONG" if pos['type'] == 0 else "SHORT"
            print(f"    - {pos['symbol']}: {direction} {pos['volume']} lots @ {pos['price_open']:.5f}")
            print(f"      Current: {pos['price_current']:.5f}, P/L: ${pos['profit']:,.2f}")
    else:
        print("  No open positions")
    
    print("\n4. Create RPM with MT5 synchronization...")
    rpm = RiskPortfolioManager(
        enable_persistence=False,  # Disable for demo
        sync_with_mt5=True,  # Enable MT5 sync
    )
    
    print(f"\n  RPM Portfolio State:")
    print(f"    Total Capital: ${rpm.portfolio_state.total_capital:,.2f}")
    print(f"    Equity: ${rpm.portfolio_state.equity:,.2f}")
    print(f"    Unrealized P/L: ${rpm.portfolio_state.unrealized_pnl:,.2f}")
    print(f"    Open Positions: {len(rpm.portfolio_state.open_positions)}")
    
    if rpm.portfolio_state.open_positions:
        print(f"\n  Position Details:")
        for key, pos in rpm.portfolio_state.open_positions.items():
            direction = "LONG" if pos.direction == 1 else "SHORT"
            print(f"    - {pos.symbol}: {direction} {pos.units:,.0f} units @ {pos.entry_price:.5f}")
    
    print("\n5. Demonstrate manual sync...")
    print("  (Simulating after some time has passed)")
    success = rpm.sync_with_mt5_account()
    
    if success:
        print("  ✓ Manual sync successful")
        print(f"    Updated Capital: ${rpm.portfolio_state.total_capital:,.2f}")
        print(f"    Updated Equity: ${rpm.portfolio_state.equity:,.2f}")
    else:
        print("  ✗ Manual sync failed")
    
    print("\n6. Check sync statistics...")
    stats = rpm.get_mt5_sync_stats()
    print(f"  MT5 Available: {stats['mt5_available']}")
    print(f"  MT5 Initialized: {stats['mt5_initialized']}")
    print(f"  Auto-initialized: {stats['initialized_by_sync']}")
    print(f"  Last Sync: {stats['last_sync_time']}")
    
    print("\n" + "="*80)
    print("MT5 SYNC BENEFITS:")
    print("="*80)
    print("✓ Automatic MT5 initialization from .env credentials")
    print("✓ Real-time account balance and equity")
    print("✓ Automatic position tracking from MT5")
    print("✓ Accurate risk calculations based on actual account state")
    print("✓ No manual configuration of capital required")
    print("✓ Positions from manual trades or other systems included")
    print("\n" + "="*80)


def demo_without_mt5():
    """Demo without MT5 - shows fallback behavior"""
    
    print("\n" + "="*80)
    print("FALLBACK MODE: RPM WITHOUT MT5")
    print("="*80)
    
    print("\n1. Create portfolio with helper function...")
    portfolio, syncer = create_mt5_synced_portfolio(default_capital=100000.0)
    
    print(f"  Portfolio created with capital: ${portfolio.total_capital:,.2f}")
    print(f"  MT5 sync status: {syncer.get_sync_stats()}")
    
    print("\n2. Create RPM with default configuration...")
    config = RPMConfig(total_capital=100000.0)
    rpm = RiskPortfolioManager(
        config=config,
        enable_persistence=False,
        sync_with_mt5=False,  # Disabled since MT5 not available
    )
    
    print(f"  RPM initialized with static capital: ${rpm.portfolio_state.total_capital:,.2f}")
    
    print("\n  NOTE: Without MT5 sync, capital is static and must be manually configured.")
    print("  Positions from MT5 manual trades won't be tracked in RPM.")


if __name__ == "__main__":
    demo_mt5_sync()
