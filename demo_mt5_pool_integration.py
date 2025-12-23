"""
Demo: MT5 Sync with Pool Integration

Shows the difference between:
1. Production mode: RPM uses existing MT5ConnectionPool (RECOMMENDED)
2. Standalone mode: RPM initializes own MT5 connection (testing only)
"""

from arbitrex.risk_portfolio_manager import RiskPortfolioManager
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
import os


def demo_with_pool():
    """PRODUCTION MODE: Use existing MT5ConnectionPool"""
    
    print("\n" + "="*80)
    print("PRODUCTION MODE: RPM WITH MT5 CONNECTION POOL")
    print("="*80)
    
    print("\n1. Create MT5ConnectionPool (already running in your system)...")
    
    # This is what your system already has running
    sessions = {
        'main': {
            'terminal_path': os.environ.get('MT5_TERMINAL') if os.environ.get('MT5_TERMINAL') else None,
            'login': int(os.environ.get('MT5_LOGIN', 0)),
            'password': os.environ.get('MT5_PASSWORD'),
            'server': os.environ.get('MT5_SERVER'),
        }
    }
    
    # Use symbols from your trading universe
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Example
    
    print("  Creating pool with session...")
    pool = MT5ConnectionPool(sessions, symbols, session_logs_dir='logs')
    
    try:
        # Get connection to ensure it's initialized
        name, session = pool.get_connection(timeout=10)
        print(f"  ✓ Pool connected: {name} (status={session.status})")
        
        print("\n2. Create RPM with pool reference (NO separate MT5 init!)...")
        rpm = RiskPortfolioManager(
            mt5_pool=pool,  # Pass existing pool - reuses connection!
            sync_with_mt5=True,
            enable_persistence=False,
        )
        
        print(f"\n  ✓ RPM initialized with pool")
        print(f"    Total Capital: ${rpm.portfolio_state.total_capital:,.2f}")
        print(f"    Equity: ${rpm.portfolio_state.equity:,.2f}")
        
        stats = rpm.get_mt5_sync_stats()
        print(f"\n3. Sync Statistics:")
        print(f"    Using MT5 Pool: {stats['using_mt5_pool']}")
        print(f"    Initialized by sync: {stats['initialized_by_sync']}")
        print(f"    MT5 Initialized: {stats['mt5_initialized']}")
        
        print("\n" + "="*80)
        print("PRODUCTION BENEFITS:")
        print("="*80)
        print("✓ Single MT5 connection shared across entire system")
        print("✓ No redundant MT5 initializations")
        print("✓ Pool manages connection lifecycle (heartbeat, reconnect)")
        print("✓ RPM just reads account data from existing connection")
        print("✓ More efficient and reliable")
        
    except Exception as e:
        print(f"\n  Error: {e}")
    finally:
        # Note: Pool cleanup happens automatically
        print("\n  Pool lifecycle managed by system")


def demo_standalone():
    """STANDALONE MODE: RPM initializes own connection (testing only)"""
    
    print("\n" + "="*80)
    print("STANDALONE MODE: RPM WITH OWN MT5 CONNECTION (Testing)")
    print("="*80)
    
    print("\n1. Create RPM without pool (initializes own connection)...")
    
    rpm = RiskPortfolioManager(
        mt5_pool=None,  # No pool - will initialize own connection
        sync_with_mt5=True,
        enable_persistence=False,
    )
    
    print(f"\n  ✓ RPM initialized with standalone connection")
    print(f"    Total Capital: ${rpm.portfolio_state.total_capital:,.2f}")
    print(f"    Equity: ${rpm.portfolio_state.equity:,.2f}")
    
    stats = rpm.get_mt5_sync_stats()
    print(f"\n2. Sync Statistics:")
    print(f"    Using MT5 Pool: {stats['using_mt5_pool']}")
    print(f"    Initialized by sync: {stats['initialized_by_sync']}")
    print(f"    MT5 Initialized: {stats['mt5_initialized']}")
    
    print("\n" + "="*80)
    print("STANDALONE LIMITATIONS:")
    print("="*80)
    print("⚠️  Creates separate MT5 connection (redundant if pool exists)")
    print("⚠️  No automatic reconnection (must handle manually)")
    print("⚠️  RPM manages connection lifecycle (more code)")
    print("✓  OK for testing/demos without full system")


def compare_modes():
    """Show comparison"""
    
    print("\n" + "="*80)
    print("COMPARISON: POOL vs STANDALONE")
    print("="*80)
    
    print("\n┌─────────────────────────┬──────────────────┬──────────────────┐")
    print("│ Feature                 │ With Pool        │ Standalone       │")
    print("├─────────────────────────┼──────────────────┼──────────────────┤")
    print("│ MT5 Connections         │ 1 (shared)       │ 2 (pool + RPM)   │")
    print("│ Reconnection Handling   │ Pool manages     │ Manual           │")
    print("│ Connection Heartbeat    │ Pool manages     │ Manual           │")
    print("│ Use Case                │ Production       │ Testing/Demos    │")
    print("│ Recommended             │ ✓ YES            │ ✗ No             │")
    print("└─────────────────────────┴──────────────────┴──────────────────┘")
    
    print("\n" + "="*80)
    print("RECOMMENDED USAGE IN YOUR SYSTEM:")
    print("="*80)
    
    print("""
# Your streaming stack already creates the pool:
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

pool = MT5ConnectionPool(sessions, symbols, ...)
pool.start_tick_collection()  # Already running

# When creating RPM, pass the pool reference:
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

rpm = RiskPortfolioManager(
    mt5_pool=pool,       # Reuse existing connection!
    sync_with_mt5=True,  # Sync account data
)

# RPM now uses pool's connection - no separate MT5 init!
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "pool":
            demo_with_pool()
        elif mode == "standalone":
            demo_standalone()
        elif mode == "compare":
            compare_modes()
        else:
            print("Usage: python demo_mt5_pool_integration.py [pool|standalone|compare]")
    else:
        print("\nDemo: MT5 Pool Integration")
        print("="*80)
        print("\nRun with:")
        print("  python demo_mt5_pool_integration.py pool       # Show pool mode")
        print("  python demo_mt5_pool_integration.py standalone # Show standalone mode")
        print("  python demo_mt5_pool_integration.py compare    # Show comparison")
        print("\n")
        
        # Show comparison by default
        compare_modes()
