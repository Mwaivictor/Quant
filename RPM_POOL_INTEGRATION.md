# RPM + MT5 Pool Integration Guide

## The Right Way: Reuse Existing MT5ConnectionPool

Your system already has `MT5ConnectionPool` managing the MT5 connection. **RPM should reuse this connection**, not create a new one.

---

## Production Integration

### Your Existing Stack

```python
# arbitrex/scripts/run_streaming_stack.py (already exists)
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

pool = MT5ConnectionPool(sessions, symbols, ...)
pool.start_tick_collection()  # Already running
```

### Add RPM with Pool Integration

```python
# When initializing RPM in your production system:
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

rpm = RiskPortfolioManager(
    mt5_pool=pool,       # ← Pass your existing pool!
    sync_with_mt5=True,  # Pull account data from pool's connection
    enable_persistence=True,
)

# Now RPM syncs with MT5 using the pool's existing connection
print(f"Capital from MT5: ${rpm.portfolio_state.total_capital:,.2f}")
```

---

## How It Works

### Without Pool (BAD - Creates Redundant Connection)

```
MT5 Terminal
     ↓
Pool Connection (for ticks) ← Your system
     ↓
RPM Connection (for account) ← Redundant!
```

**Problems:**
- ❌ Two separate MT5 connections
- ❌ Double initialization
- ❌ RPM manages its own connection (more code)
- ❌ No automatic reconnection for RPM

### With Pool (GOOD - Shared Connection)

```
MT5 Terminal
     ↓
Pool Connection ← Single connection
     ├─→ Tick collection
     └─→ RPM account sync
```

**Benefits:**
- ✅ Single MT5 connection shared
- ✅ Pool manages lifecycle (heartbeat, reconnect)
- ✅ RPM just reads account data
- ✅ More efficient and reliable

---

## Code Changes Required

### Minimal Changes to Your Existing System

**1. When you create your streaming stack:**

```python
# File: your_main_system.py or run_streaming_stack.py

from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

# Existing code - no changes
pool = MT5ConnectionPool(sessions, symbols, ...)
pool.start_tick_collection()

# NEW: Create RPM with pool reference
rpm = RiskPortfolioManager(
    mt5_pool=pool,  # Just pass the pool!
    sync_with_mt5=True,
)

# Use RPM normally
output = rpm.process_trade_intent(...)
```

**2. That's it! No other changes needed.**

---

## API Reference

### RiskPortfolioManager Constructor

```python
def __init__(
    self,
    config: Optional[RPMConfig] = None,
    enable_persistence: bool = True,
    sync_with_mt5: bool = False,
    mt5_pool = None  # NEW parameter
):
```

**Parameters:**

- `mt5_pool` (Optional[MT5ConnectionPool]): 
  - **Production:** Pass your existing `MT5ConnectionPool` instance
  - **Testing/Demos:** Leave as `None` (RPM will auto-initialize if needed)

- `sync_with_mt5` (bool):
  - `True`: Sync portfolio state with MT5 account
  - `False`: Use static configuration

### MT5AccountSync Constructor

```python
MT5AccountSync(
    mt5_pool: Optional[MT5ConnectionPool] = None,
    auto_initialize: bool = True
)
```

**Behavior:**

| mt5_pool | auto_initialize | Result |
|----------|-----------------|--------|
| Provided | Any | Uses pool's connection ✅ |
| None | True | Initializes own connection (testing) |
| None | False | No MT5 sync available |

---

## Complete Example

### Full Production Setup

```python
# main.py - Your production system entry point

import os
from dotenv import load_dotenv
load_dotenv()

from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
from arbitrex.raw_layer.config import TRADING_UNIVERSE
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

def main():
    # 1. Setup MT5 connection pool (your existing code)
    sessions = {
        'main': {
            'terminal_path': os.environ.get('MT5_TERMINAL'),
            'login': int(os.environ['MT5_LOGIN']),
            'password': os.environ['MT5_PASSWORD'],
            'server': os.environ['MT5_SERVER'],
        }
    }
    
    symbols = [s for group in TRADING_UNIVERSE.values() for s in group]
    
    pool = MT5ConnectionPool(
        sessions,
        symbols=symbols,
        session_logs_dir='logs/mt5'
    )
    
    # 2. Start tick collection (your existing code)
    pool.start_tick_collection()
    print("✓ MT5 pool started")
    
    # 3. Create RPM with pool integration (NEW)
    rpm = RiskPortfolioManager(
        mt5_pool=pool,  # Share the pool!
        sync_with_mt5=True,
        enable_persistence=True,
    )
    
    print(f"✓ RPM initialized")
    print(f"  Capital: ${rpm.portfolio_state.total_capital:,.2f}")
    print(f"  Equity: ${rpm.portfolio_state.equity:,.2f}")
    
    # 4. Your trading loop
    while True:
        # Get signals from signal engine
        signal = get_signal()
        
        # Process through RPM
        output = rpm.process_trade_intent(
            symbol=signal['symbol'],
            direction=signal['direction'],
            confidence_score=signal['confidence'],
            regime=signal['regime'],
            atr=signal['atr'],
            vol_percentile=signal['vol_percentile'],
            current_price=signal['price'],
        )
        
        if output.decision.status == 'APPROVED':
            # Send to execution engine
            execute_trade(output.decision.approved_trade)
        
        # Optionally re-sync with MT5 periodically
        if should_sync():
            rpm.sync_with_mt5_account()

if __name__ == "__main__":
    main()
```

---

## Sync Behavior

### On Startup (sync_with_mt5=True)

```python
rpm = RiskPortfolioManager(mt5_pool=pool, sync_with_mt5=True)
# Automatically pulls:
# - Account balance → total_capital
# - Account equity
# - Open positions
# - Unrealized P/L
```

### Manual Re-sync

```python
# Sync with MT5 anytime
success = rpm.sync_with_mt5_account()

if success:
    print(f"Updated capital: ${rpm.portfolio_state.total_capital:,.2f}")
```

### What Gets Synced

| MT5 Data | RPM Field |
|----------|-----------|
| `account_info().balance` | `portfolio_state.total_capital` |
| `account_info().equity` | `portfolio_state.equity` |
| `account_info().profit` | `portfolio_state.unrealized_pnl` |
| `positions_get()` | `portfolio_state.open_positions` |

---

## Testing Mode (Without Pool)

For standalone testing/demos when pool isn't running:

```python
# RPM will auto-initialize its own MT5 connection from .env
rpm = RiskPortfolioManager(
    mt5_pool=None,  # No pool - will auto-init
    sync_with_mt5=True,
)

# Uses .env credentials:
# - MT5_LOGIN
# - MT5_PASSWORD
# - MT5_SERVER
# - MT5_TERMINAL (optional)
```

**Note:** This is **NOT recommended for production** - only for testing individual RPM functionality.

---

## Verification

### Check Sync Status

```python
stats = rpm.get_mt5_sync_stats()

print(f"Using pool: {stats['using_mt5_pool']}")  # Should be True
print(f"Initialized by sync: {stats['initialized_by_sync']}")  # Should be False (pool initialized it)
print(f"MT5 connected: {stats['mt5_initialized']}")  # Should be True
print(f"Last sync: {stats['last_sync_time']}")
```

Expected output in production:
```
Using pool: True          ← Good! Reusing pool
Initialized by sync: False ← Good! Pool manages connection
MT5 connected: True       ← Good! Working
Last sync: 2025-12-23T...
```

---

## Summary

### DO ✅
```python
# Pass pool to RPM
rpm = RiskPortfolioManager(mt5_pool=pool, sync_with_mt5=True)
```

### DON'T ❌
```python
# Create separate MT5 connection
rpm = RiskPortfolioManager(sync_with_mt5=True)  # No pool = redundant init
```

---

## Next Steps

1. ✅ **Update your main system file** to pass `mt5_pool` to RPM
2. ✅ **Remove any standalone MT5 initialization** in RPM code paths
3. ✅ **Test** that RPM shows correct account balance from MT5
4. ✅ **Verify** `using_mt5_pool=True` in sync stats

Your system will be more efficient with a single shared MT5 connection!
