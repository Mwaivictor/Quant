# MT5 Account Integration Guide

## Overview

The RPM can synchronize with your live MT5 account to automatically pull:
- **Account Balance** → `total_capital`
- **Account Equity** → `equity` 
- **Open Positions** → `open_positions`
- **Unrealized P/L** → `unrealized_pnl`

This eliminates manual configuration and ensures RPM risk calculations reflect your actual account state.

---

## Quick Start

### Option 1: Auto-sync on RPM startup

```python
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

# RPM will pull account data from MT5 on initialization
rpm = RiskPortfolioManager(
    sync_with_mt5=True  # Enable MT5 sync
)

print(f"Capital from MT5: ${rpm.portfolio_state.total_capital:,.2f}")
print(f"Equity from MT5: ${rpm.portfolio_state.equity:,.2f}")
```

### Option 2: Manual sync when needed

```python
rpm = RiskPortfolioManager()  # sync_with_mt5=False by default

# Manually sync with MT5 account
success = rpm.sync_with_mt5_account()

if success:
    print("Synced with MT5!")
    print(f"Balance: ${rpm.portfolio_state.total_capital:,.2f}")
```

### Option 3: Helper function for initial setup

```python
from arbitrex.risk_portfolio_manager import create_mt5_synced_portfolio

# Tries MT5 first, falls back to default_capital if MT5 unavailable
portfolio, syncer = create_mt5_synced_portfolio(default_capital=100000.0)

print(f"Portfolio capital: ${portfolio.total_capital:,.2f}")
```

---

## How It Works

### Data Flow

```
MT5 Terminal (Live Account)
         ↓
   mt5.account_info()
         ↓
    MT5AccountSync
         ↓
    PortfolioState
         ↓
   RPM Risk Calculations
```

### What Gets Synchronized

| MT5 Field | RPM Field | Description |
|-----------|-----------|-------------|
| `account.balance` | `total_capital` | Account balance (deposits - withdrawals + realized P/L) |
| `account.equity` | `equity` | Balance + unrealized P/L |
| `account.profit` | `unrealized_pnl` | Floating P/L from open positions |
| `positions_get()` | `open_positions` | All open positions with entry price, units, direction |

### Position Conversion

MT5 uses **lots**, RPM uses **units**:
- **1 standard lot = 100,000 units** (for FX pairs)
- Direction: MT5 type 0 (BUY) → RPM direction +1 (LONG)
- Direction: MT5 type 1 (SELL) → RPM direction -1 (SHORT)

```python
# Example: MT5 position of 0.5 lots EURUSD LONG
# Converts to: 50,000 units EURUSD direction=+1
```

---

## Prerequisites

### 1. MT5 Terminal Must Be Running

```bash
# Windows: Launch MetaTrader 5
# Must be logged into your broker account
```

### 2. MetaTrader5 Python Library

```bash
pip install MetaTrader5
```

### 3. MT5 Credentials in .env (Optional)

Only needed if you want to initialize MT5 from Python:

```bash
MT5_LOGIN=12345678
MT5_PASSWORD=YourPassword123
MT5_SERVER=MetaQuotes-Demo
MT5_TERMINAL=C:\Program Files\MetaTrader 5\terminal64.exe
```

**Note:** If MT5 terminal is already running and logged in, credentials are not required for account sync.

---

## API Reference

### `RiskPortfolioManager`

#### Constructor Parameter

```python
def __init__(
    self,
    config: Optional[RPMConfig] = None,
    enable_persistence: bool = True,
    sync_with_mt5: bool = False  # NEW
):
```

**Parameters:**
- `sync_with_mt5` (bool): If True, syncs with MT5 on startup

#### New Methods

##### `sync_with_mt5_account() -> bool`

Manually sync portfolio state with MT5.

```python
rpm = RiskPortfolioManager()

# Sync with MT5 account
success = rpm.sync_with_mt5_account()

if success:
    print(f"Capital: ${rpm.portfolio_state.total_capital:,.2f}")
    print(f"Positions: {len(rpm.portfolio_state.open_positions)}")
else:
    print("MT5 not available or sync failed")
```

**Returns:**
- `True`: Sync successful
- `False`: MT5 not initialized or sync failed

**Updates:**
- `portfolio_state.total_capital` ← MT5 balance
- `portfolio_state.equity` ← MT5 equity
- `portfolio_state.unrealized_pnl` ← MT5 profit
- `portfolio_state.open_positions` ← MT5 positions
- `portfolio_state.peak_equity` ← Updated if equity > peak
- `portfolio_state.current_drawdown` ← Recalculated

##### `get_mt5_sync_stats() -> Dict`

Get MT5 synchronization statistics.

```python
stats = rpm.get_mt5_sync_stats()

print(f"MT5 Available: {stats['mt5_available']}")
print(f"MT5 Initialized: {stats['mt5_initialized']}")
print(f"Last Sync: {stats['last_sync_time']}")
```

**Returns:**
```python
{
    'mt5_available': True,  # MetaTrader5 library installed
    'mt5_initialized': True,  # MT5 terminal running and connected
    'last_sync_time': '2025-12-23T15:30:45.123456'  # ISO timestamp or None
}
```

---

### `MT5AccountSync` Class

Low-level class for MT5 integration.

#### Constructor

```python
from arbitrex.risk_portfolio_manager import MT5AccountSync

syncer = MT5AccountSync()
```

#### Methods

##### `is_mt5_initialized() -> bool`

Check if MT5 is ready for use.

```python
if syncer.is_mt5_initialized():
    print("MT5 is ready")
else:
    print("MT5 not available - install library or start terminal")
```

##### `get_account_info() -> Optional[Dict]`

Get raw MT5 account information.

```python
account = syncer.get_account_info()

if account:
    print(f"Balance: ${account['balance']:,.2f}")
    print(f"Equity: ${account['equity']:,.2f}")
    print(f"Server: {account['server']}")
    print(f"Leverage: 1:{account['leverage']}")
```

**Returns:**
```python
{
    'balance': 100000.0,
    'equity': 100250.50,
    'margin': 2500.0,
    'margin_free': 97750.50,
    'margin_level': 4010.02,  # % (or None if no margin used)
    'profit': 250.50,
    'currency': 'USD',
    'leverage': 100,
    'login': 12345678,
    'server': 'MetaQuotes-Demo'
}
```

##### `get_open_positions() -> List[Dict]`

Get all open positions from MT5.

```python
positions = syncer.get_open_positions()

for pos in positions:
    direction = "LONG" if pos['type'] == 0 else "SHORT"
    print(f"{pos['symbol']}: {direction} {pos['volume']} lots")
    print(f"  Entry: {pos['price_open']:.5f}")
    print(f"  Current: {pos['price_current']:.5f}")
    print(f"  P/L: ${pos['profit']:,.2f}")
```

**Returns:**
```python
[
    {
        'ticket': 123456789,
        'symbol': 'EURUSD',
        'type': 0,  # 0=BUY/LONG, 1=SELL/SHORT
        'volume': 0.5,  # Lots
        'price_open': 1.10500,
        'price_current': 1.10650,
        'time': datetime(2025, 12, 23, 10, 30, 45),
        'profit': 75.00,
        'sl': 1.10000,  # Stop loss
        'tp': 1.11000,  # Take profit
        'comment': 'Manual trade'
    }
]
```

##### `sync_portfolio_state(portfolio_state: PortfolioState) -> bool`

Update a `PortfolioState` object with MT5 data.

```python
from arbitrex.risk_portfolio_manager import PortfolioState

portfolio = PortfolioState(total_capital=100000.0)

success = syncer.sync_portfolio_state(portfolio)

if success:
    print(f"Updated capital: ${portfolio.total_capital:,.2f}")
```

---

### Helper Function

#### `create_mt5_synced_portfolio(default_capital: float) -> Tuple[PortfolioState, MT5AccountSync]`

Create a portfolio that tries MT5 sync, falls back to default.

```python
from arbitrex.risk_portfolio_manager import create_mt5_synced_portfolio

portfolio, syncer = create_mt5_synced_portfolio(default_capital=100000.0)

# If MT5 available: portfolio.total_capital = MT5 balance
# If MT5 unavailable: portfolio.total_capital = 100000.0

print(f"Capital: ${portfolio.total_capital:,.2f}")
print(f"MT5 Status: {syncer.get_sync_stats()}")
```

**Returns:**
- `PortfolioState`: Portfolio with MT5 data (if available) or default
- `MT5AccountSync`: Syncer instance for future updates

---

## Usage Patterns

### Pattern 1: Production System with Live Account

```python
from arbitrex.risk_portfolio_manager import RiskPortfolioManager

# Initialize with MT5 sync enabled
rpm = RiskPortfolioManager(
    enable_persistence=True,  # Save state to disk
    sync_with_mt5=True,  # Pull from MT5 on startup
)

# Periodically re-sync (e.g., every hour)
import schedule

def sync_account():
    success = rpm.sync_with_mt5_account()
    if success:
        print(f"Synced: ${rpm.portfolio_state.total_capital:,.2f}")

schedule.every(1).hour.do(sync_account)
```

### Pattern 2: Backtesting / Paper Trading

```python
# Disable MT5 sync for backtesting
rpm = RiskPortfolioManager(
    config=RPMConfig(total_capital=100000.0),  # Fixed capital
    enable_persistence=False,
    sync_with_mt5=False,  # Don't use live account
)
```

### Pattern 3: Hybrid - Start from MT5, Track Changes Internally

```python
# Sync on startup to get current state
rpm = RiskPortfolioManager(sync_with_mt5=True)

# Process trades through RPM (updates portfolio_state)
output = rpm.process_trade_intent(...)

# RPM tracks positions internally
# Only re-sync if you want to override with MT5 state
```

### Pattern 4: Multi-Account Setup

```python
# Account 1: Live trading with MT5 sync
rpm_live = RiskPortfolioManager(sync_with_mt5=True)

# Account 2: Demo with fixed capital
rpm_demo = RiskPortfolioManager(
    config=RPMConfig(total_capital=10000.0),
    sync_with_mt5=False
)
```

---

## Troubleshooting

### Issue: "MT5 not initialized"

**Cause:** MT5 terminal not running or not logged in.

**Solution:**
1. Launch MetaTrader 5 terminal
2. Log into your broker account
3. Leave terminal running
4. Re-run your Python script

### Issue: "MetaTrader5 library not available"

**Cause:** Library not installed.

**Solution:**
```bash
pip install MetaTrader5
```

### Issue: Position units don't match

**Cause:** Contract size assumption.

**Solution:** Adjust `contract_size` in `mt5_sync.py`:

```python
# Default: 1 lot = 100,000 units (standard FX lot)
contract_size = 100000.0

# For indices or CFDs, may be different:
# contract_size = 1.0  # 1 lot = 1 unit
```

### Issue: Positions from manual MT5 trades not showing in RPM

**Check:**
1. Did you call `sync_with_mt5_account()`?
2. Is MT5 terminal running?
3. Are positions actually open in MT5?

```python
# Debug
positions = rpm.mt5_sync.get_open_positions()
print(f"MT5 positions: {len(positions)}")

rpm.sync_with_mt5_account()
print(f"RPM positions: {len(rpm.portfolio_state.open_positions)}")
```

---

## Benefits

### Without MT5 Sync (Original)
❌ Manual configuration of `total_capital`  
❌ Positions from manual MT5 trades not tracked  
❌ Capital changes (deposits/withdrawals) require code update  
❌ Unrealized P/L not reflected in risk calculations  

### With MT5 Sync (New)
✅ Automatic capital from live account balance  
✅ All positions tracked (RPM + manual trades)  
✅ Real-time equity and P/L  
✅ Accurate risk calculations on actual account state  
✅ Works with multi-strategy setups  
✅ No manual configuration needed  

---

## Examples

See demos:
- `demo_mt5_sync.py` - Full MT5 integration demo
- `demo_rpm_enhanced.py` - RPM features with optional MT5 sync

Run:
```bash
python demo_mt5_sync.py
```

---

## Architecture

```
┌─────────────────────┐
│   MT5 Terminal      │
│   (Live Account)    │
└──────────┬──────────┘
           │ MetaTrader5 Python API
           │
           ↓
┌─────────────────────┐
│   MT5AccountSync    │
│  - account_info()   │
│  - positions_get()  │
│  - lot → unit conv  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  PortfolioState     │
│  - total_capital    │
│  - equity           │
│  - open_positions   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ RiskPortfolioManager│
│  - Position sizing  │
│  - Risk constraints │
│  - Kill switches    │
└─────────────────────┘
```

---

## FAQ

**Q: Does RPM place trades in MT5?**  
A: No. RPM only **reads** account data. Execution is handled by a separate Execution Engine.

**Q: What happens if MT5 disconnects during trading?**  
A: RPM continues with last known state. Re-sync when MT5 reconnects.

**Q: Can I use this without MT5?**  
A: Yes. Set `sync_with_mt5=False` and configure `RPMConfig(total_capital=...)` manually.

**Q: Does this work with MT4?**  
A: No, only MT5. MT4 doesn't have the same Python API.

**Q: What brokers are supported?**  
A: Any broker offering MT5 terminal.

**Q: Is account data sent anywhere?**  
A: No. All data stays local. MT5 Python API is local-only.

---

## Version

MT5 sync feature added in **RPM v1.1.0** (December 2025)
