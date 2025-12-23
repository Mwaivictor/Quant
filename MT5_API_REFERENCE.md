# MT5 API Integration - Execution Engine Reference

## Overview

The Execution Engine uses these **exact MT5 API calls** to execute orders and monitor fills.

---

## MT5 API Methods Used

### 1. mt5.order_send(request)

**Purpose**: Submit order to MT5 broker

**Usage in Execution Engine**:
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,      # Market order action
    "symbol": "EURUSD",                   # Symbol from RPM
    "volume": 95000,                      # Units from RPM exactly
    "type": mt5.ORDER_TYPE_BUY,           # BUY or SELL based on direction
    "price": 1.09500,                     # Current ask/bid price
    "deviation": 20,                      # 20 pips slippage tolerance
    "magic": 0,                           # Magic number for tracking
    "comment": "arbitrex_execution_a1b2c3d4"  # Execution ID
}

result = mt5.order_send(request)
```

**Returns**:
```python
result.retcode      # TRADE_RETCODE_DONE if successful
result.order        # Order ticket ID
result.deal         # Deal ID (if filled)
result.comment      # Error message if failed
```

**Used in**: Stage 4 - Order Submission

**Success Criteria**:
- `result.retcode == mt5.TRADE_RETCODE_DONE`
- `result.order > 0`

---

### 2. mt5.symbol_info_tick(symbol)

**Purpose**: Get current bid/ask prices

**Usage in Execution Engine**:
```python
tick = mt5.symbol_info_tick("EURUSD")

bid_price = tick.bid        # 1.09495
ask_price = tick.ask        # 1.09500
volume = tick.volume        # Tick volume
spread_pips = (ask_price - bid_price) / 0.0001  # In pips
```

**Used in**:
- Stage 1: Pre-execution validation (check spread)
- Stage 4: Get current price for order submission
- Stage 6: Measure slippage (compare fill price vs bid/ask)

**Error Handling**:
- If `tick is None`: Market data unavailable
- If `bid_price <= 0` or `ask_price <= 0`: Invalid data

---

### 3. mt5.orders_get()

**Purpose**: Query open orders

**Usage in Execution Engine**:
```python
# Get all open orders
total_orders = mt5.orders_total()

for i in range(total_orders):
    order = mt5.orders_get(index=i)
    
    if str(order.ticket) == order_id:
        # Found our order
        fill_price = order.price_open
        volume_current = order.volume_current
        state = order.state
        break
```

**Returns**:
```python
order.ticket           # Order ID
order.symbol           # Symbol
order.type             # ORDER_TYPE_BUY or ORDER_TYPE_SELL
order.volume_initial   # Original volume
order.volume_current   # Remaining volume
order.price_open       # Entry price
order.time_setup       # Open timestamp
order.state            # Order state
```

**Used in**: Stage 5 - Order Monitoring (check if still open)

---

### 4. mt5.history_deals_get()

**Purpose**: Query filled orders/deals

**Usage in Execution Engine**:
```python
# Check if order was filled (in history)
total_deals = mt5.history_deals_total()

for i in range(total_deals):
    deal = mt5.history_deals_get(index=i)
    
    if str(deal.ticket) == order_id:
        # Found our deal (filled order)
        fill_price = deal.price
        volume = deal.volume
        profit = deal.profit
        commission = deal.commission
        break
```

**Returns**:
```python
deal.ticket            # Deal ID
deal.symbol            # Symbol
deal.type              # DEAL_TYPE_BUY or DEAL_TYPE_SELL
deal.volume            # Filled volume
deal.price             # Fill price ← Used for slippage
deal.commission        # Commission paid
deal.profit            # Profit/loss
deal.time              # Fill timestamp
```

**Used in**: Stage 5 - Order Monitoring (check if filled)

---

### 5. mt5.account_info()

**Purpose**: Get account information

**Usage in Execution Engine**:
```python
account = mt5.account_info()

available_margin = account.margin_free    # Free margin
used_margin = account.margin_used         # Used margin
account_balance = account.balance         # Account balance
leverage = account.leverage               # Account leverage
```

**Used in**: Stage 1 - Pre-execution validation (check margin)

**Validation**:
```python
required_margin = (position_units * price) / leverage
available_margin > required_margin * 1.5  # 1.5x cushion
```

---

### 6. mt5.symbol_info(symbol)

**Purpose**: Get symbol information

**Usage in Execution Engine**:
```python
sym_info = mt5.symbol_info("EURUSD")

symbol_name = sym_info.name       # "EURUSD"
visible = sym_info.visible        # Is visible
tradable = sym_info.trade_mode    # Can trade (!=0)
bid = sym_info.bid                # Current bid
ask = sym_info.ask                # Current ask
```

**Used in**: Stage 1 - Pre-execution validation (check tradable)

**Validation**:
```python
if sym_info is None:
    return False  # Symbol not found

if not sym_info.visible:
    return False  # Symbol hidden

if sym_info.trade_mode == 0:
    return False  # Not tradable
```

---

### 7. mt5.ORDER_TYPE_BUY / mt5.ORDER_TYPE_SELL

**Purpose**: Constants for order direction

**Usage**:
```python
# For LONG (direction=1)
trade_type = mt5.ORDER_TYPE_BUY

# For SHORT (direction=-1)
trade_type = mt5.ORDER_TYPE_SELL
```

---

### 8. mt5.TRADE_ACTION_DEAL

**Purpose**: Constant for market order action

**Usage**:
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,  # Immediate market execution
    ...
}
```

---

## Execution Flow with MT5 Calls

```
┌─────────────────────────────────────────────────────────┐
│ ExecutionEngine.execute(rpm_output)                      │
│ Input: approved trade with symbol, direction, units      │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 1: VALIDATION            │
        ├────────────────────────────────┤
        │                                │
        │ ✓ mt5.symbol_info_tick()      │
        │   ↓ Check spread acceptable  │
        │                                │
        │ ✓ mt5.account_info()          │
        │   ↓ Check margin sufficient   │
        │                                │
        │ ✓ mt5.symbol_info()           │
        │   ↓ Check symbol tradable     │
        │                                │
        │ Result: APPROVE or REJECT      │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 4: SUBMIT ORDER          │
        ├────────────────────────────────┤
        │                                │
        │ Request = {                    │
        │   "action": TRADE_ACTION_DEAL, │
        │   "symbol": "EURUSD",          │
        │   "volume": 95000,             │
        │   "type": ORDER_TYPE_BUY,      │
        │   "price": ask_price,          │
        │   "deviation": 20,             │
        │   "comment": "arbitrex_..."    │
        │ }                              │
        │                                │
        │ ✓ mt5.order_send(request)     │
        │   ↓ Retry 3x on network error │
        │                                │
        │ Result: order_id or error      │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 5: MONITOR (Polling)     │
        ├────────────────────────────────┤
        │                                │
        │ Every 0.5s for 60s:            │
        │                                │
        │ ✓ mt5.orders_get()            │
        │   ↓ Check if order still open │
        │                                │
        │ ✓ mt5.history_deals_get()     │
        │   ↓ Check if deal filled      │
        │                                │
        │ Result: fill_price & units     │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 6: SLIPPAGE MEASURE      │
        ├────────────────────────────────┤
        │                                │
        │ slippage_pips =                │
        │   |fill_price - intended|     │
        │   ────────────────────── × 10 │
        │          0.0001               │
        │                                │
        │ Compare vs max_slippage_pips   │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 8: STORE IN DATABASE     │
        ├────────────────────────────────┤
        │ execution_logs table           │
        │ - execution_id                 │
        │ - order_id                     │
        │ - fill_price                   │
        │ - slippage_pips                │
        │ - timestamp                    │
        └────────┬──────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │ STAGE 9: RETURN CONFIRMATION   │
        ├────────────────────────────────┤
        │ ExecutionConfirmation {        │
        │   execution_id: "...",         │
        │   order_id: "...",             │
        │   status: FILLED,              │
        │   fill_price: 1.0950,          │
        │   slippage_pips: 0.5           │
        │ }                              │
        └────────────────────────────────┘
```

---

## Error Handling with MT5

### Network Failure
```python
try:
    result = mt5.order_send(request)
except Exception as e:
    # Network error, retry
    LOG.debug(f"Network error: {e}, retrying...")
    # Retry up to 3 times
```

### Broker Rejection
```python
result = mt5.order_send(request)
if result.retcode != mt5.TRADE_RETCODE_DONE:
    # Broker rejected
    error = result.comment
    LOG.error(f"Broker rejected: {error}")
    return False, None, error
```

### Symbol Not Found
```python
sym_info = mt5.symbol_info(symbol)
if sym_info is None:
    # Symbol doesn't exist on broker
    LOG.error(f"Symbol {symbol} not found")
    return False
```

### No Market Data
```python
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    # Market data unavailable
    LOG.error(f"No market data for {symbol}")
    return None
```

### Order Not Filled (Timeout)
```python
# After 60 seconds of polling
if not found_in_history:
    # Order never filled
    LOG.warning(f"Order {order_id} timed out")
    status = ExecutionStatus.EXPIRED
```

---

## Constants Reference

```python
# Order Actions
mt5.TRADE_ACTION_DEAL       # Market order
mt5.TRADE_ACTION_PENDING    # Pending order

# Order Types
mt5.ORDER_TYPE_BUY          # Buy market order
mt5.ORDER_TYPE_SELL         # Sell market order
mt5.ORDER_TYPE_BUY_LIMIT    # Buy limit
mt5.ORDER_TYPE_SELL_LIMIT   # Sell limit
mt5.ORDER_TYPE_BUY_STOP     # Buy stop
mt5.ORDER_TYPE_SELL_STOP    # Sell stop

# Return Codes
mt5.TRADE_RETCODE_DONE      # Successful (10009)
mt5.TRADE_RETCODE_UNKNOWN   # Unknown error
mt5.TRADE_RETCODE_INVALID_VOLUME  # Bad volume

# Deal Types
mt5.DEAL_TYPE_BUY           # Buy deal
mt5.DEAL_TYPE_SELL          # Sell deal

# Copy Modes (for tick history)
mt5.COPY_TICKS_ALL          # All ticks
```

---

## Connection Pool Integration

```python
# Get connection from pool
name, session = self.connection_pool.get_connection(timeout=10)

try:
    with session.lock:  # Thread-safe access
        # All MT5 calls happen here
        tick = mt5.symbol_info_tick(symbol)
        result = mt5.order_send(request)
        
finally:
    # Return to pool for reuse
    self.connection_pool.release_connection((name, session))
```

---

## Performance Notes

| Operation | Typical Time | Max Time |
|-----------|-------------|----------|
| mt5.symbol_info_tick() | 10-50ms | 100ms |
| mt5.order_send() | 50-200ms | 500ms |
| mt5.orders_get() | 5-20ms | 50ms |
| mt5.history_deals_get() | 10-50ms | 100ms |
| mt5.account_info() | 5-20ms | 50ms |

**Stage 5 Polling**: 0.5s interval × 120 polls = 60s max timeout

---

## Example: Complete Order Execution

```python
from arbitrex.execution_engine.engine import ExecutionEngine, BrokerInterface, ExecutionDatabase
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

# Initialize
pool = MT5ConnectionPool(sessions={...}, symbols=['EURUSD'])
broker = BrokerInterface(connection_pool=pool)
broker.connect()

db = ExecutionDatabase()
ee = ExecutionEngine(broker, db)

# Execute trade from RPM
confirmation = ee.execute(rpm_output)

# MT5 API calls made:
# 1. mt5.symbol_info_tick('EURUSD')  ← Check spread
# 2. mt5.account_info()               ← Check margin
# 3. mt5.symbol_info('EURUSD')        ← Check tradable
# 4. mt5.order_send(request)          ← Submit order
# 5. mt5.orders_get() × N             ← Monitor every 0.5s
# 6. mt5.history_deals_get()          ← Check if filled
# 7. Store to database

print(f"Executed: {confirmation.symbol}")
print(f"Status: {confirmation.status}")
print(f"Fill Price: {confirmation.fill_price}")
print(f"Slippage: {confirmation.slippage_pips} pips")
```

---

## Summary

The Execution Engine uses **7 MT5 API methods** in sequence:

1. ✅ **mt5.symbol_info_tick()** - Get bid/ask prices
2. ✅ **mt5.account_info()** - Check available margin
3. ✅ **mt5.symbol_info()** - Verify symbol tradable
4. ✅ **mt5.order_send()** - Submit market order
5. ✅ **mt5.orders_get()** - Check if order still open
6. ✅ **mt5.history_deals_get()** - Check if order filled
7. ✅ **Constants** - ORDER_TYPE_BUY, TRADE_ACTION_DEAL, etc.

All calls are **thread-safe** via MT5ConnectionPool.get_connection() with session.lock.

**Result**: Real orders executed to real MT5 broker with full audit trail.
