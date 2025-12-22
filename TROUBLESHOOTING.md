# Arbitrex Tick Stream Troubleshooting Guide

This guide walks you through diagnosing and fixing common issues with the Arbitrex tick streaming stack on Windows.

---

## The Three-Part Stack

Your system has three independent components:

1. **MT5 Connection Pool** - Pulls live tick data from MetaTrader5 terminal
2. **Tick Queue** - Stores ticks durably (Redis/SQLite)
3. **WebSocket Server** - Streams ticks to browser clients

All three must be working for the browser demo to receive ticks.

---

## 📋 Pre-flight Checklist

Before starting, verify:

- [ ] Redis 5.0+ is installed and running (`redis-server`)
- [ ] Python 3.10+ is installed with venv activated
- [ ] `.env` file exists with correct MT5 credentials
- [ ] MetaTrader5 terminal is installed
- [ ] Port 8000 is free on your machine

---

## 🔧 Common Issues & Fixes

### Issue 1: WebSocket Connection Fails (error [object Event])

**Symptom:**
```
error [object Event]
closed
error [object Event]
closed
```

**Possible Causes:**
- Backend WebSocket server (Uvicorn) is not running
- Port 8000 is already in use or blocked by firewall
- Python process killed or failed to start

**Fix:**

1. **Check if port 8000 is in use:**
   ```powershell
   netstat -ano | findstr :8000
   ```
   If a process is using it, either kill it or use a different port.

2. **Allow Python through Windows Firewall:**
   - Open Windows Defender Firewall
   - Click "Allow an app through firewall"
   - Add Python.exe from your venv: `.venv\Scripts\python.exe`

3. **Start the WebSocket server explicitly:**
   ```powershell
   $env:TICK_QUEUE_BACKEND="redis"
   $env:DISABLE_KAFKA="1"
   $env:REDIS_URL="redis://localhost:6379/0"
   
   python -m arbitrex.scripts.run_streaming_stack
   ```

   You should see:
   ```
   INFO:     Started server process [12345]
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

4. **Test connectivity:**
   ```powershell
   # In another terminal, test the HTTP endpoint
   curl http://localhost:8000
   ```
   Should return HTML.

---

### Issue 2: MT5 Session Keeps Disconnecting

**Symptom:**
```
Session main disconnected, attempting reconnect
Session main disconnected, attempting reconnect
...
```

**Possible Causes:**
- MT5 terminal not running
- Invalid terminal path in `.env`
- Incorrect login/password/server
- Market is closed (FX market closed on weekends)
- Terminal requires manual login or approval

**Fix:**

1. **Verify MT5 terminal is running:**
   - Open MetaTrader5 manually and confirm you can log in with the credentials in `.env`
   - Check that the account is active and the server is online

2. **Verify terminal path in `.env`:**
   ```env
   MT5_TERMINAL=C:\Program Files\MetaTrader 5\terminal64.exe
   ```
   Adjust the path if your MT5 is installed elsewhere. Do NOT use `.lnk` shortcuts.

3. **Check credentials:**
   ```powershell
   $env:MT5_LOGIN     # Should be numeric (e.g., 5042323533)
   $env:MT5_PASSWORD  # Should match your broker password
   $env:MT5_SERVER    # Should match your broker server (e.g., MetaQuotes-Demo)
   ```

4. **Run with verbose logging to see actual MT5 error:**
   ```powershell
   python -c "
   import logging
   logging.basicConfig(level=logging.DEBUG)
   from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
   from arbitrex.raw_layer.config import TRADING_UNIVERSE
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   sessions = {
       'test': {
           'terminal_path': os.environ.get('MT5_TERMINAL'),
           'login': int(os.environ['MT5_LOGIN']),
           'password': os.environ.get('MT5_PASSWORD'),
           'server': os.environ.get('MT5_SERVER'),
       }
   }
   symbols = [s for g in TRADING_UNIVERSE.values() for s in g]
   pool = MT5ConnectionPool(sessions, symbols)
   import time; time.sleep(5)
   pool.close()
   "
   ```

5. **Check if market is open (for FX):**
   - FX markets are open Sun 22:00 UTC → Fri 22:00 UTC
   - If it's weekend or FX holiday, MT5 may not return ticks

---

### Issue 3: Redis Connection Fails

**Symptom:**
```
Failed to init RedisTickQueue: ConnectionError('Error 111 connecting to localhost:6379...')
```

**Fix:**

1. **Start Redis if not running:**
   ```powershell
   redis-server
   ```
   You should see:
   ```
   # Server started, Redis version 5.0+
   ```

2. **Verify Redis version (must be 5.0+):**
   ```powershell
   redis-cli --version
   ```

3. **Test connection:**
   ```powershell
   redis-cli ping
   ```
   Should return `PONG`.

4. **Check `.env` for quoted strings:**
   Your `.env` must have:
   ```env
   REDIS_URL=redis://localhost:6379/0
   ```
   NOT:
   ```env
   REDIS_URL = 'redis://localhost:6379/0'  # Wrong! Has spaces and quotes
   ```

---

### Issue 4: Kafka Errors Despite DISABLE_KAFKA=1

**Symptom:**
```
%3|1766368518.459|FAIL|rdkafka#producer-1| [thrd:localhost:9092/bootstrap]: Connect failed
```

**Fix:**

1. **Ensure `.env` has:**
   ```env
   DISABLE_KAFKA=1
   ```
   Note: NO quotes, exact string `1` (not true/yes).

2. **Reload environment after editing `.env`:**
   ```powershell
   # Kill any running Python processes
   # Restart the terminal
   # Re-activate venv
   .venv\Scripts\Activate.ps1
   ```

3. **Verify it's disabled:**
   ```powershell
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('DISABLE_KAFKA:', os.environ.get('DISABLE_KAFKA'))"
   ```

---

### Issue 5: JSON Serialization Errors (uint64)

**Symptom:**
```
TypeError: Object of type uint64 is not JSON serializable
```

**Fix:**

This should be fixed in the updated `tick_queue_redis.py`. It now converts numpy types to native Python types.

Verify the fix is in place:
```powershell
grep -A 10 "def to_py" arbitrex/raw_layer/tick_queue_redis.py
```

If missing, the conversion will happen automatically.

---

## 🧪 Testing the Full Stack

### Step 1: Start Redis
```powershell
redis-server
```
Leave running in a terminal.

### Step 2: Start the Streaming Stack
```powershell
$env:TICK_QUEUE_BACKEND="redis"
$env:DISABLE_KAFKA="1"
$env:REDIS_URL="redis://localhost:6379/0"

python -m arbitrex.scripts.run_streaming_stack
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Open the Demo
1. Navigate to: `arbitrex/stream/demo.html`
2. Double-click to open in your browser
3. Symbols field should show `EURUSD`
4. Click **Connect**

### Step 4: Verify Connection
- Log should show: `connected`
- Then: `{"subscribed": ["EURUSD"]}`
- As MT5 produces ticks, you'll see: `{"symbol": "EURUSD", "ts": 1234567890, "bid": 1.0850, ...}`

---

## 📊 Debugging Commands

### Check if server is responding:
```powershell
curl http://localhost:8000
```

### Check WebSocket endpoint:
```powershell
# Use a WebSocket CLI tool or browser console:
# In browser console (F12):
ws = new WebSocket('ws://localhost:8000/ws')
ws.onopen = () => console.log('connected')
```

### Check Redis queue:
```powershell
redis-cli
> KEYS ticks:*
> XLEN ticks:EURUSD
```

### Check running processes:
```powershell
Get-Process python
```

### Check environment variables:
```powershell
$env:TICK_QUEUE_BACKEND
$env:REDIS_URL
$env:DISABLE_KAFKA
```

---

## 🚀 Next Steps

Once the stack is working:

1. **Test tick ingestion with market open:**
   - Run demo.html while FX market is open
   - You should see live ticks streaming

2. **Monitor Prometheus metrics (optional):**
   ```powershell
   $env:PROMETHEUS_PORT=8001
   # Then visit http://localhost:8001/metrics
   ```

3. **Persist ticks to disk:**
   - Ticks are stored in Redis and written to CSV via flush interval
   - Check `arbitrex/data/raw/ticks/fx/<SYMBOL>/<DATE>.csv`

---

## 📞 Support

If issues persist:

1. Check the full error traceback (scroll up in terminal)
2. Verify all three components are running (Redis, MT5, Python server)
3. Check `.env` file for typos (no quotes, spaces, or special characters)
4. Restart all services in this order: Redis → Python server → Browser
