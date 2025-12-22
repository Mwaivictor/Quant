# Quick Start Guide - Arbitrex Tick Stream

## Prerequisites
- ✓ Python 3.10+ with venv activated
- ✓ Redis 5.0+ installed
- ✓ MetaTrader5 terminal installed
- ✓ `.env` file configured with MT5 credentials

---

## 🚀 Start in 3 Steps

### Step 1: Start Redis (in Terminal 1)
```powershell
redis-server
```
You should see:
```
# Server started, Redis version 5.0...
```

### Step 2: Start Arbitrex Stack (in Terminal 2)
```powershell
cd "c:\Users\Admin\Desktop\AUTODESI\ARBITREEX MVP"
.venv\Scripts\Activate.ps1

$env:TICK_QUEUE_BACKEND="redis"
$env:DISABLE_KAFKA="1"
$env:REDIS_URL="redis://localhost:6379/0"

python -m arbitrex.scripts.run_streaming_stack
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Open Demo in Browser
Navigate to: `c:\Users\Admin\Desktop\AUTODESI\ARBITREEX MVP\arbitrex\stream\demo.html`

Click **Connect** to subscribe to ticks.

---

## ✅ Success Indicators

Browser console should show:
```
connected
{"subscribed": ["EURUSD"]}
{"symbol": "EURUSD", "ts": 1766368518, "bid": 1.08501, "ask": 1.08503, "last": 1.08502, "volume": 100}
```

---

## 🔧 If Something Fails

1. **WebSocket won't connect:**
   - Is server running on port 8000? Check terminal 2
   - Try: `curl http://localhost:8000`
   - Allow Python through Windows Firewall

2. **Session disconnecting:**
   - Is MT5 terminal running and logged in?
   - Check MT5_TERMINAL path in `.env`
   - Verify login/password/server match your broker

3. **Redis connection fails:**
   - Is Redis running? Check terminal 1
   - Run: `redis-cli ping` (should return PONG)

---

## 📚 More Help

- **Detailed troubleshooting:** See `TROUBLESHOOTING.md`
- **What was fixed:** See `FIXES_APPLIED.md`
- **Architecture overview:** See `arbitrex/raw_layer/README.md`

---

## 🛑 Stop the Stack

Press `CTRL+C` in Terminal 2 (Python server).

Redis will keep running. Stop it with:
```powershell
redis-cli shutdown
```

Or close the terminal where `redis-server` is running.
