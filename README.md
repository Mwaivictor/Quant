# 📖 Arbitrex Documentation Index

Welcome to the Arbitrex Raw Data Layer tick streaming system. This document serves as your roadmap to all documentation and how to use the system.

---

## 🚀 Getting Started

**Start here if you're new to the system:**

1. **[QUICK_START.md](QUICK_START.md)** (5 minutes)
   - 3-step startup procedure
   - Success indicators
   - Quick troubleshooting

2. **[arbitrex/raw_layer/README.md](arbitrex/raw_layer/README.md)** (10 minutes)
   - Full architecture overview
   - Module responsibilities
   - Storage layout & semantics
   - Tick collection workflow

---

## 🔧 Fixing & Understanding Issues

**Read these if something is broken or confusing:**

1. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** (Reference)
   - Pre-flight checklist
   - Common issues with solutions
   - Debugging commands
   - Testing procedures

2. **[FIXES_APPLIED.md](FIXES_APPLIED.md)** (Understanding)
   - What was broken and why
   - Detailed explanations of fixes
   - Code before/after comparisons
   - Key configuration changes

3. **[SYSTEM_REVIEW.md](SYSTEM_REVIEW.md)** (Complete Analysis)
   - Full system architecture diagram
   - All issues found and fixed
   - Components status table
   - Production readiness assessment

---

## 🤖 Automation & Scripts

**Use these to simplify operations:**

1. **[START_STACK.ps1](START_STACK.ps1)** (PowerShell Script)
   - Starts Redis automatically
   - Sets up environment variables
   - Launches Python stack
   - Opens demo in browser
   - Usage: `.\START_STACK.ps1`

2. **Manual startup** (if you prefer full control)
   - See **QUICK_START.md** for steps

---

## 📚 Core Documentation

### Architecture & Design
- **[arbitrex/raw_layer/README.md](arbitrex/raw_layer/README.md)** - Complete technical reference
  - Design goals and key concepts
  - Module responsibilities
  - Durable queue options
  - Market calendar & symbol mapping
  - Observability (Prometheus + Grafana)
  - Testing & CI guidance

### API & Configuration
- **[arbitrex/raw_layer/config.py](arbitrex/raw_layer/config.py)** - Configuration defaults
- **[.env](.env)** - Environment setup
  - MT5 credentials
  - Redis connection
  - Backend selection
  - Feature flags

### Components

#### Data Ingestion
- **mt5_pool.py** - Connection pool, session management, tick collection
- **ingest.py** - Low-level ingestion primitives
- **orchestrator.py** - Worker partitioning and orchestration

#### Storage & Streaming
- **writer.py** - Atomic CSV writers, metadata recording
- **tick_queue.py** - SQLite durable queue (fallback)
- **tick_queue_redis.py** - Redis Streams queue (recommended)
- **tick_queue_kafka.py** - Kafka producer (optional)

#### Real-time Streaming
- **stream/ws_server.py** - FastAPI WebSocket broker
- **stream/demo.html** - Browser client demo

#### Utilities
- **market_calendar.py** - Market hours detection
- **runner.py** - CLI for ingestion workflows
- **cli.py** - Command-line interface

---

## 🎯 Common Workflows

### Starting the Full Stack
```powershell
# Automated (recommended)
.\START_STACK.ps1

# OR Manual (see QUICK_START.md)
redis-server  # Terminal 1
python -m arbitrex.scripts.run_streaming_stack  # Terminal 2
```

### Ingesting OHLCV Data
```powershell
python -m arbitrex.raw_layer.runner --symbols EURUSD,GBPUSD --timeframes 1H,4H --workers 2
```

### Testing Tick Collection
```powershell
python arbitrex/scripts/run_tick_collector.py
```

### WebSocket Only (Demo)
```powershell
python arbitrex/scripts/run_tick_ws.py
```

---

## 📊 System Status

| Component | Status | Docs |
|-----------|--------|------|
| MT5 Connection Pool | ✅ Fixed | [mt5_pool.py](arbitrex/raw_layer/mt5_pool.py) |
| Tick Queue (Redis) | ✅ Fixed | [tick_queue_redis.py](arbitrex/raw_layer/tick_queue_redis.py) |
| WebSocket Server | ✅ Fixed | [ws_server.py](arbitrex/stream/ws_server.py) |
| Environment Config | ✅ Fixed | [.env](.env) |
| All Scripts | ✅ Fixed | [scripts/](arbitrex/scripts/) |
| Documentation | ✅ Complete | [README.md](arbitrex/raw_layer/README.md) |

---

## 🐛 Recent Fixes (This Session)

All of these issues have been **resolved**:

1. ✅ Missing `symbols` argument in MT5ConnectionPool
2. ✅ Quoted/spaced environment variables in `.env`
3. ✅ Kafka connecting despite DISABLE_KAFKA=1
4. ✅ WebSocket event loop capture issues
5. ✅ JSON serialization of numpy types

See **[FIXES_APPLIED.md](FIXES_APPLIED.md)** for details.

---

## 🧪 Testing & Validation

### Quick Health Check
```powershell
# Redis
redis-cli ping  # Should return: PONG

# Python Stack
curl http://localhost:8000  # Should return HTML

# WebSocket
# Open arbitrex/stream/demo.html and click Connect
# Should see: connected
```

### Full Integration Test
1. Start Redis: `redis-server`
2. Start stack: `python -m arbitrex.scripts.run_streaming_stack`
3. Open browser: `arbitrex/stream/demo.html`
4. Subscribe to symbols
5. Verify ticks appear in console

See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed testing procedures.

---

## 📈 Performance & Monitoring

### Prometheus Metrics
Set `PROMETHEUS_PORT=8001` in `.env`, then visit `http://localhost:8001/metrics`

Available metrics:
- `arbitrex_ticks_received_total` - Ticks from MT5
- `arbitrex_ticks_published_total` - Ticks sent to WebSocket
- `arbitrex_ticks_flushed_total` - Ticks written to disk
- `arbitrex_ticks_queue_size` - Pending ticks in queue

### Grafana Dashboards
Dashboard JSON included in `arbitrex/raw_layer/grafana/`. See README for import instructions.

---

## 📞 Support & Troubleshooting

### If Something Breaks

1. **Check [QUICK_START.md](QUICK_START.md)** - Verify all 3 steps
2. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Browse common issues
3. **Check [SYSTEM_REVIEW.md](SYSTEM_REVIEW.md)** - Understand architecture
4. **Check logs** - Look at terminal output for error messages

### Common Issues

| Issue | Solution |
|-------|----------|
| WebSocket won't connect | Check Redis running, verify port 8000 free |
| MT5 session disconnects | Verify terminal running, check login credentials |
| No ticks received | Ensure market is open, check symbol subscriptions |
| Redis connection fails | Verify Redis running, check REDIS_URL format |

---

## 🎓 Learning Resources

### Understanding the System
1. Read **arbitrex/raw_layer/README.md** for full architecture
2. Review **SYSTEM_REVIEW.md** for component overview
3. Check **[FIXES_APPLIED.md](FIXES_APPLIED.md)** to understand design decisions

### Extending the System
- Add new tick queues: See `tick_queue_redis.py` as template
- Add WebSocket features: See `stream/ws_server.py`
- Add metrics: See `mt5_pool.py` Prometheus section
- Add CLI commands: See `runner.py` and `cli.py`

---

## 📋 File Structure

```
c:\Users\Admin\Desktop\AUTODESI\ARBITREEX MVP\
├── .env                           # Configuration (cleaned)
├── START_STACK.ps1                # Automated startup
├── QUICK_START.md                 # 3-step guide
├── TROUBLESHOOTING.md             # Debugging guide
├── FIXES_APPLIED.md               # Detailed fixes
├── SYSTEM_REVIEW.md               # Full analysis
├── README.md                       # This file
│
├── arbitrex/
│   ├── raw_layer/
│   │   ├── README.md              # Technical reference
│   │   ├── config.py              # Configuration
│   │   ├── mt5_pool.py            # Connection pool
│   │   ├── ingest.py              # Ingestion primitives
│   │   ├── writer.py              # Atomic writers
│   │   ├── tick_queue*.py         # Queue backends
│   │   ├── market_calendar.py     # Market hours
│   │   ├── runner.py              # CLI runner
│   │   ├── cli.py                 # CLI commands
│   │   ├── orchestrator.py        # Worker orchestration
│   │   └── grafana/               # Prometheus dashboards
│   │
│   ├── stream/
│   │   ├── ws_server.py           # WebSocket server
│   │   └── demo.html              # Browser client
│   │
│   └── scripts/
│       ├── run_streaming_stack.py # Full stack
│       ├── run_tick_collector.py  # Tick collector
│       ├── run_tick_ws.py         # WebSocket only
│       └── test_*.py              # Tests
│
└── data/raw/                      # Output directory
    ├── ohlcv/                     # OHLCV files
    ├── ticks/                     # Tick files
    └── metadata/                  # Ingestion logs
```

---

## ✅ Checklist for Success

- [ ] Read **QUICK_START.md**
- [ ] Verify `.env` has no quotes
- [ ] Start Redis with `redis-server`
- [ ] Run stack with `python -m arbitrex.scripts.run_streaming_stack`
- [ ] Open `arbitrex/stream/demo.html`
- [ ] Click Connect and verify ticks
- [ ] Check **TROUBLESHOOTING.md** if issues arise

---

## 🎉 You're Ready!

Your Arbitrex tick streaming system is fully operational and well-documented.

**Next steps:**
1. Start with **[QUICK_START.md](QUICK_START.md)**
2. Run `.\START_STACK.ps1` or manual startup
3. Subscribe to tick symbols
4. Stream live market data!

For detailed understanding, see **[arbitrex/raw_layer/README.md](arbitrex/raw_layer/README.md)**.

---

**Last Updated:** December 22, 2025  
**System Status:** ✅ Production Ready  
**All Issues:** ✅ Resolved
