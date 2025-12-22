# Clean Data REST API

Complete REST API for accessing clean financial data and health monitoring.

## Features

### 📊 Data Access Endpoints
- Query clean OHLCV data by symbol/timeframe
- Filter by date range
- Retrieve only valid bars
- Get latest N bars
- List available symbols

### 🏥 Health Monitoring Endpoints  
- System health overview
- Per-symbol validation metrics
- Batch health checks across all symbols
- Processing history inspection

### ⚙️ Processing Endpoints
- Trigger raw → clean processing jobs
- Background job execution
- Job status tracking
- Batch and single-symbol processing

### 📋 Configuration Endpoints
- View current pipeline config
- Inspect output schema
- Check thresholds and rules

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
```

### 2. Start the API Server

```bash
# Method 1: Using the startup script
python -m arbitrex.scripts.run_clean_api

# Method 2: Direct uvicorn command
uvicorn arbitrex.clean_data.api:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Access Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Root**: http://localhost:8001/

## API Endpoints

### Health Monitoring

#### GET `/health`
Overall system health status.

```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-22T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "clean_data_available": true,
  "raw_data_available": true,
  "total_symbols": 3,
  "total_bars": 15000
}
```

#### GET `/health/validation/{symbol}/{timeframe}`
Validation metrics for a specific symbol.

```bash
curl http://localhost:8001/health/validation/EURUSD/1H
```

**Response:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1H",
  "total_bars": 5000,
  "valid_bars": 4850,
  "missing_bars": 50,
  "outlier_bars": 100,
  "invalid_bars": 150,
  "validation_rate": 0.97,
  "last_processed": "2025-12-22T10:00:00Z",
  "warnings": [],
  "errors": []
}
```

#### GET `/health/symbols?timeframe=1H`
Health metrics for all symbols at a timeframe.

```bash
curl http://localhost:8001/health/symbols?timeframe=1H
```

**Response:**
```json
{
  "timeframe": "1H",
  "total_symbols": 3,
  "symbols": [
    {
      "symbol": "EURUSD",
      "timeframe": "1H",
      "total_bars": 5000,
      "valid_bars": 4850,
      "validation_rate": 0.97,
      ...
    },
    ...
  ]
}
```

### Clean Data Access

#### GET `/clean/data/{symbol}/{timeframe}`
Get clean OHLCV data with quality flags.

**Query Parameters:**
- `start_date` (optional): Start date in ISO format (UTC)
- `end_date` (optional): End date in ISO format (UTC)
- `only_valid` (optional): Return only valid_bar=True rows (default: false)
- `limit` (optional): Maximum rows to return (1-10000)

```bash
# Get all data
curl http://localhost:8001/clean/data/EURUSD/1H

# Get only valid bars for date range
curl "http://localhost:8001/clean/data/EURUSD/1H?start_date=2025-12-01T00:00:00Z&end_date=2025-12-10T00:00:00Z&only_valid=true"

# Get latest 100 bars
curl "http://localhost:8001/clean/data/EURUSD/1H?limit=100"
```

**Response:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1H",
  "bars": 100,
  "start_date": "2025-12-01T00:00:00Z",
  "end_date": "2025-12-05T03:00:00Z",
  "data": [
    {
      "timestamp_utc": "2025-12-01T00:00:00Z",
      "symbol": "EURUSD",
      "timeframe": "1H",
      "open": 1.1234,
      "high": 1.1245,
      "low": 1.1230,
      "close": 1.1240,
      "volume": 1500,
      "log_return_1": 0.0005,
      "spread_estimate": null,
      "is_missing": false,
      "is_outlier": false,
      "valid_bar": true,
      "source_id": "mt5_20251201_120000",
      "schema_version": "1.0.0"
    },
    ...
  ]
}
```

#### GET `/clean/symbols?timeframe=1H`
List all symbols with clean data.

```bash
curl http://localhost:8001/clean/symbols?timeframe=1H
```

**Response:**
```json
{
  "timeframe": "1H",
  "count": 3,
  "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]
}
```

#### GET `/clean/metadata/{symbol}/{timeframe}`
Get processing metadata.

```bash
curl http://localhost:8001/clean/metadata/EURUSD/1H
```

**Response:**
```json
{
  "processing_timestamp": "2025-12-22T10:00:00Z",
  "config_version": "1.0.0",
  "schema_version": "1.0.0",
  "source_id": "mt5_20251222_100000",
  "total_bars_processed": 5000,
  "valid_bars": 4850,
  "missing_bars": 50,
  "outlier_bars": 100,
  "invalid_bars": 150,
  "warnings": [],
  "errors": []
}
```

#### GET `/clean/latest/{symbol}/{timeframe}?count=100`
Get most recent N bars.

```bash
curl "http://localhost:8001/clean/latest/EURUSD/1H?count=50"
```

### Processing Orchestration

#### POST `/processing/trigger`
Trigger raw → clean processing job.

**Request Body:**
```json
{
  "symbols": ["EURUSD", "GBPUSD"],
  "timeframe": "1H",
  "force_reprocess": false
}
```

```bash
curl -X POST http://localhost:8001/processing/trigger \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["EURUSD"], "timeframe": "1H", "force_reprocess": false}'
```

**Response:**
```json
{
  "job_id": "job_20251222_120000",
  "status": "pending",
  "symbols_total": 1,
  "symbols_processed": 0,
  "symbols_succeeded": 0,
  "symbols_failed": 0,
  "started_at": "2025-12-22T12:00:00Z",
  "completed_at": null,
  "errors": []
}
```

#### GET `/processing/status/{job_id}`
Check processing job status.

```bash
curl http://localhost:8001/processing/status/job_20251222_120000
```

**Response:**
```json
{
  "job_id": "job_20251222_120000",
  "status": "completed",
  "symbols_total": 1,
  "symbols_processed": 1,
  "symbols_succeeded": 1,
  "symbols_failed": 0,
  "started_at": "2025-12-22T12:00:00Z",
  "completed_at": "2025-12-22T12:05:00Z",
  "errors": []
}
```

### Configuration

#### GET `/config`
Get current pipeline configuration.

```bash
curl http://localhost:8001/config
```

**Response:**
```json
{
  "outlier_thresholds": {
    "price_jump_std_multiplier": 5.0,
    "volatility_window": 20,
    "max_abs_log_return": 0.15,
    "min_valid_price": 1e-06
  },
  "missing_bar_thresholds": {
    "max_consecutive_missing": 3,
    "max_missing_percentage": 0.05
  },
  ...
}
```

#### GET `/config/schema`
Get output schema specification.

```bash
curl http://localhost:8001/config/schema
```

## Usage Examples

### Example 1: Monitor All Symbols Health

```python
import requests

response = requests.get("http://localhost:8001/health/symbols?timeframe=1H")
health_data = response.json()

for symbol_health in health_data['symbols']:
    if symbol_health['validation_rate'] < 0.90:
        print(f"⚠️ {symbol_health['symbol']}: {symbol_health['validation_rate']:.1%} valid")
    else:
        print(f"✓ {symbol_health['symbol']}: {symbol_health['validation_rate']:.1%} valid")
```

### Example 2: Fetch Valid Bars for Analysis

```python
import requests
import pandas as pd

# Get only valid bars
response = requests.get(
    "http://localhost:8001/clean/data/EURUSD/1H",
    params={
        "start_date": "2025-12-01T00:00:00Z",
        "end_date": "2025-12-10T00:00:00Z",
        "only_valid": True
    }
)

data = response.json()
df = pd.DataFrame(data['data'])

print(f"Loaded {len(df)} valid bars")
print(df[['timestamp_utc', 'close', 'valid_bar']].head())
```

### Example 3: Trigger Processing Job

```python
import requests
import time

# Trigger processing
response = requests.post(
    "http://localhost:8001/processing/trigger",
    json={
        "symbols": ["EURUSD", "GBPUSD"],
        "timeframe": "1H",
        "force_reprocess": False
    }
)

job = response.json()
job_id = job['job_id']

print(f"Job {job_id} started")

# Poll status
while True:
    status_response = requests.get(f"http://localhost:8001/processing/status/{job_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']} - {status['symbols_processed']}/{status['symbols_total']} symbols")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

print(f"Job completed: {status['symbols_succeeded']} succeeded, {status['symbols_failed']} failed")
```

### Example 4: Build Real-Time Dashboard

```python
import requests
import time
from datetime import datetime

def refresh_dashboard():
    # System health
    health = requests.get("http://localhost:8001/health").json()
    
    print("\n" + "="*80)
    print(f"ARBITREX CLEAN DATA DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\nSystem Status: {health['status'].upper()}")
    print(f"Uptime: {health['uptime_seconds']/3600:.1f} hours")
    print(f"Total Symbols: {health['total_symbols']}")
    print(f"Total Bars: {health['total_bars']:,}")
    
    # Symbol health
    symbols_health = requests.get("http://localhost:8001/health/symbols?timeframe=1H").json()
    
    print(f"\nSymbol Health (1H):")
    print(f"{'Symbol':<10} {'Valid Rate':<12} {'Total Bars':<12} {'Outliers':<10}")
    print("-"*50)
    
    for s in symbols_health['symbols']:
        print(f"{s['symbol']:<10} {s['validation_rate']:>10.1%}  {s['total_bars']:>10,}  {s['outlier_bars']:>8}")

# Run dashboard
while True:
    refresh_dashboard()
    time.sleep(60)  # Refresh every minute
```

## Integration with Frontend

### JavaScript/TypeScript Example

```typescript
// api.ts
const API_BASE = 'http://localhost:8001';

export async function getCleanData(
  symbol: string,
  timeframe: string,
  options?: {
    startDate?: string;
    endDate?: string;
    onlyValid?: boolean;
    limit?: number;
  }
) {
  const params = new URLSearchParams();
  if (options?.startDate) params.append('start_date', options.startDate);
  if (options?.endDate) params.append('end_date', options.endDate);
  if (options?.onlyValid) params.append('only_valid', 'true');
  if (options?.limit) params.append('limit', options.limit.toString());
  
  const response = await fetch(
    `${API_BASE}/clean/data/${symbol}/${timeframe}?${params}`
  );
  
  return response.json();
}

export async function getHealthMetrics(symbol: string, timeframe: string) {
  const response = await fetch(
    `${API_BASE}/health/validation/${symbol}/${timeframe}`
  );
  
  return response.json();
}

// Usage in React component
import { useEffect, useState } from 'react';

function DataChart() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    getCleanData('EURUSD', '1H', { onlyValid: true, limit: 500 })
      .then(result => setData(result.data));
  }, []);
  
  // Render chart with data
}
```

## Production Deployment

### Using systemd (Linux)

Create `/etc/systemd/system/arbitrex-clean-api.service`:

```ini
[Unit]
Description=ArbitreX Clean Data API
After=network.target

[Service]
Type=simple
User=arbitrex
WorkingDirectory=/opt/arbitrex
Environment="PATH=/opt/arbitrex/.venv/bin"
ExecStart=/opt/arbitrex/.venv/bin/python -m arbitrex.scripts.run_clean_api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl enable arbitrex-clean-api
sudo systemctl start arbitrex-clean-api
sudo systemctl status arbitrex-clean-api
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY arbitrex/ arbitrex/

EXPOSE 8001

CMD ["python", "-m", "arbitrex.scripts.run_clean_api"]
```

Build and run:
```bash
docker build -t arbitrex-clean-api .
docker run -p 8001:8001 -v $(pwd)/arbitrex/data:/app/arbitrex/data arbitrex-clean-api
```

### Behind Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.arbitrex.com;
    
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security Considerations

### API Authentication (Future Enhancement)

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.get("/clean/data/{symbol}/{timeframe}", dependencies=[Depends(verify_token)])
async def get_clean_data(...):
    ...
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get("/clean/data/{symbol}/{timeframe}")
@limiter.limit("100/minute")
async def get_clean_data(...):
    ...
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
from datetime import timedelta

@lru_cache(maxsize=128)
def load_clean_data_cached(symbol: str, timeframe: str):
    return load_clean_data(symbol, timeframe)

# Cache expires after 5 minutes
```

### Async Database Queries (Future)

```python
from databases import Database

database = Database("postgresql://user:pass@localhost/arbitrex")

@app.get("/clean/data/{symbol}/{timeframe}")
async def get_clean_data(symbol: str, timeframe: str):
    query = "SELECT * FROM clean_ohlcv WHERE symbol = :symbol AND timeframe = :timeframe"
    rows = await database.fetch_all(query=query, values={"symbol": symbol, "timeframe": timeframe})
    return rows
```

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    with request_duration.time():
        response = await call_next(request)
    request_count.labels(endpoint=request.url.path, method=request.method).inc()
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest())
```

### Logging

```python
import logging
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
```

## Troubleshooting

### Issue: API returns 404 for all symbols
**Solution**: Check that clean data files exist in `arbitrex/data/clean/ohlcv/fx/`

### Issue: Background jobs don't complete
**Solution**: Check logs for processing errors. Verify raw data is available.

### Issue: High latency on data queries
**Solution**: Add `limit` parameter to queries. Consider implementing pagination.

### Issue: CORS errors from frontend
**Solution**: Configure `allow_origins` in CORS middleware with your frontend domain.

## API Versioning

Future versions will use URL versioning:
- `/v1/clean/data/{symbol}/{timeframe}` - Current version
- `/v2/clean/data/{symbol}/{timeframe}` - Future breaking changes

## Support

For issues, questions, or feature requests, see the main project README.

---

**Status**: Production Ready  
**Version**: 1.0.0  
**Last Updated**: 2025-12-22
