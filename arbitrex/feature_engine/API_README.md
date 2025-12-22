# Feature Engine REST API

REST API for ArbitreX ML Feature Computation and Retrieval.

## Overview

The Feature Engine API provides HTTP endpoints for:
- Computing ML features from OHLCV data
- Retrieving stored features
- Getting feature vectors for specific timestamps
- Monitoring system health
- Managing feature versions

## Quick Start

### Start the API Server

```bash
# Option 1: Using startup script
python start_feature_api.py

# Option 2: Direct module execution
python -m arbitrex.feature_engine.api

# Option 3: Using uvicorn directly
uvicorn arbitrex.feature_engine.api:app --host 0.0.0.0 --port 8001
```

API will be available at: `http://localhost:8001`

### Interactive Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json

## API Endpoints

### 1. Root - API Information
```http
GET /
```

Returns API version and available endpoints.

**Response:**
```json
{
  "service": "ArbitreX Feature Engine API",
  "version": "1.0.0",
  "status": "online",
  "endpoints": { ... }
}
```

---

### 2. Compute Features
```http
POST /compute
```

Compute ML features from OHLCV data.

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1H",
  "normalize": true,
  "store_features": true,
  "ohlcv_data": [
    {
      "timestamp_utc": "2025-01-01T00:00:00Z",
      "open": 1.0500,
      "high": 1.0520,
      "low": 1.0495,
      "close": 1.0510,
      "volume": 1000,
      "spread": 0.0002,
      "log_return_1": 0.0001,
      "valid_bar": true
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "1H",
  "features_computed": 16,
  "bars_processed": 200,
  "config_version": "dd1240dfd229b965",
  "computation_time_ms": 45.2,
  "features": [ ... ]
}
```

---

### 3. Get Features
```http
GET /features/{symbol}/{timeframe}?config_version={hash}&limit={n}
```

Retrieve computed features from store.

**Parameters:**
- `symbol` (path): Trading symbol (e.g., EURUSD)
- `timeframe` (path): Timeframe (1H, 4H, 1D)
- `config_version` (query, optional): Specific config version
- `limit` (query, default=100): Maximum records to return

**Example:**
```http
GET /features/EURUSD/1H?limit=50
```

**Response:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "1H",
  "records": 50,
  "features": [ ... ]
}
```

---

### 4. Get Feature Vector
```http
GET /vector/{symbol}/{timeframe}/{timestamp}?ml_only={bool}
```

Get feature vector for specific timestamp.

**Parameters:**
- `symbol` (path): Trading symbol
- `timeframe` (path): Timeframe
- `timestamp` (path): ISO format timestamp
- `ml_only` (query, default=true): Return only ML-ready features

**Example:**
```http
GET /vector/EURUSD/1H/2025-01-01T00:00:00Z?ml_only=true
```

**Response:**
```json
{
  "timestamp_utc": "2025-01-01T00:00:00Z",
  "symbol": "EURUSD",
  "timeframe": "1H",
  "feature_values": [-0.3693, -1.5325, ...],
  "feature_names": ["rolling_return_3_norm", ...],
  "config_version": "dd1240dfd229b965",
  "is_ml_ready": true
}
```

---

### 5. List Feature Versions
```http
GET /versions/{symbol}/{timeframe}
```

List available feature versions for a symbol/timeframe.

**Response:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "1H",
  "versions": ["dd1240dfd229b965", "abc123def456"],
  "count": 2
}
```

---

### 6. Get Feature Schema
```http
GET /schema/{timeframe}?ml_only={bool}
```

Get list of features for a timeframe.

**Example:**
```http
GET /schema/1H?ml_only=true
```

**Response:**
```json
{
  "success": true,
  "timeframe": "1H",
  "ml_only": true,
  "feature_count": 16,
  "features": [
    "rolling_return_3",
    "rolling_return_6",
    ...
  ]
}
```

---

### 7. Health Check
```http
GET /health
```

Get health status and metrics.

**Response:**
```json
{
  "status": "HEALTHY",
  "success_rate_pct": 98.5,
  "total_computations": 1234,
  "uptime_seconds": 3600.5,
  "metrics": {
    "computation": { ... },
    "performance": { ... },
    "data_quality": { ... },
    "storage": { ... }
  }
}
```

**Health Status Values:**
- `HEALTHY`: Success rate ≥ 95%, validation pass ≥ 95%
- `DEGRADED`: Success rate ≥ 80%, validation pass ≥ 80%
- `UNHEALTHY`: Below degraded thresholds

---

### 8. Get Configuration
```http
GET /config
```

Get current Feature Engine configuration.

**Response:**
```json
{
  "success": true,
  "config_version": "1.0.0",
  "config_hash": "dd1240dfd229b965",
  "configuration": { ... }
}
```

---

## Python Client Example

```python
import requests

# API base URL
base_url = "http://localhost:8001"

# Compute features
compute_request = {
    "symbol": "EURUSD",
    "timeframe": "1H",
    "normalize": True,
    "store_features": True,
    "ohlcv_data": [
        # ... your OHLCV data
    ]
}

response = requests.post(f"{base_url}/compute", json=compute_request)
result = response.json()

if result['success']:
    print(f"Features computed: {result['features_computed']}")
    print(f"Config version: {result['config_version']}")

# Get feature vector
timestamp = "2025-01-01T00:00:00Z"
response = requests.get(f"{base_url}/vector/EURUSD/1H/{timestamp}")
vector = response.json()

print(f"Feature values: {vector['feature_values']}")
print(f"Feature names: {vector['feature_names']}")

# Check health
response = requests.get(f"{base_url}/health")
health = response.json()
print(f"Status: {health['status']}")
print(f"Success rate: {health['success_rate_pct']}%")
```

---

## Health Monitor

The API includes comprehensive health monitoring:

### Tracked Metrics

**Computation Metrics:**
- Features computed total
- Features failed
- Symbols processed
- Timeframes processed

**Performance Metrics:**
- Average computation time
- Maximum computation time
- Total bars processed

**Data Quality:**
- Average feature coverage
- Normalization success rate
- Validation pass rate

**Storage Metrics:**
- Feature store writes
- Feature store reads
- Versions stored

### Monitoring Endpoints

1. **Real-time health**: `GET /health`
2. **Configuration**: `GET /config`
3. **Metrics export**: Automatic on shutdown

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Additional details"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

---

## Configuration

### Environment Variables

```bash
# API Host (default: 0.0.0.0)
FEATURE_API_HOST=0.0.0.0

# API Port (default: 8001)
FEATURE_API_PORT=8001

# Log Level (default: info)
FEATURE_API_LOG_LEVEL=info

# Feature Store Path
FEATURE_STORE_PATH=./arbitrex/data/features
```

### Feature Engine Config

The API uses `FeatureEngineConfig` for feature computation settings:

```python
from arbitrex.feature_engine.config import FeatureEngineConfig

config = FeatureEngineConfig(
    config_version="1.0.0",
    min_valid_bars_required=100,
    # ... other settings
)
```

---

## Testing

### Run Demo Script

```bash
# Start API server first
python start_feature_api.py

# In another terminal, run demo
python demo_feature_api.py
```

### Manual Testing with cURL

```bash
# Test root endpoint
curl http://localhost:8001/

# Get config
curl http://localhost:8001/config

# Get feature schema
curl http://localhost:8001/schema/1H?ml_only=true

# Check health
curl http://localhost:8001/health
```

---

## Performance

- **Average response time**: 10-50ms for feature retrieval
- **Computation time**: ~50-200ms for 200 bars
- **Throughput**: ~100-500 requests/second (depends on hardware)
- **Concurrent requests**: Supports multiple concurrent computations

---

## Security Considerations

**Production Deployment:**
1. Add authentication (JWT, API keys)
2. Enable HTTPS/TLS
3. Rate limiting
4. Input validation and sanitization
5. CORS configuration
6. Request size limits

**Example with Authentication:**
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/compute")
async def compute_features(
    request: ComputeFeaturesRequest,
    token: str = Depends(security)
):
    # Validate token
    if not validate_token(token):
        raise HTTPException(status_code=401)
    # ... compute features
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001
CMD ["python", "start_feature_api.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  feature-api:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./data/features:/app/arbitrex/data/features
    environment:
      - FEATURE_API_HOST=0.0.0.0
      - FEATURE_API_PORT=8001
```

---

## Troubleshooting

### API won't start
- Check port 8001 is not in use: `netstat -an | findstr 8001`
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check logs for errors

### Features not computing
- Verify input data format matches schema
- Check minimum bars requirement (default: 100)
- Review validation errors in response
- Check `/health` endpoint for errors

### Storage errors
- Verify feature store directory exists and is writable
- Check disk space
- Review storage permissions

---

## Support

For issues or questions:
1. Check API documentation: http://localhost:8001/docs
2. Review health metrics: `GET /health`
3. Check application logs
4. Export health metrics: Automatic on shutdown

---

## Version History

**v1.0.0** (2025-01-09)
- Initial release
- Feature computation endpoint
- Feature retrieval and storage
- Health monitoring
- Version management
