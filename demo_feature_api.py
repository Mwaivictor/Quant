"""
Demo Feature Engine API

Start the API server and test endpoints.
"""

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def generate_sample_data(symbol: str, bars: int = 200):
    """Generate sample OHLCV data"""
    start = datetime.now(timezone.utc) - timedelta(hours=bars)
    dates = pd.date_range(start=start, periods=bars, freq='h', tz='UTC')
    
    base_price = 1.0500 if 'USD' in symbol else 100.0
    
    data = []
    for ts in dates:
        close_price = base_price + np.random.randn() * 0.01
        data.append({
            'timestamp_utc': ts.isoformat(),
            'open': close_price + np.random.uniform(-0.0005, 0.0005),
            'high': close_price + abs(np.random.randn() * 0.001),
            'low': close_price - abs(np.random.randn() * 0.001),
            'close': close_price,
            'volume': float(np.random.uniform(1000, 2000)),
            'spread': float(np.random.uniform(0.0001, 0.0003)),
            'log_return_1': float(np.random.randn() * 0.001),
            'valid_bar': True
        })
    
    return data


def test_api():
    """Test API endpoints"""
    
    base_url = "http://localhost:8001"
    
    print("=" * 80)
    print("FEATURE ENGINE API DEMO")
    print("=" * 80)
    print()
    print("Note: Make sure API is running: python -m arbitrex.feature_engine.api")
    print()
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✓ API is online")
            print(f"   Version: {response.json()['version']}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ API not reachable: {e}")
        print("   Please start the API server first:")
        print("   python -m arbitrex.feature_engine.api")
        return
    
    print()
    
    # Test config endpoint
    print("2. Getting current configuration...")
    response = requests.get(f"{base_url}/config")
    if response.status_code == 200:
        config_data = response.json()
        print(f"   ✓ Config Version: {config_data['config_version']}")
        print(f"   ✓ Config Hash: {config_data['config_hash']}")
    print()
    
    # Test feature schema
    print("3. Getting feature schema...")
    response = requests.get(f"{base_url}/schema/1H?ml_only=true")
    if response.status_code == 200:
        schema_data = response.json()
        print(f"   ✓ ML Features for 1H: {schema_data['feature_count']}")
        print(f"   Features: {', '.join(schema_data['features'][:5])}...")
    print()
    
    # Test feature computation
    print("4. Computing features...")
    sample_data = generate_sample_data('EURUSD', 200)
    
    compute_request = {
        'symbol': 'EURUSD',
        'timeframe': '1H',
        'normalize': True,
        'store_features': True,
        'ohlcv_data': sample_data
    }
    
    response = requests.post(f"{base_url}/compute", json=compute_request)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print(f"   ✓ Features computed successfully")
            print(f"   Symbol: {result['symbol']}")
            print(f"   Timeframe: {result['timeframe']}")
            print(f"   Features: {result['features_computed']}")
            print(f"   Bars: {result['bars_processed']}")
            print(f"   Computation Time: {result['computation_time_ms']:.2f}ms")
            print(f"   Config Version: {result['config_version']}")
        else:
            print(f"   ✗ Computation failed: {result.get('error')}")
            return
    else:
        print(f"   ✗ Request failed: {response.status_code}")
        return
    
    print()
    
    # Test retrieving features
    print("5. Retrieving computed features...")
    response = requests.get(f"{base_url}/features/EURUSD/1H?limit=5")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Retrieved {result['records']} records")
        if result['records'] > 0:
            feature = result['features'][0]
            print(f"   Sample timestamp: {feature.get('timestamp_utc', 'N/A')}")
            print(f"   Sample features: {list(feature.keys())[:10]}...")
    print()
    
    # Test feature vector
    print("6. Getting feature vector for specific timestamp...")
    # Use a recent timestamp from our data
    timestamp = sample_data[-1]['timestamp_utc']
    
    response = requests.get(f"{base_url}/vector/EURUSD/1H/{timestamp}?ml_only=true")
    
    if response.status_code == 200:
        vector = response.json()
        print(f"   ✓ Feature vector retrieved")
        print(f"   Timestamp: {vector['timestamp_utc']}")
        print(f"   Features: {len(vector['feature_values'])}")
        print(f"   ML Ready: {vector['is_ml_ready']}")
        print(f"   Sample values: {vector['feature_values'][:5]}")
    print()
    
    # Test listing versions
    print("7. Listing available versions...")
    response = requests.get(f"{base_url}/versions/EURUSD/1H")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Available versions: {result['count']}")
        if result['count'] > 0:
            print(f"   Versions: {', '.join(result['versions'])}")
    print()
    
    # Test health endpoint
    print("8. Checking health status...")
    response = requests.get(f"{base_url}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"   ✓ Health Status: {health['status']}")
        print(f"   Success Rate: {health['success_rate_pct']:.2f}%")
        print(f"   Total Computations: {health['total_computations']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
    print()
    
    print("=" * 80)
    print("API DEMO COMPLETE")
    print("=" * 80)
    print()
    print("All endpoints tested successfully! ✓")
    print()
    print("Available endpoints:")
    print("  GET  /              - API info")
    print("  GET  /config        - Current configuration")
    print("  GET  /schema/{tf}   - Feature schema")
    print("  POST /compute       - Compute features")
    print("  GET  /features/{s}/{tf} - Get stored features")
    print("  GET  /vector/{s}/{tf}/{ts} - Get feature vector")
    print("  GET  /versions/{s}/{tf} - List versions")
    print("  GET  /health        - Health status")


if __name__ == "__main__":
    test_api()
