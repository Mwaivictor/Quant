"""
Test Raw Data Layer API

Quick verification that all endpoints are operational.
"""

import requests
import json
from datetime import datetime


API_BASE = "http://localhost:8000"


def test_endpoint(name: str, method: str, path: str, **kwargs):
    """Test an endpoint"""
    url = f"{API_BASE}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5, **kwargs)
        else:
            response = requests.post(url, timeout=5, **kwargs)
        
        status = "✓" if response.status_code == 200 else f"✗ ({response.status_code})"
        print(f"{status} {name}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Show key info
                if isinstance(data, dict):
                    if 'status' in data:
                        print(f"   Status: {data['status']}")
                    if 'count' in data:
                        print(f"   Count: {data['count']}")
                    if 'total_symbols' in data:
                        print(f"   Symbols: {data['total_symbols']}")
            except:
                pass
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ {name} - Error: {e}")
        return False


def main():
    """Run all tests"""
    
    print("="*60)
    print("RAW DATA API TESTS")
    print("="*60)
    print(f"Testing: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    results = []
    
    # Test endpoints
    results.append(test_endpoint("Root Endpoint", "GET", "/"))
    results.append(test_endpoint("System Health", "GET", "/health"))
    results.append(test_endpoint("Detailed Health", "GET", "/health/detailed"))
    results.append(test_endpoint("MT5 Status", "GET", "/health/mt5"))
    results.append(test_endpoint("Prometheus Metrics", "GET", "/health/metrics"))
    results.append(test_endpoint("List Symbols", "GET", "/raw/symbols"))
    results.append(test_endpoint("Trading Universe", "GET", "/symbols/universe"))
    results.append(test_endpoint("Configuration", "GET", "/config"))
    
    # Summary
    print()
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print()
        print("✅ Raw Data API with Swagger is fully operational!")
        print()
        print("📚 Access documentation:")
        print(f"   Swagger UI: {API_BASE}/docs")
        print(f"   ReDoc: {API_BASE}/redoc")
        return 0
    else:
        print(f"⚠ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
