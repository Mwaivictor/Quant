"""
Test Clean Data API Endpoints

This script tests all API endpoints to verify functionality.
"""

import requests
import json
import sys
from datetime import datetime


API_BASE = "http://localhost:8001"


def print_section(title: str):
    """Print a section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def test_endpoint(method: str, path: str, **kwargs):
    """Test an API endpoint and print results"""
    url = f"{API_BASE}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"\n{method} {path}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)[:500]}...")  # Truncate long responses
            return True, data
        else:
            print(f"Error: {response.text}")
            return False, None
    
    except Exception as e:
        print(f"\n{method} {path}")
        print(f"Error: {e}")
        return False, None


def main():
    """Run all API tests"""
    
    print_section("CLEAN DATA API TESTS")
    print(f"Testing API at: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Root endpoint
    print_section("TEST 1: Root Endpoint")
    success, _ = test_endpoint("GET", "/")
    results.append(("Root", success))
    
    # Test 2: Health check
    print_section("TEST 2: System Health")
    success, health = test_endpoint("GET", "/health")
    results.append(("Health", success))
    
    if success and health:
        print(f"\n  System Status: {health.get('status', 'unknown')}")
        print(f"  Total Symbols: {health.get('total_symbols', 0)}")
        print(f"  Total Bars: {health.get('total_bars', 0)}")
    
    # Test 3: Configuration
    print_section("TEST 3: Configuration")
    success, config = test_endpoint("GET", "/config")
    results.append(("Config", success))
    
    # Test 4: Schema
    print_section("TEST 4: Schema Definition")
    success, schema = test_endpoint("GET", "/config/schema")
    results.append(("Schema", success))
    
    if success and schema:
        print(f"\n  Schema Version: {schema.get('version', 'unknown')}")
        print(f"  Total Columns: {len(schema.get('columns', []))}")
    
    # Test 5: List symbols
    print_section("TEST 5: List Symbols (1H)")
    success, symbols_data = test_endpoint("GET", "/clean/symbols", params={"timeframe": "1H"})
    results.append(("List Symbols", success))
    
    available_symbols = []
    if success and symbols_data:
        available_symbols = symbols_data.get('symbols', [])
        print(f"\n  Available Symbols: {available_symbols}")
    
    # Test 6: Get clean data (if symbols available)
    if available_symbols:
        print_section("TEST 6: Get Clean Data")
        test_symbol = available_symbols[0]
        success, data = test_endpoint(
            "GET",
            f"/clean/data/{test_symbol}/1H",
            params={"limit": 10}
        )
        results.append(("Get Clean Data", success))
        
        if success and data:
            print(f"\n  Symbol: {data.get('symbol')}")
            print(f"  Bars: {data.get('bars')}")
            print(f"  Date Range: {data.get('start_date')} → {data.get('end_date')}")
        
        # Test 7: Get metadata
        print_section("TEST 7: Get Metadata")
        success, metadata = test_endpoint("GET", f"/clean/metadata/{test_symbol}/1H")
        results.append(("Get Metadata", success))
        
        if success and metadata:
            print(f"\n  Total Bars: {metadata.get('total_bars_processed')}")
            print(f"  Valid Bars: {metadata.get('valid_bars')}")
            print(f"  Missing: {metadata.get('missing_bars')}")
            print(f"  Outliers: {metadata.get('outlier_bars')}")
        
        # Test 8: Get validation metrics
        print_section("TEST 8: Validation Metrics")
        success, metrics = test_endpoint("GET", f"/health/validation/{test_symbol}/1H")
        results.append(("Validation Metrics", success))
        
        if success and metrics:
            print(f"\n  Validation Rate: {metrics.get('validation_rate', 0):.1%}")
            print(f"  Valid Bars: {metrics.get('valid_bars')}")
            print(f"  Total Bars: {metrics.get('total_bars')}")
        
        # Test 9: Get latest bars
        print_section("TEST 9: Latest Bars")
        success, latest = test_endpoint("GET", f"/clean/latest/{test_symbol}/1H", params={"count": 5})
        results.append(("Latest Bars", success))
        
        if success and latest:
            print(f"\n  Returned Bars: {latest.get('bars')}")
            print(f"  Latest Timestamp: {latest.get('latest_timestamp')}")
    
    else:
        print("\n⚠️ No symbols available - skipping data tests")
        print("Run the data pipeline first: python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --all")
    
    # Test 10: Health for all symbols
    print_section("TEST 10: All Symbols Health")
    success, all_health = test_endpoint("GET", "/health/symbols", params={"timeframe": "1H"})
    results.append(("All Symbols Health", success))
    
    if success and all_health:
        print(f"\n  Total Symbols: {all_health.get('total_symbols', 0)}")
        for symbol_metrics in all_health.get('symbols', [])[:3]:  # Show first 3
            print(f"    {symbol_metrics['symbol']}: {symbol_metrics['validation_rate']:.1%} valid")
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print()
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
    
    print()
    
    if passed == total:
        print("✓✓✓ ALL API TESTS PASSED ✓✓✓")
        print("\nAPI is fully operational!")
        print(f"\nAccess documentation at:")
        print(f"  Swagger UI: {API_BASE}/docs")
        print(f"  ReDoc: {API_BASE}/redoc")
        return 0
    else:
        print(f"⚠ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
