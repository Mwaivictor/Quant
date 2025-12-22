"""
Test ML Layer API

Quick test to verify API endpoints are working.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8003"


def test_health():
    """Test health endpoint"""
    print("\n1. Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"   Health: {data['status']}")
        print(f"   Config hash: {data['config_hash']}")
        print("   ✓ Health check passed")
    else:
        print(f"   ✗ Health check failed: {response.text}")
    return response.ok


def test_predict():
    """Test single prediction"""
    print("\n2. Testing /predict...")
    
    # Generate test data
    np.random.seed(42)
    n_bars = 150
    
    features = {
        'momentum_score': list(np.random.randn(n_bars) * 0.5),
        'log_return_1': list(np.random.randn(n_bars) * 0.01),
        'atr': list(np.random.uniform(0.001, 0.005, n_bars)),
        'efficiency_ratio': list(np.random.uniform(0.3, 0.8, n_bars)),
        'volatility': list(np.random.uniform(0.01, 0.03, n_bars)),
        'trend_persistence': list(np.random.uniform(0.4, 0.7, n_bars)),
        'correlation': list(np.random.uniform(-0.5, 0.5, n_bars)),
        'returns_autocorr': list(np.random.uniform(-0.3, 0.3, n_bars)),
    }
    
    qse_output = {
        'validation': {
            'signal_validity_flag': True,
            'failure_reasons': []
        },
        'metrics': {
            'trend_persistence': 0.65,
            'stationarity_pvalue': 0.03,
            'z_score_normalized': 1.2
        }
    }
    
    request_data = {
        'symbol': 'EURUSD',
        'timeframe': '4H',
        'features': features,
        'qse_output': qse_output
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data)
    
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        pred = data['prediction']
        print(f"   Symbol: {data['symbol']}")
        print(f"   Regime: {pred['regime']['regime_label']}")
        print(f"   Signal Prob: {pred['signal']['momentum_success_prob']:.3f}")
        print(f"   Allow Trade: {pred['allow_trade']}")
        print(f"   Processing Time: {data['processing_time_ms']:.2f}ms")
        print("   ✓ Prediction test passed")
    else:
        print(f"   ✗ Prediction failed: {response.text}")
    
    return response.ok


def test_batch_predict():
    """Test batch prediction"""
    print("\n3. Testing /batch_predict...")
    
    # Generate test data for 3 symbols
    np.random.seed(42)
    n_bars = 150
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    features = {}
    qse_outputs = {}
    
    for symbol in symbols:
        features[symbol] = {
            'momentum_score': list(np.random.randn(n_bars) * 0.5),
            'log_return_1': list(np.random.randn(n_bars) * 0.01),
            'atr': list(np.random.uniform(0.001, 0.005, n_bars)),
            'efficiency_ratio': list(np.random.uniform(0.3, 0.8, n_bars)),
            'volatility': list(np.random.uniform(0.01, 0.03, n_bars)),
            'trend_persistence': list(np.random.uniform(0.4, 0.7, n_bars)),
            'correlation': list(np.random.uniform(-0.5, 0.5, n_bars)),
            'returns_autocorr': list(np.random.uniform(-0.3, 0.3, n_bars)),
        }
        
        qse_outputs[symbol] = {
            'validation': {
                'signal_validity_flag': True,
                'failure_reasons': []
            },
            'metrics': {
                'trend_persistence': 0.65,
                'stationarity_pvalue': 0.03,
                'z_score_normalized': 1.2
            }
        }
    
    request_data = {
        'symbols': symbols,
        'timeframe': '4H',
        'features': features,
        'qse_outputs': qse_outputs
    }
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=request_data)
    
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"   Symbols processed: {len(data)}")
        for symbol, result in data.items():
            pred = result['prediction']
            print(f"   {symbol}: {pred['regime']['regime_label']}, P={pred['signal']['momentum_success_prob']:.3f}, Allow={pred['allow_trade']}")
        print("   ✓ Batch prediction test passed")
    else:
        print(f"   ✗ Batch prediction failed: {response.text}")
    
    return response.ok


def test_metrics():
    """Test metrics endpoint"""
    print("\n4. Testing /metrics...")
    
    response = requests.get(f"{BASE_URL}/metrics")
    
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        metrics = data['metrics']
        print(f"   Total predictions: {metrics.get('total_predictions', 0)}")
        print(f"   Allow rate: {metrics.get('allow_rate', 0):.1%}")
        print(f"   Avg processing time: {metrics.get('avg_processing_time_ms', 0):.2f}ms")
        print("   ✓ Metrics test passed")
    else:
        print(f"   ✗ Metrics failed: {response.text}")
    
    return response.ok


def test_config():
    """Test config endpoint"""
    print("\n5. Testing /config...")
    
    response = requests.get(f"{BASE_URL}/config")
    
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        config = data['config']
        print(f"   Config hash: {data['config_hash']}")
        print(f"   Entry threshold: {config['signal_filter']['entry_threshold']}")
        print(f"   Allowed regimes: {config['signal_filter']['allowed_regimes']}")
        print("   ✓ Config test passed")
    else:
        print(f"   ✗ Config failed: {response.text}")
    
    return response.ok


def test_models():
    """Test model list endpoint"""
    print("\n6. Testing /models/list...")
    
    response = requests.get(f"{BASE_URL}/models/list")
    
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        models = data.get('models', [])
        print(f"   Registered models: {len(models)}")
        if models and isinstance(models, list):
            # Show first 3 models
            for i, model in enumerate(models):
                if i >= 3:
                    break
                if isinstance(model, dict):
                    print(f"   - {model.get('model_name', 'unknown')} {model.get('version', '')}")
        print("   ✓ Models list test passed")
    else:
        print(f"   ✗ Models list failed: {response.text}")
    
    return response.ok


def run_all_tests():
    """Run all API tests"""
    print("="*60)
    print("ML Layer API Test Suite")
    print("="*60)
    
    tests = [
        test_health,
        test_predict,
        test_batch_predict,
        test_metrics,
        test_config,
        test_models,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")


if __name__ == '__main__':
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API server")
        print("   Make sure the server is running:")
        print("   python -m arbitrex.ml_layer.api")
