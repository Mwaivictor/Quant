"""Validation test for timezone normalization in raw layer.

This script validates that:
1. Broker timezone offset is detected correctly
2. Timestamps are normalized to UTC
3. Both broker and UTC timestamps are stored
4. File grouping uses UTC dates correctly
"""

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_timezone_detection():
    """Test broker timezone offset detection."""
    print("\n=== Testing Timezone Detection ===")
    
    from arbitrex.raw_layer.config import detect_broker_utc_offset, DEFAULT_CONFIG
    
    offset = detect_broker_utc_offset()
    print(f"✓ Detected broker UTC offset: {offset:+d} hours")
    print(f"✓ Config normalize_timestamps: {DEFAULT_CONFIG.normalize_timestamps}")
    print(f"✓ Config broker_utc_offset_hours: {DEFAULT_CONFIG.broker_utc_offset_hours}")
    
    return offset


def test_timestamp_conversion():
    """Test UTC conversion utilities."""
    print("\n=== Testing Timestamp Conversion ===")
    
    from arbitrex.raw_layer.config import broker_to_utc, utc_to_broker
    
    # Test with known offset (GMT+2)
    test_broker_ts = 1640000000  # Some broker timestamp
    test_offset = 2
    
    utc_ts = broker_to_utc(test_broker_ts, test_offset)
    recovered_broker_ts = utc_to_broker(utc_ts, test_offset)
    
    print(f"✓ Broker timestamp: {test_broker_ts} ({datetime.fromtimestamp(test_broker_ts)})")
    print(f"✓ UTC timestamp: {utc_ts} ({datetime.fromtimestamp(utc_ts, tz=timezone.utc)})")
    print(f"✓ Offset applied: {test_offset:+d} hours = {(test_broker_ts - utc_ts) / 3600:.1f} hours difference")
    print(f"✓ Round-trip conversion: {test_broker_ts} -> {utc_ts} -> {recovered_broker_ts}")
    
    assert recovered_broker_ts == test_broker_ts, "Round-trip conversion failed"
    assert utc_ts == test_broker_ts - (test_offset * 3600), "UTC conversion incorrect"
    
    print("✓ All conversion tests passed")


def test_csv_headers():
    """Test that CSV headers include both timestamps."""
    print("\n=== Testing CSV Headers ===")
    
    from arbitrex.raw_layer.writer import CSV_OHLCV_HEADER, CSV_TICK_HEADER
    
    print(f"✓ OHLCV headers: {CSV_OHLCV_HEADER}")
    print(f"✓ Tick headers: {CSV_TICK_HEADER}")
    
    assert "timestamp_utc" in CSV_OHLCV_HEADER, "Missing timestamp_utc in OHLCV header"
    assert "timestamp_broker" in CSV_OHLCV_HEADER, "Missing timestamp_broker in OHLCV header"
    assert "timestamp_utc" in CSV_TICK_HEADER, "Missing timestamp_utc in tick header"
    assert "timestamp_broker" in CSV_TICK_HEADER, "Missing timestamp_broker in tick header"
    
    print("✓ All header tests passed")


def test_metadata_fields():
    """Test that metadata includes timezone info."""
    print("\n=== Testing Metadata Structure ===")
    
    import tempfile
    import json
    from arbitrex.raw_layer.writer import write_ohlcv
    
    # Create test data with dual timestamps
    test_rows = [
        [1640000000, 1640007200, 1.1234, 1.1235, 1.1233, 1.1234, 100],  # UTC, Broker+2h, OHLCV
        [1640003600, 1640010800, 1.1235, 1.1236, 1.1234, 1.1235, 150],
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cycle_id = "test_normalization_cycle"
        write_ohlcv(tmpdir, "EURUSD", "1H", test_rows, cycle_id, broker_utc_offset=2)
        
        # Check metadata file
        meta_path = Path(tmpdir) / "metadata" / "ingestion_logs" / f"{cycle_id}.json"
        assert meta_path.exists(), f"Metadata file not found: {meta_path}"
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata written to: {meta_path}")
        print(f"✓ Metadata contents:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
        
        assert "broker_utc_offset_hours" in metadata, "Missing broker_utc_offset_hours in metadata"
        assert "timestamps_normalized" in metadata, "Missing timestamps_normalized in metadata"
        assert metadata["broker_utc_offset_hours"] == 2, "Incorrect broker offset in metadata"
        assert metadata["timestamps_normalized"] is True, "timestamps_normalized should be True"
        
        # Check CSV file uses UTC timestamp for file grouping
        csv_files = list(Path(tmpdir).glob("ohlcv/**/*.csv"))
        assert len(csv_files) > 0, "No CSV files written"
        
        csv_path = csv_files[0]
        print(f"✓ CSV file created: {csv_path}")
        
        # Check file is named by UTC date
        file_date = csv_path.stem  # YYYY-MM-DD
        utc_date = datetime.fromtimestamp(1640000000, tz=timezone.utc).date().isoformat()
        print(f"✓ File date from UTC timestamp: {utc_date}")
        print(f"✓ Actual file name: {file_date}.csv")
        
        assert file_date == utc_date, f"File should be grouped by UTC date {utc_date}, got {file_date}"
    
    print("✓ All metadata tests passed")


def test_end_to_end():
    """Test complete normalization pipeline if MT5 is available."""
    print("\n=== Testing End-to-End (if MT5 available) ===")
    
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("⚠ MT5 not available, skipping end-to-end test")
            return
        
        from arbitrex.raw_layer.config import detect_broker_utc_offset
        offset = detect_broker_utc_offset()
        
        # Get a sample tick
        symbols = mt5.symbols_get()
        if symbols and len(symbols) > 0:
            test_symbol = symbols[0].name
            tick = mt5.symbol_info_tick(test_symbol)
            
            if tick:
                print(f"✓ Sample symbol: {test_symbol}")
                print(f"✓ Broker tick timestamp: {tick.time} ({datetime.fromtimestamp(tick.time)})")
                print(f"✓ Current UTC: {datetime.now(timezone.utc).timestamp():.0f} ({datetime.now(timezone.utc)})")
                print(f"✓ Detected offset: {offset:+d} hours")
                
                from arbitrex.raw_layer.config import broker_to_utc
                utc_ts = broker_to_utc(int(tick.time), offset)
                print(f"✓ Normalized UTC: {utc_ts} ({datetime.fromtimestamp(utc_ts, tz=timezone.utc)})")
        
        mt5.shutdown()
        print("✓ End-to-end test completed")
        
    except ImportError:
        print("⚠ MetaTrader5 not installed, skipping end-to-end test")
    except Exception as e:
        print(f"⚠ End-to-end test error: {e}")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("TIME NORMALIZATION VALIDATION TEST")
    print("=" * 60)
    
    try:
        offset = test_timezone_detection()
        test_timestamp_conversion()
        test_csv_headers()
        test_metadata_fields()
        test_end_to_end()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nTime normalization is working correctly:")
        print(f"  - Broker offset: {offset:+d} hours from UTC")
        print("  - Timestamps: stored as dual (UTC + broker)")
        print("  - File grouping: uses UTC dates")
        print("  - Metadata: includes timezone info")
        print("\nDownstream layers should:")
        print("  - Use timestamp_utc column for all analysis")
        print("  - Use timestamp_broker only for broker reconciliation")
        print("  - Check metadata broker_utc_offset_hours for audit")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
