import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
    
    # Initialize
    ok = mt5.initialize(
        path=os.environ.get('MT5_TERMINAL'),
        login=int(os.environ.get('MT5_LOGIN')),
        password=os.environ.get('MT5_PASSWORD'),
        server=os.environ.get('MT5_SERVER'),
    )
    
    if not ok:
        print(f"Init failed: {mt5.last_error()}")
        exit(1)
    
    print("MT5 initialized")
    
    # Try to get recent ticks for EURUSD
    print("\nTesting tick retrieval for EURUSD:")
    
    from_ts = int(time.time()) - 60
    ticks = mt5.copy_ticks_from("EURUSD", from_ts, 100, mt5.COPY_TICKS_ALL)
    
    if ticks is not None and len(ticks) > 0:
        print(f"✓ Got {len(ticks)} ticks")
        print(f"  First tick: {ticks[0]}")
        print(f"  Last tick: {ticks[-1]}")
    else:
        print(f"✗ No ticks returned")
        print(f"  Error: {mt5.last_error()}")
    
    # Try different symbols
    print("\nTrying other symbols:")
    for sym in ["GBPUSD", "XAUUSD", "US100"]:
        ticks = mt5.copy_ticks_from(sym, from_ts, 10, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            print(f"  ✓ {sym}: {len(ticks)} ticks")
        else:
            print(f"  ✗ {sym}: no ticks ({mt5.last_error()})")
    
    mt5.shutdown()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
