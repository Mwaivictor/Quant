import os
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
    print("✓ MT5 library imported")
    
    # Try to initialize with credentials from .env
    creds = {
        'login': int(os.environ.get('MT5_LOGIN', 0)),
        'password': os.environ.get('MT5_PASSWORD'),
        'server': os.environ.get('MT5_SERVER'),
        'path': os.environ.get('MT5_TERMINAL'),
    }
    
    print(f"\nAttempting MT5 initialization:")
    print(f"  Login: {creds['login']}")
    print(f"  Server: {creds['server']}")
    print(f"  Terminal: {creds['path']}")
    
    ok = mt5.initialize(
        path=creds['path'],
        login=creds['login'],
        password=creds['password'],
        server=creds['server']
    )
    
    if ok:
        print("\n✓ MT5 initialized successfully!")
        print(f"  Version: {mt5.version()}")
        print(f"  Account: {mt5.account_info()}")
        mt5.shutdown()
    else:
        err = mt5.last_error()
        print(f"\n✗ MT5 initialization failed: {err}")
        
except ImportError as e:
    print(f"✗ MT5 library not found: {e}")
    print("\nInstall with: pip install MetaTrader5")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
