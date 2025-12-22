import os
import time
from dotenv import load_dotenv
load_dotenv()
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
from arbitrex.raw_layer.config import TRADING_UNIVERSE

creds = {
    'terminal_path': os.environ.get('MT5_TERMINAL'),
    'login': int(os.environ['MT5_LOGIN']) if os.environ.get('MT5_LOGIN') else None,
    'password': os.environ.get('MT5_PASSWORD'),
    'server': os.environ.get('MT5_SERVER'),
}

sessions = {'main': {'terminal_path': creds['terminal_path'], 'login': creds['login'], 'password': creds['password'], 'server': creds['server']}}

# Get symbols from TRADING_UNIVERSE
symbols = [s for group in TRADING_UNIVERSE.values() for s in group]

pool = MT5ConnectionPool(sessions, symbols=symbols, session_logs_dir=os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'mt5', 'session_logs'))
print('Pool created, starting tick collector for EURUSD...')
pool.start_tick_collector(['EURUSD'], os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw'), poll_interval=0.5, flush_interval=3.0)
try:
    time.sleep(8)
finally:
    print('Stopping tick collector...')
    pool.stop_tick_collector()
    pool.close()
    print('Stopped and closed')
