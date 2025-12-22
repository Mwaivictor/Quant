import os
from dotenv import load_dotenv

load_dotenv()

from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
from arbitrex.stream.ws_server import get_publisher
from arbitrex.raw_layer.config import TRADING_UNIVERSE
from arbitrex.raw_layer.health import init_health_monitor


def make_sessions_from_env():
    creds = {
        'terminal_path': os.environ.get('MT5_TERMINAL'),
        'login': int(os.environ['MT5_LOGIN']) if os.environ.get('MT5_LOGIN') else None,
        'password': os.environ.get('MT5_PASSWORD'),
        'server': os.environ.get('MT5_SERVER'),
    }
    return {
        'main': {
            'terminal_path': creds['terminal_path'],
            'login': creds['login'],
            'password': creds['password'],
            'server': creds['server'],
        }
    }


def get_all_symbols():
    all_symbols = []
    for grp in TRADING_UNIVERSE.values():
        all_symbols.extend(grp)
    return all_symbols


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    sessions = make_sessions_from_env()
    symbols = get_all_symbols()

    print(f"Starting Arbitrex Tick Streaming Stack")
    print(f"Symbols to track: {len(symbols)}")
    print(f"Symbols: {', '.join(symbols[:10])}..." if len(symbols) > 10 else f"Symbols: {', '.join(symbols)}")

    pool = MT5ConnectionPool(
        sessions,
        symbols=symbols,
        session_logs_dir=os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'mt5', 'session_logs')
    )

    # Initialize health monitor and register components
    print(f"\nInitializing health monitor...")
    health_monitor = init_health_monitor()
    health_monitor.set_mt5_pool(pool)
    if hasattr(pool, '_tick_queue'):
        health_monitor.set_tick_queue(pool._tick_queue)
    print(f"✓ Health monitor ready at http://localhost:8766/health")

    # Start tick collector FIRST (before setting publisher)
    print(f"\nStarting tick collector...")
    pool.start_tick_collector(symbols, os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw'),
                              poll_interval=0.5, flush_interval=3.0)

    # Set publisher - this needs to happen before Uvicorn starts
    # but the actual event loop will be captured on first publish
    print(f"Setting up WebSocket publisher...")
    pool.set_tick_publisher(get_publisher())

    # Run WebSocket ASGI server
    print(f"\nStarting WebSocket server on http://0.0.0.0:8000")
    print(f"Open arbitrex/stream/demo.html in your browser and click Connect\n")
    
    import uvicorn
    try:
        uvicorn.run("arbitrex.stream.ws_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    finally:
        print("\nShutting down...")
        pool.stop_tick_collector()
        pool.close()
