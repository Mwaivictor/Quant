"""Command-line helpers for the Raw Data Layer.

This small CLI is intentionally minimal: the raw layer should be driven by
deterministic programmatic runners (or orchestration). The CLI aids quick
manual runs for debugging.
"""
from __future__ import annotations
import argparse
import os
from .mt5_pool import MT5ConnectionPool
from .ingest import ingest_ohlcv_once


def main():
    parser = argparse.ArgumentParser(description="Arbitrex raw data ingestor (minimal CLI)")
    sub = parser.add_subparsers(dest='cmd')

    p_ingest = sub.add_parser('ingest', help='Ingest a single symbol/timeframe')
    p_ingest.add_argument('--symbol', required=True)
    p_ingest.add_argument('--timeframe', required=True, choices=['1H','4H','1D','1M'])
    p_ingest.add_argument('--cycle-id', required=True)
    p_ingest.add_argument('--login', required=False)
    p_ingest.add_argument('--password', required=False)
    p_ingest.add_argument('--server', required=False)

    p_univ = sub.add_parser('export-universe', help='Export trading universe from latest MT5 symbols JSON')
    p_univ.add_argument('--json', required=False, help='Path to mt5 symbols JSON (defaults to last export)')
    p_univ.add_argument('--out', required=False, help='Output CSV path for universe')

    args = parser.parse_args()

    if args.cmd == 'ingest':
        from .config import TRADING_UNIVERSE
        sessions = {
            'cli_session': {'terminal_path': None, 'login': int(args.login) if args.login else None, 'password': args.password, 'server': args.server}
        }
        symbols = [s for group in TRADING_UNIVERSE.values() for s in group]
        pool = MT5ConnectionPool(sessions, symbols=symbols, session_logs_dir=os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'mt5', 'session_logs'))
        try:
            res = ingest_ohlcv_once(pool, args.symbol, args.timeframe, args.cycle_id)
            print('Ingestion result:', res)
        finally:
            pool.close()

    elif args.cmd == 'export-universe':
        # find latest mt5_symbols_*.json in source_registry
        sr = os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'metadata', 'source_registry')
        files = [f for f in os.listdir(sr) if f.startswith('mt5_symbols_') and f.endswith('.json')]
        if not files and not args.json:
            raise SystemExit('No MT5 symbols JSON found; run the symbol export first')
        src = args.json if args.json else os.path.join(sr, sorted(files)[-1])
        out = args.out if args.out else os.path.join(sr, 'trading_universe.csv')
        from .ingest import generate_trading_universe_from_json
        count = generate_trading_universe_from_json(src, out, only_fx=True)
        print(f'Wrote trading universe with {count} symbols to {out}')

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
