import sys
import time
import os
import types
import json


def _make_fake_mt5(ticks_per_call=None):
    m = types.SimpleNamespace()
    # constants
    m.COPY_TICKS_ALL = 0

    class AccountInfo:
        def __init__(self):
            self.login = 12345

    def initialize(path=None):
        return True

    def login(login, password=None, server=None):
        return True

    def account_info():
        return AccountInfo()

    # iterator over provided ticks
    calls = {'i': 0}

    def copy_ticks_from(symbol, from_ts, count, flags):
        # return synthetic ticks: list of objects with attributes time,bid,ask,last,volume
        i = calls['i']
        calls['i'] += 1
        out = []
        base = int(time.time())
        for j in range(3):
            class T:
                pass
            t = T()
            t.time = base + i * 10 + j
            t.bid = 1.1000 + (j * 0.0001)
            t.ask = 1.1001 + (j * 0.0001)
            t.last = t.ask
            t.volume = 100 + j
            out.append(t)
        return out

    m.initialize = initialize
    m.login = login
    m.account_info = account_info
    m.copy_ticks_from = copy_ticks_from
    return m


def test_mt5_pool_tick_replay(tmp_path, monkeypatch):
    # place cwd into tmp_path so DB and outputs are in temp
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        fake = _make_fake_mt5()
        sys.modules['MetaTrader5'] = fake

        from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

        sessions = {'main': {'terminal_path': None, 'login': None, 'password': None, 'server': None}}
        logs_dir = os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'mt5', 'session_logs')
        pool = MT5ConnectionPool(sessions, session_logs_dir=logs_dir)
        try:
            # force market open for test
            pool._is_market_open = lambda: True
            base_dir = os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw')
            pool.start_tick_collector(['EURUSD'], base_dir, poll_interval=0.1, flush_interval=0.5)
            time.sleep(1.5)
            pool.stop_tick_collector()
            pool.close()

            # check ticks file exists
            tick_dir = os.path.join(base_dir, 'ticks', 'fx', 'EURUSD')
            assert os.path.isdir(tick_dir)
            files = os.listdir(tick_dir)
            assert any(f.endswith('.csv') for f in files)
        finally:
            try:
                pool.close()
            except Exception:
                pass
    finally:
        # cleanup
        if 'MetaTrader5' in sys.modules:
            del sys.modules['MetaTrader5']
        os.chdir(cwd)
