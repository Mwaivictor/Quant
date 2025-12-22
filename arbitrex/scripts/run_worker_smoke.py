import os
import json
from dotenv import load_dotenv
load_dotenv()
from arbitrex.raw_layer.orchestrator import _worker_init_and_ingest

creds = {
    'MT5_LOGIN': os.environ.get('MT5_LOGIN'),
    'MT5_PASSWORD': os.environ.get('MT5_PASSWORD'),
    'MT5_SERVER': os.environ.get('MT5_SERVER'),
    'MT5_TERMINAL': os.environ.get('MT5_TERMINAL'),
}

task = {
    'symbols': ['EURUSD'],
    'timeframes': ['1H','4H'],
    'bars_per_tf': {'1H': 240, '4H': 240},
    'creds': creds,
    'output_dir': os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw'),
    'tick_logging': False,
    'rate_limit': 0.05,
}

res = _worker_init_and_ingest(task)
print(json.dumps(res, indent=2))
