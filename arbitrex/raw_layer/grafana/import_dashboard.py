#!/usr/bin/env python3
"""Import the Arbitrex Raw Layer Grafana dashboard via Grafana HTTP API.

Usage:
  - set `GRAFANA_URL` (default http://localhost:3000)
  - set `GRAFANA_API_KEY` (recommended) OR set `GRAFANA_USER` and `GRAFANA_PASS`
  - run: `python grafana/import_dashboard.py`

The script reads `arbitrex_raw_layer_dashboard.json` from the same folder and POSTs
it to `/api/dashboards/db` with `overwrite=true`.
"""
import os
import sys
import json
from pathlib import Path

try:
    import requests
except Exception:
    print('Please install requests: pip install requests')
    raise

HERE = Path(__file__).parent
JSON_PATH = HERE / 'arbitrex_raw_layer_dashboard.json'

GRAFANA_URL = os.environ.get('GRAFANA_URL', 'http://localhost:3000').rstrip('/')
API_KEY = os.environ.get('GRAFANA_API_KEY')
GRAFANA_USER = os.environ.get('GRAFANA_USER')
GRAFANA_PASS = os.environ.get('GRAFANA_PASS')


def load_dashboard():
    if not JSON_PATH.exists():
        print('Dashboard JSON not found at', JSON_PATH)
        sys.exit(2)
    return json.loads(JSON_PATH.read_text(encoding='utf8'))


def build_headers():
    if API_KEY:
        return {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    return {'Content-Type': 'application/json'}


def import_dashboard(dashboard_json):
    url = f"{GRAFANA_URL}/api/dashboards/db"
    payload = { 'dashboard': dashboard_json.get('dashboard', dashboard_json), 'overwrite': True }
    headers = build_headers()

    auth = (GRAFANA_USER, GRAFANA_PASS) if (GRAFANA_USER and GRAFANA_PASS and not API_KEY) else None

    resp = requests.post(url, json=payload, headers=headers, auth=auth, timeout=15)
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return resp.status_code, body


def main():
    print('Loading dashboard JSON from', JSON_PATH)
    dashboard_json = load_dashboard()
    print('Posting to Grafana at', GRAFANA_URL)
    code, body = import_dashboard(dashboard_json)
    if 200 <= code < 300:
        print('Imported dashboard successfully:', body)
        return 0
    print('Failed to import dashboard. HTTP', code)
    print(body)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
