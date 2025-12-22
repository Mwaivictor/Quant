Grafana dashboard for Arbitrex Raw Layer
======================================

Files:
- `arbitrex_raw_layer_dashboard.json` — dashboard JSON you can import into Grafana.
- `import_dashboard.py` — convenience script to POST the dashboard to Grafana's HTTP API.

Quick import (UI):
1. Open Grafana web UI.
2. Dashboard → Manage → Import.
3. Upload `arbitrex_raw_layer_dashboard.json` or paste its contents.
4. Choose your Prometheus data source and import.

Quick import (script):
Set these environment variables (example PowerShell):
```powershell
$env:GRAFANA_URL = 'http://localhost:3000'
$env:GRAFANA_API_KEY = '<YOUR_GRAFANA_API_KEY>'
python .\arbitrex\raw_layer\grafana\import_dashboard.py
```

If you don't have an API key you may set `GRAFANA_USER` and `GRAFANA_PASS` instead, but API key is recommended.

After import: verify panels show data and that the Prometheus data source is selected.
