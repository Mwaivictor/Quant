"""
Start RPM API Server

Starts the Risk & Portfolio Manager REST API on port 8005.
"""

import sys
from arbitrex.risk_portfolio_manager.api import run_api

if __name__ == "__main__":
    print("="*80)
    print(" RISK & PORTFOLIO MANAGER (RPM) API SERVER")
    print(" The Gatekeeper with Absolute Veto Authority")
    print("="*80)
    print()
    print("Starting RPM API on http://0.0.0.0:8005")
    print()
    print("Available endpoints:")
    print("  POST /process_trade  - Process trade intent (MAIN ENDPOINT)")
    print("  GET  /health         - Get RPM health status")
    print("  GET  /portfolio      - Get portfolio state")
    print("  GET  /metrics        - Get risk metrics")
    print("  GET  /kill_switches  - Get kill switch status")
    print("  POST /halt           - Manually halt trading")
    print("  POST /resume         - Resume trading after halt")
    print("  GET  /config         - Get RPM configuration")
    print("  POST /reset/daily    - Reset daily metrics")
    print("  POST /reset/weekly   - Reset weekly metrics")
    print()
    print("API Documentation: http://localhost:8005/docs")
    print()
    print("="*80)
    print()
    
    try:
        run_api(host="0.0.0.0", port=8005)
    except KeyboardInterrupt:
        print("\n\nRPM API server stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError starting RPM API: {e}")
        sys.exit(1)
