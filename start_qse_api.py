"""
Start QSE API Server

Launches the Quantitative Statistics Engine REST API on port 8002.
"""

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("="*60)
    print("  ArbitreX - Quantitative Statistics Engine API")
    print("="*60)
    print()
    print("Starting server on http://0.0.0.0:8002")
    print()
    print("Available endpoints:")
    print("  POST /validate          - Validate signal")
    print("  GET  /regime/{symbol}   - Get market regime")
    print("  GET  /health            - System health")
    print("  GET  /health/{symbol}   - Symbol health")
    print("  GET  /failures          - Failure breakdown")
    print("  GET  /recent            - Recent validations")
    print("  GET  /config            - Configuration")
    print("  POST /reset-health      - Reset metrics (admin)")
    print()
    print("Press CTRL+C to stop")
    print("="*60)
    print()
    
    uvicorn.run(
        "arbitrex.quant_stats.api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
