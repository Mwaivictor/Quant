"""
Start the Raw Data Layer API server

Launches FastAPI server with Swagger documentation on port 8000.
"""

import uvicorn
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    """Start the Raw Data API server"""
    
    print("="*80)
    print("ArbitreX Raw Data Layer API")
    print("="*80)
    print()
    print("Starting server...")
    print("  Host: 0.0.0.0")
    print("  Port: 8000")
    print()
    print("API Documentation:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc: http://localhost:8000/redoc")
    print()
    print("Health Endpoints:")
    print("  System Health: http://localhost:8000/health")
    print("  Detailed Health: http://localhost:8000/health/detailed")
    print("  MT5 Status: http://localhost:8000/health/mt5")
    print()
    print("Data Endpoints:")
    print("  List Symbols: http://localhost:8000/raw/symbols")
    print("  Trading Universe: http://localhost:8000/symbols/universe")
    print()
    print("Press CTRL+C to stop")
    print("="*80)
    print()
    
    uvicorn.run(
        "arbitrex.raw_layer.api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
