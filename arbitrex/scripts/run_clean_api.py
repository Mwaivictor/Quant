"""
Start the Clean Data API server

This script starts the FastAPI server with appropriate configuration.
"""

import uvicorn
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    """Start the API server"""
    
    print("="*80)
    print("ArbitreX Clean Data API")
    print("="*80)
    print()
    print("Starting server...")
    print("  Host: 0.0.0.0")
    print("  Port: 8001")
    print()
    print("API Documentation:")
    print("  Swagger UI: http://localhost:8001/docs")
    print("  ReDoc: http://localhost:8001/redoc")
    print()
    print("Health Endpoint:")
    print("  http://localhost:8001/health")
    print()
    print("Press CTRL+C to stop")
    print("="*80)
    print()
    
    uvicorn.run(
        "arbitrex.clean_data.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
