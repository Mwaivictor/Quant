"""
Start Feature Engine API Server

Quick start script for the Feature Engine REST API.
"""

import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

LOG = logging.getLogger(__name__)


def main():
    """Start the API server"""
    
    print("=" * 80)
    print("ARBITREX FEATURE ENGINE API")
    print("=" * 80)
    print()
    print("Starting REST API server...")
    print()
    print("API Documentation: http://localhost:8001/docs")
    print("OpenAPI Schema: http://localhost:8001/openapi.json")
    print()
    print("Available Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /config        - Current configuration")
    print("  GET  /schema/{tf}   - Feature schema")
    print("  POST /compute       - Compute features")
    print("  GET  /features/{s}/{tf} - Get stored features")
    print("  GET  /vector/{s}/{tf}/{ts} - Get feature vector")
    print("  GET  /versions/{s}/{tf} - List versions")
    print("  GET  /health        - Health status")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        uvicorn.run(
            "arbitrex.feature_engine.api:app",
            host="0.0.0.0",
            port=8001,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("API Server stopped")
        print("=" * 80)


if __name__ == "__main__":
    main()
