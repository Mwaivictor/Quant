"""
Start ML Layer API Server

Launches the FastAPI server for ML Layer operations.
"""

import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("="*60)
    print("Starting ML Layer API Server")
    print("="*60)
    print("API Documentation: http://localhost:8003/docs")
    print("Health Check: http://localhost:8003/health")
    print("Metrics: http://localhost:8003/metrics")
    print("="*60)
    
    uvicorn.run(
        "arbitrex.ml_layer.api:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
