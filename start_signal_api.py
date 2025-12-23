"""
Start Signal Engine API Server

Run the Signal Generation Engine REST API on port 8004.
"""

import uvicorn
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/signal_engine_api.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Start Signal Engine API server"""
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SIGNAL GENERATION ENGINE API")
    logger.info("=" * 80)
    logger.info("Starting server on http://127.0.0.1:8004")
    logger.info("Swagger UI: http://127.0.0.1:8004/docs")
    logger.info("ReDoc: http://127.0.0.1:8004/redoc")
    logger.info("=" * 80)
    
    try:
        uvicorn.run(
            "arbitrex.signal_engine.api:app",
            host="127.0.0.1",
            port=8004,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down Signal Engine API...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
