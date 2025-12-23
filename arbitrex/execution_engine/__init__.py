"""
Execution Engine - Configuration & Module Initialization
"""

import logging
from typing import Optional

from .engine import (
    ExecutionEngine,
    BrokerInterface,
    ExecutionDatabase,
    ExecutionRequest,
    ExecutionLog,
    ExecutionConfirmation,
    ExecutionStatus,
    ExecutionRejectionReason,
    OrderType,
)

LOG = logging.getLogger(__name__)


class ExecutionEngineConfig:
    """Execution Engine configuration parameters"""
    
    def __init__(
        self,
        max_slippage_pips: float = 10.0,
        order_timeout_seconds: int = 60,
        max_retries: int = 3,
        min_margin_cushion: float = 1.5,
        enable_logging: bool = True,
    ):
        self.max_slippage_pips = max_slippage_pips
        self.order_timeout_seconds = order_timeout_seconds
        self.max_retries = max_retries
        self.min_margin_cushion = min_margin_cushion
        self.enable_logging = enable_logging
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        assert self.max_slippage_pips > 0, "max_slippage_pips must be positive"
        assert self.order_timeout_seconds > 0, "order_timeout_seconds must be positive"
        assert self.max_retries > 0, "max_retries must be positive"
        assert self.min_margin_cushion > 1.0, "min_margin_cushion must be > 1.0"
        return True


def create_execution_engine(
    broker: BrokerInterface,
    database: Optional[ExecutionDatabase] = None,
    config: Optional[ExecutionEngineConfig] = None,
) -> ExecutionEngine:
    """
    Factory function to create and initialize Execution Engine.
    
    Args:
        broker: BrokerInterface (MT5, etc.)
        database: ExecutionDatabase (PostgreSQL, MongoDB, etc.)
        config: ExecutionEngineConfig with parameters
    
    Returns:
        Fully initialized ExecutionEngine
    """
    
    # Use defaults if not provided
    if database is None:
        database = ExecutionDatabase()
        LOG.info("Using in-memory execution database")
    
    if config is None:
        config = ExecutionEngineConfig()
        LOG.info("Using default ExecutionEngine configuration")
    
    # Validate configuration
    config.validate()
    
    # Create engine
    engine = ExecutionEngine(
        broker=broker,
        database=database,
        max_slippage_pips=config.max_slippage_pips,
        order_timeout_seconds=config.order_timeout_seconds,
        max_retries=config.max_retries,
        min_margin_cushion=config.min_margin_cushion,
    )
    
    LOG.info(
        f"Execution Engine initialized | "
        f"Max slippage: {config.max_slippage_pips}pips | "
        f"Timeout: {config.order_timeout_seconds}s | "
        f"Retries: {config.max_retries}"
    )
    
    return engine


__all__ = [
    'ExecutionEngine',
    'BrokerInterface',
    'ExecutionDatabase',
    'ExecutionRequest',
    'ExecutionLog',
    'ExecutionConfirmation',
    'ExecutionStatus',
    'ExecutionRejectionReason',
    'OrderType',
    'ExecutionEngineConfig',
    'create_execution_engine',
]
