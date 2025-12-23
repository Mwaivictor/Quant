"""
Execution Engine REST API

Provides endpoints to:
- Submit trades for execution
- Monitor execution status
- Query execution history
- Export audit trails

Note: This layer receives RPMOutput from the Risk & Portfolio Manager
and returns ExecutionConfirmation after executing orders.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from .engine import (
    ExecutionEngine,
    ExecutionConfirmation,
    ExecutionStatus,
    ExecutionRejectionReason,
    ExecutionGroup,
    ExecutionLeg,
)

try:
    from arbitrex.event_bus import get_event_bus, Event, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

LOG = logging.getLogger(__name__)


# ========================================
# REQUEST/RESPONSE SCHEMAS
# ========================================

class ExecuteTradeRequest(BaseModel):
    """Request to execute a trade (from RPM)"""
    rpm_output_json: dict = Field(..., description="Complete RPMOutput as dict")
    max_slippage_pips: float = Field(10.0, ge=0.0, description="Max allowed slippage")
    timeout_seconds: int = Field(60, ge=1, description="Order submission timeout")


class ExecutionStatusResponse(BaseModel):
    """Response from execution"""
    execution_id: str
    order_id: str
    status: str  # ExecutionStatus enum
    symbol: str
    direction: int
    intended_units: float
    executed_units: float
    fill_price: Optional[float] = None
    slippage_pips: Optional[float] = None
    rejection_reason: Optional[str] = None
    rejection_details: Optional[str] = None
    timestamp: str


class ExecutionMetricsResponse(BaseModel):
    """Execution engine performance metrics"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float  # Percentage
    avg_slippage_pips: float


class ExecutionHistoryResponse(BaseModel):
    """Single execution history entry"""
    execution_id: str
    order_id: str
    symbol: str
    direction: int
    intended_units: float
    executed_units: float
    fill_price: Optional[float] = None
    slippage_pips: Optional[float] = None
    status: str
    created_timestamp: str


class ExecutionLegRequest(BaseModel):
    """Single leg in multi-leg trade"""
    leg_id: str
    symbol: str
    direction: int  # +1 BUY, -1 SELL
    units: float
    asset_class: str = "FX"  # FX, EQUITY, COMMODITY, CRYPTO


class ExecuteMultiLegRequest(BaseModel):
    """Request to execute multi-leg trade"""
    strategy_id: str
    legs: List[ExecutionLegRequest]
    rpm_output: dict  # RPMOutput as dict
    max_group_slippage_pips: float = 10.0
    allow_partial_fills: bool = True


class ExecutionLegResponse(BaseModel):
    """Response for single leg"""
    leg_id: str
    symbol: str
    direction: int
    units: float
    asset_class: str
    status: str
    filled_units: float
    fill_price: Optional[float]
    slippage_pips: float


class ExecutionGroupResponse(BaseModel):
    """Response from multi-leg execution"""
    group_id: str
    strategy_id: str
    status: str
    legs: List[ExecutionLegResponse]
    avg_slippage_pips: float
    rollback_executed: bool
    completed_timestamp: Optional[str]


# ========================================
# API INITIALIZATION
# ========================================

app = FastAPI(
    title="Execution Engine API",
    description="Final execution layer for FX trading system",
    version="1.0.0",
)

# Global execution engine instance
execution_engine: Optional[ExecutionEngine] = None


def get_execution_engine() -> ExecutionEngine:
    """Get global execution engine instance"""
    if execution_engine is None:
        raise HTTPException(status_code=503, detail="Execution Engine not initialized")
    return execution_engine


# ========================================
# ENDPOINTS
# ========================================

@app.post("/execute", response_model=ExecutionStatusResponse)
async def execute_trade(request: ExecuteTradeRequest):
    """
    Execute a trade approved by RPM.
    
    Input: RPMOutput (from Risk & Portfolio Manager)
    Output: ExecutionConfirmation with order details
    
    Process:
    1. Validate RPM decision
    2. Check market conditions
    3. Submit order to broker
    4. Monitor fill
    5. Store execution log
    6. Return confirmation
    """
    
    engine = get_execution_engine()
    
    try:
        # CRITICAL FIX: Reconstruct RPMOutput from dict
        # In production: proper deserialization from RPM schema
        from arbitrex.risk_portfolio_manager.schemas import RPMOutput
        
        rpm_output = RPMOutput.from_dict(request.rpm_output_json)
        
        # Execute the trade via engine
        confirmation = engine.execute(rpm_output)
        
        return ExecutionStatusResponse(
            execution_id=confirmation.execution_id,
            order_id=confirmation.order_id,
            status=confirmation.status.value,
            symbol=confirmation.symbol,
            direction=confirmation.direction,
            intended_units=confirmation.intended_units,
            executed_units=confirmation.executed_units,
            fill_price=confirmation.fill_price,
            slippage_pips=confirmation.slippage_pips,
            rejection_reason=confirmation.rejection_reason.value if confirmation.rejection_reason else None,
            rejection_details=confirmation.rejection_details,
            timestamp=confirmation.timestamp.isoformat(),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Execution error: {str(e)}"
        )


@app.post("/execute/multi-leg", response_model=ExecutionGroupResponse)
async def execute_multi_leg(request: ExecuteMultiLegRequest):
    """
    Execute multi-leg trade (stat arb, spreads, baskets, etc).
    
    Input:
    - strategy_id: Parent strategy ID
    - legs: List of legs to execute
    - rpm_output: RPM decision
    
    Process:
    1. Create ExecutionGroup
    2. Validate all legs together (all-or-nothing)
    3. Submit in risk-optimal order (shorts before longs)
    4. Monitor each leg independently
    5. Resolve group status and determine if rollback needed
    
    Rollback Triggered If:
    - Any leg < 90% filled AND
    - Any leg > 10% filled
    (Closes filled legs at market to prevent unhedged exposure)
    """
    
    engine = get_execution_engine()
    
    try:
        # Reconstruct RPMOutput
        from arbitrex.risk_portfolio_manager.schemas import RPMOutput
        
        rpm_output = RPMOutput.from_dict(request.rpm_output)
        
        # Build ExecutionGroup
        legs = [
            ExecutionLeg(
                leg_id=leg_req.leg_id,
                symbol=leg_req.symbol,
                direction=leg_req.direction,
                units=leg_req.units,
                asset_class=leg_req.asset_class,
            )
            for leg_req in request.legs
        ]
        
        group = ExecutionGroup(
            group_id=f"group_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            strategy_id=request.strategy_id,
            rpm_decision_id=rpm_output.decision.order_id or "",
            legs=legs,
            max_group_slippage_pips=request.max_group_slippage_pips,
            allow_partial_fills=request.allow_partial_fills,
        )
        
        # Execute multi-leg
        completed_group = engine.execute_group(group, rpm_output)
        
        return ExecutionGroupResponse(
            group_id=completed_group.group_id,
            strategy_id=completed_group.strategy_id,
            status=completed_group.status.value,
            legs=[
                ExecutionLegResponse(
                    leg_id=leg.leg_id,
                    symbol=leg.symbol,
                    direction=leg.direction,
                    units=leg.units,
                    asset_class=leg.asset_class,
                    status=leg.status.value,
                    filled_units=leg.filled_units,
                    fill_price=leg.fill_price,
                    slippage_pips=leg.slippage_pips,
                )
                for leg in completed_group.legs
            ],
            avg_slippage_pips=completed_group.avg_group_slippage(),
            rollback_executed=completed_group.rollback_executed,
            completed_timestamp=completed_group.completion_timestamp.isoformat() if completed_group.completion_timestamp else None,
        )
    
    except Exception as e:
        LOG.error(f"Multi-leg execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-leg execution error: {str(e)}"
        )


@app.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of specific execution"""
    
    engine = get_execution_engine()
    
    try:
        log = engine.database.get_execution_log(execution_id)
        
        if not log:
            raise HTTPException(
                status_code=404,
                detail=f"Execution not found: {execution_id}"
            )
        
        return {
            'execution_id': log.execution_id,
            'order_id': log.order_id,
            'status': log.status.value,
            'symbol': log.symbol,
            'direction': log.direction,
            'intended_units': log.intended_units,
            'executed_units': log.executed_units,
            'fill_price': log.fill_price,
            'slippage_pips': log.slippage_pips,
            'rejection_reason': log.rejection_reason.value if log.rejection_reason else None,
            'created_timestamp': log.created_timestamp.isoformat(),
            'last_update': log.last_update_timestamp.isoformat(),
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching execution: {str(e)}"
        )


@app.get("/history")
async def get_execution_history(symbol: Optional[str] = None, limit: int = 50):
    """Get recent execution history"""
    
    engine = get_execution_engine()
    
    try:
        history = engine.get_execution_history(symbol=symbol, limit=limit)
        
        return {
            'total': len(history),
            'executions': history
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching history: {str(e)}"
        )


@app.get("/metrics", response_model=ExecutionMetricsResponse)
async def get_execution_metrics():
    """Get execution engine performance metrics"""
    
    engine = get_execution_engine()
    
    try:
        metrics = engine.get_execution_metrics()
        
        return ExecutionMetricsResponse(
            total_executions=metrics['total_executions'],
            successful_executions=metrics['successful_executions'],
            failed_executions=metrics['failed_executions'],
            success_rate=metrics['success_rate'],
            avg_slippage_pips=metrics['avg_slippage_pips'],
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching metrics: {str(e)}"
        )


@app.get("/audit_trail")
async def get_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export execution audit trail for compliance.
    
    Format: ISO 8601 dates (YYYY-MM-DDTHH:MM:SS)
    """
    
    engine = get_execution_engine()
    
    try:
        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.utcnow() - timedelta(days=7)  # Last 7 days default
        
        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.utcnow()
        
        # Get audit trail
        trail = engine.export_audit_trail(start, end)
        
        return {
            'start_date': start.isoformat(),
            'end_date': end.isoformat(),
            'total_executions': len(trail),
            'executions': trail
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting audit trail: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    engine = get_execution_engine()
    metrics = engine.get_execution_metrics()
    
    return {
        'status': 'healthy',
        'engine': 'execution_engine',
        'executions': metrics['total_executions'],
        'success_rate': metrics['success_rate'],
        'timestamp': datetime.utcnow().isoformat()
    }


# ========================================
# INITIALIZATION
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize execution engine on startup"""
    global execution_engine
    
    from . import create_execution_engine, BrokerInterface, ExecutionDatabase
    
    # Create broker interface (would connect to MT5 pool in production)
    broker = BrokerInterface(broker_name="MT5")
    broker.connect()
    
    # Create database (would use PostgreSQL in production)
    database = ExecutionDatabase()
    
    # Create execution engine
    execution_engine = create_execution_engine(
        broker=broker,
        database=database,
    )
    
    # Start event bus
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.start()
        LOG.info("✓ Event bus started for Execution Engine")
    
    LOG.info("Execution Engine API initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global execution_engine
    
    if execution_engine and execution_engine.broker:
        execution_engine.broker.disconnect()
    
    # Stop event bus
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.stop()
        LOG.info("✓ Event bus stopped")
    
    LOG.info("Execution Engine API shutting down")


# ========================================
# MAIN ENTRY POINT
# ========================================

def run_execution_api(host: str = "0.0.0.0", port: int = 8006):
    """
    Run Execution Engine API server.
    
    Args:
        host: Host address
        port: Port number (default 8006)
    """
    import uvicorn
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_execution_api()
