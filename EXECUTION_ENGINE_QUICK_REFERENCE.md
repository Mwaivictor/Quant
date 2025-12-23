"""
EXECUTION ENGINE - QUICK REFERENCE GUIDE

Implementation Summary for Senior Quant/Trading Systems Developer
"""

# ════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

SYSTEM FLOW:
    Signal Engine
        ↓
    Risk & Portfolio Manager (RPM) - APPROVES/REJECTS with sizing
        ↓
    Execution Engine ← YOU ARE HERE
        ↓
    Broker (MT5)
        ↓
    Orders Executed + Database Logged


# ════════════════════════════════════════════════════════════════════════════
# CORE COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

1. ExecutionEngine
   ├─ Main orchestrator
   ├─ Coordinates validation, submission, monitoring
   └─ Returns ExecutionConfirmation

2. BrokerInterface
   ├─ Abstraction over MT5 (or other brokers)
   ├─ Handles market data, order submission, status checks
   └─ Retry logic encapsulated

3. ExecutionDatabase
   ├─ Stores all execution logs
   ├─ Immutable audit trail
   └─ In production: PostgreSQL/MongoDB

4. ExecutionRequest
   ├─ Pre-execution instruction
   ├─ Includes all context from RPM
   └─ Stored before sending to broker

5. ExecutionLog
   ├─ Complete execution record
   ├─ Updated as order progresses
   └─ Final state stored in database

6. ExecutionConfirmation
   ├─ Return value to caller
   ├─ Status, fill details, slippage
   └─ Links back to execution_id

7. REST API (/api.py)
   ├─ Expose execution via HTTP
   ├─ Monitor executions, get history
   └─ Export audit trails


# ════════════════════════════════════════════════════════════════════════════
# EXECUTION STAGES (9 Steps)
# ════════════════════════════════════════════════════════════════════════════

STAGE 1: PRE-EXECUTION VALIDATION
├─ Check: Trading halted?
├─ Check: Market open?
├─ Check: Symbol tradable?
├─ Check: Market data available?
├─ Check: Spread acceptable?
├─ Check: Margin sufficient?
├─ Check: Liquidity available?
└─ Result: APPROVE or REJECT before broker

STAGE 2: CREATE EXECUTION REQUEST
├─ Generate unique request_id
├─ Link to RPM decision
├─ Store in database (audit trail)
└─ Set parameters (timeout, max slippage, retries)

STAGE 3: CREATE EXECUTION LOG
├─ Generate unique execution_id
├─ Copy trade details from ApprovedTrade
├─ Set status = PENDING
├─ Store in database
└─ Track in memory

STAGE 4: SUBMIT ORDER WITH RETRY
├─ Attempt 1: broker.place_order()
├─ Attempt 2-3: Retry with 1s backoff
├─ Success: Get order_id, proceed
└─ Failure: Reject execution

STAGE 5: MONITOR ORDER UNTIL FILLED
├─ Poll broker status every 0.5s
├─ Wait for fill or timeout (default 60s)
├─ If filled: Record fill_price, filled_units
└─ If timeout: Set status = EXPIRED

STAGE 6: MEASURE SLIPPAGE
├─ Calculate: |fill_price - intended_price| / 0.0001
├─ Classify execution quality
├─ Log slippage for monitoring
└─ Optional: Reject if > max_slippage_pips

STAGE 7: HANDLE PARTIAL FILLS
├─ MVP: Accept partial (status = PARTIALLY_FILLED)
├─ Production: Retry remaining units
├─ Log both events
└─ Alert trader if unusual

STAGE 8: LOG & STORE
├─ Update ExecutionLog with final details
├─ Store in database (immutable)
├─ Update metrics
└─ Make searchable

STAGE 9: RETURN CONFIRMATION
├─ Create ExecutionConfirmation
├─ Return to caller
├─ Caller can query execution_id anytime
└─ Database has full audit trail


# ════════════════════════════════════════════════════════════════════════════
# KEY CONSTRAINTS
# ════════════════════════════════════════════════════════════════════════════

NEVER EVER:
✗ Re-calculate position size (use RPM's size exactly)
✗ Override RPM's risk assessment
✗ Generate new trading decisions
✗ Ignore trading_halted flag
✗ Silently fail (always log)
✗ Deviate from position_units

ALWAYS:
✓ Use position_units from ApprovedTrade exactly
✓ Validate before submitting to broker
✓ Log every step for audit trail
✓ Retry network failures (3x default)
✓ Measure and track slippage
✓ Respect kill switches
✓ Maintain margin cushion (1.5x)


# ════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

NETWORK FAILURE
├─ Action: Retry 3x with 1s backoff
├─ Success: Resume normal flow
└─ Failure: Reject, set status = REJECTED

BROKER REJECTION
├─ Action: Do NOT retry
├─ Reason: Likely technical issue (margin, symbol, etc.)
├─ Response: Log and notify trader
└─ Status: Set to REJECTED

SPREAD TOO WIDE
├─ Action: Pre-validation catches this
├─ Response: Reject before submitting to broker
└─ Status: Set to REJECTED, reason = SPREAD_TOO_WIDE

ORDER TIMEOUT
├─ Action: Wait up to 60s for fill
├─ Response: Cancel or abandon order
└─ Status: Set to EXPIRED

TRADING HALTED
├─ Action: Check portfolio_state.trading_halted FIRST
├─ Response: Reject immediately
└─ Status: Set to REJECTED, reason = TRADING_HALTED

INSUFFICIENT MARGIN
├─ Action: Check available_margin before submitting
├─ Response: Reject pre-execution
└─ Status: Set to REJECTED, reason = MARGIN_INSUFFICIENT


# ════════════════════════════════════════════════════════════════════════════
# DATABASE SCHEMA (Key Fields)
# ════════════════════════════════════════════════════════════════════════════

execution_logs:
├─ execution_id (PK, UUID)          ← Unique identifier
├─ order_id                         ← Broker order ID
├─ request_id                       ← Link to ExecutionRequest
├─ rpm_decision_id                  ← Link to RPM decision
├─ symbol, direction, units         ← Trade details
├─ fill_price, slippage_pips        ← Execution quality
├─ status (PENDING/SUBMITTED/FILLED/REJECTED/EXPIRED)
├─ rejection_reason, rejection_details
├─ timestamps (created, submitted, filled, updated)
├─ governance (model_version, rpm_version, engine_version)
└─ indexes: symbol, date, status, order_id


# ════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

POST /execute
├─ Input: RPMOutput (from RPM)
├─ Processing: 9-stage execution flow
└─ Output: ExecutionConfirmation

GET /executions/{execution_id}
├─ Input: execution_id
└─ Output: Full execution details

GET /history?symbol=EURUSD&limit=50
├─ Input: Optional symbol filter
└─ Output: List of recent executions

GET /metrics
├─ Input: None
├─ Output: Performance metrics (success rate, slippage)
└─ Purpose: Monitor execution quality

GET /audit_trail?start_date=...&end_date=...
├─ Input: Date range
├─ Output: Full execution logs (compliance-ready)
└─ Purpose: Regulatory audit

GET /health
├─ Output: Service health status
└─ Purpose: Monitoring


# ════════════════════════════════════════════════════════════════════════════
# MONITORING METRICS
# ════════════════════════════════════════════════════════════════════════════

EXECUTION QUALITY:
├─ Success rate (target: > 95%)
├─ Average slippage (target: < 2 pips)
├─ Partial fill rate
└─ Order timeout rate

RELIABILITY:
├─ Network error rate
├─ Broker rejection rate
├─ System error rate
└─ Database uptime

PERFORMANCE:
├─ Order submission latency
├─ Fill time (submission to fill)
├─ API response time
└─ Database query time


# ════════════════════════════════════════════════════════════════════════════
# PRODUCTION CHECKLIST
# ════════════════════════════════════════════════════════════════════════════

CODE:
☐ All unit tests passing
☐ Integration tests with broker
☐ Error handling tested
☐ Retry logic working
☐ Database transaction tested
☐ Audit trail generation verified

INFRASTRUCTURE:
☐ Database replicated (master-slave)
☐ Database backups running
☐ Broker API failover ready
☐ Network latency optimized
☐ Logging aggregated
☐ Monitoring configured
☐ Alerts configured

DEPLOYMENT:
☐ Load tested
☐ Chaos testing done
☐ Compliance review complete
☐ Documentation updated
☐ Runbooks written
☐ Trader training done
☐ Gradual rollout (10% → 50% → 100%)

OPERATION:
☐ Monitor success rates daily
☐ Track slippage trends
☐ Review errors weekly
☐ Audit trail exports working
☐ Performance stable
☐ Zero silent failures
☐ Regular compliance audits


# ════════════════════════════════════════════════════════════════════════════
# DESIGN PRINCIPLES
# ════════════════════════════════════════════════════════════════════════════

1. SEPARATION OF CONCERNS
   ├─ Signal Engine: WHAT to trade
   ├─ Risk Manager: WHETHER & SIZE
   ├─ Execution Engine: HOW to execute
   └─ Broker: EXECUTE orders

2. NO DECISION-MAKING
   ├─ Never re-size positions
   ├─ Never override risk decisions
   ├─ Only execute approved trades
   └─ Never generate new signals

3. DETERMINISTIC
   ├─ Same input → Same output
   ├─ No randomness or guessing
   ├─ Fully auditable
   └─ Reproducible

4. FAULT-TOLERANT
   ├─ Retry network failures
   ├─ Graceful degradation
   ├─ No silent failures
   └─ Always log

5. COMPLIANCE-READY
   ├─ Full audit trail
   ├─ Immutable records
   ├─ Searchable & exportable
   └─ Regulatory compliant


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE EXECUTION FLOW
# ════════════════════════════════════════════════════════════════════════════

INPUT (from RPM):
    RPMOutput {
        decision: {
            status: "APPROVED",
            approved_trade: {
                symbol: "EURUSD",
                direction: 1,
                position_units: 95000,  ← USE THIS EXACTLY
                confidence_score: 0.78,
                atr: 0.0015,
                ...
            }
        },
        portfolio_state: {
            trading_halted: False,
            ...
        }
    }

PROCESS:
    Step 1: Validate
        ├─ Check trading_halted = False ✓
        ├─ Check market open ✓
        ├─ Check spread < 10 pips ✓
        └─ PASS validation

    Step 2-3: Create request & log
        ├─ request_id: "req_abc123"
        └─ execution_id: "exec_def456"

    Step 4: Submit to broker
        ├─ Attempt 1: broker.place_order(
            symbol="EURUSD",
            direction=1,
            units=95000  ← EXACTLY 95000
        )
        └─ Success: order_id = "order_xyz789"

    Step 5: Monitor until fill
        ├─ Poll broker every 0.5s
        ├─ Wait max 60s
        └─ Filled: fill_price = 1.0950, units = 95000

    Step 6: Measure slippage
        ├─ intended_price = 1.0945 (market at approval)
        ├─ fill_price = 1.0950
        ├─ slippage_pips = (1.0950 - 1.0945) / 0.0001 = 5 pips
        └─ Acceptable

    Step 8: Store in database
        ├─ execution_logs table
        ├─ status = "FILLED"
        └─ All details immutable

RETURN (to caller):
    ExecutionConfirmation {
        execution_id: "exec_def456",
        order_id: "order_xyz789",
        status: "FILLED",
        symbol: "EURUSD",
        direction: 1,
        intended_units: 95000,
        executed_units: 95000,
        fill_price: 1.0950,
        slippage_pips: 5.0
    }

DATABASE AUDIT TRAIL:
    execution_logs[exec_def456]:
        ├─ execution_id: exec_def456
        ├─ order_id: order_xyz789
        ├─ rpm_decision_id: rpm_dec_123
        ├─ status: FILLED
        ├─ symbol: EURUSD
        ├─ direction: 1
        ├─ intended_units: 95000
        ├─ executed_units: 95000
        ├─ fill_price: 1.0950
        ├─ slippage_pips: 5.0
        ├─ created_timestamp: 2025-12-23T14:30:00Z
        ├─ submission_timestamp: 2025-12-23T14:30:01Z
        ├─ fill_timestamp: 2025-12-23T14:30:02Z
        ├─ rpm_version: 1.0.0
        └─ (searchable, immutable, compliant)


# ════════════════════════════════════════════════════════════════════════════
# FILE STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

arbitrex/execution_engine/
├─ __init__.py                 ← Module init + factory function
├─ engine.py                   ← Core ExecutionEngine class
├─ api.py                      ← FastAPI REST endpoints
└─ [future]
    ├─ database.py             ← PostgreSQL/MongoDB adapter
    ├─ broker.py               ← MT5 connection pool
    ├─ monitoring.py           ← Metrics & alerting
    └─ tests/
        ├─ test_engine.py      ← Unit tests
        ├─ test_broker.py      ← Broker integration tests
        └─ test_execution.py    ← End-to-end tests


# ════════════════════════════════════════════════════════════════════════════
# QUICK START
# ════════════════════════════════════════════════════════════════════════════

1. Initialize:
    from arbitrex.execution_engine import create_execution_engine, BrokerInterface, ExecutionDatabase
    
    broker = BrokerInterface(broker_name="MT5")
    broker.connect()
    
    database = ExecutionDatabase()
    
    ee = create_execution_engine(broker, database)

2. Execute trade:
    confirmation = ee.execute(rpm_output)
    
    print(f"Status: {confirmation.status}")
    print(f"Fill Price: {confirmation.fill_price}")
    print(f"Slippage: {confirmation.slippage_pips} pips")

3. Query status:
    log = ee.database.get_execution_log(execution_id)
    
    print(f"Order ID: {log.order_id}")
    print(f"Filled: {log.executed_units} units")

4. Get history:
    history = ee.get_execution_history(symbol="EURUSD", limit=50)
    
    for execution in history:
        print(f"{execution['symbol']} {execution['status']}")

5. Export audit trail:
    audit = ee.export_audit_trail(start_date, end_date)
    
    # Ready for compliance review


# ════════════════════════════════════════════════════════════════════════════
# GOLDEN RULES
# ════════════════════════════════════════════════════════════════════════════

1. EXECUTE PRECISELY
   └─ Use position_units exactly as approved

2. FAIL SAFELY
   └─ Pre-validate before broker submission

3. LOG EXHAUSTIVELY
   └─ Every step recorded for audit

4. RESPECT UPSTREAM
   └─ Never override RPM or kill switches

5. MAINTAIN DISCIPLINE
   └─ Enforce constraints rigorously

═══════════════════════════════════════════════════════════════════════════════
"""
