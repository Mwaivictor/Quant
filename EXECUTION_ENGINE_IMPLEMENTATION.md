"""
EXECUTION ENGINE - PROFESSIONAL IMPLEMENTATION GUIDE

Senior Quantitative Analyst & Trading Systems Developer
Version: 1.0.0
Date: December 23, 2025

═══════════════════════════════════════════════════════════════════════════════

SYSTEM ARCHITECTURE

                    ┌─────────────────────────────────────┐
                    │  SIGNAL GENERATION ENGINE           │
                    │  (Generate trade signals)           │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  RISK & PORTFOLIO MANAGER (RPM)     │
                    │  (Approve/reject with sizing)       │
                    │  Returns: RPMOutput                 │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  EXECUTION ENGINE (THIS LAYER)      │
                    │  (Execute approved trades)          │
                    │  Returns: ExecutionConfirmation     │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  BROKER (MT5 / Other)               │
                    │  (Real order execution)             │
                    │  Fills orders, returns results      │
                    └─────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

CORE PRINCIPLES

1. SEPARATION OF CONCERNS
   - Signal Engine decides WHAT to trade
   - RPM decides WHETHER to trade and HOW BIG
   - Execution Engine decides HOW TO TRADE SAFELY
   
2. NO DECISION-MAKING IN EXECUTION
   - Never re-size a trade
   - Never override RPM decision
   - Never generate new trading decisions
   - Only execute what RPM approves
   
3. FAULT TOLERANCE
   - Retry network failures
   - Gracefully handle broker rejections
   - Never lose execution records
   - Always log events for audit trail
   
4. AUDITABILITY
   - Every execution logged
   - Full context preserved
   - Timestamps and signatures
   - Compliance-ready format

═══════════════════════════════════════════════════════════════════════════════

INPUT SPECIFICATION: RPMOutput (From Risk & Portfolio Manager)

The Execution Engine receives:

RPMOutput {
    decision: TradeDecision {
        status: "APPROVED" | "REJECTED" | "ADJUSTED"
        
        if APPROVED:
            approved_trade: ApprovedTrade {
                symbol: "EURUSD"  ← FX pair
                direction: 1      ← LONG
                position_units: 95000  ← FINAL SIZE (DO NOT RE-SIZE)
                confidence_score: 0.78  ← Signal quality
                regime: "TRENDING"  ← Market regime
                base_units: 100000  ← Size before adjustments
                confidence_adjustment: 0.95  ← Applied multiplier
                regime_adjustment: 1.0  ← Applied multiplier
                correlation_adjustment: 0.90  ← Applied multiplier
                atr: 0.0015  ← Volatility measure
                vol_percentile: 0.65  ← Vol regime
                risk_per_trade: 500.0  ← Capital at risk
                kelly_fraction: 0.25  ← Kelly % (optional)
                expectancy: 0.0035  ← Expectancy (optional)
                ... (other optional fields)
            }
        
        if REJECTED:
            rejected_trade: RejectedTrade {
                rejection_reason: "KILL_SWITCH_TRIGGERED"
                rejection_details: "..."
            }
        
        order_id: "rpm_order_123"  ← Link to RPM order
    }
    
    portfolio_state: PortfolioState {
        trading_halted: False  ← MUST CHECK THIS
        current_drawdown: 0.05  ← For context
        equity: 98500.0  ← Account value
        ... (other state)
    }
    
    risk_metrics: RiskMetrics { ... }
    config_hash: "abc123def456"  ← RPM config
    rpm_version: "1.0.0"  ← Governance
}

═══════════════════════════════════════════════════════════════════════════════

EXECUTION FLOW (DETAILED)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PRE-EXECUTION VALIDATION (Gatekeeper)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VALIDATION CHECKLIST:                                                       │
│  ✓ Is trading halted? (Check portfolio_state.trading_halted)                │
│  ✓ Is market open? (Symbol specific)                                        │
│  ✓ Is symbol tradable? (Broker rules)                                       │
│  ✓ Can we fetch market data? (Bid/ask prices)                               │
│  ✓ Are market prices reasonable? (Sanity check)                             │
│  ✓ Is spread acceptable? (< max_slippage_pips)                              │
│  ✓ Do we have sufficient margin? (With cushion)                             │
│  ✓ Is liquidity available? (Order book depth)                               │
│                                                                              │
│  If ANY check fails:                                                         │
│    → Reject execution                                                        │
│    → Log rejection with reason                                              │
│    → Return ExecutionConfirmation(status=REJECTED)                          │
│    → Do NOT submit order to broker                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EXECUTION REQUEST CREATION                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Create ExecutionRequest:                                                    │
│    - Generate unique request_id                                             │
│    - Link to RPM decision                                                   │
│    - Store in database (for audit trail)                                    │
│    - Record all parameters                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: EXECUTION LOG CREATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Create ExecutionLog:                                                        │
│    - Generate unique execution_id                                           │
│    - Copy all trade details from ApprovedTrade                              │
│    - Set status = PENDING                                                   │
│    - Record creation timestamp                                              │
│    - Store in database                                                      │
│    - Add to active_executions (in-memory tracking)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ORDER SUBMISSION WITH RETRY LOGIC                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Submit order to broker (MT5):                                               │
│                                                                              │
│    ATTEMPT 1:                                                                │
│    ├─ Call broker.place_order()                                             │
│    ├─ If success → Get order_id, proceed to STEP 5                          │
│    └─ If failure → Check retry_count                                        │
│                                                                              │
│    ATTEMPT 2-3:                                                              │
│    ├─ Wait 1 second (back-off)                                              │
│    ├─ Retry order submission                                                │
│    ├─ If success → Proceed to STEP 5                                        │
│    └─ If failure & retry_count > max_retries → REJECT                       │
│                                                                              │
│    REJECTION:                                                                │
│    ├─ Set status = REJECTED                                                 │
│    ├─ Set rejection_reason = BROKER_REJECTION                               │
│    ├─ Update database                                                       │
│    └─ Return ExecutionConfirmation(REJECTED)                                │
│                                                                              │
│  Parameters:                                                                 │
│    symbol = ApprovedTrade.symbol                                            │
│    direction = ApprovedTrade.direction                                      │
│    units = ApprovedTrade.position_units (NEVER CHANGE)                      │
│    order_type = OrderType.MARKET (for MVP)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: ORDER MONITORING & FILL TRACKING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Poll broker until order fills:                                              │
│                                                                              │
│    While (time_elapsed < timeout_seconds):                                  │
│      ├─ Query order status from broker                                      │
│      ├─ If filled:                                                          │
│      │  ├─ Record fill_price                                                │
│      │  ├─ Record filled_units                                              │
│      │  ├─ Measure slippage                                                 │
│      │  └─ Proceed to STEP 6                                                │
│      └─ Wait 0.5 seconds, retry                                             │
│                                                                              │
│    If timeout:                                                               │
│      ├─ Set status = EXPIRED                                                │
│      ├─ Set rejection_reason = ORDER_TIMEOUT                                │
│      └─ Return ExecutionConfirmation(EXPIRED)                               │
│                                                                              │
│  Polling Parameters:                                                         │
│    - Timeout: execution_request.timeout_seconds (default 60s)               │
│    - Poll interval: 0.5 seconds                                             │
│    - Graceful degradation if broker slow                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: SLIPPAGE MEASUREMENT                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Calculate execution quality:                                                │
│                                                                              │
│    slippage_pips = |fill_price - intended_price| / point_size               │
│                                                                              │
│    Where:                                                                    │
│      - fill_price: Actual execution price from broker                       │
│      - intended_price: RPM suggested entry (market mid at approval time)    │
│      - point_size: 0.0001 for most FX pairs (1 pip)                         │
│                                                                              │
│  Slippage Categories:                                                        │
│    - 0-1 pips: Excellent execution                                          │
│    - 1-3 pips: Good execution                                               │
│    - 3-5 pips: Acceptable execution                                         │
│    - > 5 pips: Poor execution (log for analysis)                            │
│                                                                              │
│  Action:                                                                     │
│    - Always record slippage in database                                     │
│    - Use for performance monitoring                                         │
│    - Optional: Reject if slippage > threshold                               │
│      (Set max_slippage_pips in ExecutionRequest)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: PARTIAL FILL HANDLING                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  If partial fill (filled_units < intended_units):                            │
│                                                                              │
│    Option A (Recommended for MVP):                                           │
│    ├─ Accept the partial fill                                               │
│    ├─ Record actual_units = filled_units                                    │
│    ├─ Set status = PARTIALLY_FILLED                                         │
│    └─ Log for trader attention                                              │
│                                                                              │
│    Option B (Advanced):                                                      │
│    ├─ Retry remaining units                                                 │
│    ├─ Create secondary order for unfilled portion                           │
│    ├─ Batch both fills in execution log                                     │
│    └─ Complex retry logic                                                   │
│                                                                              │
│  Recommendation:                                                             │
│    For MVP: Accept partial fills (Option A)                                  │
│    Production: Implement retry logic (Option B)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: POST-EXECUTION LOGGING & STORAGE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Update ExecutionLog with final details:                                     │
│                                                                              │
│    ├─ order_id (from broker)                                                │
│    ├─ fill_price (from order)                                               │
│    ├─ executed_units (actual fill)                                          │
│    ├─ slippage_pips (calculated)                                            │
│    ├─ fill_timestamp                                                        │
│    ├─ status (FILLED, PARTIALLY_FILLED, REJECTED, etc.)                     │
│    └─ last_update_timestamp                                                 │
│                                                                              │
│  Store in Database:                                                          │
│    ├─ Database.store_execution_log(log)                                     │
│    ├─ Create audit trail record                                             │
│    ├─ Ensure durability (replication, backup)                               │
│    └─ Make searchable by execution_id, order_id, symbol, date               │
│                                                                              │
│  Update Metrics:                                                             │
│    ├─ total_executions++                                                    │
│    ├─ successful_executions++ (if FILLED)                                   │
│    ├─ failed_executions++ (if REJECTED/EXPIRED)                             │
│    ├─ Update avg_slippage_pips                                              │
│    └─ Calculate success_rate                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: RETURN CONFIRMATION                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Create ExecutionConfirmation:                                               │
│                                                                              │
│    {                                                                         │
│      execution_id: "exec_abc123",                                            │
│      order_id: "broker_order_456",                                           │
│      status: ExecutionStatus.FILLED,                                         │
│      symbol: "EURUSD",                                                       │
│      direction: 1,                                                           │
│      intended_units: 95000,                                                  │
│      executed_units: 95000,                                                  │
│      fill_price: 1.0950,                                                     │
│      slippage_pips: 0.5,                                                     │
│      timestamp: "2025-12-23T14:30:45Z"                                       │
│    }                                                                         │
│                                                                              │
│  Return to caller:                                                           │
│    ├─ Upstream monitors execution_id                                         │
│    ├─ Can query status anytime                                              │
│    ├─ Full details in database                                              │
│    └─ Audit trail immutable                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

ERROR HANDLING & RESILIENCE

Scenario 1: Network Failure During Order Submission
├─ Retry up to max_retries times
├─ Use exponential back-off (1s, 2s, 4s, ...)
├─ If all retries fail: status = REJECTED, reason = BROKER_REJECTION
└─ Log full error details for debugging

Scenario 2: Broker Rejects Order
├─ Broker returns error code (insufficient margin, symbol not available, etc.)
├─ Do NOT retry
├─ Set status = REJECTED
├─ Log rejection_reason and details
└─ Return ExecutionConfirmation immediately

Scenario 3: Market Prices Unreasonable
├─ Example: Bid price = 1.100, Ask price = 1.095 (backwards)
├─ Validation check catches this
├─ Reject execution before sending to broker
├─ Set status = REJECTED, reason = VALIDATION_FAILED
└─ Alert trader/operator

Scenario 4: Spread Too Wide
├─ Bid/Ask spread > max_slippage_pips
├─ Reject pre-execution (don't send to broker)
├─ Set status = REJECTED, reason = SPREAD_TOO_WIDE
├─ Preserve signal for later (market may improve)
└─ Log timestamp for analysis

Scenario 5: Insufficient Margin
├─ Available margin < required_margin
├─ Reject pre-execution
├─ Set status = REJECTED, reason = MARGIN_INSUFFICIENT
├─ Notify trader to deposit funds
└─ Store log for accounting

Scenario 6: Trading Halted
├─ portfolio_state.trading_halted = True
├─ Check before any execution
├─ Immediately reject with halt_reason
├─ No retries
└─ Log kill switch event

Scenario 7: Order Timeout
├─ Submitted to broker but no fill after timeout_seconds
├─ Set status = EXPIRED
├─ Log for investigation
├─ Optional: Manual follow-up to cancel stale order
└─ Notify trader

Scenario 8: Partial Fill
├─ Broker filled 70,000 units of 100,000 requested
├─ Accept partial (MVP approach)
├─ Set status = PARTIALLY_FILLED
├─ Log for trader attention
├─ Optional: Retry remaining 30,000 units
└─ Production: Implement retry logic

═══════════════════════════════════════════════════════════════════════════════

DATABASE SCHEMA EXAMPLE (PostgreSQL)

CREATE TABLE execution_logs (
    execution_id VARCHAR(36) PRIMARY KEY,
    order_id VARCHAR(36) NOT NULL,
    request_id VARCHAR(36) NOT NULL,
    rpm_decision_id VARCHAR(36) NOT NULL,
    
    -- Trade details
    symbol VARCHAR(10) NOT NULL,
    direction SMALLINT NOT NULL,  -- 1 or -1
    intended_units DECIMAL(15, 2) NOT NULL,
    executed_units DECIMAL(15, 2),
    
    -- Prices
    intended_price DECIMAL(10, 5),
    fill_price DECIMAL(10, 5),
    slippage_pips DECIMAL(6, 1),
    
    -- Status
    status VARCHAR(20) NOT NULL,
    rejection_reason VARCHAR(50),
    rejection_details TEXT,
    
    -- Metadata
    confidence_score DECIMAL(3, 2),
    regime VARCHAR(20),
    risk_per_trade DECIMAL(12, 2),
    
    -- Governance
    model_version VARCHAR(20),
    rpm_version VARCHAR(20),
    execution_engine_version VARCHAR(20),
    
    -- Timestamps
    created_timestamp TIMESTAMP NOT NULL,
    submission_timestamp TIMESTAMP,
    fill_timestamp TIMESTAMP,
    last_update_timestamp TIMESTAMP NOT NULL,
    
    -- Indexes
    INDEX idx_symbol_date (symbol, created_timestamp),
    INDEX idx_status (status),
    INDEX idx_order_id (order_id),
    INDEX idx_request_id (request_id)
);

═══════════════════════════════════════════════════════════════════════════════

API ENDPOINTS

1. POST /execute
   Input: RPMOutput
   Output: ExecutionConfirmation
   Purpose: Submit trade for execution

2. GET /executions/{execution_id}
   Output: Full execution details
   Purpose: Query execution status anytime

3. GET /history?symbol=EURUSD&limit=50
   Output: Recent execution history
   Purpose: Review past executions

4. GET /metrics
   Output: Performance metrics
   Purpose: Monitor execution quality

5. GET /audit_trail?start_date=...&end_date=...
   Output: Compliance audit trail
   Purpose: Export for regulatory review

6. GET /health
   Output: Health status
   Purpose: System monitoring

═══════════════════════════════════════════════════════════════════════════════

MONITORING & ALERTS

Key Metrics to Monitor:
├─ Success rate (target: > 95%)
├─ Average slippage (target: < 2 pips)
├─ Rejection rate (track by reason)
├─ Order timeout rate
├─ Network error rate
└─ Processing latency

Alerts to Implement:
├─ Success rate drops below 90%
├─ Average slippage exceeds 5 pips
├─ Rejection count exceeds threshold
├─ Network errors detected
├─ Database errors
└─ Broker connection lost

═══════════════════════════════════════════════════════════════════════════════

COMPLIANCE & AUDIT

Every execution must be:
✓ Uniquely identified (execution_id)
✓ Traceable to RPM decision (rpm_decision_id)
✓ Timestamped (creation, submission, fill)
✓ Linked to broker order (order_id)
✓ Recorded in immutable database
✓ Searchable and exportable
✓ Compliant with regulatory requirements

Audit Trail Contents:
├─ What was executed (symbol, size, direction)
├─ Why it was executed (RPM decision)
├─ When it was executed (timestamps)
├─ How well it was executed (fill price, slippage)
├─ Whether it succeeded (status)
├─ If it failed, why (rejection_reason)
└─ Full context for compliance

═══════════════════════════════════════════════════════════════════════════════

DEPLOYMENT CHECKLIST

Pre-Production:
☐ All unit tests passing
☐ Integration tests with broker API
☐ Database connection tested
☐ Error handling tested
☐ Retry logic validated
☐ Slippage measurement verified
☐ Audit trail generation working

Production Deployment:
☐ Database replicated (master-slave)
☐ Database backups running
☐ Monitoring & alerting configured
☐ Broker API failover ready
☐ Network latency optimized
☐ Logging aggregated (ELK stack)
☐ Compliance audit ready

Post-Deployment:
☐ Monitor success rates daily
☐ Review slippage trends
☐ Track rejection reasons
☐ Audit trail exports working
☐ Performance stable
☐ Zero silent failures

═══════════════════════════════════════════════════════════════════════════════

FINAL PRINCIPLES

1. EXECUTE PRECISELY
   - Translate RPM decision into broker order exactly
   - No interpretation, no rounding, no guessing
   - Use exact units approved by RPM

2. FAIL SAFELY
   - Pre-validate before sending to broker
   - Retry network failures
   - Gracefully handle rejections
   - Never execute partial/wrong order

3. LOG EXHAUSTIVELY
   - Every step recorded
   - Full context preserved
   - Immutable audit trail
   - Compliance-ready format

4. RESPECT UPSTREAM DECISIONS
   - Never override RPM sizing
   - Never re-calculate position
   - Never generate new trades
   - Only execute what approved

5. MAINTAIN DISCIPLINE
   - Kill switches respected
   - Trading halts honored
   - Margin cushions enforced
   - Broker constraints followed

═══════════════════════════════════════════════════════════════════════════════

"Upstream decides what to trade.
Execution decides how to trade safely.
The market decides the outcome."

═══════════════════════════════════════════════════════════════════════════════
"""
