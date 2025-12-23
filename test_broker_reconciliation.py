"""
Test Broker Reconciliation & Drift Detection

Validates the critical P0 feature that prevents catastrophic position drift.

Test Categories:
1. Drift Detection (various severity levels)
2. Action Execution (log, alert, auto-correct, halt)
3. Position Matching (matched, missing, phantom)
4. Edge Cases (empty positions, zero equity, null prices)
5. Performance (reconciliation latency)
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from arbitrex.risk_portfolio_manager.broker_reconciliation import (
    BrokerReconciliationEngine,
    DriftSeverity,
    DriftAction,
    PositionDrift,
    ReconciliationReport,
)


@pytest.fixture
def reconciliation_engine():
    """Create reconciliation engine with test thresholds"""
    return BrokerReconciliationEngine(
        reconciliation_interval=60.0,
        minimal_drift_threshold=0.005,     # 0.5%
        warning_drift_threshold=0.01,      # 1%
        critical_drift_threshold=0.02,     # 2% (NEW - auto-correct threshold)
        catastrophic_drift_threshold=0.05, # 5% (NEW - halt threshold)
        auto_correct_enabled=True,
        halt_on_catastrophic=True,
    )


@pytest.fixture
def internal_positions():
    """Sample internal positions"""
    return {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': Decimal('10000'),
            'current_price': Decimal('1.1000'),
            'unrealized_pnl': Decimal('100.00'),
            'side': 'long',
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': Decimal('5000'),
            'current_price': Decimal('1.2500'),
            'unrealized_pnl': Decimal('-50.00'),
            'side': 'short',
        },
    }


@pytest.fixture
def broker_positions_matched():
    """Broker positions that match internal (no drift)"""
    return {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10000,
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,  # BUY
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,  # SELL
        },
    }


# ========================================
# TEST 1: NO DRIFT (Perfect Match)
# ========================================

def test_no_drift_perfect_match(reconciliation_engine, internal_positions, broker_positions_matched):
    """Test perfect match between internal and broker - no drift"""
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions_matched,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.NONE
    assert report.recommended_action == DriftAction.NONE
    assert report.action_taken == DriftAction.NONE
    assert report.matched_positions == 2
    assert report.missing_positions == 0
    assert report.phantom_positions == 0
    assert report.total_drift_pct < 0.001  # < 0.1%


# ========================================
# TEST 2: MINIMAL DRIFT (Log Warning)
# ========================================

def test_minimal_drift_log_warning(reconciliation_engine, internal_positions):
    """Test minimal drift (0.6%) - should log warning only"""
    # Broker has slightly different quantity (0.6% drift - above minimal threshold)
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10060,  # +60 units (0.6% drift)
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.MINIMAL
    assert report.recommended_action == DriftAction.LOG
    assert report.action_taken == DriftAction.LOG
    assert 0.005 < report.max_drift_pct < 0.01  # 0.5-1%


# ========================================
# TEST 3: WARNING DRIFT (Alert Ops)
# ========================================

def test_warning_drift_alert_sent(reconciliation_engine, internal_positions):
    """Test warning-level drift (1.5%) - should send alert"""
    alerts_sent = []
    
    def mock_alert_callback(alert_data):
        alerts_sent.append(alert_data)
    
    reconciliation_engine.alert_callback = mock_alert_callback
    
    # 1.5% drift on EURUSD (above warning threshold of 1%)
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10150,  # +150 units (1.5% drift)
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.WARNING
    assert report.recommended_action == DriftAction.ALERT
    assert report.action_taken == DriftAction.ALERT
    assert len(alerts_sent) == 1
    assert alerts_sent[0]['severity'] == 'warning'


# ========================================
# TEST 4: CRITICAL DRIFT (Auto-Correct)
# ========================================

def test_critical_drift_auto_correct(reconciliation_engine, internal_positions):
    """Test critical drift (2%) - should trigger auto-correct"""
    # 2% drift on EURUSD
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10200,  # +200 units (2% drift)
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.CRITICAL
    assert report.recommended_action == DriftAction.AUTO_CORRECT
    assert report.action_taken == DriftAction.AUTO_CORRECT
    assert 0.009 < report.total_drift_pct < 0.03  # ~1% average (2% on one position, 0% on other)


# ========================================
# TEST 5: CATASTROPHIC DRIFT (Halt Trading)
# ========================================

def test_catastrophic_drift_halt_trading(reconciliation_engine, internal_positions):
    """Test catastrophic drift (7%) - should halt trading"""
    # 7% drift on EURUSD (above critical threshold of 5%)
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10700,  # +700 units (7% drift)
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.CATASTROPHIC
    assert report.recommended_action == DriftAction.HALT_TRADING
    assert report.action_taken == DriftAction.HALT_TRADING
    assert report.trading_halted is True
    assert report.max_drift_pct > 0.05  # > 5%


# ========================================
# TEST 6: MISSING POSITION (In Broker, Not Internal)
# ========================================

def test_missing_position_catastrophic(reconciliation_engine, internal_positions):
    """Test missing position - exists in broker but not internal"""
    # Broker has an extra position
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10000,
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
        'USDJPY': {  # MISSING FROM INTERNAL
            'symbol': 'USDJPY',
            'quantity': 20000,
            'price_current': 110.00,
            'profit': 500.00,
            'type': 0,
        },
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.CATASTROPHIC
    assert report.missing_positions == 1
    assert report.phantom_positions == 0
    assert report.matched_positions == 2
    assert report.trading_halted is True
    
    # Find the missing position drift
    missing_drift = next(d for d in report.position_drifts if d.symbol == 'USDJPY')
    assert missing_drift.internal_quantity == Decimal('0')
    assert missing_drift.broker_quantity == Decimal('20000')
    assert missing_drift.drift_severity == DriftSeverity.CATASTROPHIC


# ========================================
# TEST 7: PHANTOM POSITION (In Internal, Not Broker)
# ========================================

def test_phantom_position_catastrophic(reconciliation_engine, internal_positions):
    """Test phantom position - exists internally but not in broker"""
    # Broker missing GBPUSD position
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10000,
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        # GBPUSD is MISSING from broker
    }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.CATASTROPHIC
    assert report.missing_positions == 0
    assert report.phantom_positions == 1
    assert report.matched_positions == 1
    assert report.trading_halted is True
    
    # Find the phantom position drift
    phantom_drift = next(d for d in report.position_drifts if d.symbol == 'GBPUSD')
    assert phantom_drift.internal_quantity == Decimal('5000')
    assert phantom_drift.broker_quantity == Decimal('0')
    assert phantom_drift.drift_severity == DriftSeverity.CATASTROPHIC


# ========================================
# TEST 8: EQUITY DRIFT
# ========================================

def test_equity_drift_critical(reconciliation_engine, internal_positions, broker_positions_matched):
    """Test equity drift - positions match but equity differs"""
    # 3% equity drift
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions_matched,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('103000'),  # 3% higher
    )
    
    assert report.equity_drift == Decimal('3000')
    assert 0.02 < report.equity_drift_pct < 0.04  # ~3%
    assert report.overall_severity == DriftSeverity.CRITICAL


# ========================================
# TEST 9: EDGE CASE - Empty Positions
# ========================================

def test_empty_positions_no_drift(reconciliation_engine):
    """Test with no positions - should report no drift"""
    report = reconciliation_engine.reconcile(
        internal_positions={},
        broker_positions={},
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    assert report.overall_severity == DriftSeverity.NONE
    assert report.matched_positions == 0
    assert report.missing_positions == 0
    assert report.phantom_positions == 0
    assert report.total_drift_pct == 0.0


# ========================================
# TEST 10: PERFORMANCE - Reconciliation Latency
# ========================================

def test_reconciliation_performance(reconciliation_engine):
    """Test reconciliation completes within acceptable latency"""
    # Create 50 positions
    internal_positions = {}
    broker_positions = {}
    
    for i in range(50):
        symbol = f"SYM{i:02d}"
        internal_positions[symbol] = {
            'symbol': symbol,
            'quantity': Decimal('1000'),
            'current_price': Decimal('100.00'),
            'unrealized_pnl': Decimal('10.00'),
            'side': 'long',
        }
        broker_positions[symbol] = {
            'symbol': symbol,
            'quantity': 1000,
            'price_current': 100.00,
            'profit': 10.00,
            'type': 0,
        }
    
    report = reconciliation_engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    # Reconciliation should complete in < 100ms
    assert report.reconciliation_duration_ms < 100
    assert report.matched_positions == 50


# ========================================
# TEST 11: STATISTICS TRACKING
# ========================================

def test_statistics_tracking(reconciliation_engine, internal_positions, broker_positions_matched):
    """Test that statistics are correctly tracked"""
    # Perform multiple reconciliations
    for _ in range(5):
        reconciliation_engine.reconcile(
            internal_positions=internal_positions,
            broker_positions=broker_positions_matched,
            internal_equity=Decimal('100000'),
            broker_equity=Decimal('100000'),
        )
    
    stats = reconciliation_engine.get_stats()
    
    assert stats['total_reconciliations'] == 5
    assert stats['drift_detected_count'] == 0  # No drift in these tests
    assert stats['auto_corrections_applied'] == 0
    assert stats['halts_triggered'] == 0
    assert stats['drift_detection_rate'] == 0.0
    assert stats['last_reconciliation'] is not None


# ========================================
# TEST 12: HISTORY RETENTION
# ========================================

def test_reconciliation_history(reconciliation_engine, internal_positions, broker_positions_matched):
    """Test that reconciliation history is retained"""
    # Perform 10 reconciliations
    for i in range(10):
        reconciliation_engine.reconcile(
            internal_positions=internal_positions,
            broker_positions=broker_positions_matched,
            internal_equity=Decimal('100000'),
            broker_equity=Decimal('100000'),
        )
    
    # Check history
    assert len(reconciliation_engine.reconciliation_history) == 10
    assert reconciliation_engine.last_reconciliation is not None
    
    # Verify all reports have timestamps
    for report in reconciliation_engine.reconciliation_history:
        assert report.timestamp is not None
        assert report.reconciliation_duration_ms >= 0


# ========================================
# TEST 13: AUTO-CORRECT DISABLED
# ========================================

def test_auto_correct_disabled(internal_positions):
    """Test critical drift when auto-correct is disabled"""
    engine = BrokerReconciliationEngine(
        auto_correct_enabled=False,  # Disabled
        halt_on_catastrophic=True,
    )
    
    # 2% drift (critical level)
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10200,  # +200 units (2% drift)
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    # With auto-correct disabled, should escalate to alert
    assert report.overall_severity == DriftSeverity.CRITICAL
    assert report.recommended_action == DriftAction.ALERT  # Not AUTO_CORRECT
    assert report.action_taken == DriftAction.ALERT


# ========================================
# TEST 14: HALT DISABLED
# ========================================

def test_halt_disabled_catastrophic_drift(internal_positions):
    """Test catastrophic drift when halt is disabled"""
    engine = BrokerReconciliationEngine(
        halt_on_catastrophic=False,  # Disabled
    )
    
    # 7% drift (catastrophic)
    broker_positions = {
        'EURUSD': {
            'symbol': 'EURUSD',
            'quantity': 10700,
            'price_current': 1.1000,
            'profit': 100.00,
            'type': 0,
        },
        'GBPUSD': {
            'symbol': 'GBPUSD',
            'quantity': 5000,
            'price_current': 1.2500,
            'profit': -50.00,
            'type': 1,
        },
    }
    
    report = engine.reconcile(
        internal_positions=internal_positions,
        broker_positions=broker_positions,
        internal_equity=Decimal('100000'),
        broker_equity=Decimal('100000'),
    )
    
    # With halt disabled, should escalate to alert instead
    assert report.overall_severity == DriftSeverity.CATASTROPHIC
    assert report.recommended_action == DriftAction.ALERT  # Not HALT
    assert report.trading_halted is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
