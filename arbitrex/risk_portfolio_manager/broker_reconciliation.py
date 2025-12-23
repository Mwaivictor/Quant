"""
Broker Reconciliation & Drift Detection

Critical component that prevents catastrophic position drift between internal
portfolio state and actual broker positions.

Architecture:
- Periodic reconciliation (every 60 seconds)
- Drift detection with configurable thresholds
- Automatic correction or trading halt
- Alert generation for manual intervention
- Full audit logging

Drift Types:
1. Position quantity mismatch (internal vs broker)
2. Missing positions (exist in broker, not internal)
3. Phantom positions (exist internally, not in broker)
4. Price discrepancies (>5% difference)

Actions:
- DRIFT < 1%: Log warning, continue trading
- DRIFT 1-5%: Alert ops team, auto-correct if configured
- DRIFT > 5%: HALT TRADING, require manual intervention

"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

LOG = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Drift severity levels (with integer ordering for comparison)"""
    NONE = "none"                    # No drift detected
    MINIMAL = "minimal"              # < 0.5% drift, log only
    WARNING = "warning"              # 0.5-1% drift, alert
    CRITICAL = "critical"            # 1-5% drift, auto-correct
    CATASTROPHIC = "catastrophic"    # >5% drift, halt trading
    
    def __lt__(self, other):
        """Enable proper severity comparison"""
        if not isinstance(other, DriftSeverity):
            return NotImplemented
        order = [
            DriftSeverity.NONE,
            DriftSeverity.MINIMAL,
            DriftSeverity.WARNING,
            DriftSeverity.CRITICAL,
            DriftSeverity.CATASTROPHIC,
        ]
        return order.index(self) < order.index(other)
    
    def __le__(self, other):
        return self == other or self < other
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
        return self == other or self > other


class DriftAction(str, Enum):
    """Actions taken when drift detected"""
    NONE = "none"                    # No action needed
    LOG = "log"                      # Log warning
    ALERT = "alert"                  # Alert ops team
    AUTO_CORRECT = "auto_correct"    # Automatically correct internal state
    HALT_TRADING = "halt_trading"    # Emergency halt


@dataclass
class PositionDrift:
    """Position-level drift information"""
    symbol: str
    internal_quantity: Decimal
    broker_quantity: Decimal
    quantity_drift: Decimal          # Absolute difference
    quantity_drift_pct: float        # Percentage difference
    
    internal_price: Optional[Decimal] = None
    broker_price: Optional[Decimal] = None
    price_drift_pct: Optional[float] = None
    
    internal_pnl: Optional[Decimal] = None
    broker_pnl: Optional[Decimal] = None
    pnl_drift: Optional[Decimal] = None
    
    drift_severity: DriftSeverity = DriftSeverity.NONE
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'internal_quantity': float(self.internal_quantity),
            'broker_quantity': float(self.broker_quantity),
            'quantity_drift': float(self.quantity_drift),
            'quantity_drift_pct': self.quantity_drift_pct,
            'internal_price': float(self.internal_price) if self.internal_price else None,
            'broker_price': float(self.broker_price) if self.broker_price else None,
            'price_drift_pct': self.price_drift_pct,
            'internal_pnl': float(self.internal_pnl) if self.internal_pnl else None,
            'broker_pnl': float(self.broker_pnl) if self.broker_pnl else None,
            'pnl_drift': float(self.pnl_drift) if self.pnl_drift else None,
            'drift_severity': self.drift_severity.value,
            'detected_at': self.detected_at.isoformat(),
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report"""
    timestamp: datetime
    reconciliation_duration_ms: float
    
    # Position counts
    internal_position_count: int
    broker_position_count: int
    matched_positions: int
    missing_positions: int           # In broker, not internal
    phantom_positions: int           # In internal, not broker
    
    # Drift analysis
    position_drifts: List[PositionDrift]
    total_drift_pct: float           # Aggregate drift percentage
    max_drift_pct: float             # Worst single position drift
    
    # Capital drift
    internal_equity: Decimal
    broker_equity: Decimal
    equity_drift: Decimal
    equity_drift_pct: float
    
    # Severity assessment
    overall_severity: DriftSeverity
    recommended_action: DriftAction
    
    # Action taken
    action_taken: DriftAction
    auto_correction_applied: bool = False
    trading_halted: bool = False
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'reconciliation_duration_ms': self.reconciliation_duration_ms,
            'internal_position_count': self.internal_position_count,
            'broker_position_count': self.broker_position_count,
            'matched_positions': self.matched_positions,
            'missing_positions': self.missing_positions,
            'phantom_positions': self.phantom_positions,
            'position_drifts': [d.to_dict() for d in self.position_drifts],
            'total_drift_pct': self.total_drift_pct,
            'max_drift_pct': self.max_drift_pct,
            'internal_equity': float(self.internal_equity),
            'broker_equity': float(self.broker_equity),
            'equity_drift': float(self.equity_drift),
            'equity_drift_pct': self.equity_drift_pct,
            'overall_severity': self.overall_severity.value,
            'recommended_action': self.recommended_action.value,
            'action_taken': self.action_taken.value,
            'auto_correction_applied': self.auto_correction_applied,
            'trading_halted': self.trading_halted,
            'errors': self.errors,
        }


class BrokerReconciliationEngine:
    """
    Broker Reconciliation Engine
    
    Prevents catastrophic position drift by continuously monitoring
    and reconciling internal portfolio state with broker positions.
    
    Key Features:
    - Periodic reconciliation (configurable interval)
    - Multi-level drift detection (position, equity, P&L)
    - Automatic correction for minor drift
    - Emergency halt for catastrophic drift
    - Alert integration (Slack, PagerDuty)
    - Full audit logging
    """
    
    def __init__(
        self,
        reconciliation_interval: float = 60.0,      # seconds
        minimal_drift_threshold: float = 0.005,     # 0.5% - log warning
        warning_drift_threshold: float = 0.01,      # 1% - send alert
        critical_drift_threshold: float = 0.02,     # 2% - auto-correct (changed from 5%)
        catastrophic_drift_threshold: float = 0.05, # 5% - halt trading (NEW)
        auto_correct_enabled: bool = True,
        halt_on_catastrophic: bool = True,
        alert_callback: Optional[Callable] = None,
    ):
        """
        Initialize reconciliation engine.
        
        Args:
            reconciliation_interval: Seconds between reconciliations
            minimal_drift_threshold: Log warning threshold (default 0.5%)
            warning_drift_threshold: Alert threshold (default 1%)
            critical_drift_threshold: Auto-correct threshold (default 2%)
            catastrophic_drift_threshold: Halt threshold (default 5%)
            auto_correct_enabled: Allow automatic correction
            halt_on_catastrophic: Halt trading on >5% drift
            alert_callback: Callback for sending alerts
        """
        self.reconciliation_interval = reconciliation_interval
        self.minimal_drift_threshold = minimal_drift_threshold
        self.warning_drift_threshold = warning_drift_threshold
        self.critical_drift_threshold = critical_drift_threshold
        self.catastrophic_drift_threshold = catastrophic_drift_threshold
        self.auto_correct_enabled = auto_correct_enabled
        self.halt_on_catastrophic = halt_on_catastrophic
        self.alert_callback = alert_callback
        
        # Statistics
        self.total_reconciliations = 0
        self.drift_detected_count = 0
        self.auto_corrections_applied = 0
        self.halts_triggered = 0
        self.last_reconciliation: Optional[ReconciliationReport] = None
        self.reconciliation_history: List[ReconciliationReport] = []
        
        LOG.info(
            f"Broker Reconciliation Engine initialized "
            f"(interval={reconciliation_interval}s, "
            f"warning_threshold={warning_drift_threshold*100:.1f}%, "
            f"critical_threshold={critical_drift_threshold*100:.1f}%, "
            f"catastrophic_threshold={catastrophic_drift_threshold*100:.1f}%)"
        )
    
    def reconcile(
        self,
        internal_positions: Dict[str, Dict],
        broker_positions: Dict[str, Dict],
        internal_equity: Decimal,
        broker_equity: Decimal,
    ) -> ReconciliationReport:
        """
        Perform reconciliation between internal and broker state.
        
        Args:
            internal_positions: Dict[symbol, position_data]
            broker_positions: Dict[symbol, broker_position_data]
            internal_equity: Internal account equity
            broker_equity: Broker account equity
        
        Returns:
            ReconciliationReport with drift analysis and actions
        """
        start_time = time.time()
        
        report = ReconciliationReport(
            timestamp=datetime.utcnow(),
            reconciliation_duration_ms=0.0,
            internal_position_count=len(internal_positions),
            broker_position_count=len(broker_positions),
            matched_positions=0,
            missing_positions=0,
            phantom_positions=0,
            position_drifts=[],
            total_drift_pct=0.0,
            max_drift_pct=0.0,
            internal_equity=internal_equity,
            broker_equity=broker_equity,
            equity_drift=broker_equity - internal_equity,
            equity_drift_pct=self._calculate_drift_pct(internal_equity, broker_equity),
            overall_severity=DriftSeverity.NONE,
            recommended_action=DriftAction.NONE,
            action_taken=DriftAction.NONE,
        )
        
        try:
            # 1. Find matched, missing, and phantom positions
            all_symbols = set(internal_positions.keys()) | set(broker_positions.keys())
            
            for symbol in all_symbols:
                internal_pos = internal_positions.get(symbol)
                broker_pos = broker_positions.get(symbol)
                
                if internal_pos and broker_pos:
                    # Matched position - check for drift
                    drift = self._analyze_position_drift(symbol, internal_pos, broker_pos)
                    report.position_drifts.append(drift)
                    report.matched_positions += 1
                    
                elif broker_pos and not internal_pos:
                    # Missing position (exists in broker, not internal)
                    drift = self._create_missing_position_drift(symbol, broker_pos)
                    report.position_drifts.append(drift)
                    report.missing_positions += 1
                    
                elif internal_pos and not broker_pos:
                    # Phantom position (exists internally, not in broker)
                    drift = self._create_phantom_position_drift(symbol, internal_pos)
                    report.position_drifts.append(drift)
                    report.phantom_positions += 1
            
            # 2. Calculate aggregate drift metrics
            if report.position_drifts:
                # Use max drift instead of average for severity assessment
                report.max_drift_pct = max(abs(d.quantity_drift_pct) for d in report.position_drifts)
                # Also track average for reporting
                total_drift = sum(abs(d.quantity_drift_pct) for d in report.position_drifts)
                report.total_drift_pct = total_drift / len(report.position_drifts)
                
                # Use max drift for severity assessment (more conservative)
                # This ensures we don't mask a large drift with many small ones

            
            # 3. Assess overall severity
            report.overall_severity = self._assess_overall_severity(report)
            report.recommended_action = self._determine_action(report)
            
            # 4. Take action based on severity
            report.action_taken = self._execute_action(report)
            
            self.total_reconciliations += 1
            if report.overall_severity != DriftSeverity.NONE:
                self.drift_detected_count += 1
            
        except Exception as e:
            LOG.error(f"Reconciliation failed: {e}", exc_info=True)
            report.errors.append(str(e))
            report.overall_severity = DriftSeverity.CATASTROPHIC
            report.recommended_action = DriftAction.HALT_TRADING
        
        finally:
            # Record duration
            report.reconciliation_duration_ms = (time.time() - start_time) * 1000
            self.last_reconciliation = report
            self.reconciliation_history.append(report)
            
            # Keep only last 100 reports
            if len(self.reconciliation_history) > 100:
                self.reconciliation_history = self.reconciliation_history[-100:]
        
        return report
    
    def _analyze_position_drift(
        self, 
        symbol: str, 
        internal_pos: Dict, 
        broker_pos: Dict
    ) -> PositionDrift:
        """Analyze drift for a matched position"""
        internal_qty = Decimal(str(internal_pos.get('quantity', 0)))
        broker_qty = Decimal(str(broker_pos.get('quantity', 0)))
        
        quantity_drift = abs(internal_qty - broker_qty)
        quantity_drift_pct = self._calculate_drift_pct(internal_qty, broker_qty)
        
        # Price drift
        internal_price = internal_pos.get('current_price')
        broker_price = broker_pos.get('price_current')
        price_drift_pct = None
        if internal_price and broker_price:
            price_drift_pct = self._calculate_drift_pct(
                Decimal(str(internal_price)), 
                Decimal(str(broker_price))
            )
        
        # P&L drift
        internal_pnl = internal_pos.get('unrealized_pnl')
        broker_pnl = broker_pos.get('profit')
        pnl_drift = None
        if internal_pnl is not None and broker_pnl is not None:
            pnl_drift = Decimal(str(broker_pnl)) - Decimal(str(internal_pnl))
        
        # Assess severity
        severity = self._classify_drift_severity(quantity_drift_pct)
        
        return PositionDrift(
            symbol=symbol,
            internal_quantity=internal_qty,
            broker_quantity=broker_qty,
            quantity_drift=quantity_drift,
            quantity_drift_pct=quantity_drift_pct,
            internal_price=Decimal(str(internal_price)) if internal_price else None,
            broker_price=Decimal(str(broker_price)) if broker_price else None,
            price_drift_pct=price_drift_pct,
            internal_pnl=Decimal(str(internal_pnl)) if internal_pnl else None,
            broker_pnl=Decimal(str(broker_pnl)) if broker_pnl else None,
            pnl_drift=pnl_drift,
            drift_severity=severity,
        )
    
    def _create_missing_position_drift(self, symbol: str, broker_pos: Dict) -> PositionDrift:
        """Create drift record for position missing from internal state"""
        broker_qty = Decimal(str(broker_pos.get('quantity', 0)))
        return PositionDrift(
            symbol=symbol,
            internal_quantity=Decimal('0'),
            broker_quantity=broker_qty,
            quantity_drift=broker_qty,
            quantity_drift_pct=100.0,  # 100% drift (missing)
            broker_price=Decimal(str(broker_pos.get('price_current'))) if broker_pos.get('price_current') else None,
            broker_pnl=Decimal(str(broker_pos.get('profit'))) if broker_pos.get('profit') else None,
            drift_severity=DriftSeverity.CATASTROPHIC,
        )
    
    def _create_phantom_position_drift(self, symbol: str, internal_pos: Dict) -> PositionDrift:
        """Create drift record for phantom position (internal only)"""
        internal_qty = Decimal(str(internal_pos.get('quantity', 0)))
        return PositionDrift(
            symbol=symbol,
            internal_quantity=internal_qty,
            broker_quantity=Decimal('0'),
            quantity_drift=internal_qty,
            quantity_drift_pct=100.0,  # 100% drift (phantom)
            internal_price=Decimal(str(internal_pos.get('current_price'))) if internal_pos.get('current_price') else None,
            internal_pnl=Decimal(str(internal_pos.get('unrealized_pnl'))) if internal_pos.get('unrealized_pnl') else None,
            drift_severity=DriftSeverity.CATASTROPHIC,
        )
    
    def _calculate_drift_pct(self, internal_value: Decimal, broker_value: Decimal) -> float:
        """
        Calculate percentage drift between internal and broker values.
        
        Uses internal value as reference (what we think we have).
        """
        if internal_value == 0 and broker_value == 0:
            return 0.0
        
        # If internal is zero but broker has value, that's 100% drift
        if internal_value == 0:
            return 1.0  # 100% drift
        
        # Calculate drift as percentage of internal value
        drift = abs(internal_value - broker_value)
        return float(drift / abs(internal_value))
    
    def _classify_drift_severity(self, drift_pct: float) -> DriftSeverity:
        """Classify drift severity based on percentage"""
        if drift_pct < self.minimal_drift_threshold:
            return DriftSeverity.NONE
        elif drift_pct < self.warning_drift_threshold:
            return DriftSeverity.MINIMAL
        elif drift_pct < self.critical_drift_threshold:
            return DriftSeverity.WARNING
        elif drift_pct < self.catastrophic_drift_threshold:
            return DriftSeverity.CRITICAL
        else:
            return DriftSeverity.CATASTROPHIC
    
    def _assess_overall_severity(self, report: ReconciliationReport) -> DriftSeverity:
        """
        Assess overall drift severity for entire portfolio.
        
        Uses the WORST severity found (conservative approach).
        """
        # Check for missing/phantom positions (always catastrophic)
        if report.missing_positions > 0 or report.phantom_positions > 0:
            return DriftSeverity.CATASTROPHIC
        
        # Check equity drift
        if abs(report.equity_drift_pct) > self.catastrophic_drift_threshold:
            return DriftSeverity.CATASTROPHIC
        elif abs(report.equity_drift_pct) > self.critical_drift_threshold:
            return DriftSeverity.CRITICAL
        elif abs(report.equity_drift_pct) > self.warning_drift_threshold:
            return DriftSeverity.WARNING
        elif abs(report.equity_drift_pct) > self.minimal_drift_threshold:
            return DriftSeverity.MINIMAL
        
        # Check position-level drifts - use MAXIMUM drift (worst case)
        if report.position_drifts:
            # Just use the max severity from position drifts
            # (they've already been classified correctly)
            max_drift_severity = max(d.drift_severity for d in report.position_drifts)
            return max_drift_severity
        
        return DriftSeverity.NONE
    
    def _determine_action(self, report: ReconciliationReport) -> DriftAction:
        """Determine recommended action based on severity"""
        severity = report.overall_severity
        
        if severity == DriftSeverity.NONE:
            return DriftAction.NONE
        elif severity == DriftSeverity.MINIMAL:
            return DriftAction.LOG
        elif severity == DriftSeverity.WARNING:
            return DriftAction.ALERT
        elif severity == DriftSeverity.CRITICAL:
            return DriftAction.AUTO_CORRECT if self.auto_correct_enabled else DriftAction.ALERT
        else:  # CATASTROPHIC
            return DriftAction.HALT_TRADING if self.halt_on_catastrophic else DriftAction.ALERT
    
    def _execute_action(self, report: ReconciliationReport) -> DriftAction:
        """Execute the recommended action"""
        action = report.recommended_action
        
        if action == DriftAction.NONE:
            return DriftAction.NONE
        
        elif action == DriftAction.LOG:
            LOG.warning(
                f"Minimal drift detected: {report.total_drift_pct*100:.2f}% "
                f"(threshold: {self.minimal_drift_threshold*100:.1f}%)"
            )
            return DriftAction.LOG
        
        elif action == DriftAction.ALERT:
            self._send_alert(report)
            return DriftAction.ALERT
        
        elif action == DriftAction.AUTO_CORRECT:
            success = self._apply_auto_correction(report)
            report.auto_correction_applied = success
            if success:
                self.auto_corrections_applied += 1
                LOG.info(f"Auto-correction applied successfully")
            else:
                LOG.error(f"Auto-correction failed - escalating to alert")
                self._send_alert(report)
            return DriftAction.AUTO_CORRECT
        
        elif action == DriftAction.HALT_TRADING:
            self._halt_trading(report)
            report.trading_halted = True
            self.halts_triggered += 1
            return DriftAction.HALT_TRADING
        
        return DriftAction.NONE
    
    def _send_alert(self, report: ReconciliationReport):
        """Send alert to ops team"""
        if self.alert_callback:
            try:
                alert_data = {
                    'severity': report.overall_severity.value,
                    'timestamp': report.timestamp.isoformat(),
                    'total_drift_pct': report.total_drift_pct * 100,
                    'equity_drift_pct': report.equity_drift_pct * 100,
                    'missing_positions': report.missing_positions,
                    'phantom_positions': report.phantom_positions,
                    'message': self._create_alert_message(report),
                }
                self.alert_callback(alert_data)
            except Exception as e:
                LOG.error(f"Failed to send alert: {e}")
        else:
            LOG.error(
                f"ALERT: {report.overall_severity.value.upper()} drift detected "
                f"but no alert callback configured"
            )
    
    def _create_alert_message(self, report: ReconciliationReport) -> str:
        """Create human-readable alert message"""
        msg = f"🚨 DRIFT DETECTED ({report.overall_severity.value.upper()})\n"
        msg += f"• Total Drift: {report.total_drift_pct*100:.2f}%\n"
        msg += f"• Equity Drift: {report.equity_drift_pct*100:.2f}%\n"
        msg += f"• Positions: {report.internal_position_count} internal, {report.broker_position_count} broker\n"
        
        if report.missing_positions > 0:
            msg += f"• ⚠️ {report.missing_positions} missing position(s) (in broker, not internal)\n"
        if report.phantom_positions > 0:
            msg += f"• ⚠️ {report.phantom_positions} phantom position(s) (in internal, not broker)\n"
        
        if report.position_drifts:
            worst_drift = max(report.position_drifts, key=lambda d: abs(d.quantity_drift_pct))
            msg += f"• Worst Drift: {worst_drift.symbol} ({worst_drift.quantity_drift_pct*100:.2f}%)\n"
        
        msg += f"• Action: {report.recommended_action.value}\n"
        return msg
    
    def _apply_auto_correction(self, report: ReconciliationReport) -> bool:
        """
        Apply automatic correction to internal state.
        
        NOTE: This updates internal state to match broker.
        Only used for minor drift (1-5%).
        """
        LOG.warning(
            f"Auto-correction requested for {len(report.position_drifts)} position(s) "
            f"(total drift: {report.total_drift_pct*100:.2f}%)"
        )
        
        # TODO: Implement actual state correction
        # For now, just log the intent
        # In production, this would call portfolio_manager.force_sync_with_broker()
        
        LOG.info("Auto-correction logic not yet implemented - escalating to manual review")
        return False
    
    def _halt_trading(self, report: ReconciliationReport):
        """
        Emergency trading halt due to catastrophic drift.
        
        This should trigger kill switch activation.
        """
        LOG.critical(
            f"🛑 EMERGENCY HALT: Catastrophic drift detected "
            f"({report.total_drift_pct*100:.2f}%)"
        )
        
        # Send critical alert
        self._send_alert(report)
        
        # TODO: Trigger kill switch via event bus
        # event_bus.publish(Event(EventType.KILL_SWITCH_ACTIVATED, ...))
        
        LOG.critical("Manual intervention required - do not resume trading without reconciliation")
    
    def get_stats(self) -> Dict:
        """Get reconciliation statistics"""
        return {
            'total_reconciliations': self.total_reconciliations,
            'drift_detected_count': self.drift_detected_count,
            'auto_corrections_applied': self.auto_corrections_applied,
            'halts_triggered': self.halts_triggered,
            'drift_detection_rate': (
                self.drift_detected_count / max(self.total_reconciliations, 1)
            ),
            'last_reconciliation': (
                self.last_reconciliation.to_dict() 
                if self.last_reconciliation else None
            ),
        }
