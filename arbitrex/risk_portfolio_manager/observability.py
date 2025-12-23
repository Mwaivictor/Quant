"""
Enterprise Observability Infrastructure - Production Grade

Implements comprehensive observability for risk management:
1. Structured JSON logging with severity levels
2. Distributed tracing with correlation IDs
3. Prometheus metrics export
4. Real-time alerting system
5. Performance monitoring
6. Audit trail generation

Critical for production deployment, compliance, and debugging.

Version: 2.0.0 (Enterprise)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import threading


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Event types for structured logging"""
    TRADE_DECISION = "trade_decision"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_SIZED = "position_sized"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    STRATEGY_DISABLED = "strategy_disabled"
    PARAMETER_UPDATED = "parameter_updated"
    HEALTH_CHECK = "health_check"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_METRIC = "performance_metric"


class AlertSeverity(Enum):
    """Alert severity for notification routing"""
    INFO = "info"  # FYI only
    WARNING = "warning"  # Needs attention
    CRITICAL = "critical"  # Immediate action required
    EMERGENCY = "emergency"  # System shutdown imminent


@dataclass
class StructuredLogEntry:
    """Structured log entry with all context"""
    timestamp: datetime
    correlation_id: str
    event_type: EventType
    log_level: LogLevel
    message: str
    
    # Contextual data
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    # Performance tracking
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'event_type': self.event_type.value,
            'log_level': self.log_level.value,
            'message': self.message,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'trade_id': self.trade_id,
            'metadata': self.metadata or {},
            'execution_time_ms': self.execution_time_ms
        }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), default=str)


class CorrelationContext:
    """Thread-local correlation ID context"""
    _local = threading.local()
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread"""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get correlation ID for current thread (or generate new)"""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())
        return cls._local.correlation_id
    
    @classmethod
    def clear(cls):
        """Clear correlation ID"""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')


class StructuredLogger:
    """
    Structured logger with JSON output.
    
    Logs to:
    - File (JSON lines)
    - Console (formatted)
    - Memory buffer (for testing/debugging)
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True,
        buffer_size: int = 1000
    ):
        self.log_file = log_file
        self.console_output = console_output
        
        # In-memory buffer for recent logs
        self.log_buffer: deque = deque(maxlen=buffer_size)
        
        # Setup file logging
        if self.log_file:
            self.file_handler = logging.FileHandler(self.log_file)
            self.file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            self.file_handler = None
        
        # Console logger
        if self.console_output:
            self.console_logger = logging.getLogger('rpm_console')
            self.console_logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('[%(levelname)s] %(message)s')
            )
            self.console_logger.addHandler(console_handler)
        else:
            self.console_logger = None
    
    def log(
        self,
        event_type: EventType,
        log_level: LogLevel,
        message: str,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None
    ):
        """Log structured event"""
        entry = StructuredLogEntry(
            timestamp=datetime.utcnow(),
            correlation_id=CorrelationContext.get_correlation_id(),
            event_type=event_type,
            log_level=log_level,
            message=message,
            strategy_id=strategy_id,
            symbol=symbol,
            trade_id=trade_id,
            metadata=metadata,
            execution_time_ms=execution_time_ms
        )
        
        # Add to buffer
        self.log_buffer.append(entry)
        
        # Write to file (JSON)
        if self.file_handler:
            self.file_handler.stream.write(entry.to_json() + '\n')
            self.file_handler.flush()
        
        # Console output (human-readable)
        if self.console_logger:
            console_msg = f"[{event_type.value}] {message}"
            if strategy_id:
                console_msg += f" (strategy={strategy_id})"
            if symbol:
                console_msg += f" (symbol={symbol})"
            
            if log_level == LogLevel.DEBUG:
                self.console_logger.debug(console_msg)
            elif log_level == LogLevel.INFO:
                self.console_logger.info(console_msg)
            elif log_level == LogLevel.WARNING:
                self.console_logger.warning(console_msg)
            elif log_level == LogLevel.ERROR:
                self.console_logger.error(console_msg)
            elif log_level == LogLevel.CRITICAL:
                self.console_logger.critical(console_msg)
    
    def get_recent_logs(self, n: int = 100) -> List[StructuredLogEntry]:
        """Get N most recent log entries"""
        return list(self.log_buffer)[-n:]
    
    def search_logs(
        self,
        event_type: Optional[EventType] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        min_level: Optional[LogLevel] = None
    ) -> List[StructuredLogEntry]:
        """Search logs by criteria"""
        results = list(self.log_buffer)
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        if strategy_id:
            results = [e for e in results if e.strategy_id == strategy_id]
        
        if symbol:
            results = [e for e in results if e.symbol == symbol]
        
        if min_level:
            level_order = {
                LogLevel.DEBUG: 0,
                LogLevel.INFO: 1,
                LogLevel.WARNING: 2,
                LogLevel.ERROR: 3,
                LogLevel.CRITICAL: 4
            }
            min_level_value = level_order[min_level]
            results = [e for e in results if level_order[e.log_level] >= min_level_value]
        
        return results


class PrometheusMetrics:
    """
    Prometheus-style metrics collector.
    
    Metrics types:
    - Counter: Monotonically increasing (e.g., total_trades)
    - Gauge: Can go up/down (e.g., current_exposure)
    - Histogram: Distribution of values (e.g., execution_time_ms)
    """
    
    def __init__(self):
        # Counters: {metric_name: {labels: value}}
        self.counters: Dict[str, Dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        
        # Gauges: {metric_name: {labels: value}}
        self.gauges: Dict[str, Dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        
        # Histograms: {metric_name: {labels: [observations]}}
        self.histograms: Dict[str, Dict[tuple, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def counter_inc(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """Increment counter"""
        label_tuple = self._labels_to_tuple(labels)
        self.counters[name][label_tuple] += value
    
    def gauge_set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        label_tuple = self._labels_to_tuple(labels)
        self.gauges[name][label_tuple] = value
    
    def histogram_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram observation"""
        label_tuple = self._labels_to_tuple(labels)
        self.histograms[name][label_tuple].append(value)
    
    def _labels_to_tuple(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Convert labels dict to hashable tuple"""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))
    
    def _tuple_to_labels(self, label_tuple: tuple) -> str:
        """Convert label tuple to Prometheus label string"""
        if not label_tuple:
            return ""
        label_strs = [f'{k}="{v}"' for k, v in label_tuple]
        return "{" + ",".join(label_strs) + "}"
    
    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Format:
        # HELP metric_name Description
        # TYPE metric_name counter|gauge|histogram
        metric_name{labels} value
        """
        lines = []
        
        # Counters
        for name, label_values in self.counters.items():
            lines.append(f"# HELP rpm_{name} RPM counter metric")
            lines.append(f"# TYPE rpm_{name} counter")
            for label_tuple, value in label_values.items():
                labels_str = self._tuple_to_labels(label_tuple)
                lines.append(f"rpm_{name}{labels_str} {value}")
        
        # Gauges
        for name, label_values in self.gauges.items():
            lines.append(f"# HELP rpm_{name} RPM gauge metric")
            lines.append(f"# TYPE rpm_{name} gauge")
            for label_tuple, value in label_values.items():
                labels_str = self._tuple_to_labels(label_tuple)
                lines.append(f"rpm_{name}{labels_str} {value}")
        
        # Histograms (export as summary with percentiles)
        for name, label_observations in self.histograms.items():
            lines.append(f"# HELP rpm_{name} RPM histogram metric")
            lines.append(f"# TYPE rpm_{name} summary")
            for label_tuple, observations in label_observations.items():
                if observations:
                    labels_str = self._tuple_to_labels(label_tuple)
                    lines.append(f"rpm_{name}_count{labels_str} {len(observations)}")
                    lines.append(f"rpm_{name}_sum{labels_str} {sum(observations)}")
                    
                    # Percentiles
                    import numpy as np
                    p50 = np.percentile(observations, 50)
                    p95 = np.percentile(observations, 95)
                    p99 = np.percentile(observations, 99)
                    
                    lines.append(f"rpm_{name}{{quantile=\"0.5\",{labels_str[1:] if labels_str else ''} {p50}")
                    lines.append(f"rpm_{name}{{quantile=\"0.95\",{labels_str[1:] if labels_str else ''} {p95}")
                    lines.append(f"rpm_{name}{{quantile=\"0.99\",{labels_str[1:] if labels_str else ''} {p99}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> dict:
        """Get summary of all metrics"""
        return {
            'counters': {
                name: {str(labels): value for labels, value in label_values.items()}
                for name, label_values in self.counters.items()
            },
            'gauges': {
                name: {str(labels): value for labels, value in label_values.items()}
                for name, label_values in self.gauges.items()
            },
            'histograms': {
                name: {
                    str(labels): {
                        'count': len(observations),
                        'mean': sum(observations) / len(observations) if observations else 0.0
                    }
                    for labels, observations in label_observations.items()
                }
                for name, label_observations in self.histograms.items()
            }
        }


@dataclass
class Alert:
    """Alert event"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    correlation_id: str
    
    # Context
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    # State
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'metadata': self.metadata or {},
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertingSystem:
    """
    Real-time alerting system.
    
    Alerts are triggered for:
    - Kill switch activations
    - Risk limit breaches
    - Strategy failures
    - System errors
    """
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        
        # Active alerts: {alert_id: Alert}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=max_alerts)
        
        # Alert callbacks (for external notification systems)
        self.alert_callbacks: List[callable] = []
    
    def trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger new alert.
        
        Returns:
            alert_id
        """
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            severity=severity,
            title=title,
            message=message,
            correlation_id=CorrelationContext.get_correlation_id(),
            strategy_id=strategy_id,
            symbol=symbol,
            metadata=metadata
        )
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                # Don't let callback failures break alerting
                pass
        
        return alert.alert_id
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
    
    def get_active_alerts(
        self,
        min_severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active alerts"""
        alerts = list(self.active_alerts.values())
        
        if min_severity:
            severity_order = {
                AlertSeverity.INFO: 0,
                AlertSeverity.WARNING: 1,
                AlertSeverity.CRITICAL: 2,
                AlertSeverity.EMERGENCY: 3
            }
            min_value = severity_order[min_severity]
            alerts = [a for a in alerts if severity_order[a.severity] >= min_value]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def register_callback(self, callback: callable):
        """Register alert callback (e.g., for PagerDuty, Slack)"""
        self.alert_callbacks.append(callback)


class ObservabilityManager:
    """
    Unified observability manager.
    
    Provides single interface for:
    - Structured logging
    - Metrics collection
    - Alerting
    - Distributed tracing
    """
    
    def __init__(
        self,
        log_file: Optional[str] = "logs/rpm_structured.jsonl",
        enable_metrics: bool = True,
        enable_alerts: bool = True
    ):
        self.logger = StructuredLogger(log_file=log_file)
        self.metrics = PrometheusMetrics() if enable_metrics else None
        self.alerts = AlertingSystem() if enable_alerts else None
    
    # Logging interface
    def log_trade_decision(
        self,
        decision: str,
        strategy_id: str,
        symbol: str,
        metadata: Optional[Dict] = None
    ):
        """Log trade decision"""
        self.logger.log(
            event_type=EventType.TRADE_DECISION,
            log_level=LogLevel.INFO,
            message=f"Trade decision: {decision}",
            strategy_id=strategy_id,
            symbol=symbol,
            metadata=metadata
        )
        
        if self.metrics:
            self.metrics.counter_inc(
                'trade_decisions_total',
                labels={'strategy': strategy_id, 'decision': decision}
            )
    
    def log_kill_switch(
        self,
        kill_switch_type: str,
        reason: str,
        severity: str,
        metadata: Optional[Dict] = None
    ):
        """Log kill switch activation"""
        self.logger.log(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            log_level=LogLevel.CRITICAL,
            message=f"Kill switch triggered: {kill_switch_type} - {reason}",
            metadata=metadata
        )
        
        if self.metrics:
            self.metrics.counter_inc(
                'kill_switch_triggers_total',
                labels={'type': kill_switch_type, 'severity': severity}
            )
        
        if self.alerts:
            self.alerts.trigger_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Kill Switch: {kill_switch_type}",
                message=reason,
                metadata=metadata
            )
    
    def log_risk_breach(
        self,
        breach_type: str,
        current_value: float,
        limit: float,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ):
        """Log risk limit breach"""
        self.logger.log(
            event_type=EventType.RISK_LIMIT_BREACH,
            log_level=LogLevel.WARNING,
            message=f"Risk limit breach: {breach_type} (current={current_value:.2f}, limit={limit:.2f})",
            strategy_id=strategy_id,
            symbol=symbol,
            metadata={'breach_type': breach_type, 'current': current_value, 'limit': limit}
        )
        
        if self.metrics:
            self.metrics.counter_inc(
                'risk_breaches_total',
                labels={'type': breach_type}
            )
        
        if self.alerts:
            self.alerts.trigger_alert(
                severity=AlertSeverity.WARNING,
                title=f"Risk Limit Breach: {breach_type}",
                message=f"Current {current_value:.2f} exceeds limit {limit:.2f}",
                strategy_id=strategy_id,
                symbol=symbol
            )
    
    def record_execution_time(
        self,
        operation: str,
        duration_ms: float,
        strategy_id: Optional[str] = None
    ):
        """Record operation execution time"""
        if self.metrics:
            self.metrics.histogram_observe(
                'execution_time_ms',
                value=duration_ms,
                labels={'operation': operation, 'strategy': strategy_id or 'unknown'}
            )
    
    def update_portfolio_metrics(
        self,
        total_exposure: float,
        leverage: float,
        num_positions: int,
        unrealized_pnl: float
    ):
        """Update portfolio-level metrics"""
        if self.metrics:
            self.metrics.gauge_set('portfolio_exposure', total_exposure)
            self.metrics.gauge_set('portfolio_leverage', leverage)
            self.metrics.gauge_set('portfolio_num_positions', num_positions)
            self.metrics.gauge_set('portfolio_unrealized_pnl', unrealized_pnl)
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if self.metrics:
            return self.metrics.export_prometheus_format()
        return ""
