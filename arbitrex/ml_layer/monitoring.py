"""
ML Layer Monitoring System

Comprehensive monitoring for ML Layer operations:
- Prediction tracking (latency, decisions, regime distribution)
- Model performance monitoring (accuracy, drift detection)
- System health metrics (memory, throughput)
- Alert system (performance degradation, anomalies)
- Audit logging (all predictions for compliance)
- Prometheus/Grafana integration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path
import time

from arbitrex.ml_layer.config import MLConfig
from arbitrex.ml_layer.schemas import MLOutput, RegimeLabel

LOG = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    """Single prediction log entry"""
    timestamp: str
    symbol: str
    timeframe: str
    regime: str
    regime_confidence: float
    signal_prob: float
    allowed: bool
    decision_reasons: List[str]
    processing_time_ms: float
    config_hash: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlertRule:
    """Alert configuration"""
    name: str
    metric: str
    operator: str  # '>', '<', '=='
    threshold: float
    window_minutes: int
    severity: str  # 'info', 'warning', 'critical'
    enabled: bool = True
    
    def evaluate(self, value: float) -> bool:
        """Check if alert should trigger"""
        if not self.enabled:
            return False
        
        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '==':
            return value == self.threshold
        return False


@dataclass
class Alert:
    """Active alert"""
    timestamp: str
    rule_name: str
    severity: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MLMonitor:
    """
    ML Layer monitoring system.
    
    Tracks:
    - Prediction metrics (latency, decisions, regime distribution)
    - Model performance (accuracy, drift)
    - System health (memory, throughput)
    - Alerts and anomalies
    """
    
    def __init__(self, config: MLConfig, log_dir: str = "logs/ml_layer"):
        """
        Initialize monitor.
        
        Args:
            config: ML configuration
            log_dir: Directory for log files
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction logs (in-memory buffer + disk persistence)
        self.prediction_buffer: deque = deque(maxlen=10000)  # Last 10k predictions
        self.prediction_file = self.log_dir / "predictions.jsonl"
        
        # Metrics tracking
        self.start_time = datetime.now()
        self.metrics = {
            'total_predictions': 0,
            'allowed_predictions': 0,
            'suppressed_predictions': 0,
            'regime_counts': defaultdict(int),
            'processing_times': deque(maxlen=1000),
            'hourly_predictions': defaultdict(int),
            'symbol_predictions': defaultdict(int),
        }
        
        # Performance tracking (for drift detection)
        self.performance_history: deque = deque(maxlen=1000)
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_rules = self._initialize_alert_rules()
        
        LOG.info(f"ML Monitor initialized: log_dir={log_dir}")
    
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize default alert rules"""
        return [
            AlertRule(
                name="high_suppression_rate",
                metric="suppression_rate",
                operator=">",
                threshold=0.90,  # 90% suppression
                window_minutes=60,
                severity="warning"
            ),
            AlertRule(
                name="low_suppression_rate",
                metric="suppression_rate",
                operator="<",
                threshold=0.10,  # 10% suppression (too permissive?)
                window_minutes=60,
                severity="info"
            ),
            AlertRule(
                name="high_latency",
                metric="avg_processing_time_ms",
                operator=">",
                threshold=10.0,  # 10ms
                window_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="stressed_regime_spike",
                metric="stressed_regime_pct",
                operator=">",
                threshold=0.50,  # 50% stressed
                window_minutes=15,
                severity="critical"
            ),
            AlertRule(
                name="low_confidence",
                metric="avg_regime_confidence",
                operator="<",
                threshold=0.50,  # 50% confidence
                window_minutes=30,
                severity="warning"
            ),
        ]
    
    def log_prediction(self, symbol: str, timeframe: str, output: MLOutput):
        """
        Log a prediction for monitoring.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            output: ML Layer output
        """
        try:
            # Create log entry
            log_entry = PredictionLog(
                timestamp=str(output.timestamp),  # Ensure string
                symbol=symbol,
                timeframe=timeframe,
                regime=output.prediction.regime.regime_label.value,
                regime_confidence=output.prediction.regime.regime_confidence,
                signal_prob=output.prediction.signal.momentum_success_prob,
                allowed=output.prediction.allow_trade,
                decision_reasons=output.prediction.decision_reasons,
                processing_time_ms=output.processing_time_ms,
                config_hash=output.config_hash
            )
            
            # Add to buffer
            self.prediction_buffer.append(log_entry)
            
            # Update metrics
            self._update_metrics(log_entry)
            
            # Write to disk (if enabled)
            if self.config.governance.enable_prediction_logging:
                self._write_to_disk(log_entry)
            
            # Check alerts
            self._check_alerts()
        
        except Exception as e:
            LOG.error(f"Error logging prediction: {e}")
    
    def _update_metrics(self, log_entry: PredictionLog):
        """Update internal metrics"""
        self.metrics['total_predictions'] += 1
        
        if log_entry.allowed:
            self.metrics['allowed_predictions'] += 1
        else:
            self.metrics['suppressed_predictions'] += 1
        
        self.metrics['regime_counts'][log_entry.regime] += 1
        self.metrics['processing_times'].append(log_entry.processing_time_ms)
        
        # Hourly bucketing
        hour_key = datetime.fromisoformat(log_entry.timestamp).strftime('%Y-%m-%d %H:00')
        self.metrics['hourly_predictions'][hour_key] += 1
        
        # Symbol tracking
        self.metrics['symbol_predictions'][log_entry.symbol] += 1
    
    def _write_to_disk(self, log_entry: PredictionLog):
        """Append prediction to log file"""
        try:
            with open(self.prediction_file, 'a') as f:
                f.write(json.dumps(log_entry.to_dict()) + '\n')
        except Exception as e:
            LOG.error(f"Error writing prediction log: {e}")
    
    def _check_alerts(self):
        """Check if any alert rules should trigger"""
        try:
            current_metrics = self.get_current_metrics()
            
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = self._get_metric_value(rule.metric, current_metrics)
                
                if metric_value is None:
                    continue
                
                # Evaluate rule
                if rule.evaluate(metric_value):
                    # Create alert
                    alert = Alert(
                        timestamp=datetime.now().isoformat(),
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"{rule.metric} {rule.operator} {rule.threshold} (actual: {metric_value:.2f})",
                        metric_name=rule.metric,
                        metric_value=metric_value,
                        threshold=rule.threshold
                    )
                    
                    # Check if alert already exists (avoid duplicates)
                    if not self._alert_exists(alert):
                        self.alerts.append(alert)
                        LOG.warning(f"Alert triggered: {alert.message}")
        
        except Exception as e:
            LOG.error(f"Error checking alerts: {e}")
    
    def _get_metric_value(self, metric: str, current_metrics: Dict) -> Optional[float]:
        """Extract metric value from current metrics"""
        if metric == 'suppression_rate':
            return current_metrics.get('suppression_rate', 0)
        elif metric == 'avg_processing_time_ms':
            return current_metrics.get('avg_processing_time_ms', 0)
        elif metric == 'stressed_regime_pct':
            regime_dist = current_metrics.get('regime_distribution', {})
            return regime_dist.get('STRESSED', 0)
        elif metric == 'avg_regime_confidence':
            return current_metrics.get('avg_regime_confidence', 0)
        return None
    
    def _alert_exists(self, new_alert: Alert) -> bool:
        """Check if similar alert already exists (last 5 minutes)"""
        cutoff = datetime.now() - timedelta(minutes=5)
        
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert.timestamp)
            if alert_time > cutoff and alert.rule_name == new_alert.rule_name:
                return True
        
        return False
    
    def get_current_metrics(self) -> Dict:
        """
        Get current monitoring metrics.
        
        Returns:
            Dictionary of metrics
        """
        total = self.metrics['total_predictions']
        allowed = self.metrics['allowed_predictions']
        suppressed = self.metrics['suppressed_predictions']
        
        # Basic metrics
        metrics = {
            'total_predictions': total,
            'allowed_predictions': allowed,
            'suppressed_predictions': suppressed,
            'allow_rate': allowed / max(total, 1),
            'suppression_rate': suppressed / max(total, 1),
        }
        
        # Processing time metrics
        if self.metrics['processing_times']:
            times = list(self.metrics['processing_times'])
            metrics['avg_processing_time_ms'] = np.mean(times)
            metrics['p50_processing_time_ms'] = np.percentile(times, 50)
            metrics['p95_processing_time_ms'] = np.percentile(times, 95)
            metrics['p99_processing_time_ms'] = np.percentile(times, 99)
            metrics['max_processing_time_ms'] = np.max(times)
        
        # Regime distribution
        regime_total = sum(self.metrics['regime_counts'].values())
        if regime_total > 0:
            metrics['regime_distribution'] = {
                regime: count / regime_total
                for regime, count in self.metrics['regime_counts'].items()
            }
        
        # Regime confidence (from recent predictions)
        if self.prediction_buffer:
            recent_confidences = [p.regime_confidence for p in list(self.prediction_buffer)[-100:]]
            metrics['avg_regime_confidence'] = np.mean(recent_confidences)
            metrics['min_regime_confidence'] = np.min(recent_confidences)
        
        # Throughput (predictions per hour)
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        metrics['predictions_per_hour'] = total / max(uptime_hours, 0.01)
        
        # Symbol distribution
        metrics['symbol_distribution'] = dict(self.metrics['symbol_predictions'])
        
        return metrics
    
    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """
        Get recent prediction decisions.
        
        Args:
            limit: Maximum number of decisions to return
        
        Returns:
            List of prediction logs
        """
        recent = list(self.prediction_buffer)[-limit:]
        return [log.to_dict() for log in recent]
    
    def get_active_alerts(self) -> List[Dict]:
        """
        Get active alerts (last 24 hours).
        
        Returns:
            List of active alerts
        """
        cutoff = datetime.now() - timedelta(hours=24)
        
        active = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff
        ]
        
        return [alert.to_dict() for alert in active]
    
    def clear_alerts(self, severity: Optional[str] = None):
        """Clear alerts (optionally filtered by severity)"""
        if severity:
            self.alerts = [a for a in self.alerts if a.severity != severity]
        else:
            self.alerts.clear()
        
        LOG.info(f"Alerts cleared (severity={severity})")
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        try:
            metrics = self.get_current_metrics()
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': self.get_uptime(),
                'metrics': metrics,
                'alerts': self.get_active_alerts(),
                'recent_decisions': self.get_decision_history(limit=50)
            }
            
            # Convert any datetime objects to strings
            def convert_datetimes(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetimes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetimes(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            export_data = convert_datetimes(export_data)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            LOG.info(f"Metrics exported to {filepath}")
        
        except Exception as e:
            LOG.error(f"Error exporting metrics: {e}")
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self.get_current_metrics()
        
        lines = []
        
        # Total predictions
        lines.append(f"# TYPE ml_layer_predictions_total counter")
        lines.append(f"ml_layer_predictions_total {metrics['total_predictions']}")
        
        # Allowed/suppressed
        lines.append(f"# TYPE ml_layer_predictions_allowed counter")
        lines.append(f"ml_layer_predictions_allowed {metrics['allowed_predictions']}")
        lines.append(f"# TYPE ml_layer_predictions_suppressed counter")
        lines.append(f"ml_layer_predictions_suppressed {metrics['suppressed_predictions']}")
        
        # Rates
        lines.append(f"# TYPE ml_layer_allow_rate gauge")
        lines.append(f"ml_layer_allow_rate {metrics['allow_rate']:.4f}")
        lines.append(f"# TYPE ml_layer_suppression_rate gauge")
        lines.append(f"ml_layer_suppression_rate {metrics['suppression_rate']:.4f}")
        
        # Processing time
        if 'avg_processing_time_ms' in metrics:
            lines.append(f"# TYPE ml_layer_processing_time_ms gauge")
            lines.append(f"ml_layer_processing_time_ms {{quantile=\"0.5\"}} {metrics.get('p50_processing_time_ms', 0):.2f}")
            lines.append(f"ml_layer_processing_time_ms {{quantile=\"0.95\"}} {metrics.get('p95_processing_time_ms', 0):.2f}")
            lines.append(f"ml_layer_processing_time_ms {{quantile=\"0.99\"}} {metrics.get('p99_processing_time_ms', 0):.2f}")
        
        # Regime distribution
        if 'regime_distribution' in metrics:
            lines.append(f"# TYPE ml_layer_regime_distribution gauge")
            for regime, pct in metrics['regime_distribution'].items():
                lines.append(f'ml_layer_regime_distribution {{regime="{regime}"}} {pct:.4f}')
        
        # Throughput
        lines.append(f"# TYPE ml_layer_predictions_per_hour gauge")
        lines.append(f"ml_layer_predictions_per_hour {metrics['predictions_per_hour']:.2f}")
        
        # Active alerts
        lines.append(f"# TYPE ml_layer_active_alerts gauge")
        lines.append(f"ml_layer_active_alerts {len(self.get_active_alerts())}")
        
        return '\n'.join(lines)
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """
        Get performance summary for specified time window.
        
        Args:
            hours: Time window in hours
        
        Returns:
            Performance summary dictionary
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter predictions in time window
        recent = [
            p for p in self.prediction_buffer
            if datetime.fromisoformat(str(p.timestamp)) > cutoff
        ]
        
        if not recent:
            return {'message': 'No data in specified time window'}
        
        total = len(recent)
        allowed = sum(1 for p in recent if p.allowed)
        
        return {
            'time_window_hours': hours,
            'total_predictions': total,
            'allowed': allowed,
            'suppressed': total - allowed,
            'allow_rate': allowed / total,
            'avg_processing_time_ms': np.mean([p.processing_time_ms for p in recent]),
            'regime_distribution': {
                regime: sum(1 for p in recent if p.regime == regime) / total
                for regime in set(p.regime for p in recent)
            },
            'avg_signal_prob': np.mean([p.signal_prob for p in recent]),
            'avg_regime_confidence': np.mean([p.regime_confidence for p in recent]),
            'symbols': list(set(p.symbol for p in recent))
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules.append(rule)
        LOG.info(f"Alert rule added: {rule.name}")
    
    def disable_alert_rule(self, rule_name: str):
        """Disable alert rule by name"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                LOG.info(f"Alert rule disabled: {rule_name}")
                return True
        return False
    
    def enable_alert_rule(self, rule_name: str):
        """Enable alert rule by name"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = True
                LOG.info(f"Alert rule enabled: {rule_name}")
                return True
        return False
    
    def get_alert_rules(self) -> List[Dict]:
        """Get all alert rules"""
        return [asdict(rule) for rule in self.alert_rules]


# Convenience function for easy monitoring setup
def create_monitor(config: Optional[MLConfig] = None) -> MLMonitor:
    """
    Create ML monitor with default or custom config.
    
    Args:
        config: ML configuration (creates default if None)
    
    Returns:
        Initialized MLMonitor
    """
    if config is None:
        config = MLConfig()
    
    return MLMonitor(config)
