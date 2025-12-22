"""
QSE Health Monitor

Tracks statistical validation performance, quality metrics, and system health.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import logging
import json

LOG = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for validation performance"""
    total_validations: int = 0
    valid_signals: int = 0
    invalid_signals: int = 0
    
    # Failure breakdown
    trend_failures: int = 0
    stationarity_failures: int = 0
    distribution_failures: int = 0
    correlation_failures: int = 0
    volatility_failures: int = 0
    
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    
    # Quality metrics
    avg_trend_score: float = 0.0
    avg_adf_pvalue: float = 0.0
    avg_z_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_validations': self.total_validations,
            'valid_signals': self.valid_signals,
            'invalid_signals': self.invalid_signals,
            'validity_rate': self.valid_signals / max(1, self.total_validations),
            'trend_failures': self.trend_failures,
            'stationarity_failures': self.stationarity_failures,
            'distribution_failures': self.distribution_failures,
            'correlation_failures': self.correlation_failures,
            'volatility_failures': self.volatility_failures,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'max_processing_time_ms': self.max_processing_time_ms,
            'min_processing_time_ms': self.min_processing_time_ms if self.min_processing_time_ms != float('inf') else 0.0,
            'avg_trend_score': self.avg_trend_score,
            'avg_adf_pvalue': self.avg_adf_pvalue,
            'avg_z_score': self.avg_z_score
        }


@dataclass
class SymbolHealth:
    """Health status for a single symbol"""
    symbol: str
    last_validation_time: Optional[datetime] = None
    consecutive_failures: int = 0
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'last_validation_time': self.last_validation_time.isoformat() if self.last_validation_time else None,
            'consecutive_failures': self.consecutive_failures,
            'metrics': self.metrics.to_dict(),
            'recent_failures': list(self.recent_failures)
        }


class QSEHealthMonitor:
    """
    Health monitoring for Quantitative Statistics Engine.
    
    Tracks:
        - Validation success/failure rates
        - Processing times
        - Statistical quality metrics
        - Per-symbol health status
        - System uptime
    """
    
    def __init__(self):
        """Initialize health monitor"""
        self.start_time = datetime.now()
        self.symbol_health: Dict[str, SymbolHealth] = {}
        self.global_metrics = ValidationMetrics()
        
        # Recent validation history (last 100)
        self.recent_validations: deque = deque(maxlen=100)
        
        # Processing time tracking
        self.processing_times: deque = deque(maxlen=1000)
        
        LOG.info("QSE Health Monitor initialized")
    
    def record_validation_start(self, symbol: str) -> float:
        """
        Record start of validation.
        
        Args:
            symbol: Symbol being validated
        
        Returns:
            Start time (for elapsed calculation)
        """
        if symbol not in self.symbol_health:
            self.symbol_health[symbol] = SymbolHealth(symbol=symbol)
        
        return time.time()
    
    def record_validation_success(
        self,
        symbol: str,
        start_time: float,
        metrics: dict
    ):
        """
        Record successful validation.
        
        Args:
            symbol: Symbol validated
            start_time: Validation start time
            metrics: Statistical metrics from validation
        """
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update symbol health
        health = self.symbol_health[symbol]
        health.last_validation_time = datetime.now()
        health.consecutive_failures = 0
        health.metrics.total_validations += 1
        health.metrics.valid_signals += 1
        
        # Update processing times
        health.metrics.avg_processing_time_ms = (
            (health.metrics.avg_processing_time_ms * (health.metrics.total_validations - 1) + elapsed_ms)
            / health.metrics.total_validations
        )
        health.metrics.max_processing_time_ms = max(health.metrics.max_processing_time_ms, elapsed_ms)
        health.metrics.min_processing_time_ms = min(health.metrics.min_processing_time_ms, elapsed_ms)
        
        # Update quality metrics
        trend_score = metrics.get('trend_persistence_score', 0.0)
        adf_pval = metrics.get('adf_pvalue', 0.0)
        z_score = abs(metrics.get('z_score', 0.0))
        
        health.metrics.avg_trend_score = (
            (health.metrics.avg_trend_score * (health.metrics.total_validations - 1) + trend_score)
            / health.metrics.total_validations
        )
        health.metrics.avg_adf_pvalue = (
            (health.metrics.avg_adf_pvalue * (health.metrics.total_validations - 1) + adf_pval)
            / health.metrics.total_validations
        )
        health.metrics.avg_z_score = (
            (health.metrics.avg_z_score * (health.metrics.total_validations - 1) + z_score)
            / health.metrics.total_validations
        )
        
        # Update global metrics
        self._update_global_metrics(elapsed_ms, valid=True)
        
        # Add to recent validations
        self.recent_validations.append({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'valid': True,
            'elapsed_ms': elapsed_ms
        })
        
        self.processing_times.append(elapsed_ms)
        
        LOG.debug(f"Validation success: {symbol} ({elapsed_ms:.2f}ms)")
    
    def record_validation_failure(
        self,
        symbol: str,
        start_time: float,
        failure_reasons: List[str],
        metrics: dict
    ):
        """
        Record failed validation.
        
        Args:
            symbol: Symbol validated
            start_time: Validation start time
            failure_reasons: List of failure reasons
            metrics: Statistical metrics from validation
        """
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update symbol health
        health = self.symbol_health[symbol]
        health.last_validation_time = datetime.now()
        health.consecutive_failures += 1
        health.metrics.total_validations += 1
        health.metrics.invalid_signals += 1
        
        # Update processing times
        health.metrics.avg_processing_time_ms = (
            (health.metrics.avg_processing_time_ms * (health.metrics.total_validations - 1) + elapsed_ms)
            / health.metrics.total_validations
        )
        health.metrics.max_processing_time_ms = max(health.metrics.max_processing_time_ms, elapsed_ms)
        health.metrics.min_processing_time_ms = min(health.metrics.min_processing_time_ms, elapsed_ms)
        
        # Track failure types
        for reason in failure_reasons:
            if 'trend' in reason.lower() or 'persistence' in reason.lower():
                health.metrics.trend_failures += 1
            if 'stationar' in reason.lower() or 'adf' in reason.lower():
                health.metrics.stationarity_failures += 1
            if 'distribution' in reason.lower() or 'outlier' in reason.lower():
                health.metrics.distribution_failures += 1
            if 'correlation' in reason.lower():
                health.metrics.correlation_failures += 1
            if 'volatility' in reason.lower() or 'regime' in reason.lower():
                health.metrics.volatility_failures += 1
        
        # Add to recent failures
        health.recent_failures.append({
            'timestamp': datetime.now().isoformat(),
            'reasons': failure_reasons
        })
        
        # Update quality metrics
        trend_score = metrics.get('trend_persistence_score', 0.0)
        adf_pval = metrics.get('adf_pvalue', 0.0)
        z_score = abs(metrics.get('z_score', 0.0))
        
        health.metrics.avg_trend_score = (
            (health.metrics.avg_trend_score * (health.metrics.total_validations - 1) + trend_score)
            / health.metrics.total_validations
        )
        health.metrics.avg_adf_pvalue = (
            (health.metrics.avg_adf_pvalue * (health.metrics.total_validations - 1) + adf_pval)
            / health.metrics.total_validations
        )
        health.metrics.avg_z_score = (
            (health.metrics.avg_z_score * (health.metrics.total_validations - 1) + z_score)
            / health.metrics.total_validations
        )
        
        # Update global metrics
        self._update_global_metrics(elapsed_ms, valid=False)
        
        # Add to recent validations
        self.recent_validations.append({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'valid': False,
            'reasons': failure_reasons,
            'elapsed_ms': elapsed_ms
        })
        
        self.processing_times.append(elapsed_ms)
        
        LOG.warning(f"Validation failure: {symbol} - {failure_reasons}")
    
    def _update_global_metrics(self, elapsed_ms: float, valid: bool):
        """Update global metrics"""
        self.global_metrics.total_validations += 1
        
        if valid:
            self.global_metrics.valid_signals += 1
        else:
            self.global_metrics.invalid_signals += 1
        
        # Update processing times
        self.global_metrics.avg_processing_time_ms = (
            (self.global_metrics.avg_processing_time_ms * (self.global_metrics.total_validations - 1) + elapsed_ms)
            / self.global_metrics.total_validations
        )
        self.global_metrics.max_processing_time_ms = max(self.global_metrics.max_processing_time_ms, elapsed_ms)
        self.global_metrics.min_processing_time_ms = min(self.global_metrics.min_processing_time_ms, elapsed_ms)
    
    def get_health_status(self) -> Dict:
        """
        Get overall health status.
        
        Returns:
            Health status dictionary
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Determine overall status
        validity_rate = self.global_metrics.valid_signals / max(1, self.global_metrics.total_validations)
        
        if validity_rate >= 0.80:
            status = "HEALTHY"
        elif validity_rate >= 0.50:
            status = "DEGRADED"
        else:
            status = "UNHEALTHY"
        
        # Count symbols with consecutive failures
        unhealthy_symbols = sum(
            1 for h in self.symbol_health.values()
            if h.consecutive_failures >= 5
        )
        
        return {
            'status': status,
            'uptime_seconds': uptime,
            'global_metrics': self.global_metrics.to_dict(),
            'symbols_tracked': len(self.symbol_health),
            'unhealthy_symbols': unhealthy_symbols,
            'recent_validations_count': len(self.recent_validations),
            'avg_processing_time_ms': self.global_metrics.avg_processing_time_ms,
            'validity_rate': validity_rate
        }
    
    def get_symbol_health(self, symbol: str) -> Optional[Dict]:
        """
        Get health status for specific symbol.
        
        Args:
            symbol: Symbol to check
        
        Returns:
            Symbol health dictionary or None
        """
        if symbol not in self.symbol_health:
            return None
        
        return self.symbol_health[symbol].to_dict()
    
    def get_all_symbol_health(self) -> Dict[str, Dict]:
        """
        Get health status for all symbols.
        
        Returns:
            Dictionary mapping symbol to health status
        """
        return {
            symbol: health.to_dict()
            for symbol, health in self.symbol_health.items()
        }
    
    def get_recent_validations(self, limit: int = 20) -> List[Dict]:
        """
        Get recent validation history.
        
        Args:
            limit: Maximum number of validations to return
        
        Returns:
            List of recent validations
        """
        return list(self.recent_validations)[-limit:]
    
    def get_failure_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of failure types across all symbols.
        
        Returns:
            Dictionary with failure counts by type
        """
        return {
            'trend_failures': sum(h.metrics.trend_failures for h in self.symbol_health.values()),
            'stationarity_failures': sum(h.metrics.stationarity_failures for h in self.symbol_health.values()),
            'distribution_failures': sum(h.metrics.distribution_failures for h in self.symbol_health.values()),
            'correlation_failures': sum(h.metrics.correlation_failures for h in self.symbol_health.values()),
            'volatility_failures': sum(h.metrics.volatility_failures for h in self.symbol_health.values())
        }
    
    def export_health_report(self, filepath: str):
        """
        Export health report to JSON file.
        
        Args:
            filepath: Path to save report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_status': self.get_health_status(),
            'symbol_health': self.get_all_symbol_health(),
            'failure_breakdown': self.get_failure_breakdown(),
            'recent_validations': self.get_recent_validations(50)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        LOG.info(f"Health report exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.symbol_health.clear()
        self.global_metrics = ValidationMetrics()
        self.recent_validations.clear()
        self.processing_times.clear()
        self.start_time = datetime.now()
        LOG.info("Health monitor metrics reset")
