"""
Feature Engine Health Monitor

Monitors feature computation health, performance, and data quality.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
import json

LOG = logging.getLogger(__name__)


@dataclass
class FeatureHealthMetrics:
    """Health metrics for feature computation"""
    
    # Computation metrics
    features_computed_total: int = 0
    features_failed: int = 0
    symbols_processed: int = 0
    timeframes_processed: int = 0
    
    # Performance metrics
    avg_computation_time_ms: float = 0.0
    max_computation_time_ms: float = 0.0
    total_bars_processed: int = 0
    
    # Data quality metrics
    avg_feature_coverage_pct: float = 0.0  # % of non-null features
    normalization_success_pct: float = 0.0
    validation_pass_pct: float = 0.0
    
    # Storage metrics
    feature_store_writes: int = 0
    feature_store_reads: int = 0
    feature_versions_stored: int = 0
    
    # Error tracking
    validation_errors: List[str] = field(default_factory=list)
    computation_errors: List[str] = field(default_factory=list)
    storage_errors: List[str] = field(default_factory=list)
    
    # Timestamps
    last_computation_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    monitor_start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'computation': {
                'features_computed_total': self.features_computed_total,
                'features_failed': self.features_failed,
                'symbols_processed': self.symbols_processed,
                'timeframes_processed': self.timeframes_processed,
            },
            'performance': {
                'avg_computation_time_ms': round(self.avg_computation_time_ms, 2),
                'max_computation_time_ms': round(self.max_computation_time_ms, 2),
                'total_bars_processed': self.total_bars_processed,
            },
            'data_quality': {
                'avg_feature_coverage_pct': round(self.avg_feature_coverage_pct, 2),
                'normalization_success_pct': round(self.normalization_success_pct, 2),
                'validation_pass_pct': round(self.validation_pass_pct, 2),
            },
            'storage': {
                'feature_store_writes': self.feature_store_writes,
                'feature_store_reads': self.feature_store_reads,
                'feature_versions_stored': self.feature_versions_stored,
            },
            'errors': {
                'validation_errors': len(self.validation_errors),
                'computation_errors': len(self.computation_errors),
                'storage_errors': len(self.storage_errors),
            },
            'timestamps': {
                'last_computation_time': self.last_computation_time.isoformat() if self.last_computation_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
                'monitor_start_time': self.monitor_start_time.isoformat(),
                'uptime_seconds': (datetime.now(timezone.utc) - self.monitor_start_time).total_seconds(),
            }
        }


class FeatureEngineHealthMonitor:
    """
    Monitor Feature Engine health and performance.
    
    Tracks:
        - Feature computation success/failure rates
        - Performance metrics (computation time)
        - Data quality (feature coverage, normalization)
        - Storage operations
        - Errors and warnings
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize health monitor.
        
        Args:
            log_dir: Directory for health logs (optional)
        """
        self.metrics = FeatureHealthMetrics()
        self.log_dir = log_dir
        
        # Computation tracking
        self.computation_times: List[float] = []
        self.validation_results: List[bool] = []
        self.normalization_results: List[bool] = []
        self.feature_coverage: List[float] = []
        
        # Symbol/timeframe tracking
        self.processed_symbols = set()
        self.processed_timeframes = set()
        
        LOG.info("Feature Engine Health Monitor initialized")
    
    def record_computation_start(self, symbol: str, timeframe: str) -> datetime:
        """
        Record start of feature computation.
        
        Args:
            symbol: Symbol being processed
            timeframe: Timeframe being processed
        
        Returns:
            Start timestamp
        """
        start_time = datetime.now(timezone.utc)
        self.processed_symbols.add(symbol)
        self.processed_timeframes.add(timeframe)
        
        LOG.debug(f"Started feature computation: {symbol} {timeframe}")
        
        return start_time
    
    def record_computation_success(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        features_computed: int,
        bars_processed: int,
        feature_coverage_pct: float,
        normalization_applied: bool
    ):
        """
        Record successful feature computation.
        
        Args:
            symbol: Symbol processed
            timeframe: Timeframe processed
            start_time: Computation start time
            features_computed: Number of features computed
            bars_processed: Number of bars processed
            feature_coverage_pct: Percentage of non-null features
            normalization_applied: Whether normalization was applied
        """
        # Calculate computation time
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        self.computation_times.append(elapsed_ms)
        
        # Update metrics
        self.metrics.features_computed_total += features_computed
        self.metrics.total_bars_processed += bars_processed
        self.metrics.symbols_processed = len(self.processed_symbols)
        self.metrics.timeframes_processed = len(self.processed_timeframes)
        self.metrics.last_computation_time = datetime.now(timezone.utc)
        self.metrics.last_success_time = datetime.now(timezone.utc)
        
        # Update performance metrics
        self.metrics.avg_computation_time_ms = sum(self.computation_times) / len(self.computation_times)
        self.metrics.max_computation_time_ms = max(self.computation_times)
        
        # Track data quality
        self.feature_coverage.append(feature_coverage_pct)
        self.metrics.avg_feature_coverage_pct = sum(self.feature_coverage) / len(self.feature_coverage)
        
        if normalization_applied:
            self.normalization_results.append(True)
        
        self.metrics.normalization_success_pct = (
            (sum(self.normalization_results) / len(self.normalization_results) * 100)
            if self.normalization_results else 0.0
        )
        
        LOG.info(f"✓ Feature computation success: {symbol} {timeframe} "
                f"({elapsed_ms:.1f}ms, {features_computed} features, {bars_processed} bars)")
    
    def record_computation_failure(
        self,
        symbol: str,
        timeframe: str,
        error: str
    ):
        """
        Record failed feature computation.
        
        Args:
            symbol: Symbol that failed
            timeframe: Timeframe that failed
            error: Error message
        """
        self.metrics.features_failed += 1
        self.metrics.computation_errors.append(
            f"{datetime.now(timezone.utc).isoformat()} | {symbol} {timeframe} | {error}"
        )
        
        # Keep only last 100 errors
        if len(self.metrics.computation_errors) > 100:
            self.metrics.computation_errors = self.metrics.computation_errors[-100:]
        
        LOG.error(f"✗ Feature computation failed: {symbol} {timeframe} | {error}")
    
    def record_validation_result(self, success: bool, errors: List[str] = None):
        """
        Record validation result.
        
        Args:
            success: Whether validation passed
            errors: List of validation errors
        """
        self.validation_results.append(success)
        
        self.metrics.validation_pass_pct = (
            (sum(self.validation_results) / len(self.validation_results) * 100)
            if self.validation_results else 0.0
        )
        
        if not success and errors:
            for error in errors:
                self.metrics.validation_errors.append(
                    f"{datetime.now(timezone.utc).isoformat()} | {error}"
                )
            
            # Keep only last 100 errors
            if len(self.metrics.validation_errors) > 100:
                self.metrics.validation_errors = self.metrics.validation_errors[-100:]
    
    def record_storage_write(self, success: bool, version: str = None):
        """
        Record feature store write operation.
        
        Args:
            success: Whether write succeeded
            version: Feature version written
        """
        if success:
            self.metrics.feature_store_writes += 1
            if version:
                self.metrics.feature_versions_stored += 1
            LOG.debug(f"Feature store write success: version {version}")
        else:
            self.metrics.storage_errors.append(
                f"{datetime.now(timezone.utc).isoformat()} | Write failed"
            )
            LOG.error("Feature store write failed")
    
    def record_storage_read(self, success: bool):
        """
        Record feature store read operation.
        
        Args:
            success: Whether read succeeded
        """
        if success:
            self.metrics.feature_store_reads += 1
            LOG.debug("Feature store read success")
        else:
            self.metrics.storage_errors.append(
                f"{datetime.now(timezone.utc).isoformat()} | Read failed"
            )
            LOG.error("Feature store read failed")
    
    def get_health_status(self) -> Dict:
        """
        Get current health status.
        
        Returns:
            Health status dictionary
        """
        total_computations = self.metrics.features_computed_total + self.metrics.features_failed
        success_rate = (
            (self.metrics.features_computed_total / total_computations * 100)
            if total_computations > 0 else 0.0
        )
        
        # Determine overall health
        if success_rate >= 95 and self.metrics.validation_pass_pct >= 95:
            health = "HEALTHY"
        elif success_rate >= 80 and self.metrics.validation_pass_pct >= 80:
            health = "DEGRADED"
        else:
            health = "UNHEALTHY"
        
        return {
            'status': health,
            'success_rate_pct': round(success_rate, 2),
            'total_computations': total_computations,
            'metrics': self.metrics.to_dict(),
            'recent_errors': {
                'validation': self.metrics.validation_errors[-5:] if self.metrics.validation_errors else [],
                'computation': self.metrics.computation_errors[-5:] if self.metrics.computation_errors else [],
                'storage': self.metrics.storage_errors[-5:] if self.metrics.storage_errors else [],
            }
        }
    
    def export_metrics(self, filepath: Optional[Path] = None) -> Path:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Optional output filepath
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                filepath = self.log_dir / f"feature_health_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            else:
                filepath = Path(f"feature_health_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        
        health_status = self.get_health_status()
        
        with open(filepath, 'w') as f:
            json.dump(health_status, f, indent=2)
        
        LOG.info(f"Health metrics exported to {filepath}")
        return filepath
    
    def get_summary(self) -> str:
        """
        Get human-readable health summary.
        
        Returns:
            Summary string
        """
        status = self.get_health_status()
        
        summary = f"""
{'=' * 80}
FEATURE ENGINE HEALTH SUMMARY
{'=' * 80}

Status: {status['status']}
Success Rate: {status['success_rate_pct']:.2f}%
Total Computations: {status['total_computations']}

Computation Metrics:
  - Features Computed: {self.metrics.features_computed_total}
  - Features Failed: {self.metrics.features_failed}
  - Symbols Processed: {self.metrics.symbols_processed}
  - Timeframes Processed: {self.metrics.timeframes_processed}

Performance Metrics:
  - Avg Computation Time: {self.metrics.avg_computation_time_ms:.2f}ms
  - Max Computation Time: {self.metrics.max_computation_time_ms:.2f}ms
  - Total Bars Processed: {self.metrics.total_bars_processed}

Data Quality:
  - Avg Feature Coverage: {self.metrics.avg_feature_coverage_pct:.2f}%
  - Normalization Success: {self.metrics.normalization_success_pct:.2f}%
  - Validation Pass Rate: {self.metrics.validation_pass_pct:.2f}%

Storage:
  - Store Writes: {self.metrics.feature_store_writes}
  - Store Reads: {self.metrics.feature_store_reads}
  - Versions Stored: {self.metrics.feature_versions_stored}

Errors:
  - Validation Errors: {len(self.metrics.validation_errors)}
  - Computation Errors: {len(self.metrics.computation_errors)}
  - Storage Errors: {len(self.metrics.storage_errors)}

Uptime: {(datetime.now(timezone.utc) - self.metrics.monitor_start_time).total_seconds():.1f}s
Last Success: {self.metrics.last_success_time.strftime('%Y-%m-%d %H:%M:%S UTC') if self.metrics.last_success_time else 'N/A'}

{'=' * 80}
"""
        return summary
    
    def reset_metrics(self):
        """Reset all metrics (for testing or new monitoring period)"""
        self.metrics = FeatureHealthMetrics()
        self.computation_times.clear()
        self.validation_results.clear()
        self.normalization_results.clear()
        self.feature_coverage.clear()
        self.processed_symbols.clear()
        self.processed_timeframes.clear()
        
        LOG.info("Health metrics reset")
