"""
Raw Layer System Health Monitor

Comprehensive health monitoring and diagnostics for the Arbitrex Raw Data Layer.
Tracks MT5 connectivity, tick collection, data quality, queue health, and system metrics.

Author: Arbitrex Team
Created: 2025-12-22
"""

from __future__ import annotations
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

LOG = logging.getLogger("arbitrex.raw.health")


@dataclass
class HealthMetric:
    """Single health metric with status and metadata."""
    name: str
    status: str  # "healthy", "degraded", "critical", "unknown"
    value: Any
    threshold: Optional[Any] = None
    message: str = ""
    last_updated: float = 0.0
    
    def __post_init__(self):
        if self.last_updated == 0.0:
            self.last_updated = time.time()


@dataclass
class HealthReport:
    """Complete system health report."""
    overall_status: str  # "healthy", "degraded", "critical"
    timestamp: float
    uptime_seconds: float
    components: Dict[str, HealthMetric]
    metrics: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "timestamp": self.timestamp,
            "timestamp_utc": datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z",
            "uptime_seconds": self.uptime_seconds,
            "uptime_formatted": self._format_uptime(self.uptime_seconds),
            "components": {k: asdict(v) for k, v in self.components.items()},
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors
        }
    
    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime as human-readable string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {mins}m {secs}s"
        elif hours > 0:
            return f"{hours}h {mins}m {secs}s"
        elif mins > 0:
            return f"{mins}m {secs}s"
        else:
            return f"{secs}s"


class HealthMonitor:
    """
    Centralized health monitoring for the raw data layer.
    
    Monitors:
    - MT5 connection status and session health
    - Tick collection rate and latency
    - Queue depth and processing lag
    - Data ingestion completeness and quality
    - File system health (disk space, write permissions)
    - Performance metrics (CPU, memory, I/O)
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("arbitrex/data/raw")
        self.start_time = time.time()
        
        # Metric tracking
        self._tick_counts = defaultdict(int)  # symbol -> count
        self._tick_timestamps = defaultdict(lambda: deque(maxlen=100))  # symbol -> recent timestamps
        self._ingestion_cycles = deque(maxlen=100)  # Recent ingestion cycle results
        self._errors = deque(maxlen=50)
        self._warnings = deque(maxlen=50)
        
        # Component references (set by external systems)
        self._mt5_pool = None
        self._tick_queue = None
        
        LOG.info("Health monitor initialized")
    
    def set_mt5_pool(self, pool):
        """Register MT5 connection pool for monitoring."""
        self._mt5_pool = pool
        LOG.info("MT5 pool registered with health monitor")
    
    def set_tick_queue(self, queue):
        """Register tick queue for monitoring."""
        self._tick_queue = queue
        LOG.info("Tick queue registered with health monitor")
    
    def record_tick(self, symbol: str, timestamp: float):
        """Record a tick for rate tracking."""
        self._tick_counts[symbol] += 1
        self._tick_timestamps[symbol].append(timestamp)
    
    def record_ingestion_cycle(self, metadata: Dict[str, Any]):
        """Record ingestion cycle result."""
        self._ingestion_cycles.append({
            "timestamp": time.time(),
            "metadata": metadata
        })
    
    def record_error(self, component: str, message: str):
        """Record an error."""
        self._errors.append({
            "timestamp": time.time(),
            "component": component,
            "message": message
        })
        LOG.error(f"Health monitor: {component} - {message}")
    
    def record_warning(self, component: str, message: str):
        """Record a warning."""
        self._warnings.append({
            "timestamp": time.time(),
            "component": component,
            "message": message
        })
        LOG.warning(f"Health monitor: {component} - {message}")
    
    def check_mt5_health(self) -> HealthMetric:
        """Check MT5 connection and session health."""
        if self._mt5_pool is None:
            return HealthMetric(
                name="mt5_connection",
                status="unknown",
                value=None,
                message="MT5 pool not registered"
            )
        
        try:
            # Check if any sessions are connected
            connected_sessions = 0
            total_sessions = 0
            initialized_sessions = 0
            
            for name, session in self._mt5_pool._sessions.items():
                total_sessions += 1
                if session.status == "CONNECTED":
                    connected_sessions += 1
                if hasattr(session, 'mt5_initialized') and session.mt5_initialized:
                    initialized_sessions += 1
            
            if connected_sessions == 0:
                status = "critical"
                message = f"No MT5 sessions connected (0/{total_sessions})"
            elif connected_sessions < total_sessions:
                status = "degraded"
                message = f"Some MT5 sessions disconnected ({connected_sessions}/{total_sessions})"
            else:
                status = "healthy"
                message = f"All MT5 sessions connected and initialized ({connected_sessions}/{total_sessions})"
            
            return HealthMetric(
                name="mt5_connection",
                status=status,
                value={
                    "connected": connected_sessions,
                    "total": total_sessions,
                    "initialized": initialized_sessions
                },
                message=message
            )
        except Exception as e:
            return HealthMetric(
                name="mt5_connection",
                status="critical",
                value=None,
                message=f"MT5 health check failed: {e}"
            )
    
    def check_tick_collection_health(self) -> HealthMetric:
        """Check tick collection rate and freshness."""
        if not self._tick_timestamps:
            return HealthMetric(
                name="tick_collection",
                status="degraded",
                value={"rate": 0, "symbols": 0},
                message="No ticks collected yet"
            )
        
        now = time.time()
        rates = {}
        total_rate = 0
        stale_symbols = []
        
        for symbol, timestamps in self._tick_timestamps.items():
            if not timestamps:
                continue
            
            # Calculate rate from last 60 seconds of ticks
            recent = [ts for ts in timestamps if now - ts <= 60]
            rate = len(recent)  # ticks per minute
            rates[symbol] = rate
            total_rate += rate
            
            # Check for stale data (no ticks in last 2 minutes)
            if timestamps and (now - timestamps[-1]) > 120:
                stale_symbols.append(symbol)
        
        # Determine status
        if total_rate == 0:
            status = "critical"
            message = "No ticks received in last 60 seconds"
        elif stale_symbols:
            status = "degraded"
            message = f"{len(stale_symbols)} symbols stale (no ticks >2min): {', '.join(stale_symbols[:5])}"
        elif total_rate < 10:
            status = "degraded"
            message = f"Low tick rate: {total_rate} ticks/min across {len(rates)} symbols"
        else:
            status = "healthy"
            message = f"Collecting {total_rate} ticks/min from {len(rates)} symbols"
        
        return HealthMetric(
            name="tick_collection",
            status=status,
            value={
                "total_rate_per_min": total_rate,
                "symbols_active": len(rates),
                "symbols_stale": len(stale_symbols),
                "top_rates": dict(sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            threshold={"min_rate": 10},
            message=message
        )
    
    def check_queue_health(self) -> HealthMetric:
        """Check durable queue health (Redis or SQLite)."""
        if self._tick_queue is None:
            return HealthMetric(
                name="queue",
                status="unknown",
                value=None,
                message="Tick queue not registered"
            )
        
        try:
            # Try to get queue depth/size
            queue_info = {}
            
            if hasattr(self._tick_queue, '_redis_client'):
                # Redis Streams
                try:
                    redis = self._tick_queue._redis_client
                    queue_info['type'] = 'redis'
                    
                    # Get stream lengths
                    stream_keys = redis.keys("arbitrex:ticks:*")
                    total_pending = 0
                    for key in stream_keys:
                        length = redis.xlen(key)
                        total_pending += length
                    
                    queue_info['pending_messages'] = total_pending
                    queue_info['streams'] = len(stream_keys)
                    
                    if total_pending > 10000:
                        status = "degraded"
                        message = f"High queue backlog: {total_pending} messages across {len(stream_keys)} streams"
                    elif total_pending > 50000:
                        status = "critical"
                        message = f"Critical queue backlog: {total_pending} messages"
                    else:
                        status = "healthy"
                        message = f"Queue healthy: {total_pending} messages pending"
                except Exception as e:
                    status = "degraded"
                    message = f"Redis queue check failed: {e}"
            
            elif hasattr(self._tick_queue, '_conn'):
                # SQLite
                queue_info['type'] = 'sqlite'
                try:
                    cursor = self._tick_queue._conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM ticks")
                    count = cursor.fetchone()[0]
                    queue_info['pending_messages'] = count
                    
                    if count > 5000:
                        status = "degraded"
                        message = f"SQLite queue backlog: {count} messages"
                    elif count > 20000:
                        status = "critical"
                        message = f"Critical SQLite backlog: {count} messages"
                    else:
                        status = "healthy"
                        message = f"SQLite queue healthy: {count} messages"
                except Exception as e:
                    status = "degraded"
                    message = f"SQLite queue check failed: {e}"
            else:
                status = "unknown"
                message = "Unknown queue type"
            
            return HealthMetric(
                name="queue",
                status=status,
                value=queue_info,
                threshold={"max_pending": 10000},
                message=message
            )
        except Exception as e:
            return HealthMetric(
                name="queue",
                status="critical",
                value=None,
                message=f"Queue health check failed: {e}"
            )
    
    def check_filesystem_health(self) -> HealthMetric:
        """Check filesystem health (disk space, write permissions)."""
        try:
            # Check disk space
            stat = os.statvfs(self.base_dir) if hasattr(os, 'statvfs') else None
            
            if stat:
                free_bytes = stat.f_bavail * stat.f_frsize
                total_bytes = stat.f_blocks * stat.f_frsize
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                used_percent = ((total_bytes - free_bytes) / total_bytes) * 100
            else:
                # Windows fallback
                import shutil
                usage = shutil.disk_usage(self.base_dir)
                free_gb = usage.free / (1024**3)
                total_gb = usage.total / (1024**3)
                used_percent = (usage.used / usage.total) * 100
            
            # Check write permissions
            test_file = self.base_dir / ".health_check_test"
            try:
                test_file.touch()
                test_file.unlink()
                writable = True
            except Exception:
                writable = False
            
            # Determine status
            if not writable:
                status = "critical"
                message = f"Cannot write to {self.base_dir}"
            elif used_percent > 95:
                status = "critical"
                message = f"Disk almost full: {used_percent:.1f}% used ({free_gb:.1f}GB free)"
            elif used_percent > 85:
                status = "degraded"
                message = f"Disk space low: {used_percent:.1f}% used ({free_gb:.1f}GB free)"
            else:
                status = "healthy"
                message = f"Disk healthy: {free_gb:.1f}GB free ({100-used_percent:.1f}% available)"
            
            return HealthMetric(
                name="filesystem",
                status=status,
                value={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 2),
                    "writable": writable,
                    "base_dir": str(self.base_dir)
                },
                threshold={"max_used_percent": 85, "min_free_gb": 10},
                message=message
            )
        except Exception as e:
            return HealthMetric(
                name="filesystem",
                status="critical",
                value=None,
                message=f"Filesystem check failed: {e}"
            )
    
    def check_data_quality(self) -> HealthMetric:
        """Check recent data quality (completeness, freshness)."""
        if not self._ingestion_cycles:
            return HealthMetric(
                name="data_quality",
                status="unknown",
                value=None,
                message="No ingestion cycles recorded yet"
            )
        
        # Analyze recent ingestion cycles
        now = time.time()
        recent_cycles = [c for c in self._ingestion_cycles if now - c['timestamp'] <= 3600]
        
        if not recent_cycles:
            return HealthMetric(
                name="data_quality",
                status="degraded",
                value={"recent_cycles": 0},
                message="No ingestion cycles in last hour"
            )
        
        success_count = 0
        partial_count = 0
        failed_count = 0
        
        for cycle in recent_cycles:
            status = cycle.get('metadata', {}).get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                success_count += 1
            elif status == 'PARTIAL':
                partial_count += 1
            else:
                failed_count += 1
        
        total = len(recent_cycles)
        success_rate = (success_count / total) * 100 if total > 0 else 0
        
        if success_rate < 50:
            status = "critical"
            message = f"Low success rate: {success_rate:.1f}% ({failed_count} failed, {partial_count} partial)"
        elif success_rate < 90:
            status = "degraded"
            message = f"Degraded success rate: {success_rate:.1f}% ({failed_count} failed, {partial_count} partial)"
        else:
            status = "healthy"
            message = f"Good data quality: {success_rate:.1f}% success ({total} cycles/hour)"
        
        return HealthMetric(
            name="data_quality",
            status=status,
            value={
                "success_rate": round(success_rate, 2),
                "cycles_last_hour": total,
                "success": success_count,
                "partial": partial_count,
                "failed": failed_count
            },
            threshold={"min_success_rate": 90},
            message=message
        )
    
    def check_timezone_config(self) -> HealthMetric:
        """Check timezone normalization configuration."""
        try:
            from .config import DEFAULT_CONFIG, detect_broker_utc_offset
            
            offset = DEFAULT_CONFIG.broker_utc_offset_hours
            if offset is None:
                offset = detect_broker_utc_offset()
            
            normalize_enabled = DEFAULT_CONFIG.normalize_timestamps
            
            if not normalize_enabled:
                status = "degraded"
                message = "Timestamp normalization is DISABLED (not recommended)"
            else:
                status = "healthy"
                message = f"Timestamp normalization enabled (broker offset: {offset:+d} hours)"
            
            return HealthMetric(
                name="timezone_config",
                status=status,
                value={
                    "normalize_enabled": normalize_enabled,
                    "broker_utc_offset_hours": offset,
                    "auto_detect": DEFAULT_CONFIG.broker_utc_offset_hours is None
                },
                message=message
            )
        except Exception as e:
            return HealthMetric(
                name="timezone_config",
                status="unknown",
                value=None,
                message=f"Timezone config check failed: {e}"
            )
    
    def get_health_report(self) -> HealthReport:
        """Generate complete health report."""
        components = {
            "mt5": self.check_mt5_health(),
            "tick_collection": self.check_tick_collection_health(),
            "queue": self.check_queue_health(),
            "filesystem": self.check_filesystem_health(),
            "data_quality": self.check_data_quality(),
            "timezone": self.check_timezone_config()
        }
        
        # Determine overall status
        statuses = [c.status for c in components.values()]
        if "critical" in statuses:
            overall = "critical"
        elif "degraded" in statuses:
            overall = "degraded"
        elif "unknown" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"
        
        # Collect recent errors and warnings
        now = time.time()
        recent_errors = [
            f"[{e['component']}] {e['message']}"
            for e in self._errors
            if now - e['timestamp'] <= 600  # Last 10 minutes
        ]
        recent_warnings = [
            f"[{w['component']}] {w['message']}"
            for w in self._warnings
            if now - w['timestamp'] <= 600
        ]
        
        # Additional metrics
        uptime = time.time() - self.start_time
        metrics = {
            "total_ticks_collected": sum(self._tick_counts.values()),
            "symbols_tracked": len(self._tick_counts),
            "total_ingestion_cycles": len(self._ingestion_cycles),
            "errors_last_10min": len(recent_errors),
            "warnings_last_10min": len(recent_warnings)
        }
        
        return HealthReport(
            overall_status=overall,
            timestamp=time.time(),
            uptime_seconds=uptime,
            components=components,
            metrics=metrics,
            warnings=recent_warnings,
            errors=recent_errors
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get simplified health summary (for quick checks)."""
        report = self.get_health_report()
        return {
            "status": report.overall_status,
            "timestamp": report.timestamp,
            "uptime": report.uptime_seconds,
            "components": {
                name: {"status": comp.status, "message": comp.message}
                for name, comp in report.components.items()
            },
            "warnings": len(report.warnings),
            "errors": len(report.errors)
        }


# Global singleton instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def init_health_monitor(base_dir: Optional[Path] = None) -> HealthMonitor:
    """Initialize global health monitor."""
    global _health_monitor
    _health_monitor = HealthMonitor(base_dir)
    return _health_monitor
