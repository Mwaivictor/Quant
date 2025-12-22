"""
Health Check HTTP Endpoint for Raw Layer

FastAPI endpoint to expose system health via HTTP REST API.
Can be run standalone or integrated with existing WebSocket server.

Usage:
    python -m arbitrex.raw_layer.health_api
    
    Then access:
    - http://localhost:8766/health (summary)
    - http://localhost:8766/health/detailed (full report)
    - http://localhost:8766/health/metrics (Prometheus-compatible)
"""

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from .health import get_health_monitor, init_health_monitor

LOG = logging.getLogger("arbitrex.raw.health_api")

# Create FastAPI app
health_app = FastAPI(title="Arbitrex Raw Layer Health API", version="1.0.0")


@health_app.on_event("startup")
async def startup():
    """Initialize health monitor on startup."""
    init_health_monitor()
    LOG.info("Health API started")


@health_app.get("/health")
async def health_check():
    """
    Quick health check endpoint.
    
    Returns summary status suitable for load balancers and monitoring tools.
    
    Response:
        - 200 OK: System healthy
        - 503 Service Unavailable: System degraded or critical
    """
    monitor = get_health_monitor()
    summary = monitor.get_health_summary()
    
    status_code = 200 if summary['status'] == 'healthy' else 503
    
    return JSONResponse(
        content=summary,
        status_code=status_code
    )


@health_app.get("/health/detailed")
async def health_detailed():
    """
    Detailed health report with all metrics and diagnostics.
    
    Returns comprehensive health information including:
    - Component-level health (MT5, tick collection, queue, filesystem)
    - Performance metrics
    - Recent errors and warnings
    - Uptime and system info
    """
    monitor = get_health_monitor()
    report = monitor.get_health_report()
    
    return JSONResponse(
        content=report.to_dict(),
        status_code=200
    )


@health_app.get("/health/metrics")
async def health_metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus exposition format for scraping.
    """
    monitor = get_health_monitor()
    report = monitor.get_health_report()
    
    # Generate Prometheus metrics
    lines = [
        "# HELP arbitrex_raw_layer_up System up status (1=up, 0=down)",
        "# TYPE arbitrex_raw_layer_up gauge",
        f"arbitrex_raw_layer_up 1",
        "",
        "# HELP arbitrex_raw_layer_uptime_seconds System uptime in seconds",
        "# TYPE arbitrex_raw_layer_uptime_seconds counter",
        f"arbitrex_raw_layer_uptime_seconds {report.uptime_seconds}",
        "",
        "# HELP arbitrex_raw_layer_health_status Component health status (0=critical, 1=degraded, 2=healthy)",
        "# TYPE arbitrex_raw_layer_health_status gauge"
    ]
    
    status_map = {"critical": 0, "degraded": 1, "unknown": 1, "healthy": 2}
    for comp_name, comp in report.components.items():
        status_value = status_map.get(comp.status, 1)
        lines.append(f'arbitrex_raw_layer_health_status{{component="{comp_name}"}} {status_value}')
    
    lines.append("")
    lines.append("# HELP arbitrex_raw_layer_ticks_total Total ticks collected")
    lines.append("# TYPE arbitrex_raw_layer_ticks_total counter")
    lines.append(f"arbitrex_raw_layer_ticks_total {report.metrics.get('total_ticks_collected', 0)}")
    
    lines.append("")
    lines.append("# HELP arbitrex_raw_layer_symbols_tracked Number of symbols being tracked")
    lines.append("# TYPE arbitrex_raw_layer_symbols_tracked gauge")
    lines.append(f"arbitrex_raw_layer_symbols_tracked {report.metrics.get('symbols_tracked', 0)}")
    
    lines.append("")
    lines.append("# HELP arbitrex_raw_layer_errors_total Recent errors in last 10 minutes")
    lines.append("# TYPE arbitrex_raw_layer_errors_total gauge")
    lines.append(f"arbitrex_raw_layer_errors_total {report.metrics.get('errors_last_10min', 0)}")
    
    lines.append("")
    lines.append("# HELP arbitrex_raw_layer_warnings_total Recent warnings in last 10 minutes")
    lines.append("# TYPE arbitrex_raw_layer_warnings_total gauge")
    lines.append(f"arbitrex_raw_layer_warnings_total {report.metrics.get('warnings_last_10min', 0)}")
    
    metrics_text = "\n".join(lines)
    
    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4"
    )


@health_app.get("/health/components/{component}")
async def health_component(component: str):
    """
    Get health status for a specific component.
    
    Args:
        component: Component name (mt5, tick_collection, queue, filesystem, data_quality, timezone)
    
    Returns:
        Component health details or 404 if component not found
    """
    monitor = get_health_monitor()
    report = monitor.get_health_report()
    
    if component not in report.components:
        return JSONResponse(
            content={"error": f"Component '{component}' not found"},
            status_code=404
        )
    
    comp = report.components[component]
    return JSONResponse(
        content={
            "component": component,
            "status": comp.status,
            "value": comp.value,
            "threshold": comp.threshold,
            "message": comp.message,
            "last_updated": comp.last_updated,
            "last_updated_utc": comp.last_updated
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    LOG.info("Starting Health API server on http://localhost:8766")
    print("=" * 70)
    print("Arbitrex Raw Layer Health API")
    print("=" * 70)
    print()
    print("Endpoints:")
    print("  - http://localhost:8766/health           (quick check)")
    print("  - http://localhost:8766/health/detailed  (full report)")
    print("  - http://localhost:8766/health/metrics   (Prometheus)")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(health_app, host="0.0.0.0", port=8766, log_level="info")
