"""
Health Monitor CLI Tool

Command-line interface for checking raw layer health status.

Usage:
    python -m arbitrex.raw_layer.health_cli
    python -m arbitrex.raw_layer.health_cli --watch
    python -m arbitrex.raw_layer.health_cli --json
    python -m arbitrex.raw_layer.health_cli --component mt5
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from .health import init_health_monitor


def format_status(status: str) -> str:
    """Format status with color codes."""
    colors = {
        "healthy": "\033[92m",  # Green
        "degraded": "\033[93m",  # Yellow
        "critical": "\033[91m",  # Red
        "unknown": "\033[90m"    # Gray
    }
    reset = "\033[0m"
    return f"{colors.get(status, '')}{status.upper()}{reset}"


def format_metric_value(value):
    """Format metric value for display."""
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, (int, float)):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    else:
        return str(value)


def print_summary(report):
    """Print health summary."""
    print("=" * 70)
    print("ARBITREX RAW LAYER HEALTH SUMMARY")
    print("=" * 70)
    print()
    print(f"Overall Status: {format_status(report.overall_status)}")
    print(f"Timestamp:      {datetime.utcfromtimestamp(report.timestamp).isoformat()}Z")
    print(f"Uptime:         {report._format_uptime(report.uptime_seconds)}")
    print()
    print("Components:")
    print("-" * 70)
    
    for name, comp in report.components.items():
        status_str = format_status(comp.status)
        print(f"  {name:20s} {status_str:30s} {comp.message}")
    
    print()
    print("Metrics:")
    print("-" * 70)
    for key, value in report.metrics.items():
        print(f"  {key:35s} {value}")
    
    if report.warnings:
        print()
        print(f"Warnings ({len(report.warnings)}):")
        print("-" * 70)
        for w in report.warnings[:10]:
            print(f"  ⚠  {w}")
    
    if report.errors:
        print()
        print(f"Errors ({len(report.errors)}):")
        print("-" * 70)
        for e in report.errors[:10]:
            print(f"  ✗  {e}")
    
    print()
    print("=" * 70)


def print_detailed(report):
    """Print detailed health report."""
    print("=" * 70)
    print("ARBITREX RAW LAYER DETAILED HEALTH REPORT")
    print("=" * 70)
    print()
    print(f"Overall Status: {format_status(report.overall_status)}")
    print(f"Timestamp:      {datetime.utcfromtimestamp(report.timestamp).isoformat()}Z")
    print(f"Uptime:         {report._format_uptime(report.uptime_seconds)}")
    print()
    
    for name, comp in report.components.items():
        print("-" * 70)
        print(f"Component: {name.upper()}")
        print("-" * 70)
        print(f"  Status:       {format_status(comp.status)}")
        print(f"  Message:      {comp.message}")
        print(f"  Last Updated: {datetime.utcfromtimestamp(comp.last_updated).isoformat()}Z")
        
        if comp.value:
            print(f"  Value:")
            if isinstance(comp.value, dict):
                for k, v in comp.value.items():
                    print(f"    {k:20s} {format_metric_value(v)}")
            else:
                print(f"    {format_metric_value(comp.value)}")
        
        if comp.threshold:
            print(f"  Threshold:")
            if isinstance(comp.threshold, dict):
                for k, v in comp.threshold.items():
                    print(f"    {k:20s} {v}")
        print()
    
    print("=" * 70)
    print("METRICS")
    print("=" * 70)
    for key, value in report.metrics.items():
        print(f"  {key:35s} {format_metric_value(value)}")
    
    if report.warnings:
        print()
        print("=" * 70)
        print(f"WARNINGS ({len(report.warnings)})")
        print("=" * 70)
        for w in report.warnings:
            print(f"  {w}")
    
    if report.errors:
        print()
        print("=" * 70)
        print(f"ERRORS ({len(report.errors)})")
        print("=" * 70)
        for e in report.errors:
            print(f"  {e}")
    
    print()
    print("=" * 70)


def print_component(report, component_name):
    """Print health for specific component."""
    if component_name not in report.components:
        print(f"Error: Component '{component_name}' not found")
        print(f"Available components: {', '.join(report.components.keys())}")
        return
    
    comp = report.components[component_name]
    
    print("=" * 70)
    print(f"COMPONENT: {component_name.upper()}")
    print("=" * 70)
    print()
    print(f"Status:       {format_status(comp.status)}")
    print(f"Message:      {comp.message}")
    print(f"Last Updated: {datetime.utcfromtimestamp(comp.last_updated).isoformat()}Z")
    print()
    
    if comp.value:
        print("Value:")
        if isinstance(comp.value, dict):
            for k, v in comp.value.items():
                print(f"  {k:25s} {format_metric_value(v)}")
        else:
            print(f"  {format_metric_value(comp.value)}")
        print()
    
    if comp.threshold:
        print("Thresholds:")
        if isinstance(comp.threshold, dict):
            for k, v in comp.threshold.items():
                print(f"  {k:25s} {v}")
        print()
    
    print("=" * 70)


def watch_health(interval=5):
    """Watch health status with periodic refresh."""
    print("Health Monitor Watch Mode (Ctrl+C to exit)")
    print(f"Refresh interval: {interval} seconds")
    print()
    
    try:
        while True:
            # Clear screen (cross-platform)
            print("\033[2J\033[H", end="")
            
            monitor = init_health_monitor()
            report = monitor.get_health_report()
            print_summary(report)
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description="Arbitrex Raw Layer Health Monitor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick summary
  python -m arbitrex.raw_layer.health_cli
  
  # Detailed report
  python -m arbitrex.raw_layer.health_cli --detailed
  
  # JSON output
  python -m arbitrex.raw_layer.health_cli --json
  
  # Check specific component
  python -m arbitrex.raw_layer.health_cli --component mt5
  
  # Watch mode (refresh every 5 seconds)
  python -m arbitrex.raw_layer.health_cli --watch
  
  # Custom watch interval
  python -m arbitrex.raw_layer.health_cli --watch --interval 10
        """
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed health report"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--component", "-c",
        type=str,
        help="Show health for specific component (mt5, tick_collection, queue, filesystem, data_quality, timezone)"
    )
    
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode: continuously refresh health status"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Refresh interval for watch mode (seconds, default: 5)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for raw data (default: arbitrex/data/raw)"
    )
    
    args = parser.parse_args()
    
    # Watch mode
    if args.watch:
        watch_health(args.interval)
        return
    
    # Initialize health monitor
    monitor = init_health_monitor(args.base_dir)
    report = monitor.get_health_report()
    
    # Output format
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.component:
        print_component(report, args.component)
    elif args.detailed:
        print_detailed(report)
    else:
        print_summary(report)
    
    # Exit with appropriate code
    if report.overall_status == "critical":
        sys.exit(2)
    elif report.overall_status == "degraded":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
