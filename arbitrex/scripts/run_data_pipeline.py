"""
Unified Data Processing Runner

Orchestrates the complete data pipeline:
    Raw Layer Ingestion → Clean Data Processing

Usage:
    python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --symbols EURUSD GBPUSD
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbitrex.clean_data.integration import RawToCleanBridge
from arbitrex.clean_data.config import CleanDataConfig
from arbitrex.raw_layer.config import TRADING_UNIVERSE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'data_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Arbitrex Unified Data Pipeline: Raw → Clean",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific symbols
    python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --symbols EURUSD GBPUSD USDJPY
    
    # Process entire FX asset class
    python -m arbitrex.scripts.run_data_pipeline --timeframe 4H --asset-class FX
    
    # Process all available data
    python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --all
    
    # Process with custom paths
    python -m arbitrex.scripts.run_data_pipeline --timeframe 1D --all \
        --raw-dir arbitrex/data/raw \
        --clean-dir arbitrex/data/clean
        """
    )
    
    # Timeframe (required)
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        choices=["1H", "4H", "1D", "1M"],
        help="Timeframe to process"
    )
    
    # Symbol selection (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to process (e.g., EURUSD GBPUSD)"
    )
    symbol_group.add_argument(
        "--asset-class",
        choices=list(TRADING_UNIVERSE.keys()),
        help="Process entire asset class from TRADING_UNIVERSE"
    )
    symbol_group.add_argument(
        "--all",
        action="store_true",
        help="Process all available symbols"
    )
    
    # Paths
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Raw data directory (default: arbitrex/data/raw)"
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        help="Clean data output directory (default: arbitrex/data/clean)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom clean data config JSON"
    )
    
    # Output options
    parser.add_argument(
        "--report",
        type=str,
        help="Save processing report to JSON file"
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Process data but don't write output (dry run)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Load clean data config
    if args.config:
        LOG.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = CleanDataConfig.from_dict(config_dict)
    else:
        LOG.info("Using default clean data configuration")
        config = CleanDataConfig()
    
    # Initialize bridge
    bridge = RawToCleanBridge(
        raw_base_dir=Path(args.raw_dir) if args.raw_dir else None,
        clean_base_dir=Path(args.clean_dir) if args.clean_dir else None,
        config=config
    )
    
    # Determine symbols to process
    if args.symbols:
        symbols = args.symbols
        LOG.info(f"Processing specific symbols: {symbols}")
    elif args.asset_class:
        symbols = TRADING_UNIVERSE[args.asset_class]
        LOG.info(f"Processing {args.asset_class} asset class: {len(symbols)} symbols")
    else:  # args.all
        symbols = None  # Will use all available
        LOG.info("Processing all available symbols")
    
    # Print configuration summary
    print("\n" + "="*80)
    print("ARBITREX DATA PIPELINE")
    print("="*80)
    print(f"Timeframe:          {args.timeframe}")
    print(f"Symbols:            {len(symbols) if symbols else 'All available'}")
    print(f"Raw directory:      {bridge.raw_base_dir}")
    print(f"Clean directory:    {bridge.clean_base_dir}")
    print(f"Config version:     {config.config_version}")
    print(f"Write output:       {not args.no_write}")
    print("="*80 + "\n")
    
    # Process data
    LOG.info("Starting data pipeline processing...")
    start_time = datetime.utcnow()
    
    try:
        if args.asset_class:
            results = bridge.process_universe(
                timeframe=args.timeframe,
                asset_classes=[args.asset_class]
            )
        else:
            results = bridge.process_universe(
                timeframe=args.timeframe,
                symbols=symbols
            )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate report
        report = bridge.get_processing_report(results)
        report["processing_time_seconds"] = processing_time
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Processing time:    {processing_time:.2f} seconds")
        print(f"Total symbols:      {report['summary']['total_symbols']}")
        print(f"Successful:         {report['summary']['successful']}")
        print(f"Failed:             {report['summary']['failed']}")
        print(f"Success rate:       {report['summary']['success_rate']:.2f}%")
        print()
        print(f"Total bars:         {report['aggregated_statistics']['total_bars_processed']:,}")
        print(f"Valid bars:         {report['aggregated_statistics']['total_valid_bars']:,}")
        print(f"Missing bars:       {report['aggregated_statistics']['total_missing_bars']:,}")
        print(f"Outlier bars:       {report['aggregated_statistics']['total_outlier_bars']:,}")
        print(f"Validation rate:    {report['aggregated_statistics']['overall_validation_rate']:.2f}%")
        print("="*80)
        
        # Print per-symbol summary
        print("\nPER-SYMBOL RESULTS:")
        print("-"*80)
        for symbol, details in report['symbol_details'].items():
            if "status" in details:
                print(f"✗ {symbol:10s} FAILED")
            else:
                val_rate = details['validation_rate']
                status = "✓" if val_rate >= 90.0 else "⚠"
                print(
                    f"{status} {symbol:10s} "
                    f"{details['valid_bars']:6,}/{details['total_bars']:6,} bars "
                    f"({val_rate:5.2f}%) "
                    f"missing={details['missing_bars']:4,} "
                    f"outliers={details['outlier_bars']:4,}"
                )
        print("="*80 + "\n")
        
        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            LOG.info(f"Report saved to {report_path}")
            print(f"✓ Report saved to {report_path}\n")
        
        # Exit code based on success rate
        if report['summary']['success_rate'] == 100.0:
            sys.exit(0)
        elif report['summary']['success_rate'] >= 50.0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Mostly failed
    
    except Exception as e:
        LOG.exception("Pipeline execution failed")
        print(f"\n✗ Pipeline failed: {e}\n")
        sys.exit(3)


if __name__ == "__main__":
    main()
