"""
Clean Data CLI

Command-line interface for the Clean Data Pipeline.

Usage:
    python -m arbitrex.clean_data.cli --symbol EURUSD --timeframe 1H --input raw_data.csv
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import json

from arbitrex.clean_data.pipeline import CleanDataPipeline
from arbitrex.clean_data.config import CleanDataConfig
from arbitrex.clean_data.schemas import CleanOHLCVSchema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Arbitrex Clean Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single symbol
    python -m arbitrex.clean_data.cli --input raw_data/EURUSD_1H.csv --symbol EURUSD --timeframe 1H --output clean_data/

    # Process with custom config
    python -m arbitrex.clean_data.cli --input raw_data/EURUSD_1H.csv --symbol EURUSD --timeframe 1H --config my_config.json

    # Batch process directory
    python -m arbitrex.clean_data.cli --input-dir raw_data/ --timeframe 1H --output clean_data/ --batch
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Input CSV file path (single symbol)"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Input directory path (batch mode)"
    )
    
    # Symbol and timeframe
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol identifier (required for single file mode)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        choices=["1H", "4H", "1D", "1M"],
        help="Timeframe"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="clean_data_output",
        help="Output directory (default: clean_data_output)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config JSON file"
    )
    
    # Batch mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all CSV files in input directory"
    )
    
    # Validation threshold
    parser.add_argument(
        "--min-valid-pct",
        type=float,
        default=90.0,
        help="Minimum valid bar percentage for acceptance (default: 90.0)"
    )
    
    # Verbosity
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
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Load configuration
    if args.config:
        LOG.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = CleanDataConfig.from_dict(config_dict)
    else:
        LOG.info("Using default configuration")
        config = CleanDataConfig()
    
    # Initialize pipeline
    pipeline = CleanDataPipeline(config)
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute
    if args.batch or args.input_dir:
        # Batch mode
        if not args.input_dir:
            LOG.error("--input-dir required for batch mode")
            sys.exit(1)
        
        success = process_batch(
            pipeline, 
            Path(args.input_dir), 
            args.timeframe,
            output_dir,
            args.min_valid_pct
        )
    else:
        # Single file mode
        if not args.symbol:
            LOG.error("--symbol required for single file mode")
            sys.exit(1)
        
        success = process_single(
            pipeline,
            Path(args.input),
            args.symbol,
            args.timeframe,
            output_dir,
            args.min_valid_pct
        )
    
    sys.exit(0 if success else 1)


def process_single(
    pipeline: CleanDataPipeline,
    input_path: Path,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    min_valid_pct: float
) -> bool:
    """Process single symbol"""
    
    LOG.info(f"Processing {symbol} {timeframe} from {input_path}")
    
    # Load raw data
    try:
        raw_df = pd.read_csv(input_path)
        LOG.info(f"Loaded {len(raw_df)} raw bars")
    except Exception as e:
        LOG.error(f"Failed to load {input_path}: {e}")
        return False
    
    # Process
    cleaned_df, metadata = pipeline.process_symbol(
        raw_df=raw_df,
        symbol=symbol,
        timeframe=timeframe,
        source_id=str(input_path)
    )
    
    if cleaned_df is None:
        LOG.error(f"Processing failed: {metadata.errors}")
        return False
    
    # Validate
    try:
        CleanOHLCVSchema.validate_dataframe(cleaned_df)
        LOG.info("✓ Schema validation passed")
    except ValueError as e:
        LOG.error(f"✗ Schema validation failed: {e}")
        return False
    
    # Check acceptance threshold
    validation_rate = (metadata.valid_bars / metadata.total_bars_processed) * 100
    
    if validation_rate < min_valid_pct:
        LOG.warning(
            f"✗ Dataset rejected: validation rate {validation_rate:.2f}% "
            f"below threshold {min_valid_pct}%"
        )
        # Still write but mark as rejected
    else:
        LOG.info(f"✓ Dataset accepted: {validation_rate:.2f}% valid bars")
    
    # Write output
    output_file = pipeline.write_clean_data(cleaned_df, metadata, output_dir)
    LOG.info(f"✓ Written to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print(f"CLEAN DATA SUMMARY: {symbol} {timeframe}")
    print("="*70)
    print(f"Total bars:       {metadata.total_bars_processed}")
    print(f"Valid bars:       {metadata.valid_bars} ({validation_rate:.2f}%)")
    print(f"Missing bars:     {metadata.missing_bars}")
    print(f"Outlier bars:     {metadata.outlier_bars}")
    print(f"Invalid bars:     {metadata.invalid_bars}")
    print(f"Warnings:         {len(metadata.warnings)}")
    print(f"Errors:           {len(metadata.errors)}")
    print(f"Output file:      {output_file}")
    print("="*70)
    
    return validation_rate >= min_valid_pct


def process_batch(
    pipeline: CleanDataPipeline,
    input_dir: Path,
    timeframe: str,
    output_dir: Path,
    min_valid_pct: float
) -> bool:
    """Process all CSV files in directory"""
    
    LOG.info(f"Batch processing {input_dir} for {timeframe}")
    
    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        LOG.error(f"No CSV files found in {input_dir}")
        return False
    
    LOG.info(f"Found {len(csv_files)} CSV files")
    
    results = []
    
    for csv_file in csv_files:
        # Extract symbol from filename (assumes format: SYMBOL_TIMEFRAME.csv)
        symbol = csv_file.stem.split('_')[0]
        
        LOG.info(f"\nProcessing {symbol}...")
        
        try:
            raw_df = pd.read_csv(csv_file)
            
            cleaned_df, metadata = pipeline.process_symbol(
                raw_df=raw_df,
                symbol=symbol,
                timeframe=timeframe,
                source_id=str(csv_file)
            )
            
            if cleaned_df is not None:
                validation_rate = (metadata.valid_bars / metadata.total_bars_processed) * 100
                
                if validation_rate >= min_valid_pct:
                    pipeline.write_clean_data(cleaned_df, metadata, output_dir)
                    results.append((symbol, True, validation_rate, None))
                    LOG.info(f"✓ {symbol}: {validation_rate:.2f}% valid")
                else:
                    results.append((symbol, False, validation_rate, "Below threshold"))
                    LOG.warning(f"✗ {symbol}: {validation_rate:.2f}% valid (rejected)")
            else:
                results.append((symbol, False, 0.0, metadata.errors[0] if metadata.errors else "Unknown error"))
                LOG.error(f"✗ {symbol}: Processing failed")
        
        except Exception as e:
            LOG.error(f"✗ {symbol}: Exception - {e}")
            results.append((symbol, False, 0.0, str(e)))
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total files:      {len(csv_files)}")
    print(f"Successful:       {sum(1 for _, success, _, _ in results if success)}")
    print(f"Failed:           {sum(1 for _, success, _, _ in results if not success)}")
    print("\nResults:")
    print("-"*70)
    for symbol, success, valid_pct, error in results:
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {symbol:10s} {valid_pct:6.2f}% valid")
        else:
            print(f"{status} {symbol:10s} FAILED: {error}")
    print("="*70)
    
    return all(success for _, success, _, _ in results)


if __name__ == "__main__":
    main()
