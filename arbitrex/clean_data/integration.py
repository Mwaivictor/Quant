"""
Raw-to-Clean Integration Module

Bridges the Raw Data Layer with the Clean Data Layer.
Provides utilities for reading raw data and processing through clean pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
import json

from arbitrex.raw_layer.config import TRADING_UNIVERSE, DEFAULT_TIMEFRAMES
from arbitrex.clean_data.pipeline import CleanDataPipeline
from arbitrex.clean_data.config import CleanDataConfig
from arbitrex.clean_data.schemas import CleanDataMetadata

LOG = logging.getLogger(__name__)


class RawToCleanBridge:
    """
    Integration layer between Raw and Clean data layers.
    
    Responsibilities:
        - Read raw OHLCV data from raw layer storage
        - Convert raw format to clean pipeline input format
        - Process through clean pipeline
        - Write clean output to designated location
        - Maintain metadata linkage
    """
    
    def __init__(
        self,
        raw_base_dir: Path = None,
        clean_base_dir: Path = None,
        config: Optional[CleanDataConfig] = None
    ):
        """
        Initialize bridge.
        
        Args:
            raw_base_dir: Raw layer data directory (default: arbitrex/data/raw)
            clean_base_dir: Clean layer output directory (default: arbitrex/data/clean)
            config: Clean pipeline configuration
        """
        # Default paths
        if raw_base_dir is None:
            raw_base_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        if clean_base_dir is None:
            clean_base_dir = Path(__file__).resolve().parent.parent / "data" / "clean"
        
        self.raw_base_dir = Path(raw_base_dir)
        self.clean_base_dir = Path(clean_base_dir)
        
        # Initialize clean pipeline
        self.config = config or CleanDataConfig()
        self.pipeline = CleanDataPipeline(self.config)
        
        LOG.info(f"Bridge initialized: raw={self.raw_base_dir}, clean={self.clean_base_dir}")
    
    def get_available_symbols(self, timeframe: str) -> List[str]:
        """
        Get list of symbols with available raw data.
        
        Args:
            timeframe: Timeframe to check
        
        Returns:
            List of symbol identifiers
        """
        symbols = []
        
        # Check FX directory
        fx_dir = self.raw_base_dir / "ohlcv" / "fx"
        
        if fx_dir.exists():
            for symbol_dir in fx_dir.iterdir():
                if symbol_dir.is_dir():
                    timeframe_dir = symbol_dir / timeframe
                    if timeframe_dir.exists() and list(timeframe_dir.glob("*.csv")):
                        symbols.append(symbol_dir.name)
        
        LOG.info(f"Found {len(symbols)} symbols with {timeframe} data")
        return sorted(symbols)
    
    def read_raw_symbol_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Read all raw data for a symbol/timeframe.
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            Concatenated dataframe of all raw data
        """
        symbol_dir = self.raw_base_dir / "ohlcv" / "fx" / symbol / timeframe
        
        if not symbol_dir.exists():
            raise FileNotFoundError(f"No raw data found for {symbol} {timeframe} at {symbol_dir}")
        
        # Find all CSV files
        csv_files = sorted(symbol_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {symbol_dir}")
        
        LOG.info(f"Reading {len(csv_files)} raw files for {symbol} {timeframe}")
        
        # Read and concatenate
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                LOG.warning(f"Failed to read {csv_file}: {e}")
        
        if not dfs:
            raise ValueError(f"No data could be read for {symbol} {timeframe}")
        
        # Concatenate all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamps if needed
        if "timestamp_utc" not in combined_df.columns and "timestamp" in combined_df.columns:
            # Raw layer provides timestamp in seconds
            combined_df["timestamp_utc"] = pd.to_datetime(combined_df["timestamp"], unit="s", utc=True)
        elif "timestamp_utc" in combined_df.columns:
            combined_df["timestamp_utc"] = pd.to_datetime(combined_df["timestamp_utc"], utc=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp_utc").reset_index(drop=True)
        
        # Filter by date range if specified
        if start_date:
            combined_df = combined_df[combined_df["timestamp_utc"] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df["timestamp_utc"] <= end_date]
        
        # Remove duplicates (keep first occurrence)
        combined_df = combined_df.drop_duplicates(subset=["timestamp_utc"], keep="first")
        
        LOG.info(f"Loaded {len(combined_df)} bars for {symbol} {timeframe}")
        
        return combined_df
    
    def process_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        write_output: bool = True
    ) -> Tuple[Optional[pd.DataFrame], CleanDataMetadata]:
        """
        Process single symbol through raw → clean pipeline.
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            write_output: Whether to write output to disk
        
        Returns:
            (cleaned_df, metadata)
        """
        LOG.info(f"Processing {symbol} {timeframe} through raw→clean pipeline")
        
        try:
            # Read raw data
            raw_df = self.read_raw_symbol_data(symbol, timeframe, start_date, end_date)
            
            # Find source metadata
            source_id = self._get_source_metadata_id(symbol, timeframe)
            
            # Process through clean pipeline
            cleaned_df, metadata = self.pipeline.process_symbol(
                raw_df=raw_df,
                symbol=symbol,
                timeframe=timeframe,
                source_id=source_id
            )
            
            # Write output if requested
            if write_output and cleaned_df is not None:
                output_dir = self.clean_base_dir / "ohlcv" / "fx" / symbol / timeframe
                output_dir.mkdir(parents=True, exist_ok=True)
                
                self.pipeline.write_clean_data(cleaned_df, metadata, output_dir)
            
            return cleaned_df, metadata
            
        except Exception as e:
            LOG.error(f"Failed to process {symbol} {timeframe}: {e}")
            raise
    
    def process_universe(
        self,
        timeframe: str,
        symbols: Optional[List[str]] = None,
        asset_classes: Optional[List[str]] = None
    ) -> Dict[str, Tuple[Optional[pd.DataFrame], CleanDataMetadata]]:
        """
        Process entire trading universe or subset.
        
        Args:
            timeframe: Timeframe to process
            symbols: Optional list of specific symbols (overrides asset_classes)
            asset_classes: Optional list of asset classes from TRADING_UNIVERSE
        
        Returns:
            Dict mapping symbol to (cleaned_df, metadata)
        """
        # Determine which symbols to process
        if symbols is not None:
            target_symbols = symbols
        elif asset_classes is not None:
            target_symbols = []
            for asset_class in asset_classes:
                if asset_class in TRADING_UNIVERSE:
                    target_symbols.extend(TRADING_UNIVERSE[asset_class])
        else:
            # Process all available symbols
            target_symbols = self.get_available_symbols(timeframe)
        
        LOG.info(f"Processing {len(target_symbols)} symbols for {timeframe}")
        
        results = {}
        success_count = 0
        failure_count = 0
        
        for symbol in target_symbols:
            LOG.info(f"\n{'='*60}")
            LOG.info(f"Processing {symbol} {timeframe}")
            LOG.info(f"{'='*60}")
            
            try:
                cleaned_df, metadata = self.process_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    write_output=True
                )
                
                results[symbol] = (cleaned_df, metadata)
                
                if cleaned_df is not None:
                    validation_rate = (metadata.valid_bars / metadata.total_bars_processed) * 100
                    LOG.info(f"✓ {symbol}: {validation_rate:.2f}% valid bars")
                    success_count += 1
                else:
                    LOG.warning(f"✗ {symbol}: Processing returned None")
                    failure_count += 1
                
            except Exception as e:
                LOG.error(f"✗ {symbol}: Exception - {e}")
                results[symbol] = (None, None)
                failure_count += 1
        
        # Summary
        LOG.info(f"\n{'='*60}")
        LOG.info(f"UNIVERSE PROCESSING COMPLETE")
        LOG.info(f"{'='*60}")
        LOG.info(f"Total symbols: {len(target_symbols)}")
        LOG.info(f"Successful: {success_count}")
        LOG.info(f"Failed: {failure_count}")
        LOG.info(f"{'='*60}\n")
        
        return results
    
    def _get_source_metadata_id(self, symbol: str, timeframe: str) -> str:
        """
        Find most recent source metadata for symbol/timeframe.
        
        Returns source_id for audit trail linkage.
        """
        metadata_dir = self.raw_base_dir / "metadata" / "ingestion_logs"
        
        if not metadata_dir.exists():
            return f"raw_layer_{symbol}_{timeframe}"
        
        # Find matching metadata files
        pattern = f"*_{symbol}_{timeframe}.json"
        metadata_files = sorted(metadata_dir.glob(pattern), reverse=True)
        
        if metadata_files:
            # Return most recent
            return metadata_files[0].stem
        
        return f"raw_layer_{symbol}_{timeframe}"
    
    def get_processing_report(self, results: Dict) -> Dict:
        """
        Generate comprehensive processing report.
        
        Args:
            results: Output from process_universe()
        
        Returns:
            Report dictionary
        """
        total = len(results)
        successful = sum(1 for df, _ in results.values() if df is not None)
        failed = total - successful
        
        # Aggregate statistics
        total_bars = sum(
            meta.total_bars_processed 
            for df, meta in results.values() 
            if meta is not None
        )
        total_valid = sum(
            meta.valid_bars 
            for df, meta in results.values() 
            if meta is not None
        )
        total_missing = sum(
            meta.missing_bars 
            for df, meta in results.values() 
            if meta is not None
        )
        total_outliers = sum(
            meta.outlier_bars 
            for df, meta in results.values() 
            if meta is not None
        )
        
        # Per-symbol breakdown
        symbol_details = {}
        for symbol, (df, meta) in results.items():
            if meta is not None:
                symbol_details[symbol] = {
                    "total_bars": meta.total_bars_processed,
                    "valid_bars": meta.valid_bars,
                    "validation_rate": (meta.valid_bars / meta.total_bars_processed * 100) if meta.total_bars_processed > 0 else 0,
                    "missing_bars": meta.missing_bars,
                    "outlier_bars": meta.outlier_bars,
                    "warnings": len(meta.warnings),
                    "errors": len(meta.errors),
                }
            else:
                symbol_details[symbol] = {"status": "failed"}
        
        return {
            "processing_timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_symbols": total,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total * 100) if total > 0 else 0,
            },
            "aggregated_statistics": {
                "total_bars_processed": total_bars,
                "total_valid_bars": total_valid,
                "total_missing_bars": total_missing,
                "total_outlier_bars": total_outliers,
                "overall_validation_rate": (total_valid / total_bars * 100) if total_bars > 0 else 0,
            },
            "symbol_details": symbol_details,
            "config_version": self.config.config_version,
        }
