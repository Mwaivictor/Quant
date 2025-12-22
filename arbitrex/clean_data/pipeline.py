"""
Clean Data Pipeline

Orchestrates the complete cleaning process with fail-fast error handling.

Philosophy:
    - Abort on critical failure
    - No partial writes
    - Explicit error messages
    - Complete auditability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import logging
import json

from arbitrex.clean_data.config import CleanDataConfig, DEFAULT_CONFIG
from arbitrex.clean_data.schemas import CleanOHLCVSchema, CleanDataMetadata
from arbitrex.clean_data.time_alignment import TimeAligner
from arbitrex.clean_data.missing_bar_detection import MissingBarDetector
from arbitrex.clean_data.outlier_detection import OutlierDetector
from arbitrex.clean_data.return_calculation import ReturnCalculator
from arbitrex.clean_data.spread_estimation import SpreadEstimator
from arbitrex.clean_data.validator import BarValidator

LOG = logging.getLogger(__name__)


class CleanDataPipeline:
    """
    Main pipeline for transforming raw OHLCV to clean OHLCV.
    
    Execution Flow:
        Raw Data
         → Time Alignment
         → Missing Detection
         → Outlier Detection
         → Return Calculation
         → Spread Estimation (optional)
         → Validity Check
         → Write fx_ohlcv_clean
    
    Guarantees:
        - Deterministic output
        - Full auditability
        - Fail-fast on errors
        - No partial writes
    """
    
    def __init__(self, config: Optional[CleanDataConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: CleanDataConfig instance (uses DEFAULT_CONFIG if None)
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize all components
        self.time_aligner = TimeAligner(self.config)
        self.missing_detector = MissingBarDetector(self.config)
        self.outlier_detector = OutlierDetector(self.config)
        self.return_calculator = ReturnCalculator(self.config)
        self.spread_estimator = SpreadEstimator(self.config)
        self.validator = BarValidator(self.config)
        
        # Statistics tracking
        self.processing_stats = {}
        self.warnings = []
        self.errors = []
    
    def process_symbol(
        self,
        raw_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source_id: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], CleanDataMetadata]:
        """
        Process single symbol through complete pipeline.
        
        Args:
            raw_df: Raw OHLCV dataframe
            symbol: Symbol identifier
            timeframe: Timeframe (1H, 4H, 1D)
            source_id: Raw data source identifier
        
        Returns:
            (cleaned_df, metadata) or (None, metadata) on failure
        
        Process:
            1. Validate input
            2. Time alignment
            3. Missing bar detection
            4. Outlier detection
            5. Return calculation
            6. Spread estimation
            7. Validation
            8. Schema conformance
            9. Final acceptance decision
        """
        LOG.info(f"Processing {symbol} {timeframe}: {len(raw_df)} raw bars")
        
        start_time = datetime.utcnow()
        
        try:
            # Step 0: Validate input is not empty
            if raw_df.empty:
                self._record_error(f"{symbol} {timeframe}: Empty input dataframe")
                if self.config.fail_on_critical_error:
                    raise ValueError("Empty input dataframe")
                return None, self._create_metadata(symbol, timeframe, source_id, start_time)
            
            # Step 1: Time Alignment
            LOG.info(f"Step 1/7: Time alignment for {symbol} {timeframe}")
            df = self.time_aligner.align_to_grid(raw_df, symbol, timeframe)
            
            # Validate alignment
            is_valid, issues = self.time_aligner.validate_alignment(df, timeframe)
            if not is_valid:
                self._record_error(f"Alignment validation failed: {'; '.join(issues)}")
                if self.config.fail_on_critical_error:
                    raise ValueError(f"Time alignment failed: {issues}")
            
            # Step 2: Missing Bar Detection
            LOG.info(f"Step 2/7: Missing bar detection for {symbol} {timeframe}")
            df = self.missing_detector.detect_missing_bars(df, symbol, timeframe)
            
            # Check if symbol should be excluded
            should_exclude, exclude_reason = self.missing_detector.should_exclude_symbol(
                df, symbol, timeframe
            )
            if should_exclude:
                self._record_warning(f"Symbol excluded: {exclude_reason}")
                # Continue but mark in metadata
            
            # Step 3: Outlier Detection
            LOG.info(f"Step 3/7: Outlier detection for {symbol} {timeframe}")
            df = self.outlier_detector.detect_outliers(df, symbol, timeframe)
            
            # Step 4: Return Calculation
            LOG.info(f"Step 4/7: Return calculation for {symbol} {timeframe}")
            df = self.return_calculator.calculate_returns(df, symbol, timeframe)
            
            # Validate returns
            is_valid, issues = self.return_calculator.validate_returns(df, symbol, timeframe)
            if not is_valid:
                self._record_warning(f"Return validation issues: {'; '.join(issues)}")
            
            # Step 5: Spread Estimation (optional)
            LOG.info(f"Step 5/7: Spread estimation for {symbol} {timeframe}")
            df = self.spread_estimator.estimate_spreads(df, symbol, timeframe)
            
            # Step 6: Validation
            LOG.info(f"Step 6/7: Bar validation for {symbol} {timeframe}")
            df = self.validator.validate_bars(df, symbol, timeframe)
            
            # Step 7: Schema Conformance
            LOG.info(f"Step 7/7: Schema conformance for {symbol} {timeframe}")
            df = self._ensure_schema_conformance(df, symbol, timeframe, source_id)
            
            # Validate schema
            try:
                CleanOHLCVSchema.validate_dataframe(df)
            except ValueError as e:
                self._record_error(f"Schema validation failed: {e}")
                if self.config.fail_on_critical_error:
                    raise
                return None, self._create_metadata(symbol, timeframe, source_id, start_time)
            
            # Step 8: Final Acceptance Decision
            should_accept, accept_reason = self.validator.should_accept_dataset(
                df, symbol, timeframe
            )
            
            if not should_accept:
                self._record_warning(f"Dataset rejected: {accept_reason}")
                # Return data but mark as rejected in metadata
            else:
                LOG.info(f"Dataset accepted: {accept_reason}")
            
            # Create metadata
            metadata = self._create_metadata(
                symbol, timeframe, source_id, start_time,
                df=df, accepted=should_accept
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            LOG.info(
                f"Completed {symbol} {timeframe} in {processing_time:.2f}s: "
                f"{metadata.valid_bars}/{metadata.total_bars_processed} valid bars"
            )
            
            return df, metadata
            
        except Exception as e:
            self._record_error(f"Pipeline failed for {symbol} {timeframe}: {str(e)}")
            LOG.exception(f"Pipeline error for {symbol} {timeframe}")
            
            if self.config.fail_on_critical_error:
                raise
            
            return None, self._create_metadata(symbol, timeframe, source_id, start_time)
    
    def process_multiple_symbols(
        self,
        raw_data: Dict[str, pd.DataFrame],
        timeframe: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Tuple[Optional[pd.DataFrame], CleanDataMetadata]]:
        """
        Process multiple symbols in batch.
        
        Args:
            raw_data: Dict mapping symbol to raw dataframe
            timeframe: Timeframe for all symbols
            output_dir: Optional output directory for writing results
        
        Returns:
            Dict mapping symbol to (cleaned_df, metadata)
        """
        results = {}
        
        LOG.info(f"Processing {len(raw_data)} symbols for {timeframe}")
        
        for symbol, raw_df in raw_data.items():
            cleaned_df, metadata = self.process_symbol(
                raw_df, symbol, timeframe
            )
            
            results[symbol] = (cleaned_df, metadata)
            
            # Optionally write to disk
            if output_dir and cleaned_df is not None:
                self.write_clean_data(cleaned_df, metadata, output_dir)
        
        # Summary statistics
        total_symbols = len(results)
        successful = sum(1 for df, _ in results.values() if df is not None)
        failed = total_symbols - successful
        
        LOG.info(
            f"Batch processing complete: {successful}/{total_symbols} successful, "
            f"{failed} failed"
        )
        
        return results
    
    def write_clean_data(
        self,
        df: pd.DataFrame,
        metadata: CleanDataMetadata,
        output_dir: Path
    ) -> Path:
        """
        Write cleaned data to disk with metadata.
        
        Args:
            df: Cleaned dataframe
            metadata: Metadata object
            output_dir: Output directory
        
        Returns:
            Path to written file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        symbol = metadata.symbols_processed[0] if metadata.symbols_processed else "UNKNOWN"
        timeframe = metadata.timeframes_processed[0] if metadata.timeframes_processed else "UNKNOWN"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Write data
        data_file = output_dir / f"{symbol}_{timeframe}_{timestamp}_clean.csv"
        df.to_csv(data_file, index=False)
        LOG.info(f"Written clean data to {data_file}")
        
        # Write metadata
        metadata_file = output_dir / f"{symbol}_{timeframe}_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        LOG.info(f"Written metadata to {metadata_file}")
        
        return data_file
    
    def _ensure_schema_conformance(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source_id: Optional[str]
    ) -> pd.DataFrame:
        """
        Ensure dataframe conforms to CleanOHLCVSchema.
        
        Adds missing columns, reorders, sets correct dtypes.
        """
        # Add required columns if missing
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        if "timeframe" not in df.columns:
            df["timeframe"] = timeframe
        if "source_id" not in df.columns:
            df["source_id"] = source_id if source_id else np.nan
        if "schema_version" not in df.columns:
            df["schema_version"] = CleanOHLCVSchema.schema_version
        
        # Ensure all required columns exist
        required_cols = CleanOHLCVSchema.get_column_names()
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Reorder columns
        df = df[required_cols]
        
        # Set dtypes
        dtype_map = CleanOHLCVSchema.get_dtype_map()
        for col, dtype in dtype_map.items():
            if "datetime64" in dtype:
                df[col] = pd.to_datetime(df[col], utc=True)
            elif dtype == "bool":
                # Ensure boolean (fill NaN with False for flags)
                if col in ["is_missing", "is_outlier", "valid_bar"]:
                    df[col] = df[col].fillna(False).astype(bool)
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _create_metadata(
        self,
        symbol: str,
        timeframe: str,
        source_id: Optional[str],
        start_time: datetime,
        df: Optional[pd.DataFrame] = None,
        accepted: bool = False
    ) -> CleanDataMetadata:
        """Create metadata object for processing run"""
        
        if df is not None:
            total_bars = len(df)
            valid_bars = int(df["valid_bar"].sum())
            missing_bars = int(df["is_missing"].sum())
            outlier_bars = int(df["is_outlier"].sum())
            invalid_bars = total_bars - valid_bars
        else:
            total_bars = 0
            valid_bars = 0
            missing_bars = 0
            outlier_bars = 0
            invalid_bars = 0
        
        return CleanDataMetadata(
            processing_timestamp=datetime.utcnow(),
            config_version=self.config.config_version,
            schema_version=CleanOHLCVSchema.schema_version,
            raw_source_path=source_id or "unknown",
            raw_source_timestamp=None,
            total_bars_processed=total_bars,
            valid_bars=valid_bars,
            missing_bars=missing_bars,
            outlier_bars=outlier_bars,
            invalid_bars=invalid_bars,
            symbols_processed=[symbol],
            timeframes_processed=[timeframe],
            config_snapshot=self.config.to_dict(),
            warnings=self.warnings.copy(),
            errors=self.errors.copy(),
        )
    
    def _record_warning(self, message: str):
        """Record warning message"""
        self.warnings.append(message)
        LOG.warning(message)
    
    def _record_error(self, message: str):
        """Record error message"""
        self.errors.append(message)
        LOG.error(message)
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline execution"""
        return {
            "config_version": self.config.config_version,
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "warnings_list": self.warnings,
            "errors_list": self.errors,
        }
