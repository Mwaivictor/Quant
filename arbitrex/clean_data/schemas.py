"""
Clean Data Layer Schemas

Defines the exact output contract for fx_ohlcv_clean.
This schema is immutable and versioned.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import pandas as pd


@dataclass
class CleanOHLCVSchema:
    """
    Output schema for fx_ohlcv_clean.
    
    Contract Guarantees:
        - Raw OHLC values never altered
        - All timestamps in UTC
        - All flags explicitly set
        - NULL values only where data missing/invalid
    """
    
    # Primary key
    timestamp_utc: datetime  # Canonical bar close time (UTC)
    symbol: str              # FX pair (e.g., "EURUSD")
    timeframe: str           # 1H / 4H / 1D / 1M
    
    # Raw OHLCV (NEVER MODIFIED)
    open: Optional[float]    # Raw open price (NULL if missing)
    high: Optional[float]    # Raw high price (NULL if missing)
    low: Optional[float]     # Raw low price (NULL if missing)
    close: Optional[float]   # Raw close price (NULL if missing)
    volume: Optional[float]  # Tick volume (NULL if missing)
    
    # Derived quantities (minimal)
    log_return_1: Optional[float]    # log(close_t / close_{t-1})
    spread_estimate: Optional[float] # Optional bid-ask spread estimate
    
    # Quality flags (EXPLICIT)
    is_missing: bool         # True if expected bar not found in raw data
    is_outlier: bool         # True if statistical anomaly detected
    valid_bar: bool          # True only if all validation passes
    
    # Auditability
    source_id: Optional[str] # Reference to raw ingestion cycle
    
    # Schema version
    schema_version: str = "1.0.0"
    
    @classmethod
    def get_column_names(cls) -> List[str]:
        """Return ordered list of column names"""
        return [
            "timestamp_utc",
            "symbol",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "log_return_1",
            "spread_estimate",
            "is_missing",
            "is_outlier",
            "valid_bar",
            "source_id",
            "schema_version",
        ]
    
    @classmethod
    def get_dtype_map(cls) -> dict:
        """Return pandas dtype mapping"""
        return {
            "timestamp_utc": "datetime64[ns, UTC]",
            "symbol": "str",
            "timeframe": "str",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "log_return_1": "float64",
            "spread_estimate": "float64",
            "is_missing": "bool",
            "is_outlier": "bool",
            "valid_bar": "bool",
            "source_id": "str",
            "schema_version": "str",
        }
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame conforms to schema.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = cls.get_column_names()
        
        # Check all required columns present
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check no extra columns
        extra_cols = set(df.columns) - set(required_columns)
        if extra_cols:
            raise ValueError(f"Unexpected columns found: {extra_cols}")
        
        # Check dtypes match (allow compatible types)
        dtype_map = cls.get_dtype_map()
        for col, expected_dtype in dtype_map.items():
            actual_dtype = str(df[col].dtype)
            
            # Allow compatible numeric types
            if expected_dtype == "float64" and actual_dtype in ["float32", "float64", "int64", "int32"]:
                continue
            
            # Allow compatible datetime types
            if "datetime64" in expected_dtype and "datetime64" in actual_dtype:
                continue
            
            # Allow compatible string types
            if expected_dtype == "str" and actual_dtype in ["object", "string"]:
                continue
            
            # Allow compatible bool types
            if expected_dtype == "bool" and actual_dtype in ["bool", "boolean"]:
                continue
            
            # Strict match for others
            if expected_dtype not in actual_dtype and actual_dtype not in expected_dtype:
                raise ValueError(f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'")
        
        # Check boolean columns have no nulls
        bool_cols = ["is_missing", "is_outlier", "valid_bar"]
        for col in bool_cols:
            if df[col].isna().any():
                raise ValueError(f"Boolean column '{col}' contains NULL values")
        
        # Check primary key uniqueness
        pk_cols = ["timestamp_utc", "symbol", "timeframe"]
        if df.duplicated(subset=pk_cols).any():
            raise ValueError(f"Duplicate rows found in primary key: {pk_cols}")
        
        return True
    
    @classmethod
    def create_empty_dataframe(cls) -> pd.DataFrame:
        """Create empty DataFrame with correct schema"""
        columns = cls.get_column_names()
        df = pd.DataFrame(columns=columns)
        
        # Set dtypes
        dtype_map = cls.get_dtype_map()
        for col, dtype in dtype_map.items():
            if "datetime64" in dtype:
                df[col] = pd.to_datetime(df[col], utc=True)
            elif dtype == "bool":
                df[col] = df[col].astype("boolean")
            elif dtype == "float64":
                df[col] = df[col].astype("float64")
            elif dtype == "str":
                df[col] = df[col].astype("str")
        
        return df


@dataclass
class CleanDataMetadata:
    """
    Metadata for each clean dataset batch.
    
    Required for full auditability and reproducibility.
    """
    
    # Processing metadata
    processing_timestamp: datetime
    config_version: str
    schema_version: str
    
    # Source references
    raw_source_path: str
    raw_source_timestamp: Optional[datetime]
    
    # Processing statistics
    total_bars_processed: int
    valid_bars: int
    missing_bars: int
    outlier_bars: int
    invalid_bars: int
    
    # Symbols and timeframes processed
    symbols_processed: List[str]
    timeframes_processed: List[str]
    
    # Thresholds used
    config_snapshot: dict
    
    # Warnings and errors
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> dict:
        """Serialize metadata to dictionary"""
        return {
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "config_version": self.config_version,
            "schema_version": self.schema_version,
            "raw_source_path": self.raw_source_path,
            "raw_source_timestamp": self.raw_source_timestamp.isoformat() if self.raw_source_timestamp else None,
            "statistics": {
                "total_bars_processed": self.total_bars_processed,
                "valid_bars": self.valid_bars,
                "missing_bars": self.missing_bars,
                "outlier_bars": self.outlier_bars,
                "invalid_bars": self.invalid_bars,
                "valid_bar_percentage": (self.valid_bars / self.total_bars_processed * 100) if self.total_bars_processed > 0 else 0.0,
            },
            "scope": {
                "symbols_processed": self.symbols_processed,
                "timeframes_processed": self.timeframes_processed,
            },
            "config_snapshot": self.config_snapshot,
            "issues": {
                "warnings": self.warnings,
                "errors": self.errors,
            }
        }
