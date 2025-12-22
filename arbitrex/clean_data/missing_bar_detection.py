"""
Missing Bar Detection Module

Identifies gaps in time series data and enforces strict quality gates.

Philosophy:
    - Missing data is a critical issue
    - Never forward-fill or interpolate
    - Explicitly flag all missing bars
    - Enforce symbol exclusion rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from arbitrex.clean_data.config import CleanDataConfig, MissingBarThresholds

LOG = logging.getLogger(__name__)


class MissingBarDetector:
    """
    Detects and flags missing bars in aligned time series.
    
    Rules:
        - Single missing bar → flag it
        - Multiple consecutive missing → symbol may be excluded
        - High missing percentage → symbol excluded
        - Missing bars invalidate downstream calculations
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
        self.thresholds = config.missing_bar_thresholds
    
    def detect_missing_bars(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Detect and flag missing bars.
        
        Args:
            df: Aligned dataframe (from TimeAligner)
            symbol: Symbol being processed
            timeframe: Timeframe
        
        Returns:
            DataFrame with is_missing flag updated
        
        Note:
            is_missing should already be set by TimeAligner.
            This method validates and adds additional context.
        """
        if df.empty:
            return df
        
        # Ensure is_missing column exists
        if "is_missing" not in df.columns:
            df["is_missing"] = df["close"].isna()
        
        # Count consecutive missing sequences
        df = self._mark_consecutive_missing(df)
        
        # Log statistics
        missing_count = df["is_missing"].sum()
        missing_pct = (missing_count / len(df)) * 100
        
        LOG.info(
            f"{symbol} {timeframe}: {missing_count}/{len(df)} bars missing ({missing_pct:.2f}%)"
        )
        
        return df
    
    def _mark_consecutive_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify consecutive missing bar sequences.
        
        Adds column: consecutive_missing_count
        """
        # Create groups of consecutive missing bars
        df["_missing_group"] = (df["is_missing"] != df["is_missing"].shift()).cumsum()
        
        # Count consecutive missing in each group
        df["consecutive_missing_count"] = df.groupby("_missing_group")["is_missing"].transform(
            lambda x: x.sum() if x.iloc[0] else 0
        )
        
        # Drop temporary column
        df = df.drop(columns=["_missing_group"])
        
        return df
    
    def validate_missing_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate missing bar thresholds.
        
        Returns:
            (is_valid, list_of_issues)
        
        Validation Rules:
            - Total missing percentage < threshold
            - Max consecutive missing < threshold
            - Critical gaps identified
        """
        issues = []
        
        if df.empty:
            issues.append("Empty dataframe")
            return False, issues
        
        # Calculate statistics
        missing_count = df["is_missing"].sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Check 1: Total missing percentage
        if missing_pct > self.thresholds.max_missing_percentage * 100:
            issues.append(
                f"Missing bar percentage ({missing_pct:.2f}%) exceeds threshold "
                f"({self.thresholds.max_missing_percentage * 100}%)"
            )
        
        # Check 2: Consecutive missing bars
        if "consecutive_missing_count" in df.columns:
            max_consecutive = df["consecutive_missing_count"].max()
            if max_consecutive > self.thresholds.max_consecutive_missing:
                issues.append(
                    f"Consecutive missing bars ({max_consecutive}) exceeds threshold "
                    f"({self.thresholds.max_consecutive_missing})"
                )
        
        # Check 3: Identify critical gaps (start or end)
        if len(df) > 0:
            if df["is_missing"].iloc[0]:
                issues.append("First bar is missing")
            if df["is_missing"].iloc[-1]:
                issues.append("Last bar is missing")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            LOG.warning(
                f"{symbol} {timeframe} validation failed: {len(issues)} issues"
            )
        
        return is_valid, issues
    
    def get_missing_bar_report(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Generate comprehensive missing bar report.
        
        Returns:
            Dictionary with missing bar statistics
        """
        if df.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_bars": 0,
                "missing_bars": 0,
                "missing_percentage": 0.0,
                "max_consecutive_missing": 0,
                "gaps": [],
            }
        
        missing_count = df["is_missing"].sum()
        missing_pct = (missing_count / len(df)) * 100
        
        # Find all gaps
        gaps = []
        if "consecutive_missing_count" in df.columns:
            gap_starts = df[
                (df["is_missing"]) & 
                (df["consecutive_missing_count"] > 0) &
                ((~df["is_missing"].shift(1)) | (df.index == 0))
            ]
            
            for idx in gap_starts.index:
                gap_size = df.loc[idx, "consecutive_missing_count"]
                gap_start = df.loc[idx, "timestamp_utc"]
                
                # Find gap end
                gap_end_idx = idx + gap_size - 1
                if gap_end_idx < len(df):
                    gap_end = df.loc[gap_end_idx, "timestamp_utc"]
                else:
                    gap_end = df["timestamp_utc"].max()
                
                gaps.append({
                    "start": gap_start.isoformat(),
                    "end": gap_end.isoformat(),
                    "size": int(gap_size),
                })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": len(df),
            "missing_bars": int(missing_count),
            "missing_percentage": float(missing_pct),
            "max_consecutive_missing": int(df.get("consecutive_missing_count", pd.Series([0])).max()),
            "gaps": gaps,
            "timestamp_range": {
                "start": df["timestamp_utc"].min().isoformat(),
                "end": df["timestamp_utc"].max().isoformat(),
            }
        }
    
    def should_exclude_symbol(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[bool, str]:
        """
        Determine if symbol should be excluded due to missing data.
        
        Returns:
            (should_exclude, reason)
        """
        if df.empty:
            return True, "Empty dataframe"
        
        # Validate against thresholds
        is_valid, issues = self.validate_missing_bars(df, symbol, timeframe)
        
        if not is_valid:
            reason = "; ".join(issues)
            LOG.warning(f"Symbol {symbol} {timeframe} excluded: {reason}")
            return True, reason
        
        return False, ""
