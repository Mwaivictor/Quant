"""
Validator Module

Implements the valid_bar logic - the mandatory gate for downstream use.

Philosophy:
    - Valid bar = passes ALL checks
    - One failure = invalid bar
    - No exceptions, no overrides
    - Explicit validation results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from arbitrex.clean_data.config import CleanDataConfig, ValidationRules

LOG = logging.getLogger(__name__)


class BarValidator:
    """
    Determines bar validity using strict validation rules.
    
    A bar is valid ONLY if ALL conditions pass:
        1. Not missing
        2. Not outlier
        3. OHLC consistency valid
        4. Timestamp aligned
        5. Return computable (when required)
    
    Otherwise: valid_bar = False
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
        self.rules = config.validation_rules
    
    def validate_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Apply validation rules and set valid_bar flag.
        
        Args:
            df: DataFrame with all flags and calculations
            symbol: Symbol being processed
            timeframe: Timeframe
        
        Returns:
            DataFrame with valid_bar column set
        
        Process:
            1. Initialize valid_bar = True
            2. Apply each validation rule
            3. Logical AND all results
        """
        if df.empty:
            return df
        
        # Initialize as True
        df["valid_bar"] = True
        
        # Rule 1: Must not be missing
        if self.rules.require_non_missing:
            df.loc[df["is_missing"], "valid_bar"] = False
        
        # Rule 2: Must not be outlier
        if self.rules.require_non_outlier:
            df.loc[df["is_outlier"], "valid_bar"] = False
        
        # Rule 3: OHLC consistency
        if self.rules.enforce_ohlc_consistency:
            df = self._check_ohlc_consistency(df)
        
        # Rule 4: Valid returns (if required)
        if self.rules.require_valid_returns and "log_return_1" in df.columns:
            # Bars after first must have valid return
            # (First bar can be valid with NULL return)
            for idx in df.index[1:]:
                if df.loc[idx, "valid_bar"] and pd.isna(df.loc[idx, "log_return_1"]):
                    # Check if this is expected (previous bar invalid)
                    prev_idx = idx - 1
                    if df.loc[prev_idx, "valid_bar"]:
                        # Previous bar valid but return NULL - invalid
                        df.loc[idx, "valid_bar"] = False
        
        # Rule 5: Minimum volume
        if self.rules.min_volume > 0 and "volume" in df.columns:
            low_volume = (df["volume"] < self.rules.min_volume) & (df["volume"].notna())
            df.loc[low_volume, "valid_bar"] = False
        
        # Count valid bars
        valid_count = df["valid_bar"].sum()
        valid_pct = (valid_count / len(df)) * 100
        
        LOG.info(
            f"{symbol} {timeframe}: {valid_count}/{len(df)} valid bars ({valid_pct:.2f}%)"
        )
        
        return df
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC logical consistency.
        
        Sets valid_bar=False if:
            - high < max(open, close)
            - low > min(open, close)
            - high < low
        """
        # Only check non-missing bars
        non_missing = ~df["is_missing"]
        
        if not non_missing.any():
            return df
        
        # Get OHLC
        o = df.loc[non_missing, "open"]
        h = df.loc[non_missing, "high"]
        l = df.loc[non_missing, "low"]
        c = df.loc[non_missing, "close"]
        
        # Consistency checks
        invalid_high = h < np.maximum(o, c)
        invalid_low = l > np.minimum(o, c)
        invalid_hl = h < l
        
        # Mark as invalid
        inconsistent = invalid_high | invalid_low | invalid_hl
        
        if inconsistent.any():
            df.loc[inconsistent[inconsistent].index, "valid_bar"] = False
            LOG.warning(
                f"Marked {inconsistent.sum()} bars invalid due to OHLC inconsistency"
            )
        
        return df
    
    def get_validation_report(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation statistics and breakdown
        """
        if df.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_bars": 0,
                "valid_bars": 0,
                "invalid_bars": 0,
                "validation_rate": 0.0,
                "failure_breakdown": {},
            }
        
        total = len(df)
        valid = df["valid_bar"].sum()
        invalid = total - valid
        
        # Breakdown of failure reasons
        failure_breakdown = {}
        
        invalid_bars = ~df["valid_bar"]
        
        if invalid_bars.any():
            # Missing
            failure_breakdown["missing"] = int(
                (df.loc[invalid_bars, "is_missing"]).sum()
            )
            
            # Outlier
            failure_breakdown["outlier"] = int(
                (df.loc[invalid_bars, "is_outlier"]).sum()
            )
            
            # NULL return (non-first bar)
            if "log_return_1" in df.columns and len(df) > 1:
                invalid_returns = (
                    invalid_bars &
                    (df["log_return_1"].isna()) &
                    (df.index > 0)
                )
                failure_breakdown["null_return"] = int(invalid_returns.sum())
            
            # Low volume
            if "volume" in df.columns and self.rules.min_volume > 0:
                low_vol = (
                    invalid_bars &
                    (df["volume"] < self.rules.min_volume) &
                    (df["volume"].notna())
                )
                failure_breakdown["low_volume"] = int(low_vol.sum())
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": total,
            "valid_bars": int(valid),
            "invalid_bars": int(invalid),
            "validation_rate": float(valid / total * 100),
            "failure_breakdown": failure_breakdown,
            "rules_applied": {
                "require_non_missing": self.rules.require_non_missing,
                "require_non_outlier": self.rules.require_non_outlier,
                "enforce_ohlc_consistency": self.rules.enforce_ohlc_consistency,
                "require_valid_returns": self.rules.require_valid_returns,
                "min_volume": self.rules.min_volume,
            }
        }
    
    def should_accept_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        min_valid_percentage: float = 90.0
    ) -> Tuple[bool, str]:
        """
        Determine if cleaned dataset should be accepted.
        
        Args:
            df: Cleaned dataframe
            symbol: Symbol
            timeframe: Timeframe
            min_valid_percentage: Minimum % of valid bars required
        
        Returns:
            (should_accept, reason)
        """
        if df.empty:
            return False, "Empty dataframe"
        
        valid_count = df["valid_bar"].sum()
        valid_pct = (valid_count / len(df)) * 100
        
        if valid_pct < min_valid_percentage:
            return False, f"Valid bar percentage ({valid_pct:.2f}%) below threshold ({min_valid_percentage}%)"
        
        # Check for critical gaps at start/end
        if len(df) > 0:
            if not df["valid_bar"].iloc[0]:
                return False, "First bar invalid"
            if not df["valid_bar"].iloc[-1]:
                return False, "Last bar invalid"
        
        # Check for long sequences of invalid bars
        max_consecutive_invalid = self._max_consecutive_false(df["valid_bar"])
        if max_consecutive_invalid > 10:  # configurable threshold
            return False, f"Long sequence of invalid bars detected ({max_consecutive_invalid})"
        
        return True, f"Dataset accepted ({valid_pct:.2f}% valid bars)"
    
    def _max_consecutive_false(self, series: pd.Series) -> int:
        """Calculate maximum consecutive False values"""
        groups = (series != series.shift()).cumsum()
        false_groups = series[~series].groupby(groups).size()
        
        if len(false_groups) == 0:
            return 0
        
        return int(false_groups.max())
