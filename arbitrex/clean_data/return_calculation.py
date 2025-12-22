"""
Return Calculation Module

Computes log returns with strict validity rules.

Philosophy:
    - Returns only computed between valid bars
    - Never compute across missing bars
    - Never compute across outliers
    - NULL returns explicitly marked
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from arbitrex.clean_data.config import CleanDataConfig

LOG = logging.getLogger(__name__)


class ReturnCalculator:
    """
    Computes log returns with strict safety rules.
    
    Rules:
        - log_return_1 = log(close_t / close_{t-1})
        - Only if both bars valid (not missing, not outlier)
        - Never across gaps
        - NULL if any condition violated
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Calculate log returns.
        
        Args:
            df: DataFrame with aligned bars and flags
            symbol: Symbol being processed
            timeframe: Timeframe
        
        Returns:
            DataFrame with log_return_1 column
        
        Process:
            1. Initialize log_return_1 = NULL
            2. Identify valid bar pairs (consecutive, non-missing, non-outlier)
            3. Compute log returns only for valid pairs
            4. Leave rest as NULL
        """
        if df.empty:
            return df
        
        # Initialize return column
        df["log_return_1"] = np.nan
        
        # Skip if all bars missing
        if df["is_missing"].all():
            LOG.warning(f"{symbol} {timeframe}: All bars missing, no returns computed")
            return df
        
        # Identify valid bars (not missing, not outlier)
        valid_bars = (~df["is_missing"]) & (~df["is_outlier"])
        
        if valid_bars.sum() < 2:
            LOG.warning(f"{symbol} {timeframe}: < 2 valid bars, no returns computed")
            return df
        
        # Get close prices for valid bars
        close_series = df.loc[valid_bars, "close"]
        
        # Compute log returns
        # This automatically skips gaps because we're only operating on valid_bars subset
        returns = np.log(close_series / close_series.shift(1))
        
        # Assign back to dataframe
        df.loc[valid_bars, "log_return_1"] = returns
        
        # Additional safety: set return to NULL if previous bar was invalid
        # This handles cases where valid_bars might not be consecutive
        for idx in df.index[1:]:
            prev_idx = idx - 1
            
            # If current bar valid but previous bar invalid, return should be NULL
            if valid_bars[idx] and not valid_bars[prev_idx]:
                df.loc[idx, "log_return_1"] = np.nan
        
        # Count valid returns
        valid_returns = df["log_return_1"].notna().sum()
        return_pct = (valid_returns / len(df)) * 100
        
        LOG.info(
            f"{symbol} {timeframe}: {valid_returns}/{len(df)} returns computed ({return_pct:.2f}%)"
        )
        
        return df
    
    def validate_returns(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> tuple[bool, list[str]]:
        """
        Validate return calculations.
        
        Returns:
            (is_valid, list_of_issues)
        
        Checks:
            - No returns across missing bars
            - No returns across outliers
            - No infinite returns
            - No returns at first bar
        """
        issues = []
        
        if df.empty:
            return True, issues
        
        if "log_return_1" not in df.columns:
            issues.append("log_return_1 column missing")
            return False, issues
        
        # Check 1: First bar should have NULL return
        if len(df) > 0 and pd.notna(df["log_return_1"].iloc[0]):
            issues.append("First bar has non-NULL return")
        
        # Check 2: No returns where previous bar was missing
        for idx in df.index[1:]:
            prev_idx = idx - 1
            
            if pd.notna(df.loc[idx, "log_return_1"]):
                # Check previous bar valid
                if df.loc[prev_idx, "is_missing"]:
                    issues.append(
                        f"Return at {df.loc[idx, 'timestamp_utc']} computed across missing bar"
                    )
                
                if df.loc[prev_idx, "is_outlier"]:
                    issues.append(
                        f"Return at {df.loc[idx, 'timestamp_utc']} computed across outlier"
                    )
        
        # Check 3: No infinite returns
        if df["log_return_1"].apply(lambda x: np.isinf(x) if pd.notna(x) else False).any():
            issues.append("Infinite returns detected")
        
        # Check 4: Returns should be NULL where current bar is missing/outlier
        invalid_bars = df["is_missing"] | df["is_outlier"]
        returns_on_invalid = df.loc[invalid_bars, "log_return_1"].notna()
        
        if returns_on_invalid.any():
            issues.append(f"{returns_on_invalid.sum()} returns computed on invalid bars")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            LOG.warning(f"{symbol} {timeframe} return validation failed: {len(issues)} issues")
        
        return is_valid, issues
    
    def get_return_statistics(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """
        Generate return statistics report.
        
        Returns:
            Dictionary with return statistics
        """
        if df.empty or "log_return_1" not in df.columns:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_bars": 0,
                "valid_returns": 0,
                "null_returns": 0,
                "statistics": None,
            }
        
        valid_returns = df["log_return_1"].notna()
        return_values = df.loc[valid_returns, "log_return_1"]
        
        if len(return_values) == 0:
            stats = None
        else:
            stats = {
                "mean": float(return_values.mean()),
                "std": float(return_values.std()),
                "min": float(return_values.min()),
                "max": float(return_values.max()),
                "median": float(return_values.median()),
                "skew": float(return_values.skew()),
                "kurtosis": float(return_values.kurtosis()),
            }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": len(df),
            "valid_returns": int(valid_returns.sum()),
            "null_returns": int((~valid_returns).sum()),
            "return_coverage": float(valid_returns.sum() / len(df) * 100),
            "statistics": stats,
        }
