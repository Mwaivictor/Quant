"""
Spread Estimation Module (Optional)

Estimates bid-ask spread from OHLC data.

Philosophy:
    - Optional component (disabled by default)
    - Uses high-low range as proxy
    - Smoothed with exponential moving average
    - Flagged if unreliable
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from arbitrex.clean_data.config import CleanDataConfig, SpreadEstimation

LOG = logging.getLogger(__name__)


class SpreadEstimator:
    """
    Estimates bid-ask spread from OHLC bars.
    
    Method:
        spread_estimate = (high - low) / close
        
    Optional smoothing with EMA.
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
        self.spread_config = config.spread_estimation
        self.enabled = self.spread_config.enabled
    
    def estimate_spreads(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Estimate bid-ask spreads.
        
        Args:
            df: DataFrame with OHLC data
            symbol: Symbol being processed
            timeframe: Timeframe
        
        Returns:
            DataFrame with spread_estimate column
        """
        # Initialize spread column
        df["spread_estimate"] = np.nan
        
        # Skip if disabled
        if not self.enabled:
            return df
        
        if df.empty:
            return df
        
        # Only compute for valid bars
        valid_bars = (~df["is_missing"]) & (~df["is_outlier"])
        
        if not valid_bars.any():
            LOG.warning(f"{symbol} {timeframe}: No valid bars for spread estimation")
            return df
        
        # Get OHLC for valid bars
        valid_df = df.loc[valid_bars]
        
        if not all(col in valid_df.columns for col in ["high", "low", "close"]):
            LOG.warning(f"{symbol} {timeframe}: Missing OHLC columns for spread estimation")
            return df
        
        # Compute raw spread
        if self.spread_config.use_hl_spread:
            raw_spread = (valid_df["high"] - valid_df["low"]) / valid_df["close"]
        else:
            # Alternative: use tick volume-based estimation (not implemented)
            raw_spread = np.nan
        
        # Apply smoothing if configured
        if self.spread_config.smoothing_alpha > 0:
            smoothed_spread = raw_spread.ewm(
                alpha=self.spread_config.smoothing_alpha,
                adjust=False
            ).mean()
        else:
            smoothed_spread = raw_spread
        
        # Assign back to dataframe
        df.loc[valid_bars, "spread_estimate"] = smoothed_spread
        
        # Log statistics
        valid_spreads = df["spread_estimate"].notna().sum()
        if valid_spreads > 0:
            mean_spread = df["spread_estimate"].mean() * 10000  # in bps
            LOG.info(
                f"{symbol} {timeframe}: {valid_spreads} spreads estimated "
                f"(mean: {mean_spread:.2f} bps)"
            )
        
        return df
    
    def validate_spreads(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> tuple[bool, list[str]]:
        """
        Validate spread estimates.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.enabled:
            return True, issues
        
        if df.empty or "spread_estimate" not in df.columns:
            return True, issues
        
        valid_spreads = df["spread_estimate"].notna()
        
        if not valid_spreads.any():
            return True, issues
        
        spread_values = df.loc[valid_spreads, "spread_estimate"]
        
        # Check 1: No negative spreads
        if (spread_values < 0).any():
            issues.append("Negative spreads detected")
        
        # Check 2: No extreme spreads (>10%)
        if (spread_values > 0.10).any():
            issues.append("Extreme spreads detected (>10%)")
        
        # Check 3: Spreads only on valid bars
        invalid_bars = df["is_missing"] | df["is_outlier"]
        spreads_on_invalid = df.loc[invalid_bars, "spread_estimate"].notna()
        
        if spreads_on_invalid.any():
            issues.append(f"Spreads estimated on {spreads_on_invalid.sum()} invalid bars")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def get_spread_statistics(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[dict]:
        """
        Generate spread statistics.
        
        Returns:
            Dictionary with spread stats or None if disabled
        """
        if not self.enabled:
            return None
        
        if df.empty or "spread_estimate" not in df.columns:
            return None
        
        valid_spreads = df["spread_estimate"].notna()
        
        if not valid_spreads.any():
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "enabled": True,
                "valid_estimates": 0,
                "statistics": None,
            }
        
        spread_values = df.loc[valid_spreads, "spread_estimate"]
        
        # Convert to basis points
        spread_bps = spread_values * 10000
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "enabled": True,
            "valid_estimates": int(valid_spreads.sum()),
            "statistics": {
                "mean_bps": float(spread_bps.mean()),
                "median_bps": float(spread_bps.median()),
                "min_bps": float(spread_bps.min()),
                "max_bps": float(spread_bps.max()),
                "std_bps": float(spread_bps.std()),
            }
        }
