"""
Outlier Detection Module

Flags statistical anomalies and OHLC inconsistencies.

Philosophy:
    - Detection only, NEVER correction
    - Flag outliers, don't remove them
    - Multiple detection methods for robustness
    - Explicit is_outlier flag
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from arbitrex.clean_data.config import CleanDataConfig, OutlierThresholds

LOG = logging.getLogger(__name__)


class OutlierDetector:
    """
    Detects outliers using statistical and logical methods.
    
    Detection Methods:
        1. Price jump test (rolling volatility)
        2. OHLC consistency checks
        3. Zero/negative price detection
        4. Extreme return magnitude
    
    All detection is flag-only. Raw values are NEVER modified.
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
        self.thresholds = config.outlier_thresholds
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Detect and flag outliers.
        
        Args:
            df: DataFrame with aligned bars
            symbol: Symbol being processed
            timeframe: Timeframe
        
        Returns:
            DataFrame with is_outlier flag set
        
        Process:
            1. Initialize is_outlier = False
            2. Run price jump test
            3. Run OHLC consistency checks
            4. Run zero/negative price checks
            5. Logical OR all flags
        """
        if df.empty:
            return df
        
        # Initialize outlier flag
        df["is_outlier"] = False
        
        # Skip if all bars are missing
        if df["is_missing"].all():
            return df
        
        # Test 1: Price jump detection
        df = self._detect_price_jumps(df)
        
        # Test 2: OHLC consistency
        df = self._detect_ohlc_inconsistencies(df)
        
        # Test 3: Zero or negative prices
        df = self._detect_invalid_prices(df)
        
        # Test 4: Extreme returns (if computed)
        if "log_return_1" in df.columns:
            df = self._detect_extreme_returns(df)
        
        # Count outliers
        outlier_count = df["is_outlier"].sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        LOG.info(
            f"{symbol} {timeframe}: {outlier_count}/{len(df)} outliers detected ({outlier_pct:.2f}%)"
        )
        
        return df
    
    def _detect_price_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect price jumps using rolling volatility.
        
        Method:
            - Compute rolling std of log returns
            - Flag returns > k * rolling_std
        """
        # Need enough data for rolling window
        if len(df) < self.thresholds.min_bars_required:
            return df
        
        # Compute simple returns for volatility estimation
        valid_close = df.loc[~df["is_missing"], "close"]
        
        if len(valid_close) < self.thresholds.volatility_window:
            return df
        
        # Compute log returns on valid bars only
        returns = np.log(valid_close / valid_close.shift(1))
        
        # Rolling volatility
        rolling_std = returns.rolling(
            window=self.thresholds.volatility_window,
            min_periods=self.thresholds.min_bars_required
        ).std()
        
        # Detect jumps
        threshold = self.thresholds.price_jump_std_multiplier * rolling_std
        is_jump = (returns.abs() > threshold) & (~returns.isna()) & (~threshold.isna())
        
        # Map back to original dataframe
        df.loc[is_jump[is_jump].index, "is_outlier"] = True
        
        if is_jump.sum() > 0:
            LOG.debug(f"Detected {is_jump.sum()} price jumps")
        
        return df
    
    def _detect_ohlc_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect OHLC logical inconsistencies.
        
        Rules:
            - high >= max(open, close)
            - low <= min(open, close)
            - high >= low
            - All prices > 0
        """
        # Only check non-missing bars
        non_missing = ~df["is_missing"]
        
        if not non_missing.any():
            return df
        
        # Get OHLC for non-missing bars
        o = df.loc[non_missing, "open"]
        h = df.loc[non_missing, "high"]
        l = df.loc[non_missing, "low"]
        c = df.loc[non_missing, "close"]
        
        # Check 1: high < max(open, close)
        invalid_high = h < np.maximum(o, c)
        
        # Check 2: low > min(open, close)
        invalid_low = l > np.minimum(o, c)
        
        # Check 3: high < low
        invalid_hl = h < l
        
        # Combine all inconsistencies
        inconsistent = invalid_high | invalid_low | invalid_hl
        
        # Flag in original dataframe
        df.loc[inconsistent[inconsistent].index, "is_outlier"] = True
        
        if inconsistent.sum() > 0:
            LOG.warning(f"Detected {inconsistent.sum()} OHLC inconsistencies")
        
        return df
    
    def _detect_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect zero, negative, or extremely small prices.
        """
        # Only check non-missing bars
        non_missing = ~df["is_missing"]
        
        if not non_missing.any():
            return df
        
        # Check all OHLC columns
        price_cols = ["open", "high", "low", "close"]
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            prices = df.loc[non_missing, col]
            
            # Flag zero or negative
            invalid = (prices <= self.thresholds.zero_price_tolerance) | (prices.isna())
            
            if invalid.any():
                df.loc[invalid[invalid].index, "is_outlier"] = True
                LOG.warning(f"Detected {invalid.sum()} invalid prices in {col}")
        
        return df
    
    def _detect_extreme_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag extreme return magnitudes.
        
        Catches catastrophic errors like decimal point shifts.
        """
        if "log_return_1" not in df.columns:
            return df
        
        # Only check valid returns
        valid_returns = df["log_return_1"].notna()
        
        if not valid_returns.any():
            return df
        
        # Flag extreme magnitudes
        extreme = (
            df.loc[valid_returns, "log_return_1"].abs() > 
            self.thresholds.max_abs_log_return
        )
        
        if extreme.any():
            df.loc[extreme[extreme].index, "is_outlier"] = True
            LOG.warning(f"Detected {extreme.sum()} extreme returns")
        
        return df
    
    def get_outlier_report(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """
        Generate outlier statistics report.
        
        Returns:
            Dictionary with outlier details
        """
        if df.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_bars": 0,
                "outlier_bars": 0,
                "outlier_percentage": 0.0,
                "outlier_timestamps": [],
            }
        
        outlier_count = df["is_outlier"].sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        # Get outlier timestamps
        outlier_timestamps = []
        if outlier_count > 0:
            outliers = df[df["is_outlier"]]
            outlier_timestamps = [
                {
                    "timestamp": ts.isoformat(),
                    "close": float(close) if pd.notna(close) else None,
                    "return": float(ret) if pd.notna(ret) else None,
                }
                for ts, close, ret in zip(
                    outliers["timestamp_utc"],
                    outliers.get("close", [None] * len(outliers)),
                    outliers.get("log_return_1", [None] * len(outliers)),
                )
            ]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": len(df),
            "outlier_bars": int(outlier_count),
            "outlier_percentage": float(outlier_pct),
            "outlier_timestamps": outlier_timestamps[:10],  # Limit to first 10
            "outlier_count_detail": {
                "price_jumps": int((df.get("_price_jump", False)).sum()) if "_price_jump" in df.columns else 0,
                "ohlc_inconsistent": int((df.get("_ohlc_inconsistent", False)).sum()) if "_ohlc_inconsistent" in df.columns else 0,
                "invalid_prices": int((df.get("_invalid_price", False)).sum()) if "_invalid_price" in df.columns else 0,
            }
        }
