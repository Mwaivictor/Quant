"""
Input Validation Module

Enforces data trust boundary: Only consume valid_bar == True.
No internal cleaning or repair logic allowed.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

from arbitrex.feature_engine.config import FeatureEngineConfig

LOG = logging.getLogger(__name__)


class FeatureInputValidator:
    """
    Validates input data from Clean Data Layer.
    
    Rules:
        - Must have valid_bar column
        - Only consume rows where valid_bar == True
        - Check for required columns
        - Verify minimum bar count
        - Check for timeframe consistency
    """
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
    
    def validate_input(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Tuple[bool, pd.DataFrame, List[str]]:
        """
        Validate input DataFrame from Clean Data Layer.
        
        Args:
            df: Clean OHLCV DataFrame
            symbol: Symbol being processed
            timeframe: Timeframe (1H, 4H, 1D)
        
        Returns:
            (is_valid, filtered_df, errors)
        
        Rules:
            - Only valid_bar == True rows passed through
            - No repair or imputation allowed
            - Minimum bar count enforced
        """
        errors = []
        
        # Check for required columns
        required_cols = [
            'timestamp_utc', 'symbol', 'timeframe',
            'open', 'high', 'low', 'close', 'volume',
            'log_return_1', 'valid_bar'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, df, errors
        
        # Check valid_bar column exists and is boolean
        if df['valid_bar'].dtype != bool:
            errors.append(f"valid_bar must be boolean, got {df['valid_bar'].dtype}")
            return False, df, errors
        
        # Filter to valid bars only
        valid_bar_count = df['valid_bar'].sum()
        total_bar_count = len(df)
        
        if self.config.verbose_logging:
            LOG.info(f"{symbol} {timeframe}: {valid_bar_count}/{total_bar_count} valid bars "
                    f"({valid_bar_count/total_bar_count*100:.1f}%)")
        
        # Check minimum valid bar percentage
        if total_bar_count > 0:
            valid_pct = valid_bar_count / total_bar_count
            if valid_pct < self.config.min_valid_bar_pct:
                errors.append(
                    f"Valid bar percentage ({valid_pct:.1%}) below threshold "
                    f"({self.config.min_valid_bar_pct:.1%})"
                )
                return False, df, errors
        
        # Filter to valid bars only
        df_valid = df[df['valid_bar']].copy()
        
        # Check minimum bar count
        if len(df_valid) < self.config.min_valid_bars_required:
            errors.append(
                f"Insufficient valid bars: {len(df_valid)} < "
                f"{self.config.min_valid_bars_required} required"
            )
            return False, df_valid, errors
        
        # Verify timeframe consistency
        unique_timeframes = df_valid['timeframe'].unique()
        if len(unique_timeframes) > 1:
            errors.append(f"Multiple timeframes in data: {unique_timeframes}")
            return False, df_valid, errors
        
        if unique_timeframes[0] != timeframe:
            errors.append(
                f"Timeframe mismatch: expected {timeframe}, "
                f"got {unique_timeframes[0]}"
            )
            return False, df_valid, errors
        
        # Verify symbol consistency
        unique_symbols = df_valid['symbol'].unique()
        if len(unique_symbols) > 1:
            errors.append(f"Multiple symbols in data: {unique_symbols}")
            return False, df_valid, errors
        
        if unique_symbols[0] != symbol:
            errors.append(
                f"Symbol mismatch: expected {symbol}, got {unique_symbols[0]}"
            )
            return False, df_valid, errors
        
        # Verify timestamp monotonicity
        if not df_valid['timestamp_utc'].is_monotonic_increasing:
            errors.append("Timestamps not monotonic increasing")
            return False, df_valid, errors
        
        # Verify no NaN in OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            nan_count = df_valid[col].isna().sum()
            if nan_count > 0:
                errors.append(f"NaN values in {col}: {nan_count} bars")
                return False, df_valid, errors
        
        # All checks passed
        if self.config.verbose_logging:
            LOG.info(f"✓ Input validation passed: {symbol} {timeframe}")
        
        return True, df_valid, errors
    
    def check_sufficient_history(
        self,
        df: pd.DataFrame,
        required_window: int
    ) -> bool:
        """
        Check if sufficient history exists for feature computation.
        
        Args:
            df: Valid bar DataFrame
            required_window: Maximum window size needed
        
        Returns:
            True if sufficient history available
        """
        return len(df) >= required_window
