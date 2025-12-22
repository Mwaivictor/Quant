"""
Category E: Regime Features (Daily Only)

Purpose: Trade permission and market state detection.

Features:
    - Binary trend regime flag
    - Stress indicator (σ_short / σ_long)

⚠️ DAILY TIMEFRAME ONLY
These features determine trade permission, not direction.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import RegimeConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class RegimeFeatures:
    """
    Compute regime detection features.
    
    DAILY TIMEFRAME ONLY.
    
    These features provide trade permission, not signals.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Compute regime features.
        
        Args:
            df: DataFrame with valid bars only
            timeframe: Must be '1D' if daily_only=True
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - trend_regime: Binary trend flag (1=uptrend, 0=no trend, -1=downtrend)
            - stress_indicator: σ_short / σ_long
        
        Rules:
            - Only computed for daily timeframe
            - Trend regime uses MA crossover with buffer
            - Stress > 1.0 = increasing volatility
        """
        if not self.config.enabled:
            LOG.info("Regime features disabled")
            return df
        
        # Check daily-only constraint
        if self.config.daily_only and timeframe != '1D':
            LOG.warning(f"Regime features only for daily, skipping {timeframe}")
            # Add null columns to maintain schema consistency
            df = df.copy()
            df['trend_regime'] = np.nan
            df['stress_indicator'] = np.nan
            return df
        
        df = df.copy()
        
        # Trend regime detection
        ma_fast = PT.simple_moving_average(df['close'], self.config.trend_ma_fast)
        ma_slow = PT.simple_moving_average(df['close'], self.config.trend_ma_slow)
        
        # Calculate relative position with buffer
        rel_pos = (ma_fast - ma_slow) / ma_slow
        
        # Binary trend regime with buffer
        # +1: Fast > Slow + buffer (uptrend)
        #  0: Within buffer (no trend)
        # -1: Fast < Slow - buffer (downtrend)
        trend_regime = np.where(
            rel_pos > self.config.trend_buffer,
            1,  # Uptrend
            np.where(
                rel_pos < -self.config.trend_buffer,
                -1,  # Downtrend
                0  # No clear trend
            )
        )
        df['trend_regime'] = trend_regime
        
        LOG.debug(f"Trend regime distribution: "
                 f"Up={np.sum(trend_regime==1)}, "
                 f"Neutral={np.sum(trend_regime==0)}, "
                 f"Down={np.sum(trend_regime==-1)}")
        
        # Stress indicator
        log_returns = df['log_return_1']
        stress = PT.stress_indicator(
            log_returns,
            self.config.stress_short_window,
            self.config.stress_long_window
        )
        df['stress_indicator'] = stress
        
        LOG.debug(f"Computed stress_indicator: "
                 f"{df['stress_indicator'].notna().sum()}/{len(df)} non-null values")
        
        LOG.info(f"✓ Computed 2 regime features (daily only)")
        
        return df
