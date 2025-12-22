"""
Category B: Volatility Structure Features

Purpose: Risk context and regime awareness.

Features:
    - Rolling volatility (σ_6, σ_12, σ_24)
    - Normalized ATR (ATR / close)

All features are stationary (normalized by price or intrinsic scale).
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import VolatilityConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class VolatilityFeatures:
    """
    Compute volatility structure features.
    
    All features describe risk context, not direction.
    """
    
    def __init__(self, config: VolatilityConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute volatility features.
        
        Args:
            df: DataFrame with valid bars only
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - vol_6: 6-bar rolling volatility of log returns
            - vol_12: 12-bar rolling volatility
            - vol_24: 24-bar rolling volatility
            - atr_normalized: ATR / close_t
        """
        if not self.config.enabled:
            LOG.info("Volatility features disabled")
            return df
        
        df = df.copy()
        
        # Extract log returns
        log_returns = df['log_return_1']
        
        # Rolling volatility for different windows
        for window in self.config.vol_windows:
            col_name = f'vol_{window}'
            df[col_name] = PT.rolling_std(log_returns, window)
            
            LOG.debug(f"Computed {col_name}: "
                     f"{df[col_name].notna().sum()}/{len(df)} non-null values")
        
        # Normalized ATR
        atr = PT.atr(
            df['high'],
            df['low'],
            df['close'],
            self.config.atr_window
        )
        
        df['atr_normalized'] = atr / df['close']
        
        LOG.info(f"✓ Computed {len(self.config.vol_windows) + 1} "
                f"volatility features")
        
        return df
