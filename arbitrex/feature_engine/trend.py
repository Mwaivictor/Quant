"""
Category C: Trend Structure Features (Descriptive)

Purpose: Trend geometry description, NOT prediction.

Features:
    - MA slope (normalized by ATR)
    - Price-to-MA distance (normalized by ATR)

⚠️ These describe current structure, not future direction.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import TrendConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class TrendFeatures:
    """
    Compute trend structure features.
    
    All features describe geometry, not direction prediction.
    """
    
    def __init__(self, config: TrendConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute trend structure features.
        
        Args:
            df: DataFrame with valid bars only
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - ma_12_slope: Normalized slope of 12-bar MA
            - ma_24_slope: Normalized slope of 24-bar MA
            - ma_50_slope: Normalized slope of 50-bar MA
            - distance_to_ma_12: (close - MA_12) / ATR
            - distance_to_ma_24: (close - MA_24) / ATR
            - distance_to_ma_50: (close - MA_50) / ATR
        """
        if not self.config.enabled:
            LOG.info("Trend features disabled")
            return df
        
        df = df.copy()
        
        # Compute ATR for normalization
        atr = PT.atr(
            df['high'],
            df['low'],
            df['close'],
            self.config.distance_atr_window
        )
        
        close = df['close']
        
        # Compute features for each MA window
        for ma_window in self.config.ma_windows:
            # Moving average
            ma = PT.simple_moving_average(close, ma_window)
            
            # MA slope (normalized)
            ma_slope = PT.ma_slope(ma, self.config.slope_window, atr)
            df[f'ma_{ma_window}_slope'] = ma_slope
            
            # Price-to-MA distance (normalized)
            distance = PT.price_distance_to_ma(close, ma, atr)
            df[f'distance_to_ma_{ma_window}'] = distance
            
            LOG.debug(f"Computed MA_{ma_window} features: "
                     f"{df[f'ma_{ma_window}_slope'].notna().sum()}/{len(df)} "
                     f"non-null values")
        
        LOG.info(f"✓ Computed {len(self.config.ma_windows) * 2} trend features")
        
        return df
