"""
Category D: Range & Market Efficiency Features

Purpose: Detect chop vs flow, consolidation vs expansion.

Features:
    - Efficiency Ratio (Kaufman)
    - Range compression ratio

These features describe market structure, not direction.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import EfficiencyConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class EfficiencyFeatures:
    """
    Compute market efficiency and range features.
    
    All features describe structure quality, not direction.
    """
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute efficiency and range features.
        
        Args:
            df: DataFrame with valid bars only
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - efficiency_ratio: Kaufman ER (0 = chop, 1 = trend)
            - range_compression: (high_w - low_w) / ATR
        """
        if not self.config.enabled:
            LOG.info("Efficiency features disabled")
            return df
        
        df = df.copy()
        
        # Efficiency Ratio
        er = PT.efficiency_ratio(
            df['close'],
            self.config.er_direction_window,
            self.config.er_volatility_window
        )
        df['efficiency_ratio'] = er
        
        LOG.debug(f"Computed efficiency_ratio: "
                 f"{df['efficiency_ratio'].notna().sum()}/{len(df)} non-null values")
        
        # Range compression
        atr = PT.atr(
            df['high'],
            df['low'],
            df['close'],
            self.config.range_atr_window
        )
        
        range_comp = PT.range_compression(
            df['high'],
            df['low'],
            self.config.range_window,
            atr
        )
        df['range_compression'] = range_comp
        
        LOG.debug(f"Computed range_compression: "
                 f"{df['range_compression'].notna().sum()}/{len(df)} non-null values")
        
        LOG.info(f"✓ Computed 2 efficiency features")
        
        return df
