"""
Category F: Execution/Cost Filter Features (Optional)

Purpose: Prevent untradable signals due to execution costs.

Features:
    - Spread ratio (avg_spread / ATR)

⚠️ NEVER PASSED TO ML MODELS
These are execution filters only.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import ExecutionConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class ExecutionFeatures:
    """
    Compute execution cost features.
    
    ⚠️ ML_EXCLUDED = True
    
    These features are for execution filtering, not ML models.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute execution cost features.
        
        Args:
            df: DataFrame with valid bars only
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - spread_ratio: avg_spread / ATR
        
        Rules:
            - Only if spread_estimate column exists
            - ML_EXCLUDED flag set
        """
        if not self.config.enabled:
            LOG.info("Execution features disabled")
            return df
        
        df = df.copy()
        
        # Check if spread estimate exists
        if 'spread_estimate' not in df.columns:
            LOG.warning("spread_estimate column not found, skipping execution features")
            df['spread_ratio'] = np.nan
            return df
        
        # Compute ATR for normalization
        atr = PT.atr(
            df['high'],
            df['low'],
            df['close'],
            self.config.spread_atr_window
        )
        
        # Average spread
        avg_spread = PT.rolling_mean(
            df['spread_estimate'],
            self.config.spread_avg_window
        )
        
        # Spread ratio
        df['spread_ratio'] = avg_spread / atr.replace(0, np.nan)
        
        LOG.debug(f"Computed spread_ratio: "
                 f"{df['spread_ratio'].notna().sum()}/{len(df)} non-null values")
        
        LOG.info(f"✓ Computed 1 execution feature (ML excluded)")
        
        return df
