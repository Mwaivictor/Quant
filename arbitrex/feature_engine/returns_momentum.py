"""
Category A: Returns & Momentum Features

Purpose: Directional persistence and momentum strength.

Features:
    - Rolling returns (3, 6, 12 bars)
    - Momentum score (R_12 / σ_12)

All features are stationary (returns, not prices).
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from arbitrex.feature_engine.config import ReturnsMomentumConfig
from arbitrex.feature_engine.primitives import PrimitiveTransforms as PT

LOG = logging.getLogger(__name__)


class ReturnsMomentumFeatures:
    """
    Compute returns and momentum features.
    
    All features describe directional persistence, not prediction.
    """
    
    def __init__(self, config: ReturnsMomentumConfig):
        self.config = config
    
    def compute(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute returns and momentum features.
        
        Args:
            df: DataFrame with valid bars only
        
        Returns:
            DataFrame with added feature columns
        
        Features Added:
            - rolling_return_3: 3-bar cumulative log return
            - rolling_return_6: 6-bar cumulative log return
            - rolling_return_12: 12-bar cumulative log return
            - momentum_score: R_12 / σ_12
        """
        if not self.config.enabled:
            LOG.info("Returns & Momentum features disabled")
            return df
        
        df = df.copy()
        
        # Extract log returns (already computed by Clean Data Layer)
        log_returns = df['log_return_1']
        
        # Rolling returns for different windows
        for window in self.config.return_windows:
            col_name = f'rolling_return_{window}'
            df[col_name] = PT.rolling_return(log_returns, window)
            
            LOG.debug(f"Computed {col_name}: "
                     f"{df[col_name].notna().sum()}/{len(df)} non-null values")
        
        # Momentum score: return / volatility
        return_12 = PT.rolling_return(log_returns, self.config.momentum_window)
        vol_12 = PT.rolling_std(log_returns, self.config.momentum_vol_window)
        
        df['momentum_score'] = return_12 / vol_12.replace(0, np.nan)
        
        LOG.info(f"✓ Computed {len(self.config.return_windows) + 1} "
                f"returns & momentum features")
        
        return df
