"""
Normalization Module

Applies rolling z-score normalization to feature vectors.

Rules:
    - Rolling window only (no global statistics)
    - No future information
    - Normalization metadata stored with features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from arbitrex.feature_engine.config import NormalizationConfig

LOG = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalize features using rolling z-score.
    
    Formula:
        x_norm(t) = (x_t - μ_{t-W}) / σ_{t-W}
    
    All normalization parameters must be stored for live inference.
    """
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.normalization_metadata: Dict[str, Dict] = {}
    
    def normalize(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply rolling z-score normalization to features.
        
        Args:
            df: DataFrame with computed features
            feature_columns: List of feature column names to normalize
        
        Returns:
            (normalized_df, normalization_metadata)
        
        Normalization:
            - Rolling mean and std computed with window
            - Z-score clipped to prevent extreme outliers
            - Metadata stored for each feature
        """
        df = df.copy()
        metadata = {}
        
        for col in feature_columns:
            if col not in df.columns:
                LOG.warning(f"Feature column {col} not found, skipping normalization")
                continue
            
            # Skip if insufficient data
            if df[col].notna().sum() < self.config.min_bars_required:
                LOG.warning(f"Insufficient data for {col} normalization")
                df[f'{col}_norm'] = np.nan
                continue
            
            # Compute rolling statistics
            if self.config.use_robust:
                # Robust statistics: median and MAD
                rolling_median = df[col].rolling(
                    window=self.config.norm_window,
                    min_periods=self.config.min_bars_required
                ).median()
                
                # Median Absolute Deviation
                mad = df[col].rolling(
                    window=self.config.norm_window,
                    min_periods=self.config.min_bars_required
                ).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                
                # Convert MAD to std-equivalent
                rolling_std = mad * 1.4826
                rolling_mean = rolling_median
                
            else:
                # Standard statistics: mean and std
                rolling_mean = df[col].rolling(
                    window=self.config.norm_window,
                    min_periods=self.config.min_bars_required
                ).mean()
                
                rolling_std = df[col].rolling(
                    window=self.config.norm_window,
                    min_periods=self.config.min_bars_required
                ).std()
            
            # Z-score normalization
            z_score = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
            
            # Clip extreme outliers
            z_score_clipped = z_score.clip(
                -self.config.z_score_clip,
                self.config.z_score_clip
            )
            
            # Store normalized feature
            df[f'{col}_norm'] = z_score_clipped
            
            # Store normalization metadata
            metadata[col] = {
                'method': 'rolling_z_score',
                'window': self.config.norm_window,
                'min_bars': self.config.min_bars_required,
                'clip': self.config.z_score_clip,
                'robust': self.config.use_robust,
                'mean_col': f'{col}_mean',
                'std_col': f'{col}_std',
            }
            
            # Optionally store rolling statistics for inspection
            df[f'{col}_mean'] = rolling_mean
            df[f'{col}_std'] = rolling_std
            
            LOG.debug(f"Normalized {col}: "
                     f"{df[f'{col}_norm'].notna().sum()}/{len(df)} non-null values")
        
        self.normalization_metadata = metadata
        
        LOG.info(f"✓ Normalized {len(feature_columns)} features")
        
        return df, metadata
    
    def get_feature_vector_columns(
        self,
        feature_columns: List[str],
        include_unnormalized: bool = False
    ) -> List[str]:
        """
        Get list of normalized feature column names.
        
        Args:
            feature_columns: Original feature column names
            include_unnormalized: If True, include both raw and normalized
        
        Returns:
            List of column names for feature vector
        """
        if include_unnormalized:
            # Include both raw and normalized
            cols = []
            for col in feature_columns:
                cols.append(col)
                cols.append(f'{col}_norm')
            return cols
        else:
            # Normalized only
            return [f'{col}_norm' for col in feature_columns]
