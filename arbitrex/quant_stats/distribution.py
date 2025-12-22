"""
Distribution Stability & Outlier Detection

Z-score based outlier detection and distribution stability checks.
z_t = (x_t - μ_window) / σ_window

Outlier if |z_t| > threshold (typically 3.0)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

LOG = logging.getLogger(__name__)


class DistributionAnalyzer:
    """
    Analyze distribution stability and detect outliers using z-scores.
    
    Computes rolling mean/std and flags observations with extreme z-scores.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,
        z_score_threshold: float = 3.0,
        min_samples: int = 30
    ):
        """
        Initialize distribution analyzer.
        
        Args:
            rolling_window: Window for rolling statistics
            z_score_threshold: Z-score threshold for outliers
            min_samples: Minimum samples required
        """
        self.rolling_window = rolling_window
        self.z_score_threshold = z_score_threshold
        self.min_samples = min_samples
        
        LOG.info(f"Distribution analyzer initialized: window={rolling_window}, "
                f"z_threshold={z_score_threshold}")
    
    def compute_rolling_stats(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling mean and std.
        
        Args:
            series: Time series
        
        Returns:
            (rolling_mean, rolling_std)
        """
        rolling_mean = series.rolling(
            window=self.rolling_window,
            min_periods=self.min_samples
        ).mean()
        
        rolling_std = series.rolling(
            window=self.rolling_window,
            min_periods=self.min_samples
        ).std()
        
        return rolling_mean, rolling_std
    
    def compute_z_scores(
        self,
        series: pd.Series,
        rolling_mean: pd.Series,
        rolling_std: pd.Series
    ) -> pd.Series:
        """
        Compute z-scores.
        
        Args:
            series: Original series
            rolling_mean: Rolling mean
            rolling_std: Rolling std
        
        Returns:
            Z-scores
        """
        # Avoid division by zero
        rolling_std_safe = rolling_std.replace(0, np.nan)
        
        z_scores = (series - rolling_mean) / rolling_std_safe
        
        return z_scores
    
    def detect_outliers(
        self,
        z_scores: pd.Series
    ) -> pd.Series:
        """
        Detect outliers based on z-score threshold.
        
        Args:
            z_scores: Z-scores
        
        Returns:
            Boolean series indicating outliers
        """
        return np.abs(z_scores) > self.z_score_threshold
    
    def check_distribution_stability(
        self,
        series: pd.Series,
        lookback: int = 20
    ) -> bool:
        """
        Check if distribution is stable over recent period.
        
        Uses coefficient of variation to assess stability.
        
        Args:
            series: Time series
            lookback: Bars to look back
        
        Returns:
            True if distribution stable
        """
        if len(series) < lookback:
            return False
        
        recent = series.iloc[-lookback:]
        
        # Remove NaNs
        recent_clean = recent.dropna()
        
        if len(recent_clean) < lookback // 2:
            return False
        
        mean = recent_clean.mean()
        std = recent_clean.std()
        
        # Check coefficient of variation
        # CoV = σ / |μ| (if μ ≠ 0)
        if abs(mean) < 1e-8:
            # Near-zero mean, check if std is also small
            return std < 1e-6
        
        cov = abs(std / mean)
        
        # Distribution is stable if CoV is not too large
        # For returns, CoV can be quite large, so use generous threshold
        stable = cov < 10.0
        
        LOG.debug(f"Distribution stability: CoV={cov:.4f}, stable={stable}")
        
        return stable
    
    def analyze_bar(
        self,
        series: pd.Series,
        bar_index: int
    ) -> Dict[str, float]:
        """
        Analyze distribution for specific bar.
        
        Args:
            series: Full time series
            bar_index: Index of bar to analyze
        
        Returns:
            Dict with z_score, is_outlier, rolling_mean, rolling_std, distribution_stable
        """
        # Get window ending at bar_index (causal)
        window_data = series.iloc[:bar_index+1]
        
        if len(window_data) < self.min_samples:
            return {
                'z_score': 0.0,
                'is_outlier': False,
                'rolling_mean': 0.0,
                'rolling_std': 0.0,
                'distribution_stable': False
            }
        
        # Compute rolling stats
        rolling_mean, rolling_std = self.compute_rolling_stats(window_data)
        
        # Get values at bar_index
        mean_val = rolling_mean.iloc[-1] if not pd.isna(rolling_mean.iloc[-1]) else 0.0
        std_val = rolling_std.iloc[-1] if not pd.isna(rolling_std.iloc[-1]) else 0.0
        
        # Compute z-score for current bar
        if std_val > 1e-8:
            z_score = (series.iloc[bar_index] - mean_val) / std_val
        else:
            z_score = 0.0
        
        is_outlier = abs(z_score) > self.z_score_threshold
        
        # Check distribution stability
        dist_stable = self.check_distribution_stability(
            window_data,
            lookback=min(20, len(window_data))
        )
        
        return {
            'z_score': float(z_score),
            'is_outlier': bool(is_outlier),
            'rolling_mean': float(mean_val),
            'rolling_std': float(std_val),
            'distribution_stable': bool(dist_stable)
        }
    
    def analyze_series(
        self,
        series: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze full series.
        
        Args:
            series: Time series
        
        Returns:
            DataFrame with analysis results
        """
        rolling_mean, rolling_std = self.compute_rolling_stats(series)
        z_scores = self.compute_z_scores(series, rolling_mean, rolling_std)
        outliers = self.detect_outliers(z_scores)
        
        results = pd.DataFrame({
            'value': series,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_scores,
            'is_outlier': outliers
        })
        
        return results
