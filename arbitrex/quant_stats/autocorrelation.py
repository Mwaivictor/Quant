"""
Autocorrelation & Trend Persistence Analysis

Computes autocorrelation of returns to detect trend persistence.
High autocorrelation → trend likely persists.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

LOG = logging.getLogger(__name__)


class AutocorrelationAnalyzer:
    """
    Analyze autocorrelation structure of return series.
    
    ρ_k = corr(r_t, r_{t-k})
    
    High autocorr at short lags → momentum/trend persistence
    Decaying autocorr → mean reversion expected
    """
    
    def __init__(self, lags: List[int], rolling_window: int, min_threshold: float):
        """
        Initialize autocorrelation analyzer.
        
        Args:
            lags: List of lags to compute (e.g., [1, 5, 10, 20])
            rolling_window: Window for rolling autocorr
            min_threshold: Minimum autocorr to consider significant
        """
        self.lags = sorted(lags)
        self.rolling_window = rolling_window
        self.min_threshold = min_threshold
        
        LOG.info(f"Autocorrelation analyzer initialized: lags={lags}, "
                f"window={rolling_window}, threshold={min_threshold}")
    
    def compute_autocorrelation(
        self,
        returns: pd.Series,
        lag: int
    ) -> pd.Series:
        """
        Compute rolling autocorrelation for a given lag.
        
        Args:
            returns: Log returns series
            lag: Lag for autocorrelation
        
        Returns:
            Rolling autocorrelation series
        """
        # Use rolling window correlation
        autocorr = returns.rolling(window=self.rolling_window).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False
        )
        
        return autocorr
    
    def compute_all_lags(
        self,
        returns: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Compute autocorrelation for all configured lags.
        
        Args:
            returns: Log returns series
        
        Returns:
            Dictionary of {lag: autocorr_series}
        """
        autocorr_results = {}
        
        for lag in self.lags:
            col_name = f'autocorr_lag{lag}'
            autocorr_results[col_name] = self.compute_autocorrelation(returns, lag)
            
            LOG.debug(f"Computed autocorr lag {lag}: "
                     f"{autocorr_results[col_name].notna().sum()} non-null values")
        
        return autocorr_results
    
    def compute_trend_persistence_score(
        self,
        autocorr_values: Dict[str, float]
    ) -> float:
        """
        Aggregate autocorrelation into trend persistence score.
        
        Score = average of significant autocorrelations (> threshold)
        
        Args:
            autocorr_values: Dict of autocorr values by lag
        
        Returns:
            Trend persistence score (0-1)
        """
        significant_autocorrs = []
        
        for lag in self.lags:
            key = f'autocorr_lag{lag}'
            if key in autocorr_values:
                val = autocorr_values[key]
                if not np.isnan(val) and abs(val) >= self.min_threshold:
                    significant_autocorrs.append(abs(val))
        
        if len(significant_autocorrs) == 0:
            return 0.0
        
        # Average significant autocorrelations
        score = np.mean(significant_autocorrs)
        
        return float(score)
    
    def check_persistence(
        self,
        trend_score: float
    ) -> bool:
        """
        Check if trend persistence is sufficient for signal.
        
        Args:
            trend_score: Trend persistence score
        
        Returns:
            True if persistence check passed
        """
        return trend_score >= self.min_threshold
    
    def analyze_bar(
        self,
        returns: pd.Series,
        bar_index: int
    ) -> Dict[str, float]:
        """
        Analyze autocorrelation for a specific bar.
        
        Args:
            returns: Full return series
            bar_index: Index of bar to analyze
        
        Returns:
            Dictionary with autocorr values and trend score
        """
        # Get window ending at bar_index
        window_returns = returns.iloc[:bar_index+1].tail(self.rolling_window)
        
        if len(window_returns) < self.rolling_window:
            # Insufficient data
            result = {f'autocorr_lag{lag}': np.nan for lag in self.lags}
            result['trend_persistence_score'] = 0.0
            return result
        
        # Compute autocorrelations
        result = {}
        for lag in self.lags:
            if len(window_returns) > lag:
                autocorr = window_returns.autocorr(lag=lag)
                result[f'autocorr_lag{lag}'] = float(autocorr) if not np.isnan(autocorr) else np.nan
            else:
                result[f'autocorr_lag{lag}'] = np.nan
        
        # Compute trend persistence score
        result['trend_persistence_score'] = self.compute_trend_persistence_score(result)
        
        return result
