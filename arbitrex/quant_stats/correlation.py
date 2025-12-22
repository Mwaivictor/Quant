"""
Cross-Pair Correlation Analysis

Compute rolling correlations between multiple symbols.
Suppress signals when correlation is too high (> threshold).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

LOG = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyze cross-pair correlations.
    
    Computes rolling correlation matrices and identifies
    highly correlated pairs that may pose risk concentration.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,
        max_correlation_threshold: float = 0.85,
        min_pairs: int = 2
    ):
        """
        Initialize correlation analyzer.
        
        Args:
            rolling_window: Window for correlation computation
            max_correlation_threshold: Maximum allowed correlation
            min_pairs: Minimum number of pairs required
        """
        self.rolling_window = rolling_window
        self.max_correlation_threshold = max_correlation_threshold
        self.min_pairs = min_pairs
        
        LOG.info(f"Correlation analyzer initialized: window={rolling_window}, "
                f"max_corr={max_correlation_threshold}")
    
    def compute_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
        bar_index: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix from multiple return series.
        
        Args:
            returns_dict: Dict mapping symbol -> return series
            bar_index: If provided, compute causal correlation up to this bar
        
        Returns:
            Correlation matrix
        """
        if len(returns_dict) < self.min_pairs:
            LOG.warning(f"Insufficient pairs: {len(returns_dict)} < {self.min_pairs}")
            return pd.DataFrame()
        
        # Combine into DataFrame
        df = pd.DataFrame(returns_dict)
        
        # If bar_index provided, slice causally
        if bar_index is not None:
            # Get window ending at bar_index
            start_idx = max(0, bar_index - self.rolling_window + 1)
            df = df.iloc[start_idx:bar_index+1]
        else:
            # Use last rolling_window bars
            df = df.tail(self.rolling_window)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        return corr_matrix
    
    def get_correlation_metrics(
        self,
        corr_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract correlation metrics from matrix.
        
        Args:
            corr_matrix: Correlation matrix
        
        Returns:
            Dict with avg_correlation, max_correlation, correlation_dispersion
        """
        if corr_matrix.empty:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0
            }
        
        # Extract upper triangle (excluding diagonal)
        n = len(corr_matrix)
        if n < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0
            }
        
        # Get upper triangle values
        upper_triangle = []
        for i in range(n):
            for j in range(i+1, n):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    upper_triangle.append(abs(corr_val))  # Use absolute correlations
        
        if len(upper_triangle) == 0:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0
            }
        
        avg_corr = np.mean(upper_triangle)
        max_corr = np.max(upper_triangle)
        
        # Dispersion = standard deviation of correlations
        dispersion = np.std(upper_triangle)
        
        return {
            'avg_correlation': float(avg_corr),
            'max_correlation': float(max_corr),
            'correlation_dispersion': float(dispersion)
        }
    
    def find_highly_correlated_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs with correlation above threshold.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold (uses self.max_correlation_threshold if None)
        
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if threshold is None:
            threshold = self.max_correlation_threshold
        
        if corr_matrix.empty:
            return []
        
        high_corr_pairs = []
        symbols = corr_matrix.columns.tolist()
        
        for i, sym1 in enumerate(symbols):
            for j in range(i+1, len(symbols)):
                sym2 = symbols[j]
                corr_val = corr_matrix.loc[sym1, sym2]
                
                if not pd.isna(corr_val) and abs(corr_val) > threshold:
                    high_corr_pairs.append((sym1, sym2, float(corr_val)))
        
        return high_corr_pairs
    
    def check_correlation(
        self,
        returns_dict: Dict[str, pd.Series],
        bar_index: int
    ) -> bool:
        """
        Check if correlations are within acceptable range.
        
        Args:
            returns_dict: Dict mapping symbol -> return series
            bar_index: Current bar index
        
        Returns:
            True if correlations are acceptable (max_corr < threshold)
        """
        corr_matrix = self.compute_correlation_matrix(returns_dict, bar_index)
        
        if corr_matrix.empty:
            return True  # No correlation issues if insufficient data
        
        metrics = self.get_correlation_metrics(corr_matrix)
        max_corr = metrics['max_correlation']
        
        is_acceptable = max_corr < self.max_correlation_threshold
        
        LOG.debug(f"Correlation check at bar {bar_index}: "
                 f"max_corr={max_corr:.4f}, acceptable={is_acceptable}")
        
        return is_acceptable
    
    def analyze_bar(
        self,
        returns_dict: Dict[str, pd.Series],
        bar_index: int
    ) -> Dict[str, any]:
        """
        Analyze correlations for specific bar.
        
        Args:
            returns_dict: Dict mapping symbol -> return series
            bar_index: Index of bar to analyze
        
        Returns:
            Dict with correlation metrics and high correlation pairs
        """
        if len(returns_dict) < self.min_pairs:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0,
                'high_correlation_pairs': [],
                'correlation_acceptable': True
            }
        
        corr_matrix = self.compute_correlation_matrix(returns_dict, bar_index)
        
        if corr_matrix.empty:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0,
                'high_correlation_pairs': [],
                'correlation_acceptable': True
            }
        
        metrics = self.get_correlation_metrics(corr_matrix)
        high_corr_pairs = self.find_highly_correlated_pairs(corr_matrix)
        
        is_acceptable = metrics['max_correlation'] < self.max_correlation_threshold
        
        return {
            **metrics,
            'high_correlation_pairs': high_corr_pairs,
            'correlation_acceptable': is_acceptable
        }
    
    def rolling_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> pd.Series:
        """
        Compute rolling correlation between two series.
        
        Args:
            series1: First series
            series2: Second series
        
        Returns:
            Rolling correlation
        """
        return series1.rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window // 2
        ).corr(series2)
