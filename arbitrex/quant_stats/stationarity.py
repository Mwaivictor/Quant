"""
Stationarity Testing (ADF)

Augmented Dickey-Fuller test for stationarity of return series.
p-value < 0.05 → stationary → valid for modeling
p-value ≥ 0.05 → non-stationary → flag for review
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available, ADF tests will be disabled")

LOG = logging.getLogger(__name__)


class StationarityTester:
    """
    Test stationarity of time series using Augmented Dickey-Fuller test.
    
    ADF null hypothesis: series has unit root (non-stationary)
    p-value < α → reject null → stationary
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        rolling_window: int = 60,
        max_lag: int = 10
    ):
        """
        Initialize stationarity tester.
        
        Args:
            significance_level: p-value threshold (default 0.05)
            rolling_window: Minimum bars for ADF test
            max_lag: Maximum lag for ADF regression
        """
        self.significance_level = significance_level
        self.rolling_window = rolling_window
        self.max_lag = max_lag
        
        if not STATSMODELS_AVAILABLE:
            LOG.warning("Statsmodels not available, ADF tests will return default values")
        
        LOG.info(f"Stationarity tester initialized: α={significance_level}, "
                f"window={rolling_window}, max_lag={max_lag}")
    
    def run_adf_test(
        self,
        series: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Run Augmented Dickey-Fuller test on series.
        
        Args:
            series: Time series to test
        
        Returns:
            (is_stationary, p_value, test_statistic)
        """
        if not STATSMODELS_AVAILABLE:
            # Return conservative defaults if statsmodels not available
            return False, 1.0, 0.0
        
        if len(series) < self.rolling_window:
            LOG.debug(f"Insufficient data for ADF: {len(series)} < {self.rolling_window}")
            return False, 1.0, 0.0
        
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < self.rolling_window:
            LOG.debug(f"Insufficient non-null data for ADF: {len(series_clean)}")
            return False, 1.0, 0.0
        
        try:
            # Run ADF test
            # maxlag='auto' uses 12*(nobs/100)^{1/4} formula
            result = adfuller(
                series_clean,
                maxlag=self.max_lag,
                regression='c',  # Include constant
                autolag='AIC'  # Use AIC to select lag
            )
            
            test_statistic = result[0]
            p_value = result[1]
            
            # Reject null (non-stationary) if p-value < significance level
            is_stationary = p_value < self.significance_level
            
            LOG.debug(f"ADF test: statistic={test_statistic:.4f}, "
                     f"p-value={p_value:.4f}, stationary={is_stationary}")
            
            return is_stationary, float(p_value), float(test_statistic)
            
        except Exception as e:
            LOG.error(f"ADF test failed: {e}")
            return False, 1.0, 0.0
    
    def rolling_adf_test(
        self,
        series: pd.Series
    ) -> pd.DataFrame:
        """
        Compute rolling ADF test.
        
        Args:
            series: Time series
        
        Returns:
            DataFrame with rolling ADF results
        """
        results = {
            'adf_stationary': [],
            'adf_pvalue': [],
            'adf_test_statistic': []
        }
        
        for i in range(len(series)):
            if i < self.rolling_window - 1:
                # Insufficient data
                results['adf_stationary'].append(False)
                results['adf_pvalue'].append(1.0)
                results['adf_test_statistic'].append(0.0)
            else:
                # Use window ending at position i
                window_data = series.iloc[max(0, i-self.rolling_window+1):i+1]
                is_stat, pval, test_stat = self.run_adf_test(window_data)
                
                results['adf_stationary'].append(is_stat)
                results['adf_pvalue'].append(pval)
                results['adf_test_statistic'].append(test_stat)
        
        return pd.DataFrame(results, index=series.index)
    
    def test_bar(
        self,
        series: pd.Series,
        bar_index: int
    ) -> Tuple[bool, float, float]:
        """
        Test stationarity for specific bar using rolling window.
        
        Args:
            series: Full time series
            bar_index: Index of bar to test
        
        Returns:
            (is_stationary, p_value, test_statistic)
        """
        # Get window ending at bar_index
        start_idx = max(0, bar_index - self.rolling_window + 1)
        window_data = series.iloc[start_idx:bar_index+1]
        
        return self.run_adf_test(window_data)
    
    def check_stationarity(
        self,
        p_value: float
    ) -> bool:
        """
        Check if p-value indicates stationarity.
        
        Args:
            p_value: ADF test p-value
        
        Returns:
            True if stationary
        """
        return p_value < self.significance_level
