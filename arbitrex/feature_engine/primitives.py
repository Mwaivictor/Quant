"""
Primitive Transform Module

Basic building blocks for feature computation.
All transforms are causal (no lookahead) and deterministic.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

LOG = logging.getLogger(__name__)


class PrimitiveTransforms:
    """
    Primitive transforms used across feature categories.
    
    All functions:
        - Accept Series or DataFrame
        - Return same shape as input
        - Use only data ≤ t for value at t
        - Are deterministic
    """
    
    @staticmethod
    def rolling_mean(series: pd.Series, window: int) -> pd.Series:
        """
        Rolling mean ending at each timestamp.
        
        Args:
            series: Input series
            window: Window size
        
        Returns:
            Rolling mean (NaN for first window-1 values)
        """
        return series.rolling(window=window, min_periods=window).mean()
    
    @staticmethod
    def rolling_std(series: pd.Series, window: int, ddof: int = 1) -> pd.Series:
        """
        Rolling standard deviation ending at each timestamp.
        
        Args:
            series: Input series
            window: Window size
            ddof: Delta degrees of freedom (default 1 for sample std)
        
        Returns:
            Rolling std (NaN for first window-1 values)
        """
        return series.rolling(window=window, min_periods=window).std(ddof=ddof)
    
    @staticmethod
    def rolling_sum(series: pd.Series, window: int) -> pd.Series:
        """Rolling sum ending at each timestamp"""
        return series.rolling(window=window, min_periods=window).sum()
    
    @staticmethod
    def rolling_min(series: pd.Series, window: int) -> pd.Series:
        """Rolling minimum ending at each timestamp"""
        return series.rolling(window=window, min_periods=window).min()
    
    @staticmethod
    def rolling_max(series: pd.Series, window: int) -> pd.Series:
        """Rolling maximum ending at each timestamp"""
        return series.rolling(window=window, min_periods=window).max()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Average True Range.
        
        TR = max(high - low, |high - close_prev|, |low - close_prev|)
        ATR = rolling_mean(TR, window)
        
        Causal: Uses close from previous bar only.
        """
        # True Range components
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        
        # True Range = max of three components
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # Average True Range
        atr_val = PrimitiveTransforms.rolling_mean(tr, window)
        
        return atr_val
    
    @staticmethod
    def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average (SMA)"""
        return PrimitiveTransforms.rolling_mean(series, window)
    
    @staticmethod
    def price_distance_to_ma(
        price: pd.Series,
        ma: pd.Series,
        atr: pd.Series
    ) -> pd.Series:
        """
        Normalized price distance to moving average.
        
        distance = (price - MA) / ATR
        
        Stationarity: Normalized by ATR (volatility)
        """
        return (price - ma) / atr.replace(0, np.nan)
    
    @staticmethod
    def ma_slope(ma: pd.Series, window: int, atr: pd.Series) -> pd.Series:
        """
        Normalized MA slope.
        
        slope = (MA_t - MA_{t-window}) / ATR
        
        Stationarity: Normalized by ATR
        """
        slope = ma - ma.shift(window)
        return slope / atr.replace(0, np.nan)
    
    @staticmethod
    def rolling_return(log_returns: pd.Series, window: int) -> pd.Series:
        """
        Rolling cumulative return over window.
        
        R_w = sum(log_return from t-w+1 to t)
        
        Causal: Only uses returns up to and including t
        """
        return PrimitiveTransforms.rolling_sum(log_returns, window)
    
    @staticmethod
    def efficiency_ratio(
        close: pd.Series,
        direction_window: int,
        volatility_window: int
    ) -> pd.Series:
        """
        Efficiency Ratio (Kaufman).
        
        ER = |price_change| / sum(|price_changes|)
        
        Measures trend efficiency:
            ER → 1.0: Strong trend
            ER → 0.0: Random walk / chop
        """
        # Net price change
        net_change = (close - close.shift(direction_window)).abs()
        
        # Sum of absolute changes
        abs_changes = close.diff().abs()
        volatility = PrimitiveTransforms.rolling_sum(abs_changes, volatility_window)
        
        # Efficiency Ratio
        er = net_change / volatility.replace(0, np.nan)
        
        return er.clip(0, 1)  # Bound to [0, 1]
    
    @staticmethod
    def range_compression(
        high: pd.Series,
        low: pd.Series,
        window: int,
        atr: pd.Series
    ) -> pd.Series:
        """
        Range compression ratio.
        
        Measures how compressed recent range is relative to volatility.
        
        range_ratio = (high_w - low_w) / ATR
        
        Low values → consolidation
        High values → expansion
        """
        range_high = PrimitiveTransforms.rolling_max(high, window)
        range_low = PrimitiveTransforms.rolling_min(low, window)
        
        range_w = range_high - range_low
        
        return range_w / atr.replace(0, np.nan)
    
    @staticmethod
    def stress_indicator(
        returns: pd.Series,
        short_window: int,
        long_window: int
    ) -> pd.Series:
        """
        Market stress indicator.
        
        stress = σ_short / σ_long
        
        > 1.0: Increasing volatility (stress)
        < 1.0: Decreasing volatility (calm)
        
        Regime detection tool.
        """
        vol_short = PrimitiveTransforms.rolling_std(returns, short_window)
        vol_long = PrimitiveTransforms.rolling_std(returns, long_window)
        
        return vol_short / vol_long.replace(0, np.nan)
