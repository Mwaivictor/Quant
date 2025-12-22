"""
Volatility-Based Signal Filtering

Percentile-based volatility regime classification.
- LOW: < 10th percentile
- NORMAL: 10-90th percentile  
- HIGH: > 90th percentile

Suppress signals in extreme or low volatility regimes.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from enum import Enum
import logging

LOG = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class VolatilityFilter:
    """
    Filter signals based on volatility regime.
    
    Computes rolling volatility and classifies into regimes
    based on historical percentiles.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,
        min_percentile: float = 10.0,
        max_percentile: float = 90.0,
        lookback_window: int = 252  # ~1 year of daily bars
    ):
        """
        Initialize volatility filter.
        
        Args:
            rolling_window: Window for volatility computation
            min_percentile: Lower percentile threshold
            max_percentile: Upper percentile threshold
            lookback_window: Window for percentile calculation
        """
        self.rolling_window = rolling_window
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.lookback_window = lookback_window
        
        LOG.info(f"Volatility filter initialized: window={rolling_window}, "
                f"percentiles=[{min_percentile}, {max_percentile}]")
    
    def compute_volatility(
        self,
        returns: pd.Series,
        annualize: bool = False
    ) -> pd.Series:
        """
        Compute rolling volatility (standard deviation).
        
        Args:
            returns: Return series
            annualize: Whether to annualize (√252 factor for daily)
        
        Returns:
            Rolling volatility
        """
        vol = returns.rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window // 2
        ).std()
        
        if annualize:
            # Assume daily bars, annualize with √252
            vol = vol * np.sqrt(252)
        
        return vol
    
    def compute_percentile_thresholds(
        self,
        volatility: pd.Series,
        bar_index: int
    ) -> Tuple[float, float]:
        """
        Compute percentile thresholds causally.
        
        Args:
            volatility: Volatility series
            bar_index: Current bar index
        
        Returns:
            (lower_threshold, upper_threshold)
        """
        # Use lookback window ending at bar_index
        start_idx = max(0, bar_index - self.lookback_window + 1)
        historical_vol = volatility.iloc[start_idx:bar_index+1]
        
        # Remove NaNs
        historical_vol_clean = historical_vol.dropna()
        
        if len(historical_vol_clean) < 20:
            # Insufficient data, return conservative thresholds
            return 0.0, np.inf
        
        lower_threshold = np.percentile(historical_vol_clean, self.min_percentile)
        upper_threshold = np.percentile(historical_vol_clean, self.max_percentile)
        
        return float(lower_threshold), float(upper_threshold)
    
    def classify_regime(
        self,
        current_vol: float,
        lower_threshold: float,
        upper_threshold: float
    ) -> VolatilityRegime:
        """
        Classify volatility regime.
        
        Args:
            current_vol: Current volatility
            lower_threshold: Lower percentile threshold
            upper_threshold: Upper percentile threshold
        
        Returns:
            VolatilityRegime
        """
        if pd.isna(current_vol):
            return VolatilityRegime.UNKNOWN
        
        if current_vol < lower_threshold:
            return VolatilityRegime.LOW
        elif current_vol > upper_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL
    
    def should_filter(
        self,
        regime: VolatilityRegime,
        allow_low: bool = False,
        allow_high: bool = False
    ) -> bool:
        """
        Determine if signal should be filtered based on regime.
        
        Args:
            regime: Current volatility regime
            allow_low: Allow signals in low volatility
            allow_high: Allow signals in high volatility
        
        Returns:
            True if signal should be filtered (suppressed)
        """
        if regime == VolatilityRegime.UNKNOWN:
            return True  # Filter unknown regime
        
        if regime == VolatilityRegime.LOW and not allow_low:
            return True
        
        if regime == VolatilityRegime.HIGH and not allow_high:
            return True
        
        return False
    
    def analyze_bar(
        self,
        returns: pd.Series,
        bar_index: int
    ) -> dict:
        """
        Analyze volatility regime for specific bar.
        
        Args:
            returns: Return series
            bar_index: Index of bar to analyze
        
        Returns:
            Dict with volatility metrics and regime
        """
        # Compute volatility for full series up to bar_index
        window_data = returns.iloc[:bar_index+1]
        volatility = self.compute_volatility(window_data)
        
        if len(volatility) == 0 or pd.isna(volatility.iloc[-1]):
            return {
                'current_volatility': 0.0,
                'volatility_percentile': 0.0,
                'regime': VolatilityRegime.UNKNOWN,
                'lower_threshold': 0.0,
                'upper_threshold': 0.0
            }
        
        current_vol = volatility.iloc[-1]
        
        # Compute percentile thresholds
        lower_thresh, upper_thresh = self.compute_percentile_thresholds(
            volatility,
            bar_index
        )
        
        # Classify regime
        regime = self.classify_regime(current_vol, lower_thresh, upper_thresh)
        
        # Compute percentile of current volatility
        historical_vol = volatility.iloc[max(0, bar_index-self.lookback_window+1):bar_index+1]
        historical_vol_clean = historical_vol.dropna()
        
        if len(historical_vol_clean) > 0:
            percentile = (historical_vol_clean < current_vol).sum() / len(historical_vol_clean) * 100
        else:
            percentile = 50.0
        
        return {
            'current_volatility': float(current_vol),
            'volatility_percentile': float(percentile),
            'regime': regime,
            'lower_threshold': float(lower_thresh),
            'upper_threshold': float(upper_thresh)
        }
    
    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Compute Average True Range (ATR).
        
        ATR is more robust volatility measure for OHLC data.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
        
        Returns:
            ATR series
        """
        # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
