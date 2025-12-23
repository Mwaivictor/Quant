"""
Adaptive & Regime-Aware Risk Thresholds - Institutional Grade

Replaces static thresholds with dynamic, data-driven limits that adapt to:
1. Market volatility regimes
2. Rolling statistical distributions
3. Stress conditions

Version: 2.0.0 (Enterprise)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from scipy import stats


@dataclass
class RegimeParameters:
    """Risk parameters for a specific market regime"""
    regime_name: str
    risk_per_trade: float
    max_position_size_multiplier: float
    max_gross_exposure: float
    max_leverage: float
    volatility_multiplier: float
    kill_switch_sensitivity: float  # Multiplier for kill switch thresholds
    correlation_stress_factor: float  # Inflate correlations by this factor
    
    def to_dict(self) -> dict:
        return {
            'regime_name': self.regime_name,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size_multiplier': self.max_position_size_multiplier,
            'max_gross_exposure': self.max_gross_exposure,
            'max_leverage': self.max_leverage,
            'volatility_multiplier': self.volatility_multiplier,
            'kill_switch_sensitivity': self.kill_switch_sensitivity,
            'correlation_stress_factor': self.correlation_stress_factor
        }


class RegimeAwareRiskLimits:
    """
    Maintains distinct risk profiles for each market regime.
    
    Regimes:
    - TRENDING: Favorable for momentum strategies
    - RANGING: Mean-reversion friendly
    - VOLATILE: Elevated risk, reduce exposure
    - STRESSED: Crisis mode, maximum conservatism
    """
    
    def __init__(self):
        self.regimes = self._initialize_default_regimes()
        self.current_regime = 'TRENDING'
    
    def _initialize_default_regimes(self) -> Dict[str, RegimeParameters]:
        """Initialize institutional-grade regime parameters"""
        return {
            'TRENDING': RegimeParameters(
                regime_name='TRENDING',
                risk_per_trade=0.01,  # 1% per trade
                max_position_size_multiplier=1.2,
                max_gross_exposure=2.0,  # 200% gross exposure
                max_leverage=2.0,
                volatility_multiplier=1.0,
                kill_switch_sensitivity=1.0,
                correlation_stress_factor=1.0  # No stress
            ),
            'RANGING': RegimeParameters(
                regime_name='RANGING',
                risk_per_trade=0.01,
                max_position_size_multiplier=1.0,
                max_gross_exposure=1.5,
                max_leverage=1.5,
                volatility_multiplier=1.0,
                kill_switch_sensitivity=1.0,
                correlation_stress_factor=1.1  # Slight correlation increase
            ),
            'VOLATILE': RegimeParameters(
                regime_name='VOLATILE',
                risk_per_trade=0.005,  # 0.5% per trade (reduced)
                max_position_size_multiplier=0.7,
                max_gross_exposure=1.0,  # 100% gross exposure (reduced)
                max_leverage=1.0,
                volatility_multiplier=0.7,  # Scale down by 30%
                kill_switch_sensitivity=1.5,  # More sensitive kill switches
                correlation_stress_factor=1.3  # Correlations increase 30%
            ),
            'STRESSED': RegimeParameters(
                regime_name='STRESSED',
                risk_per_trade=0.002,  # 0.2% per trade (maximum conservatism)
                max_position_size_multiplier=0.3,
                max_gross_exposure=0.5,  # 50% gross exposure (defensive)
                max_leverage=0.5,
                volatility_multiplier=0.3,  # Scale down by 70%
                kill_switch_sensitivity=0.5,  # VERY sensitive (half the threshold)
                correlation_stress_factor=1.5  # Assume 50% correlation increase
            )
        }
    
    def get_regime_parameters(self, regime: str) -> RegimeParameters:
        """Get parameters for specified regime"""
        return self.regimes.get(regime, self.regimes['TRENDING'])
    
    def set_regime(self, regime: str):
        """Update current regime"""
        if regime in self.regimes:
            self.current_regime = regime
    
    def get_current_parameters(self) -> RegimeParameters:
        """Get parameters for current regime"""
        return self.regimes[self.current_regime]
    
    def customize_regime(self, regime: str, **kwargs):
        """Customize regime parameters"""
        if regime in self.regimes:
            current = self.regimes[regime]
            for key, value in kwargs.items():
                if hasattr(current, key):
                    setattr(current, key, value)


class AdaptiveVolatilityThresholds:
    """
    Replaces static volatility thresholds with adaptive, percentile-based limits.
    
    Features:
    - Rolling window of historical volatility
    - Percentile-based thresholds (e.g., 90th percentile)
    - Stress-adjusted scaling
    - Automatic outlier detection
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        high_vol_percentile: float = 90.0,
        extreme_vol_percentile: float = 95.0,
        crisis_vol_percentile: float = 99.0,
        min_samples: int = 30
    ):
        self.lookback_days = lookback_days
        self.high_vol_percentile = high_vol_percentile
        self.extreme_vol_percentile = extreme_vol_percentile
        self.crisis_vol_percentile = crisis_vol_percentile
        self.min_samples = min_samples
        
        # Historical volatility storage
        self.volatility_history: deque = deque(maxlen=lookback_days)
        
        # Cached percentiles (updated when history changes)
        self._percentiles_cache: Optional[dict] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def record_volatility(self, volatility: float, timestamp: Optional[datetime] = None):
        """Record daily volatility observation"""
        self.volatility_history.append({
            'volatility': volatility,
            'timestamp': timestamp or datetime.utcnow()
        })
        self._percentiles_cache = None  # Invalidate cache
    
    def get_adaptive_thresholds(self) -> dict:
        """
        Calculate adaptive volatility thresholds.
        
        Returns:
            dict with 'high', 'extreme', 'crisis' thresholds
        """
        if len(self.volatility_history) < self.min_samples:
            # Insufficient data - return conservative defaults
            return {
                'high': 0.015,  # 1.5% daily vol
                'extreme': 0.025,  # 2.5%
                'crisis': 0.040,  # 4.0%
                'median': 0.010,
                'is_adaptive': False,
                'sample_size': len(self.volatility_history)
            }
        
        # Check cache
        if self._percentiles_cache is not None:
            cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds() / 3600
            if cache_age < 1.0:  # Cache valid for 1 hour
                return self._percentiles_cache
        
        # Calculate percentiles
        vols = np.array([v['volatility'] for v in self.volatility_history])
        
        thresholds = {
            'high': np.percentile(vols, self.high_vol_percentile),
            'extreme': np.percentile(vols, self.extreme_vol_percentile),
            'crisis': np.percentile(vols, self.crisis_vol_percentile),
            'median': np.median(vols),
            'mean': np.mean(vols),
            'std': np.std(vols),
            'is_adaptive': True,
            'sample_size': len(self.volatility_history),
            'percentiles': {
                'high': self.high_vol_percentile,
                'extreme': self.extreme_vol_percentile,
                'crisis': self.crisis_vol_percentile
            }
        }
        
        # Cache results
        self._percentiles_cache = thresholds
        self._cache_timestamp = datetime.utcnow()
        
        return thresholds
    
    def classify_volatility(self, current_volatility: float) -> Tuple[str, dict]:
        """
        Classify current volatility level.
        
        Returns:
            (classification, details)
            
        Classifications:
        - 'NORMAL': Below high threshold
        - 'ELEVATED': Between high and extreme
        - 'EXTREME': Between extreme and crisis
        - 'CRISIS': Above crisis threshold
        """
        thresholds = self.get_adaptive_thresholds()
        
        if current_volatility < thresholds['high']:
            classification = 'NORMAL'
            multiplier = 1.0
        elif current_volatility < thresholds['extreme']:
            classification = 'ELEVATED'
            multiplier = 0.8  # Reduce position sizes by 20%
        elif current_volatility < thresholds['crisis']:
            classification = 'EXTREME'
            multiplier = 0.5  # Reduce position sizes by 50%
        else:
            classification = 'CRISIS'
            multiplier = 0.2  # Reduce position sizes by 80%
        
        percentile = self._calculate_percentile(current_volatility)
        
        return classification, {
            'classification': classification,
            'current_volatility': current_volatility,
            'percentile': percentile,
            'position_size_multiplier': multiplier,
            'thresholds': thresholds,
            'is_adaptive': thresholds['is_adaptive']
        }
    
    def _calculate_percentile(self, value: float) -> float:
        """Calculate percentile of value in historical distribution"""
        if len(self.volatility_history) < self.min_samples:
            return 50.0  # Default to median if insufficient data
        
        vols = np.array([v['volatility'] for v in self.volatility_history])
        percentile = stats.percentileofscore(vols, value)
        return percentile
    
    def get_stats(self) -> dict:
        """Get volatility statistics"""
        if len(self.volatility_history) == 0:
            return {
                'sample_size': 0,
                'has_sufficient_data': False
            }
        
        vols = np.array([v['volatility'] for v in self.volatility_history])
        latest = self.volatility_history[-1]
        
        return {
            'sample_size': len(self.volatility_history),
            'has_sufficient_data': len(self.volatility_history) >= self.min_samples,
            'current_volatility': latest['volatility'],
            'mean': np.mean(vols),
            'median': np.median(vols),
            'std': np.std(vols),
            'min': np.min(vols),
            'max': np.max(vols),
            'last_update': latest['timestamp'].isoformat()
        }


class StressAdjustedLimits:
    """
    Applies stress multipliers to risk limits during adverse conditions.
    
    Stress factors:
    - Volatility clustering (GARCH effects)
    - Correlation breakdowns
    - Liquidity shocks
    - Fat-tail events
    """
    
    def __init__(
        self,
        stress_lookback_days: int = 10,
        vol_clustering_threshold: float = 2.0,  # 2x recent vol = stress
        correlation_stress_threshold: float = 1.3  # 30% correlation increase
    ):
        self.stress_lookback_days = stress_lookback_days
        self.vol_clustering_threshold = vol_clustering_threshold
        self.correlation_stress_threshold = correlation_stress_threshold
        
        # Stress indicators
        self.is_stressed = False
        self.stress_score = 0.0  # 0.0 = normal, 1.0 = maximum stress
        self.stress_factors: Dict[str, float] = {}
    
    def calculate_stress_score(
        self,
        current_volatility: float,
        avg_volatility: float,
        correlation_inflation: float = 1.0,
        liquidity_score: float = 1.0  # 1.0 = normal, <1.0 = stressed
    ) -> float:
        """
        Calculate composite stress score.
        
        Returns:
            float in [0, 1] where 1 = maximum stress
        """
        stress_components = []
        
        # Volatility clustering stress
        vol_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        vol_stress = min(1.0, (vol_ratio - 1.0) / (self.vol_clustering_threshold - 1.0)) if vol_ratio > 1.0 else 0.0
        stress_components.append(('volatility_clustering', vol_stress))
        
        # Correlation stress
        corr_stress = min(1.0, (correlation_inflation - 1.0) / (self.correlation_stress_threshold - 1.0)) if correlation_inflation > 1.0 else 0.0
        stress_components.append(('correlation_stress', corr_stress))
        
        # Liquidity stress
        liquidity_stress = max(0.0, 1.0 - liquidity_score)
        stress_components.append(('liquidity_stress', liquidity_stress))
        
        # Composite stress score (max of all components)
        self.stress_score = max([s[1] for s in stress_components])
        self.stress_factors = dict(stress_components)
        self.is_stressed = self.stress_score > 0.3  # Stressed if score > 30%
        
        return self.stress_score
    
    def get_stress_adjusted_limits(
        self,
        base_limits: dict
    ) -> dict:
        """
        Apply stress adjustments to base risk limits.
        
        Args:
            base_limits: dict with 'risk_per_trade', 'max_position', etc.
        
        Returns:
            Stress-adjusted limits
        """
        if self.stress_score <= 0:
            return base_limits.copy()
        
        # Stress multiplier: reduces linearly with stress score
        # stress_score=0 → multiplier=1.0
        # stress_score=1 → multiplier=0.2 (80% reduction)
        multiplier = 1.0 - (self.stress_score * 0.8)
        
        adjusted = {}
        for key, value in base_limits.items():
            if isinstance(value, (int, float)):
                adjusted[key] = value * multiplier
            else:
                adjusted[key] = value
        
        adjusted['stress_multiplier'] = multiplier
        adjusted['stress_score'] = self.stress_score
        adjusted['is_stressed'] = self.is_stressed
        
        return adjusted
    
    def get_stress_report(self) -> dict:
        """Get detailed stress report"""
        return {
            'is_stressed': self.is_stressed,
            'stress_score': self.stress_score,
            'stress_factors': self.stress_factors,
            'stress_multiplier': 1.0 - (self.stress_score * 0.8)
        }


class AdaptiveRiskManager:
    """
    Orchestrates all adaptive risk management components.
    
    Integrates:
    - Regime-aware limits
    - Adaptive volatility thresholds
    - Stress adjustments
    """
    
    def __init__(
        self,
        regime_limits_config: Optional[dict] = None,
        adaptive_vol_config: Optional[dict] = None,
        stress_config: Optional[dict] = None
    ):
        self.regime_limits = RegimeAwareRiskLimits()
        self.adaptive_vol = AdaptiveVolatilityThresholds(**(adaptive_vol_config or {}))
        self.stress_adjuster = StressAdjustedLimits(**(stress_config or {}))
        
        # Current state
        self.current_regime = 'TRENDING'
    
    def update_regime(self, regime: str):
        """Update market regime"""
        self.current_regime = regime
        self.regime_limits.set_regime(regime)
    
    def get_current_risk_limits(
        self,
        current_volatility: float,
        correlation_inflation: float = 1.0,
        liquidity_score: float = 1.0
    ) -> dict:
        """
        Get comprehensive risk limits incorporating all adaptive components.
        
        Returns:
            Complete risk limit dict with regime, volatility, and stress adjustments
        """
        # Step 1: Get regime-based parameters
        regime_params = self.regime_limits.get_current_parameters()
        
        # Step 2: Classify volatility
        vol_classification, vol_details = self.adaptive_vol.classify_volatility(current_volatility)
        
        # Step 3: Calculate stress score
        avg_vol = self.adaptive_vol.get_stats().get('mean', current_volatility)
        stress_score = self.stress_adjuster.calculate_stress_score(
            current_volatility=current_volatility,
            avg_volatility=avg_vol,
            correlation_inflation=correlation_inflation,
            liquidity_score=liquidity_score
        )
        
        # Step 4: Build base limits from regime
        base_limits = {
            'risk_per_trade': regime_params.risk_per_trade,
            'max_position_size_multiplier': regime_params.max_position_size_multiplier,
            'max_gross_exposure': regime_params.max_gross_exposure,
            'max_leverage': regime_params.max_leverage
        }
        
        # Step 5: Apply volatility multiplier
        vol_multiplier = vol_details['position_size_multiplier']
        for key in ['risk_per_trade', 'max_position_size_multiplier']:
            base_limits[key] *= vol_multiplier
        
        # Step 6: Apply stress adjustments
        final_limits = self.stress_adjuster.get_stress_adjusted_limits(base_limits)
        
        # Step 7: Add metadata
        final_limits.update({
            'regime': self.current_regime,
            'regime_parameters': regime_params.to_dict(),
            'volatility_classification': vol_classification,
            'volatility_details': vol_details,
            'stress_report': self.stress_adjuster.get_stress_report(),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return final_limits
    
    def get_comprehensive_stats(self) -> dict:
        """Get complete statistics from all components"""
        return {
            'current_regime': self.current_regime,
            'regime_parameters': self.regime_limits.get_current_parameters().to_dict(),
            'volatility_stats': self.adaptive_vol.get_stats(),
            'volatility_thresholds': self.adaptive_vol.get_adaptive_thresholds(),
            'stress_report': self.stress_adjuster.get_stress_report()
        }
