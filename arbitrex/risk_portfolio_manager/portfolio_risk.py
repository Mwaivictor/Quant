"""
Portfolio-Level Risk Mathematics - Institutional Grade

Implements sophisticated portfolio risk models:
1. Portfolio variance and volatility targeting (σp = √(w'Σw))
2. Value-at-Risk (VaR) and Expected Shortfall (CVaR)
3. Fat-tail and non-normal distribution modeling
4. Historical simulation and stress correlation

Replaces correlation penalties with proper multivariate risk.

Version: 2.0.0 (Enterprise)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta
from collections import deque


@dataclass
class PortfolioRiskMetrics:
    """Complete portfolio risk metrics"""
    portfolio_variance: float
    portfolio_volatility: float
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_cvar_95: float
    portfolio_cvar_99: float
    positions_correlation: Dict[str, Dict[str, float]]
    target_volatility: float
    volatility_utilization: float  # Current / Target
    breaches_target: bool
    fat_tail_alpha: float  # Student-t degrees of freedom (if fitted)
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'portfolio_variance': self.portfolio_variance,
            'portfolio_volatility': self.portfolio_volatility,
            'portfolio_var_95': self.portfolio_var_95,
            'portfolio_var_99': self.portfolio_var_99,
            'portfolio_cvar_95': self.portfolio_cvar_95,
            'portfolio_cvar_99': self.portfolio_cvar_99,
            'target_volatility': self.target_volatility,
            'volatility_utilization': self.volatility_utilization,
            'breaches_target': self.breaches_target,
            'fat_tail_alpha': self.fat_tail_alpha,
            'timestamp': self.timestamp.isoformat()
        }


class CovarianceMatrixEstimator:
    """
    Estimates rolling covariance matrix for portfolio assets.
    
    Methods:
    - Sample covariance (Ledoit-Wolf shrinkage)
    - Exponentially weighted moving average (EWMA)
    - Stress-adjusted (correlation inflation)
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        min_observations: int = 30,
        ewma_lambda: float = 0.94,
        shrinkage_intensity: float = 0.1
    ):
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.ewma_lambda = ewma_lambda
        self.shrinkage_intensity = shrinkage_intensity
        
        # Historical returns storage: {symbol: deque of returns}
        self.returns_history: Dict[str, deque] = {}
        
        # Cached covariance matrix
        self._cov_matrix_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def record_return(self, symbol: str, return_value: float, timestamp: Optional[datetime] = None):
        """Record asset return"""
        if symbol not in self.returns_history:
            self.returns_history[symbol] = deque(maxlen=self.lookback_days)
        
        self.returns_history[symbol].append({
            'return': return_value,
            'timestamp': timestamp or datetime.utcnow()
        })
        
        # Invalidate cache
        self._cov_matrix_cache = None
    
    def estimate_covariance_matrix(
        self,
        symbols: List[str],
        method: str = 'sample',
        stress_correlation_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix for given symbols.
        
        Args:
            symbols: List of asset symbols
            method: 'sample', 'ewma', or 'stressed'
            stress_correlation_factor: Multiply off-diagonals by this factor
        
        Returns:
            Covariance matrix as DataFrame
        """
        # Check cache (only for non-stressed, sample method)
        if method == 'sample' and stress_correlation_factor == 1.0:
            if self._cov_matrix_cache is not None:
                cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds() / 3600
                if cache_age < 1.0:  # Cache valid for 1 hour
                    cached_symbols = set(self._cov_matrix_cache.index)
                    if set(symbols).issubset(cached_symbols):
                        return self._cov_matrix_cache.loc[symbols, symbols]
        
        # Build returns DataFrame
        returns_data = {}
        for symbol in symbols:
            if symbol not in self.returns_history:
                continue
            returns = [r['return'] for r in self.returns_history[symbol]]
            if len(returns) >= self.min_observations:
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            # Insufficient data - return identity matrix scaled by typical volatility
            return pd.DataFrame(
                np.eye(len(symbols)) * 0.01**2,  # 1% daily vol
                index=symbols,
                columns=symbols
            )
        
        # Align lengths (take minimum)
        min_length = min(len(v) for v in returns_data.values())
        for symbol in returns_data:
            returns_data[symbol] = returns_data[symbol][-min_length:]
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate covariance matrix based on method
        if method == 'sample':
            cov_matrix = returns_df.cov()
        elif method == 'ewma':
            cov_matrix = self._ewma_covariance(returns_df)
        elif method == 'stressed':
            cov_matrix = returns_df.cov()
            # Inflate correlations (stress scenario)
            cov_matrix = self._stress_covariance(cov_matrix, stress_correlation_factor)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply Ledoit-Wolf shrinkage to sample covariance
        if method == 'sample' and self.shrinkage_intensity > 0:
            cov_matrix = self._apply_shrinkage(cov_matrix)
        
        # Cache result (only for sample, non-stressed)
        if method == 'sample' and stress_correlation_factor == 1.0:
            self._cov_matrix_cache = cov_matrix
            self._cache_timestamp = datetime.utcnow()
        
        # Ensure all requested symbols are present (fill with zeros if missing)
        for symbol in symbols:
            if symbol not in cov_matrix.index:
                # Add symbol with zero covariance
                cov_matrix.loc[symbol, :] = 0.0
                cov_matrix.loc[:, symbol] = 0.0
                cov_matrix.loc[symbol, symbol] = 0.01**2  # Default variance
        
        return cov_matrix.loc[symbols, symbols]
    
    def _ewma_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EWMA covariance matrix"""
        n = len(returns_df)
        weights = np.array([(1 - self.ewma_lambda) * self.ewma_lambda**i for i in range(n)])
        weights = weights[::-1] / weights.sum()
        
        # Weighted covariance
        returns_centered = returns_df - returns_df.mean()
        cov_matrix = returns_centered.T @ np.diag(weights) @ returns_centered
        
        return cov_matrix
    
    def _stress_covariance(self, cov_matrix: pd.DataFrame, stress_factor: float) -> pd.DataFrame:
        """Apply stress to correlations (inflate off-diagonals)"""
        # Extract correlation matrix
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix.values / np.outer(std, std)
        
        # Stress correlations: move towards 1.0
        np.fill_diagonal(corr_matrix, 1.0)  # Preserve diagonal
        corr_matrix = corr_matrix * stress_factor
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Reconstruct covariance
        stressed_cov = corr_matrix * np.outer(std, std)
        
        return pd.DataFrame(stressed_cov, index=cov_matrix.index, columns=cov_matrix.columns)
    
    def _apply_shrinkage(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Apply Ledoit-Wolf shrinkage"""
        # Shrink towards identity matrix scaled by average variance
        target = np.eye(len(cov_matrix)) * np.trace(cov_matrix) / len(cov_matrix)
        shrunk = (1 - self.shrinkage_intensity) * cov_matrix + self.shrinkage_intensity * target
        return pd.DataFrame(shrunk, index=cov_matrix.index, columns=cov_matrix.columns)
    
    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Get correlation matrix"""
        cov_matrix = self.estimate_covariance_matrix(symbols)
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix.values / np.outer(std, std)
        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)


class PortfolioVarianceCalculator:
    """
    Calculates portfolio variance: σp² = w'Σw
    
    Enforces target portfolio volatility through position scaling.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        max_volatility: float = 0.25,  # 25% annualized (hard cap)
        rebalance_threshold: float = 0.10  # Rebalance if 10% over target
    ):
        self.target_volatility = target_volatility
        self.max_volatility = max_volatility
        self.rebalance_threshold = rebalance_threshold
        
        self.cov_estimator = CovarianceMatrixEstimator()
    
    def calculate_portfolio_variance(
        self,
        positions: Dict[str, float],  # {symbol: units}
        prices: Dict[str, float],  # {symbol: current_price}
        total_capital: float,
        stress_correlation_factor: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate portfolio variance and volatility.
        
        Returns:
            (variance, volatility) as decimals (e.g., 0.15 = 15%)
        """
        if not positions:
            return 0.0, 0.0
        
        symbols = list(positions.keys())
        
        # Calculate weights
        position_values = {s: positions[s] * prices[s] for s in symbols}
        total_exposure = sum(abs(v) for v in position_values.values())
        
        if total_exposure == 0:
            return 0.0, 0.0
        
        weights = np.array([position_values[s] / total_capital for s in symbols])
        
        # Get covariance matrix
        cov_matrix = self.cov_estimator.estimate_covariance_matrix(
            symbols,
            method='stressed' if stress_correlation_factor > 1.0 else 'sample',
            stress_correlation_factor=stress_correlation_factor
        )
        
        # Portfolio variance: w'Σw
        variance = weights @ cov_matrix.values @ weights
        volatility = np.sqrt(variance)
        
        # Annualize (assuming daily returns)
        volatility_annualized = volatility * np.sqrt(252)
        
        return variance, volatility_annualized
    
    def get_position_scaling_factor(
        self,
        current_volatility: float
    ) -> float:
        """
        Calculate scaling factor to bring portfolio vol to target.
        
        Returns:
            Multiplier for all positions (e.g., 0.8 = scale down by 20%)
        """
        if current_volatility <= self.target_volatility:
            return 1.0  # No scaling needed
        
        if current_volatility > self.max_volatility:
            # Emergency: scale to max
            return self.max_volatility / current_volatility
        
        # Check if rebalancing needed
        overshoot = (current_volatility - self.target_volatility) / self.target_volatility
        
        if overshoot > self.rebalance_threshold:
            # Scale down to target
            return self.target_volatility / current_volatility
        
        # Within threshold - no action
        return 1.0


class VaRCalculator:
    """
    Calculate Value-at-Risk (VaR) and Expected Shortfall (CVaR).
    
    Methods:
    - Parametric VaR (assumes normal or Student-t)
    - Historical simulation VaR
    - CVaR (expected loss beyond VaR)
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        lookback_days: int = 252,
        use_fat_tails: bool = True
    ):
        self.confidence_levels = confidence_levels
        self.lookback_days = lookback_days
        self.use_fat_tails = use_fat_tails
        
        # Historical returns for simulation
        self.portfolio_returns: deque = deque(maxlen=lookback_days)
        
        # Fitted distribution parameters
        self.fitted_distribution: Optional[dict] = None
    
    def record_portfolio_return(self, return_value: float, timestamp: Optional[datetime] = None):
        """Record portfolio return"""
        self.portfolio_returns.append({
            'return': return_value,
            'timestamp': timestamp or datetime.utcnow()
        })
        
        # Re-fit distribution if enough data
        if len(self.portfolio_returns) >= 30:
            self._fit_distribution()
    
    def _fit_distribution(self):
        """Fit Student-t distribution to returns (captures fat tails)"""
        returns = np.array([r['return'] for r in self.portfolio_returns])
        
        if self.use_fat_tails:
            # Fit Student-t distribution
            try:
                df, loc, scale = stats.t.fit(returns)
                self.fitted_distribution = {
                    'type': 'student_t',
                    'df': df,  # Degrees of freedom (lower = fatter tails)
                    'loc': loc,  # Location (mean)
                    'scale': scale  # Scale (like std dev)
                }
            except:
                # Fallback to normal
                self.fitted_distribution = {
                    'type': 'normal',
                    'mean': np.mean(returns),
                    'std': np.std(returns)
                }
        else:
            # Normal distribution
            self.fitted_distribution = {
                'type': 'normal',
                'mean': np.mean(returns),
                'std': np.std(returns)
            }
    
    def calculate_var(
        self,
        portfolio_value: float,
        confidence_level: float = 0.95,
        method: str = 'parametric',
        horizon_days: int = 1
    ) -> float:
        """
        Calculate Value-at-Risk.
        
        Args:
            portfolio_value: Current portfolio value ($)
            confidence_level: Confidence level (e.g., 0.95 = 95%)
            method: 'parametric' or 'historical'
            horizon_days: Risk horizon in days
        
        Returns:
            VaR in dollars (positive = loss)
        """
        if len(self.portfolio_returns) < 30:
            # Insufficient data - return conservative estimate
            return portfolio_value * 0.02 * np.sqrt(horizon_days)  # 2% daily VaR
        
        if method == 'parametric':
            return self._parametric_var(portfolio_value, confidence_level, horizon_days)
        elif method == 'historical':
            return self._historical_var(portfolio_value, confidence_level, horizon_days)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _parametric_var(
        self,
        portfolio_value: float,
        confidence_level: float,
        horizon_days: int
    ) -> float:
        """Parametric VaR using fitted distribution"""
        if not self.fitted_distribution:
            self._fit_distribution()
        
        dist = self.fitted_distribution
        
        if dist['type'] == 'student_t':
            # Student-t VaR (captures fat tails)
            quantile = stats.t.ppf(1 - confidence_level, df=dist['df'], loc=dist['loc'], scale=dist['scale'])
        else:
            # Normal VaR
            quantile = stats.norm.ppf(1 - confidence_level, loc=dist['mean'], scale=dist['std'])
        
        # Scale to horizon
        quantile_scaled = quantile * np.sqrt(horizon_days)
        
        # Convert to dollar VaR (loss is negative return)
        var = -quantile_scaled * portfolio_value
        
        return var
    
    def _historical_var(
        self,
        portfolio_value: float,
        confidence_level: float,
        horizon_days: int
    ) -> float:
        """Historical simulation VaR"""
        returns = np.array([r['return'] for r in self.portfolio_returns])
        
        # Scale to horizon (simple scaling - assumes IID)
        returns_scaled = returns * np.sqrt(horizon_days)
        
        # Get quantile
        quantile = np.percentile(returns_scaled, (1 - confidence_level) * 100)
        
        # Convert to dollar VaR
        var = -quantile * portfolio_value
        
        return var
    
    def calculate_cvar(
        self,
        portfolio_value: float,
        confidence_level: float = 0.95,
        method: str = 'historical',
        horizon_days: int = 1
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        CVaR = expected loss given that loss exceeds VaR.
        
        Returns:
            CVaR in dollars (positive = loss)
        """
        if len(self.portfolio_returns) < 30:
            return portfolio_value * 0.03 * np.sqrt(horizon_days)  # Conservative
        
        returns = np.array([r['return'] for r in self.portfolio_returns])
        returns_scaled = returns * np.sqrt(horizon_days)
        
        # Get VaR threshold
        var_threshold = np.percentile(returns_scaled, (1 - confidence_level) * 100)
        
        # CVaR = mean of returns below VaR threshold
        tail_returns = returns_scaled[returns_scaled <= var_threshold]
        
        if len(tail_returns) == 0:
            # No tail events - return VaR as approximation
            return self.calculate_var(portfolio_value, confidence_level, method, horizon_days)
        
        cvar_return = np.mean(tail_returns)
        cvar = -cvar_return * portfolio_value
        
        return cvar
    
    def get_tail_statistics(self) -> dict:
        """Get fat-tail statistics"""
        if not self.fitted_distribution or len(self.portfolio_returns) < 30:
            return {
                'has_data': False
            }
        
        returns = np.array([r['return'] for r in self.portfolio_returns])
        
        # Kurtosis (fat tails if > 3)
        kurtosis = stats.kurtosis(returns, fisher=False)
        
        # Skewness
        skewness = stats.skew(returns)
        
        dist = self.fitted_distribution
        
        return {
            'has_data': True,
            'sample_size': len(self.portfolio_returns),
            'distribution_type': dist['type'],
            'degrees_of_freedom': dist.get('df', None),
            'kurtosis': kurtosis,
            'skewness': skewness,
            'has_fat_tails': kurtosis > 4.0,  # Excess kurtosis
            'is_left_skewed': skewness < -0.5  # Negative tail risk
        }


class PortfolioRiskEngine:
    """
    Orchestrates all portfolio risk calculations.
    
    Provides unified interface for:
    - Portfolio variance/volatility targeting
    - VaR/CVaR constraints
    - Fat-tail modeling
    - Position scaling decisions
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        var_limit_pct: float = 0.02,  # 2% VaR limit
        cvar_limit_pct: float = 0.03,  # 3% CVaR limit
        confidence_level: float = 0.95
    ):
        self.target_volatility = target_volatility
        self.var_limit_pct = var_limit_pct
        self.cvar_limit_pct = cvar_limit_pct
        self.confidence_level = confidence_level
        
        self.variance_calculator = PortfolioVarianceCalculator(target_volatility=target_volatility)
        self.var_calculator = VaRCalculator()
    
    def calculate_comprehensive_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        total_capital: float,
        stress_correlation_factor: float = 1.0
    ) -> PortfolioRiskMetrics:
        """
        Calculate complete portfolio risk metrics.
        
        Returns:
            PortfolioRiskMetrics with all risk measures
        """
        # Portfolio variance/volatility
        variance, volatility = self.variance_calculator.calculate_portfolio_variance(
            positions, prices, total_capital, stress_correlation_factor
        )
        
        # Portfolio value
        portfolio_value = sum(positions[s] * prices[s] for s in positions)
        
        # VaR and CVaR
        var_95 = self.var_calculator.calculate_var(portfolio_value, 0.95, method='parametric')
        var_99 = self.var_calculator.calculate_var(portfolio_value, 0.99, method='parametric')
        cvar_95 = self.var_calculator.calculate_cvar(portfolio_value, 0.95)
        cvar_99 = self.var_calculator.calculate_cvar(portfolio_value, 0.99)
        
        # Correlation matrix
        symbols = list(positions.keys())
        corr_matrix = self.variance_calculator.cov_estimator.get_correlation_matrix(symbols)
        correlations = {
            s1: {s2: corr_matrix.loc[s1, s2] for s2 in symbols}
            for s1 in symbols
        }
        
        # Fat-tail statistics
        tail_stats = self.var_calculator.get_tail_statistics()
        fat_tail_alpha = tail_stats.get('degrees_of_freedom', np.inf)
        
        # Check breaches
        volatility_utilization = volatility / self.target_volatility if self.target_volatility > 0 else 0.0
        breaches_target = volatility > self.target_volatility
        
        return PortfolioRiskMetrics(
            portfolio_variance=variance,
            portfolio_volatility=volatility,
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            portfolio_cvar_95=cvar_95,
            portfolio_cvar_99=cvar_99,
            positions_correlation=correlations,
            target_volatility=self.target_volatility,
            volatility_utilization=volatility_utilization,
            breaches_target=breaches_target,
            fat_tail_alpha=fat_tail_alpha,
            timestamp=datetime.utcnow()
        )
    
    def check_risk_limits(
        self,
        risk_metrics: PortfolioRiskMetrics,
        total_capital: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if portfolio breaches risk limits.
        
        Returns:
            (is_acceptable, rejection_reason)
        """
        # Check volatility target
        if risk_metrics.breaches_target:
            if risk_metrics.volatility_utilization > 1.5:  # 50% over target
                return False, f"Portfolio volatility {risk_metrics.portfolio_volatility*100:.1f}% exceeds target {risk_metrics.target_volatility*100:.1f}% by >50%"
        
        # Check VaR limit
        var_limit = total_capital * self.var_limit_pct
        if risk_metrics.portfolio_var_95 > var_limit:
            return False, f"Portfolio VaR ${risk_metrics.portfolio_var_95:,.0f} exceeds limit ${var_limit:,.0f}"
        
        # Check CVaR limit
        cvar_limit = total_capital * self.cvar_limit_pct
        if risk_metrics.portfolio_cvar_95 > cvar_limit:
            return False, f"Portfolio CVaR ${risk_metrics.portfolio_cvar_95:,.0f} exceeds limit ${cvar_limit:,.0f}"
        
        return True, None
    
    def get_position_scaling_recommendation(
        self,
        risk_metrics: PortfolioRiskMetrics
    ) -> Tuple[float, str]:
        """
        Recommend position scaling factor based on risk metrics.
        
        Returns:
            (scaling_factor, reason)
        """
        scaling_factor = self.variance_calculator.get_position_scaling_factor(
            risk_metrics.portfolio_volatility
        )
        
        if scaling_factor < 1.0:
            reason = f"Scale down {(1-scaling_factor)*100:.1f}% to meet volatility target"
        else:
            reason = "No scaling needed - within risk limits"
        
        return scaling_factor, reason
