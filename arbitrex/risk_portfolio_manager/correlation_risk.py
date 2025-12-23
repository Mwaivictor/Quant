"""
Correlation-Aware Sizing Module

Implements portfolio-level risk management considering cross-asset correlations.
Addresses the critical gap where univariate ATR-based sizing underestimates portfolio risk.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings

from .schemas import Position


class CorrelationMatrix:
    """
    Manages correlation matrix for portfolio risk calculations.
    
    Critical during stress events when correlations → +1, amplifying portfolio risk by 50-200%.
    """
    
    def __init__(self, default_correlation: float = 0.3):
        """
        Initialize correlation matrix.
        
        Args:
            default_correlation: Default correlation for unknown pairs
        """
        self.default_correlation = default_correlation
        self.correlations: Dict[Tuple[str, str], float] = {}
        self.last_updated: Dict[Tuple[str, str], datetime] = {}
        
        # Regime-dependent correlation adjustments
        self.regime_correlation_floor = {
            'TRENDING': 0.2,   # Lower correlations in trends
            'RANGING': 0.3,    # Moderate correlations
            'VOLATILE': 0.5,   # Elevated correlations
            'STRESSED': 0.8,   # CRITICAL: Correlations spike to 0.8-1.0 in crises
        }
    
    def get_correlation(
        self,
        symbol1: str,
        symbol2: str,
        regime: str = 'RANGING',
    ) -> float:
        """
        Get correlation between two symbols with regime adjustment.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            regime: Current market regime
        
        Returns:
            float: Correlation coefficient [-1, 1]
        """
        if symbol1 == symbol2:
            return 1.0
        
        # Normalize order (EURUSD, GBPUSD) same as (GBPUSD, EURUSD)
        key = tuple(sorted([symbol1, symbol2]))
        
        # Get stored correlation or default
        base_corr = self.correlations.get(key, self.default_correlation)
        
        # Apply regime floor - correlations increase in stress
        regime_floor = self.regime_correlation_floor.get(regime, 0.3)
        
        # Take maximum of base correlation and regime floor
        # This ensures we don't underestimate risk during crises
        adjusted_corr = max(base_corr, regime_floor)
        
        return adjusted_corr
    
    def update_correlation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float,
    ):
        """
        Update correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient
        """
        if not (-1.0 <= correlation <= 1.0):
            raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")
        
        key = tuple(sorted([symbol1, symbol2]))
        self.correlations[key] = correlation
        self.last_updated[key] = datetime.now()
    
    def build_correlation_matrix(
        self,
        symbols: List[str],
        regime: str = 'RANGING',
    ) -> np.ndarray:
        """
        Build full correlation matrix for portfolio.
        
        Args:
            symbols: List of symbols
            regime: Current market regime
        
        Returns:
            np.ndarray: N×N correlation matrix
        """
        n = len(symbols)
        corr_matrix = np.eye(n)  # Start with identity matrix
        
        for i in range(n):
            for j in range(i+1, n):
                corr = self.get_correlation(symbols[i], symbols[j], regime)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetric
        
        return corr_matrix
    
    def set_fx_pair_correlations(self):
        """
        Set realistic FX pair correlations.
        
        Based on historical data:
        - EURUSD vs GBPUSD: ~0.70 (both vs USD)
        - AUDUSD vs NZDUSD: ~0.85 (commodity currencies)
        - USDJPY vs EURJPY: ~0.75 (both vs JPY)
        - Cross-regional: ~0.3-0.5
        """
        fx_correlations = [
            ('EURUSD', 'GBPUSD', 0.70),
            ('EURUSD', 'AUDUSD', 0.55),
            ('EURUSD', 'NZDUSD', 0.50),
            ('EURUSD', 'USDJPY', -0.40),  # Negative correlation
            ('GBPUSD', 'AUDUSD', 0.60),
            ('GBPUSD', 'NZDUSD', 0.55),
            ('AUDUSD', 'NZDUSD', 0.85),  # High correlation (commodities)
            ('AUDUSD', 'USDJPY', -0.35),
            ('USDCAD', 'USDCHF', 0.65),
            ('EURUSD', 'USDCHF', -0.75),  # Strong negative
        ]
        
        for sym1, sym2, corr in fx_correlations:
            self.update_correlation(sym1, sym2, corr)


class PortfolioRiskCalculator:
    """
    Calculates portfolio-level risk metrics considering correlations.
    
    CRITICAL ENHANCEMENT: Addresses 50-200% risk underestimation during stress events.
    """
    
    def __init__(self, correlation_matrix: CorrelationMatrix):
        """
        Initialize portfolio risk calculator.
        
        Args:
            correlation_matrix: Correlation matrix instance
        """
        self.correlation_matrix = correlation_matrix
    
    def calculate_portfolio_volatility(
        self,
        positions: List[Position],
        symbol_volatilities: Dict[str, float],
        regime: str = 'RANGING',
    ) -> float:
        """
        Calculate portfolio volatility considering correlations.
        
        Formula: σ_portfolio = √(w'Σw)
        where:
        - w is weight vector (position sizes)
        - Σ is covariance matrix (correlations × volatilities)
        
        Args:
            positions: Current positions
            symbol_volatilities: Volatility (ATR/price) for each symbol
            regime: Current market regime
        
        Returns:
            float: Portfolio volatility (annualized)
        """
        if not positions:
            return 0.0
        
        # Get unique symbols
        symbols = list(set(p.symbol for p in positions))
        
        # Build weight vector (notional exposure / total capital)
        total_capital = sum(abs(p.units * p.entry_price) for p in positions)
        if total_capital == 0:
            return 0.0
        
        weights = []
        symbol_list = []
        for symbol in symbols:
            # Sum all positions in this symbol
            symbol_exposure = sum(
                p.units * p.entry_price * np.sign(p.units)
                for p in positions
                if p.symbol == symbol
            )
            weights.append(symbol_exposure / total_capital)
            symbol_list.append(symbol)
        
        weights = np.array(weights)
        
        # Get volatilities
        vols = np.array([symbol_volatilities.get(s, 0.01) for s in symbol_list])
        
        # Build correlation matrix
        corr_matrix = self.correlation_matrix.build_correlation_matrix(symbol_list, regime)
        
        # Build covariance matrix: Σ = D × C × D
        # where D is diagonal matrix of volatilities, C is correlation matrix
        vol_matrix = np.diag(vols)
        cov_matrix = vol_matrix @ corr_matrix @ vol_matrix
        
        # Calculate portfolio variance
        portfolio_var = weights.T @ cov_matrix @ weights
        
        # Convert to volatility (annualized - assuming daily data)
        portfolio_vol = np.sqrt(portfolio_var * 252)  # 252 trading days
        
        return float(portfolio_vol)
    
    def calculate_marginal_risk_contribution(
        self,
        current_positions: List[Position],
        new_position: Position,
        symbol_volatilities: Dict[str, float],
        regime: str = 'RANGING',
    ) -> float:
        """
        Calculate how much a new position increases portfolio risk.
        
        Marginal Risk Contribution = ∂(Portfolio Vol) / ∂(Position Size)
        
        Args:
            current_positions: Existing positions
            new_position: Proposed new position
            symbol_volatilities: Volatility for each symbol
            regime: Current market regime
        
        Returns:
            float: Marginal risk contribution (% increase in portfolio vol)
        """
        # Portfolio vol before new position
        vol_before = self.calculate_portfolio_volatility(
            current_positions,
            symbol_volatilities,
            regime
        )
        
        # Portfolio vol after new position
        vol_after = self.calculate_portfolio_volatility(
            current_positions + [new_position],
            symbol_volatilities,
            regime
        )
        
        # Marginal contribution
        if vol_before == 0:
            return vol_after
        
        marginal_contribution = (vol_after - vol_before) / vol_before
        
        return float(marginal_contribution)
    
    def calculate_diversification_benefit(
        self,
        positions: List[Position],
        symbol_volatilities: Dict[str, float],
        regime: str = 'RANGING',
    ) -> float:
        """
        Calculate diversification benefit of current portfolio.
        
        Diversification = 1 - (Portfolio Vol / Sum of Individual Vols)
        
        Perfect diversification = 1.0 (zero correlation)
        No diversification = 0.0 (perfect correlation)
        
        Args:
            positions: Current positions
            symbol_volatilities: Volatility for each symbol
            regime: Current market regime
        
        Returns:
            float: Diversification benefit [0, 1]
        """
        if not positions:
            return 1.0
        
        # Calculate portfolio volatility (with correlations)
        portfolio_vol = self.calculate_portfolio_volatility(
            positions,
            symbol_volatilities,
            regime
        )
        
        # Calculate sum of individual volatilities (no correlations)
        total_capital = sum(abs(p.units * p.entry_price) for p in positions)
        if total_capital == 0:
            return 1.0
        
        weighted_vol_sum = 0.0
        for position in positions:
            weight = abs(position.units * position.entry_price) / total_capital
            symbol_vol = symbol_volatilities.get(position.symbol, 0.01)
            weighted_vol_sum += weight * symbol_vol
        
        # Diversification benefit
        if weighted_vol_sum == 0:
            return 1.0
        
        diversification = 1.0 - (portfolio_vol / weighted_vol_sum)
        
        return float(np.clip(diversification, 0.0, 1.0))


class CorrelationAwareSizer:
    """
    Enhanced position sizer that adjusts for portfolio correlations.
    
    CRITICAL: Prevents catastrophic risk accumulation during stress events.
    """
    
    def __init__(
        self,
        correlation_matrix: CorrelationMatrix,
        risk_calculator: PortfolioRiskCalculator,
        max_portfolio_volatility: float = 0.15,  # 15% annualized
    ):
        """
        Initialize correlation-aware sizer.
        
        Args:
            correlation_matrix: Correlation matrix
            risk_calculator: Portfolio risk calculator
            max_portfolio_volatility: Maximum allowed portfolio volatility
        """
        self.correlation_matrix = correlation_matrix
        self.risk_calculator = risk_calculator
        self.max_portfolio_volatility = max_portfolio_volatility
    
    def calculate_correlation_adjustment(
        self,
        symbol: str,
        proposed_units: float,
        current_positions: List[Position],
        symbol_volatilities: Dict[str, float],
        regime: str = 'RANGING',
        entry_price: float = 1.0,
    ) -> Tuple[float, dict]:
        """
        Calculate position size adjustment based on portfolio correlations.
        
        Args:
            symbol: New position symbol
            proposed_units: Proposed position size (from ATR-based sizing)
            current_positions: Existing positions
            symbol_volatilities: Volatility for each symbol
            regime: Current market regime
            entry_price: Entry price for notional calculation
        
        Returns:
            Tuple[float, dict]: (adjusted_units, adjustment_details)
        """
        if not current_positions:
            # No portfolio yet, no correlation adjustment needed
            return proposed_units, {'adjustment_factor': 1.0, 'reason': 'empty_portfolio'}
        
        # Create mock position for proposed trade
        mock_position = Position(
            symbol=symbol,
            direction=1 if proposed_units > 0 else -1,
            entry_price=entry_price,
            units=abs(proposed_units),
            entry_time=datetime.now(),
        )
        
        # Calculate marginal risk contribution
        marginal_risk = self.risk_calculator.calculate_marginal_risk_contribution(
            current_positions,
            mock_position,
            symbol_volatilities,
            regime
        )
        
        # Calculate current diversification
        diversification = self.risk_calculator.calculate_diversification_benefit(
            current_positions,
            symbol_volatilities,
            regime
        )
        
        # Adjustment logic:
        # 1. If marginal risk is high (>10%), reduce position
        # 2. If diversification is low (<0.3), we're concentrated - reduce position
        # 3. In STRESSED regime with high correlations, be very conservative
        
        adjustment_factor = 1.0
        reason = "no_adjustment"
        
        # Check 1: High marginal risk
        if marginal_risk > 0.15:  # >15% increase in portfolio vol
            adjustment_factor = min(adjustment_factor, 0.5)  # Cut in half
            reason = "high_marginal_risk"
        elif marginal_risk > 0.10:  # >10% increase
            adjustment_factor = min(adjustment_factor, 0.7)  # Reduce by 30%
            reason = "elevated_marginal_risk"
        
        # Check 2: Low diversification
        if diversification < 0.3:  # Poor diversification
            adjustment_factor = min(adjustment_factor, 0.6)  # Reduce by 40%
            reason = "low_diversification"
        
        # Check 3: Stressed regime (CRITICAL)
        if regime == 'STRESSED':
            # In stressed markets, correlations → 1, portfolio risk explodes
            adjustment_factor = min(adjustment_factor, 0.4)  # Reduce to 40%
            reason = "stressed_regime_high_correlation"
        elif regime == 'VOLATILE':
            adjustment_factor = min(adjustment_factor, 0.7)  # Reduce to 70%
            reason = "volatile_regime_elevated_correlation"
        
        adjusted_units = proposed_units * adjustment_factor
        
        adjustment_details = {
            'adjustment_factor': float(adjustment_factor),
            'marginal_risk_contribution': float(marginal_risk),
            'portfolio_diversification': float(diversification),
            'regime': regime,
            'reason': reason,
            'original_units': float(proposed_units),
            'adjusted_units': float(adjusted_units),
        }
        
        return adjusted_units, adjustment_details
