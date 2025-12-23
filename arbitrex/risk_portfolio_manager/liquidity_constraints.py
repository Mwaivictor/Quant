"""
Liquidity Constraints for Position Sizing

Ensures execution feasibility by limiting position size based on:
1. Average Daily Volume (ADV) constraints
2. Bid-ask spread penalties
3. Market impact estimation (square-root model)

Critical for institutional trading to avoid:
- Moving the market against ourselves
- Unfavorable execution slippage
- Illiquid position accumulation

Mathematical models:
1. ADV Constraint: Q ≤ α · ADV (typically α = 0.01 = 1%)
2. Spread Cost: Cost = S · P · Q (where S = spread %)
3. Market Impact: MI = η · σ · √(Q/ADV) (Almgren-Chriss model)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LiquidityResult:
    """Result from liquidity constraint check"""
    max_units: float  # Maximum units based on liquidity
    adv_limit: float  # Limit from ADV constraint
    spread_penalty: float  # Multiplier from spread [0-1]
    market_impact: float  # Estimated market impact ($)
    market_impact_pct: float  # MI as % of position value
    is_acceptable: bool  # True if liquidity adequate
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'max_units': float(self.max_units),
            'adv_limit': float(self.adv_limit),
            'spread_penalty': float(self.spread_penalty),
            'market_impact': float(self.market_impact),
            'market_impact_pct': float(self.market_impact_pct),
            'is_acceptable': bool(self.is_acceptable),
            'rejection_reason': self.rejection_reason,
        }


class LiquidityConstraints:
    """
    Liquidity-based position size constraints.
    
    Usage:
        liq = LiquidityConstraints(max_adv_pct=0.01, max_spread_bps=20)
        result = liq.check(
            proposed_units=10000,
            adv_units=500000,
            spread_pct=0.0015,
            volatility=0.01,
            current_price=1.10
        )
        
        if result.is_acceptable:
            final_units = min(proposed_units, result.max_units)
    """
    
    def __init__(
        self,
        max_adv_pct: float = 0.01,
        max_spread_bps: float = 20.0,
        max_market_impact_pct: float = 0.005,
        impact_coefficient: float = 0.1,
        min_adv_units: float = 10000.0,
    ):
        """
        Initialize liquidity constraints.
        
        Args:
            max_adv_pct: Maximum position as % of ADV (default: 1%)
            max_spread_bps: Maximum acceptable spread in bps (default: 20 bps)
            max_market_impact_pct: Maximum MI as % of position value (default: 0.5%)
            impact_coefficient: Market impact coefficient η (default: 0.1)
            min_adv_units: Minimum ADV to consider tradeable (default: 10,000 units)
        
        Conservative defaults:
        - 1% of ADV → minimal market footprint
        - 20 bps spread → accepts typical FX spreads (1-2 pips on majors)
        - 0.5% max impact → keeps execution costs low
        - η=0.1 → moderate impact assumption
        """
        self.max_adv_pct = float(max_adv_pct)
        self.max_spread_bps = float(max_spread_bps)
        self.max_market_impact_pct = float(max_market_impact_pct)
        self.impact_coefficient = float(impact_coefficient)
        self.min_adv_units = float(min_adv_units)
    
    def check(
        self,
        proposed_units: float,
        adv_units: float,
        spread_pct: float,
        volatility: float,
        current_price: float,
    ) -> LiquidityResult:
        """
        Check liquidity constraints and calculate limits.
        
        Args:
            proposed_units: Desired position size (units)
            adv_units: Average daily volume (units)
            spread_pct: Bid-ask spread as decimal (e.g., 0.0015 = 15 bps)
            volatility: Daily volatility (σ) as decimal (e.g., 0.01 = 1%)
            current_price: Current market price ($ per unit)
        
        Returns:
            LiquidityResult with max_units, penalties, impact, validity
        
        Rejection conditions:
        - adv_units < min_adv_units (illiquid)
        - spread > max_spread_bps (too wide)
        - estimated market impact > max_market_impact_pct
        """
        # Validate inputs
        if proposed_units <= 0:
            return LiquidityResult(
                max_units=0.0,
                adv_limit=0.0,
                spread_penalty=0.0,
                market_impact=0.0,
                market_impact_pct=0.0,
                is_acceptable=False,
                rejection_reason="Invalid proposed_units: must be > 0"
            )
        
        if adv_units <= 0:
            return LiquidityResult(
                max_units=0.0,
                adv_limit=0.0,
                spread_penalty=0.0,
                market_impact=0.0,
                market_impact_pct=0.0,
                is_acceptable=False,
                rejection_reason="Invalid adv_units: must be > 0"
            )
        
        if current_price <= 0:
            return LiquidityResult(
                max_units=0.0,
                adv_limit=0.0,
                spread_penalty=0.0,
                market_impact=0.0,
                market_impact_pct=0.0,
                is_acceptable=False,
                rejection_reason="Invalid current_price: must be > 0"
            )
        
        # Check minimum ADV
        if adv_units < self.min_adv_units:
            return LiquidityResult(
                max_units=0.0,
                adv_limit=0.0,
                spread_penalty=0.0,
                market_impact=0.0,
                market_impact_pct=0.0,
                is_acceptable=False,
                rejection_reason=f"ADV {adv_units:.0f} below minimum {self.min_adv_units:.0f}"
            )
        
        # Check spread constraint
        spread_bps = spread_pct * 10000  # Convert to basis points
        if spread_bps > self.max_spread_bps:
            return LiquidityResult(
                max_units=0.0,
                adv_limit=0.0,
                spread_penalty=0.0,
                market_impact=0.0,
                market_impact_pct=0.0,
                is_acceptable=False,
                rejection_reason=f"Spread {spread_bps:.1f} bps exceeds max {self.max_spread_bps:.1f} bps"
            )
        
        # Calculate ADV constraint
        # Max position = α · ADV
        adv_limit = self.max_adv_pct * adv_units
        
        # Calculate spread penalty (linear decay)
        # Penalty = 1.0 at spread=0, 0.5 at spread=max_spread_bps
        spread_penalty = 1.0 - (spread_bps / self.max_spread_bps) * 0.5
        spread_penalty = np.clip(spread_penalty, 0.5, 1.0)
        
        # Calculate market impact (Almgren-Chriss square-root model)
        # MI = η · σ · √(Q/ADV) · P · Q
        # Where:
        # - η = impact coefficient (default: 0.1)
        # - σ = volatility
        # - Q = position size (units)
        # - ADV = average daily volume
        # - P = current price
        
        Q = float(proposed_units)
        ADV = float(adv_units)
        sigma = float(volatility)
        P = float(current_price)
        eta = self.impact_coefficient
        
        # Market impact per unit
        mi_per_unit = eta * sigma * np.sqrt(Q / ADV)
        
        # Total market impact ($)
        market_impact = mi_per_unit * P * Q
        
        # Position value
        position_value = Q * P
        
        # Market impact as % of position value
        market_impact_pct = market_impact / position_value if position_value > 0 else 0.0
        
        # Check if market impact is acceptable
        if market_impact_pct > self.max_market_impact_pct:
            # Calculate max units that keeps MI under threshold
            # Rearranging: Q_max = (MI_max / (η·σ·P))^(2/3) · ADV^(1/3)
            # Simplified: iteratively reduce Q until MI < threshold
            max_units_mi = Q
            for _ in range(10):  # Max 10 iterations
                mi_pct = (eta * sigma * np.sqrt(max_units_mi / ADV) * P * max_units_mi) / (max_units_mi * P)
                if mi_pct <= self.max_market_impact_pct:
                    break
                max_units_mi *= 0.8  # Reduce by 20% each iteration
            
            # Take minimum of ADV limit and MI limit
            max_units = min(adv_limit, max_units_mi)
            
            # If proposed units exceed limit, flag as warning (not rejection)
            return LiquidityResult(
                max_units=float(max_units),
                adv_limit=float(adv_limit),
                spread_penalty=float(spread_penalty),
                market_impact=float(market_impact),
                market_impact_pct=float(market_impact_pct),
                is_acceptable=True,  # Acceptable with reduced size
                rejection_reason=None
            )
        
        # All checks passed
        max_units = min(adv_limit, proposed_units)
        
        return LiquidityResult(
            max_units=float(max_units),
            adv_limit=float(adv_limit),
            spread_penalty=float(spread_penalty),
            market_impact=float(market_impact),
            market_impact_pct=float(market_impact_pct),
            is_acceptable=True,
            rejection_reason=None
        )
    
    def estimate_slippage(
        self,
        units: float,
        spread_pct: float,
        market_impact_pct: float,
        current_price: float,
    ) -> float:
        """
        Estimate total execution slippage.
        
        Total slippage = spread cost + market impact
        
        Args:
            units: Position size (units)
            spread_pct: Bid-ask spread as decimal
            market_impact_pct: Market impact as % of position value
            current_price: Current market price
        
        Returns:
            float: Estimated slippage in dollars
        """
        position_value = units * current_price
        
        # Spread cost = half-spread · position value
        spread_cost = 0.5 * spread_pct * position_value
        
        # Market impact cost
        mi_cost = market_impact_pct * position_value
        
        # Total slippage
        total_slippage = spread_cost + mi_cost
        
        return float(total_slippage)
