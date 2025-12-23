"""
Kelly Criterion Position Sizing - Adaptive Enterprise Edition

Growth-optimal position sizing with conservative safety factors.

Mathematical foundation:
Kelly Fraction = (p*W - (1-p)*L) / W

Where:
- p = win rate (probability of profit)
- W = average win
- L = average loss

Enterprise modifications (v2.0.0):
1. Fractional Kelly (lambda * kelly_fraction) with lambda in [0.1, 0.3]
2. ADAPTIVE hard cap based on market regime:
   - TRENDING: 1.0% of capital (aggressive)
   - RANGING: 0.8% of capital (moderate)
   - VOLATILE: 0.5% of capital (conservative)
   - STRESSED: 0.2% of capital (defensive)
3. Reject if Kelly <= 0 (negative edge)

Integration: Works seamlessly with AdaptiveRiskManager for regime awareness.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KellyResult:
    """Result from Kelly Criterion calculation"""
    kelly_fraction: float
    fractional_kelly: float
    kelly_cap: float
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'kelly_fraction': float(self.kelly_fraction),
            'fractional_kelly': float(self.fractional_kelly),
            'kelly_cap': float(self.kelly_cap),
            'is_valid': bool(self.is_valid),
            'rejection_reason': self.rejection_reason,
        }


class KellyCriterion:
    """Kelly Criterion calculator with conservative institutional adjustments"""

    def __init__(
        self,
        safety_factor: float = 0.25,
        max_kelly_pct: float = 0.01,
        min_win_rate: float = 0.51,
        min_sample_size: int = 30,
        use_adaptive_cap: bool = True,
    ):
        # Validate safety factor
        if not 0.1 <= safety_factor <= 0.3:
            raise ValueError(f"safety_factor must be in [0.1, 0.3], got {safety_factor}")
        # Validate max Kelly
        if not 0.001 <= max_kelly_pct <= 0.05:
            raise ValueError(f"max_kelly_pct must be in [0.1%, 5%], got {max_kelly_pct}")

        self.safety_factor = float(safety_factor)
        self.max_kelly_pct = float(max_kelly_pct)
        self.min_win_rate = float(min_win_rate)
        self.min_sample_size = int(min_sample_size)
        self.use_adaptive_cap = use_adaptive_cap

        # Regime-based Kelly cap multipliers
        self.regime_multipliers = {
            'TRENDING': 1.0,
            'RANGING': 0.8,
            'VOLATILE': 0.5,
            'STRESSED': 0.2,
            'CRISIS': 0.1,
        }

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> KellyResult:
        """Calculate Kelly fraction with adaptive safety factor"""
        # Input validations
        if not 0 <= win_rate <= 1:
            return KellyResult(0, 0, 0, False, f"Invalid win_rate: {win_rate} (must be in [0,1])")
        if avg_win <= 0:
            return KellyResult(0, 0, 0, False, f"Invalid avg_win: {avg_win} (must be > 0)")
        if avg_loss <= 0:
            return KellyResult(0, 0, 0, False, f"Invalid avg_loss: {avg_loss} (must be > 0)")
        if num_trades is not None and num_trades < self.min_sample_size:
            return KellyResult(0, 0, 0, False, f"Insufficient sample size: {num_trades} trades (min: {self.min_sample_size})")
        if win_rate < self.min_win_rate:
            return KellyResult(0, 0, 0, False, f"Win rate {win_rate:.2%} below minimum {self.min_win_rate:.2%}")

        # Kelly formula
        kelly_fraction = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win
        if kelly_fraction <= 0:
            return KellyResult(kelly_fraction, 0, 0, False, f"Negative edge: Kelly = {kelly_fraction:.4f}")

        fractional_kelly = self.safety_factor * kelly_fraction

        # Apply adaptive cap
        regime_upper = regime.upper() if regime else None
        if self.use_adaptive_cap and regime_upper in self.regime_multipliers:
            regime_adjusted_max = self.max_kelly_pct * self.regime_multipliers[regime_upper]
        else:
            regime_adjusted_max = self.max_kelly_pct

        kelly_cap = min(fractional_kelly, regime_adjusted_max)

        return KellyResult(
            kelly_fraction=kelly_fraction,
            fractional_kelly=fractional_kelly,
            kelly_cap=kelly_cap,
            is_valid=True,
        )

    def calculate_from_stats(
        self,
        total_trades: int,
        winning_trades: int,
        total_profit: float,
        total_loss: float,
    ) -> KellyResult:
        """Calculate Kelly from aggregate trading statistics"""
        if total_trades <= 0:
            return KellyResult(0, 0, 0, False, "No trades in history")
        if winning_trades <= 0:
            return KellyResult(0, 0, 0, False, "No winning trades in history")

        win_rate = winning_trades / total_trades
        losing_trades = total_trades - winning_trades
        avg_win = total_profit / winning_trades if total_profit > 0 else 0.01
        avg_loss = total_loss / losing_trades if losing_trades > 0 and total_loss > 0 else avg_win * 0.5

        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=total_trades,
        )

    def get_recommended_units(
        self,
        total_capital: float,
        current_price: float,
        kelly_result: KellyResult,
    ) -> float:
        """Convert Kelly % to position units"""
        if not kelly_result.is_valid or current_price <= 0:
            return 0.0
        max_position_value = total_capital * kelly_result.kelly_cap
        max_units = max_position_value / current_price
        return float(max_units)
