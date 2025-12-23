"""
Expectancy-Based Position Sizing

Mathematical expectancy (expected value per trade) used to scale position sizes.

Formula:
E = p·W - (1-p)·L

Where:
- E = expectancy (expected return per trade)
- p = win rate (probability of profit)
- W = average win
- L = average loss

Trading system interpretation:
- E > 0: Positive edge → scale up position
- E = 0: Breakeven → reject trade
- E < 0: Negative edge → reject trade

Position scaling:
- High expectancy (E > 0.02) → 1.5× multiplier
- Medium expectancy (0.01 < E ≤ 0.02) → 1.0× multiplier
- Low expectancy (0 < E ≤ 0.01) → 0.5× multiplier
- Negative/zero expectancy → REJECT
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExpectancyResult:
    """Result from expectancy calculation"""
    expectancy: float  # E = p·W - (1-p)·L
    expectancy_multiplier: float  # Position size adjustment [0.5, 1.5]
    is_valid: bool  # True if E > 0
    rejection_reason: Optional[str] = None
    
    # Breakdown
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # (p·W) / ((1-p)·L)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'expectancy': float(self.expectancy),
            'expectancy_multiplier': float(self.expectancy_multiplier),
            'is_valid': bool(self.is_valid),
            'rejection_reason': self.rejection_reason,
            'win_rate': float(self.win_rate),
            'avg_win': float(self.avg_win),
            'avg_loss': float(self.avg_loss),
            'profit_factor': float(self.profit_factor),
        }


class ExpectancyCalculator:
    """
    Calculate trading expectancy and position size adjustment.
    
    Usage:
        exp_calc = ExpectancyCalculator()
        result = exp_calc.calculate(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
        
        if result.is_valid:
            position_multiplier = result.expectancy_multiplier  # Apply to base size
    """
    
    def __init__(
        self,
        min_expectancy: float = 0.001,
        high_expectancy_threshold: float = 0.02,
        medium_expectancy_threshold: float = 0.01,
        high_multiplier: float = 1.5,
        medium_multiplier: float = 1.0,
        low_multiplier: float = 0.5,
        min_sample_size: int = 30,
    ):
        """
        Initialize expectancy calculator.
        
        Args:
            min_expectancy: Minimum expectancy to accept trade (default: 0.1%)
            high_expectancy_threshold: Expectancy for high multiplier (default: 2%)
            medium_expectancy_threshold: Expectancy for medium multiplier (default: 1%)
            high_multiplier: Multiplier for high expectancy (default: 1.5×)
            medium_multiplier: Multiplier for medium expectancy (default: 1.0×)
            low_multiplier: Multiplier for low expectancy (default: 0.5×)
            min_sample_size: Minimum trades for statistical confidence (default: 30)
        
        Conservative design:
        - Rejects E ≤ 0.1% (must have measurable edge)
        - High bar for 1.5× scaling (E > 2%)
        - Reduces size for marginal systems (0 < E ≤ 1%)
        """
        self.min_expectancy = float(min_expectancy)
        self.high_expectancy_threshold = float(high_expectancy_threshold)
        self.medium_expectancy_threshold = float(medium_expectancy_threshold)
        self.high_multiplier = float(high_multiplier)
        self.medium_multiplier = float(medium_multiplier)
        self.low_multiplier = float(low_multiplier)
        self.min_sample_size = int(min_sample_size)
    
    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: Optional[int] = None,
    ) -> ExpectancyResult:
        """
        Calculate expectancy and position size multiplier.
        
        Formula:
        E = p·W - (1-p)·L
        
        Where:
        - p = win_rate
        - W = avg_win (as decimal, e.g., 0.02 = 2%)
        - L = avg_loss (as decimal, e.g., 0.015 = 1.5%)
        
        Args:
            win_rate: Win rate p ∈ [0, 1]
            avg_win: Average win in decimal
            avg_loss: Average loss in decimal (positive value)
            num_trades: Number of historical trades (for confidence check)
        
        Returns:
            ExpectancyResult with expectancy, multiplier, validity
        
        Rejection conditions:
        - win_rate ∉ [0, 1]
        - avg_win ≤ 0 or avg_loss ≤ 0
        - expectancy ≤ min_expectancy
        - num_trades < min_sample_size
        """
        # Validate inputs
        if not 0 <= win_rate <= 1:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason=f"Invalid win_rate: {win_rate} (must be in [0,1])"
            )
        
        if avg_win <= 0:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason=f"Invalid avg_win: {avg_win} (must be > 0)"
            )
        
        if avg_loss <= 0:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason=f"Invalid avg_loss: {avg_loss} (must be > 0)"
            )
        
        # Check sample size (if provided)
        if num_trades is not None and num_trades < self.min_sample_size:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason=f"Insufficient sample size: {num_trades} trades (min: {self.min_sample_size})"
            )
        
        # Calculate expectancy
        # E = p·W - (1-p)·L
        p = float(win_rate)
        W = float(avg_win)
        L = float(avg_loss)
        
        expectancy = (p * W) - ((1 - p) * L)
        
        # Calculate profit factor for diagnostics
        # PF = (p·W) / ((1-p)·L)
        denominator = (1 - p) * L
        if denominator > 0:
            profit_factor = (p * W) / denominator
        else:
            profit_factor = np.inf if p == 1.0 else 0.0
        
        # Check if expectancy meets minimum threshold
        if expectancy <= self.min_expectancy:
            return ExpectancyResult(
                expectancy=float(expectancy),
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason=f"Expectancy {expectancy:.4f} below minimum {self.min_expectancy:.4f}",
                win_rate=p,
                avg_win=W,
                avg_loss=L,
                profit_factor=float(profit_factor)
            )
        
        # Determine multiplier based on expectancy magnitude
        if expectancy > self.high_expectancy_threshold:
            multiplier = self.high_multiplier
        elif expectancy > self.medium_expectancy_threshold:
            multiplier = self.medium_multiplier
        else:
            multiplier = self.low_multiplier
        
        return ExpectancyResult(
            expectancy=float(expectancy),
            expectancy_multiplier=float(multiplier),
            is_valid=True,
            rejection_reason=None,
            win_rate=p,
            avg_win=W,
            avg_loss=L,
            profit_factor=float(profit_factor)
        )
    
    def calculate_from_stats(
        self,
        total_trades: int,
        winning_trades: int,
        total_profit: float,
        total_loss: float,
    ) -> ExpectancyResult:
        """
        Calculate expectancy from aggregate trading statistics.
        
        Convenience method when you have historical P&L data.
        
        Args:
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            total_profit: Sum of all profits (absolute value)
            total_loss: Sum of all losses (absolute value)
        
        Returns:
            ExpectancyResult
        """
        if total_trades <= 0:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason="No trades in history"
            )
        
        if winning_trades <= 0:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason="No winning trades in history"
            )
        
        # Calculate win rate
        win_rate = winning_trades / total_trades
        
        # Calculate average win/loss
        losing_trades = total_trades - winning_trades
        
        if losing_trades == 0:
            # Perfect track record - use conservative avg_loss estimate
            avg_win = total_profit / winning_trades if total_profit > 0 else 0.01
            avg_loss = avg_win * 0.5  # Assume 2:1 RR
        else:
            avg_win = total_profit / winning_trades if total_profit > 0 else 0.01
            avg_loss = total_loss / losing_trades if total_loss > 0 else 0.01
        
        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=total_trades
        )
    
    def calculate_from_trades(
        self,
        trade_returns: list[float],
    ) -> ExpectancyResult:
        """
        Calculate expectancy from list of trade returns.
        
        Args:
            trade_returns: List of trade returns (e.g., [0.02, -0.01, 0.03, ...])
        
        Returns:
            ExpectancyResult
        """
        if not trade_returns:
            return ExpectancyResult(
                expectancy=0.0,
                expectancy_multiplier=0.0,
                is_valid=False,
                rejection_reason="No trades provided"
            )
        
        total_trades = len(trade_returns)
        winning_trades = sum(1 for r in trade_returns if r > 0)
        
        profits = [r for r in trade_returns if r > 0]
        losses = [abs(r) for r in trade_returns if r < 0]
        
        total_profit = sum(profits) if profits else 0.0
        total_loss = sum(losses) if losses else 0.0
        
        return self.calculate_from_stats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_profit=total_profit,
            total_loss=total_loss
        )
