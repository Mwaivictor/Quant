"""
Position Sizing Module - Institutional Grade

Implements multi-layered position sizing with:
1. Kelly Criterion (growth-optimal with safety factor)
2. Expectancy-based adjustment (edge-sensitive scaling)
3. Non-linear confidence scaling (sigmoid transformation)
4. Portfolio-aware volatility (multivariate risk)
5. Liquidity constraints (ADV, spread, market impact)

Conservative by design - multiple multiplicative constraints ensure capital protection.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime

from .kelly_criterion import KellyCriterion, KellyResult
from .expectancy import ExpectancyCalculator, ExpectancyResult
from .liquidity_constraints import LiquidityConstraints, LiquidityResult


class PositionSizer:
    """
    Institutional-grade position sizer with multi-layered risk controls.
    
    Position sizing flow:
    1. Base ATR sizing: risk_capital / (ATR * multiplier)
    2. Kelly Criterion cap: limit to growth-optimal %
    3. Expectancy adjustment: scale by edge quality
    4. Sigmoid confidence: non-linear ML confidence transform
    5. Regime adjustment: market-condition multiplier
    6. Portfolio volatility: scale if portfolio σ exceeds target
    7. Liquidity constraints: ADV, spread, market impact limits
    
    All adjustments are multiplicative and conservative.
    """
    
    def __init__(self, config):
        """
        Initialize institutional position sizer.
        
        Args:
            config: RPMConfig instance with sizing parameters
        """
        self.config = config
        
        # Initialize institutional modules
        self.kelly = KellyCriterion(
            safety_factor=0.25,  # Quarter Kelly (conservative)
            max_kelly_pct=0.01,  # 1% hard cap
            min_win_rate=0.51,
            min_sample_size=30
        )
        
        self.expectancy = ExpectancyCalculator(
            min_expectancy=0.001,  # 0.1% minimum edge
            high_expectancy_threshold=0.02,  # 2% for 1.5× multiplier
            medium_expectancy_threshold=0.01,  # 1% for 1.0× multiplier
            high_multiplier=1.5,
            medium_multiplier=1.0,
            low_multiplier=0.5,
            min_sample_size=30
        )
        
        self.liquidity = LiquidityConstraints(
            max_adv_pct=0.01,  # 1% of ADV
            max_spread_bps=20.0,  # 20 bps max spread
            max_market_impact_pct=0.005,  # 0.5% max impact
            impact_coefficient=0.1,
            min_adv_units=10000.0
        )
    
    def calculate_position_size(
        self,
        symbol: str,
        atr: float,
        confidence_score: float,
        regime: str,
        vol_percentile: float,
        current_price: Optional[float] = None,
        # Institutional parameters (optional)
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        num_trades: Optional[int] = None,
        adv_units: Optional[float] = None,
        spread_pct: Optional[float] = None,
        daily_volatility: Optional[float] = None,
        portfolio_volatility: Optional[float] = None,
        target_portfolio_vol: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """
        Calculate institutional-grade position size with all constraints.
        
        Flow:
        1. ATR-based base sizing
        2. Kelly Criterion cap (if stats provided)
        3. Expectancy adjustment (if stats provided)
        4. Non-linear confidence scaling (sigmoid)
        5. Regime adjustment
        6. Portfolio volatility constraint (if provided)
        7. Liquidity constraints (if provided)
        
        Args:
            symbol: Trading symbol
            atr: Average True Range for volatility scaling
            confidence_score: Model confidence [0-1]
            regime: Market regime (TRENDING, RANGING, VOLATILE, STRESSED)
            vol_percentile: Volatility percentile [0-1]
            current_price: Current market price (required for Kelly/liquidity)
            
            # Kelly Criterion (optional)
            win_rate: Historical win rate [0-1]
            avg_win: Average win as decimal (e.g., 0.02 = 2%)
            avg_loss: Average loss as decimal (e.g., 0.015 = 1.5%)
            num_trades: Number of historical trades
            
            # Liquidity (optional)
            adv_units: Average daily volume in units
            spread_pct: Bid-ask spread as decimal (e.g., 0.0015 = 15 bps)
            daily_volatility: Daily volatility σ as decimal
            
            # Portfolio risk (optional)
            portfolio_volatility: Current portfolio volatility σp
            target_portfolio_vol: Target portfolio volatility σtarget
        
        Returns:
            Tuple[float, dict]: (final_position_units, sizing_breakdown_dict)
            
        Note:
            All institutional constraints are OPTIONAL but recommended:
            - Without Kelly/expectancy stats: uses ATR+confidence+regime only
            - Without liquidity data: no ADV/spread constraints applied
            - Without portfolio vol: no multivariate risk adjustment
        """
        
        # Initialize sizing breakdown
        breakdown = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: ATR-Based Base Sizing
        # ═══════════════════════════════════════════════════════════════
        if atr <= 0:
            return 0.0, {'error': 'ATR must be > 0', **breakdown}
        
        risk_capital = self.config.total_capital * self.config.risk_per_trade
        stop_distance = atr * self.config.atr_multiplier
        base_units = risk_capital / stop_distance
        
        breakdown['atr'] = float(atr)
        breakdown['risk_capital'] = float(risk_capital)
        breakdown['base_units'] = float(base_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Kelly Criterion Cap (ADAPTIVE - regime-aware)
        # ═══════════════════════════════════════════════════════════════
        kelly_result = None
        kelly_cap_units = None
        
        if all(v is not None for v in [win_rate, avg_win, avg_loss, current_price]):
            kelly_result = self.kelly.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                num_trades=num_trades,
                regime=regime  # Pass regime for adaptive Kelly cap
            )
            
            breakdown['kelly'] = kelly_result.to_dict()
            
            if not kelly_result.is_valid:
                # Reject if Kelly indicates negative edge
                return 0.0, {
                    'rejection_reason': f"Kelly rejection: {kelly_result.rejection_reason}",
                    **breakdown
                }
            
            # Calculate Kelly-based maximum units
            kelly_cap_units = self.kelly.get_recommended_units(
                total_capital=self.config.total_capital,
                current_price=current_price,
                kelly_result=kelly_result
            )
            
            # Apply Kelly cap (take minimum)
            base_units = min(base_units, kelly_cap_units)
            breakdown['kelly_cap_units'] = float(kelly_cap_units)
            breakdown['kelly_capped'] = kelly_cap_units < breakdown['base_units']
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Expectancy Adjustment (if stats provided)
        # ═══════════════════════════════════════════════════════════════
        expectancy_result = None
        expectancy_multiplier = 1.0
        
        if all(v is not None for v in [win_rate, avg_win, avg_loss]):
            expectancy_result = self.expectancy.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                num_trades=num_trades
            )
            
            breakdown['expectancy'] = expectancy_result.to_dict()
            
            if not expectancy_result.is_valid:
                # Reject if expectancy is non-positive
                return 0.0, {
                    'rejection_reason': f"Expectancy rejection: {expectancy_result.rejection_reason}",
                    **breakdown
                }
            
            expectancy_multiplier = expectancy_result.expectancy_multiplier
            base_units *= expectancy_multiplier
            breakdown['expectancy_adjusted_units'] = float(base_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Non-Linear Confidence Scaling (Sigmoid)
        # ═══════════════════════════════════════════════════════════════
        confidence_multiplier = self._calculate_confidence_multiplier_sigmoid(confidence_score)
        confidence_adjusted_units = base_units * confidence_multiplier
        
        breakdown['confidence_score'] = float(confidence_score)
        breakdown['confidence_multiplier'] = float(confidence_multiplier)
        breakdown['confidence_adjusted_units'] = float(confidence_adjusted_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Regime Adjustment
        # ═══════════════════════════════════════════════════════════════
        regime_multiplier = self._get_regime_multiplier(regime)
        regime_adjusted_units = confidence_adjusted_units * regime_multiplier
        
        breakdown['regime'] = regime
        breakdown['regime_multiplier'] = float(regime_multiplier)
        breakdown['regime_adjusted_units'] = float(regime_adjusted_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Portfolio Volatility Constraint (if provided)
        # ═══════════════════════════════════════════════════════════════
        portfolio_vol_multiplier = 1.0
        
        if portfolio_volatility is not None and target_portfolio_vol is not None:
            if portfolio_volatility > target_portfolio_vol:
                # Scale down if portfolio vol exceeds target
                # Multiplier = target / current (always ≤ 1.0)
                portfolio_vol_multiplier = target_portfolio_vol / portfolio_volatility
                regime_adjusted_units *= portfolio_vol_multiplier
                
                breakdown['portfolio_volatility'] = float(portfolio_volatility)
                breakdown['target_portfolio_vol'] = float(target_portfolio_vol)
                breakdown['portfolio_vol_multiplier'] = float(portfolio_vol_multiplier)
                breakdown['portfolio_vol_adjusted_units'] = float(regime_adjusted_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Volatility Percentile Adjustment
        # ═══════════════════════════════════════════════════════════════
        vol_adjustment = self._calculate_volatility_multiplier(vol_percentile)
        vol_adjusted_units = regime_adjusted_units * vol_adjustment
        
        breakdown['vol_percentile'] = float(vol_percentile)
        breakdown['vol_adjustment'] = float(vol_adjustment)
        breakdown['vol_adjusted_units'] = float(vol_adjusted_units)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 8: Liquidity Constraints (if provided)
        # ═══════════════════════════════════════════════════════════════
        liquidity_result = None
        final_units = vol_adjusted_units
        
        if all(v is not None for v in [adv_units, spread_pct, daily_volatility, current_price]):
            liquidity_result = self.liquidity.check(
                proposed_units=vol_adjusted_units,
                adv_units=adv_units,
                spread_pct=spread_pct,
                volatility=daily_volatility,
                current_price=current_price
            )
            
            breakdown['liquidity'] = liquidity_result.to_dict()
            
            if not liquidity_result.is_acceptable:
                # Reject if liquidity inadequate
                return 0.0, {
                    'rejection_reason': f"Liquidity rejection: {liquidity_result.rejection_reason}",
                    **breakdown
                }
            
            # Apply liquidity limit (take minimum)
            final_units = min(vol_adjusted_units, liquidity_result.max_units)
            
            # Apply spread penalty
            final_units *= liquidity_result.spread_penalty
            
            breakdown['liquidity_capped'] = liquidity_result.max_units < vol_adjusted_units
            breakdown['spread_penalty_applied'] = liquidity_result.spread_penalty < 1.0
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL: Round and Validate
        # ═══════════════════════════════════════════════════════════════
        # Ensure non-negative
        final_units = max(0.0, final_units)
        
        # Round to 2 decimals (standard lot sizing)
        final_units = round(final_units, 2)
        
        breakdown['final_units'] = float(final_units)
        
        # Return final position size and comprehensive breakdown
        return final_units, breakdown
    
    def _calculate_confidence_multiplier_sigmoid(self, confidence_score: float) -> float:
        """
        Calculate confidence-based size multiplier using sigmoid transformation.
        
        Non-linear scaling with sigmoid centered at confidence=0.5:
        - Flatter response for low/high confidence (diminishing returns)
        - Steeper gradient around 0.5 (sensitive to uncertainty)
        - Output range: [0.5, 1.5]
        
        Formula:
        sigmoid(x) = 1 / (1 + exp(-k*(x - 0.5)))
        multiplier = 0.5 + sigmoid(confidence) * 1.0
        
        Args:
            confidence_score: Model confidence [0-1]
        
        Returns:
            float: Confidence multiplier [0.5, 1.5]
        """
        if not self.config.confidence_scaling:
            return 1.0
        
        # Clamp confidence to [0, 1]
        confidence = np.clip(confidence_score, 0.0, 1.0)
        
        # Sigmoid transformation
        # k = steepness parameter (higher = steeper gradient at center)
        k = 10.0  # Moderate steepness
        
        # Sigmoid centered at 0.5
        # Maps [0, 1] → [0, 1] with non-linear S-curve
        sigmoid_value = 1.0 / (1.0 + np.exp(-k * (confidence - 0.5)))
        
        # Scale to [0.5, 1.5]
        multiplier = 0.5 + sigmoid_value * 1.0
        
        # Safety clamp (should be redundant, but ensures bounds)
        multiplier = np.clip(
            multiplier,
            self.config.confidence_min_multiplier,
            self.config.confidence_max_multiplier
        )
        
        return float(multiplier)
    
    def _calculate_confidence_multiplier(self, confidence_score: float) -> float:
        """
        LEGACY: Linear confidence multiplier (deprecated in favor of sigmoid).
        
        Linear scaling between min and max multipliers:
        - confidence=0.5 → min_multiplier (0.5)
        - confidence=0.75 → 1.0
        - confidence=1.0 → max_multiplier (1.5)
        
        Args:
            confidence_score: Model confidence [0-1]
        
        Returns:
            float: Confidence multiplier
        """
        if not self.config.confidence_scaling:
            return 1.0
        
        # Clamp confidence to [0, 1]
        confidence = np.clip(confidence_score, 0.0, 1.0)
        
        # Linear interpolation
        if confidence <= 0.75:
            # Scale from min_multiplier at 0.5 to 1.0 at 0.75
            min_mult = self.config.confidence_min_multiplier
            slope = (1.0 - min_mult) / 0.25  # (1.0 - 0.5) / (0.75 - 0.5)
            multiplier = min_mult + slope * (confidence - 0.5)
        else:
            # Scale from 1.0 at 0.75 to max_multiplier at 1.0
            max_mult = self.config.confidence_max_multiplier
            slope = (max_mult - 1.0) / 0.25  # (1.5 - 1.0) / (1.0 - 0.75)
            multiplier = 1.0 + slope * (confidence - 0.75)
        
        # Clamp to configured bounds
        multiplier = np.clip(
            multiplier,
            self.config.confidence_min_multiplier,
            self.config.confidence_max_multiplier
        )
        
        return float(multiplier)
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Get regime-based size multiplier.
        
        Args:
            regime: Market regime
        
        Returns:
            float: Regime multiplier
        """
        regime = regime.upper()
        
        # Get multiplier from config
        multiplier = self.config.regime_adjustments.get(regime, 1.0)
        
        return float(multiplier)
    
    def _calculate_volatility_multiplier(self, vol_percentile: float) -> float:
        """
        Calculate volatility-based size adjustment.
        
        Reduce size in extreme volatility:
        - vol_percentile < 0.80 → 1.0 (no adjustment)
        - vol_percentile 0.80-0.90 → 0.9 (10% reduction)
        - vol_percentile 0.90-0.95 → 0.8 (20% reduction)
        - vol_percentile > 0.95 → 0.7 (30% reduction)
        
        Args:
            vol_percentile: Volatility percentile [0-1]
        
        Returns:
            float: Volatility multiplier
        """
        vol_pct = np.clip(vol_percentile, 0.0, 1.0)
        
        if vol_pct < 0.80:
            return 1.0
        elif vol_pct < 0.90:
            return 0.9
        elif vol_pct < 0.95:
            return 0.8
        else:
            return 0.7
    
    def adjust_for_correlation(
        self,
        base_units: float,
        correlation: float,
    ) -> Tuple[float, float]:
        """
        Adjust position size for correlation with existing positions.
        
        If correlation > max_correlation_exposure, apply penalty.
        
        Args:
            base_units: Base position size before correlation adjustment
            correlation: Correlation with existing positions [0-1]
        
        Returns:
            Tuple[float, float]: (adjusted_units, correlation_adjustment)
        """
        if correlation < self.config.max_correlation_exposure:
            # No adjustment needed
            return base_units, 1.0
        
        # Apply correlation penalty
        adjustment = self.config.correlation_penalty
        adjusted_units = base_units * adjustment
        
        return adjusted_units, adjustment
    
    def validate_position_size(
        self,
        units: float,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate position size against limits.
        
        Args:
            units: Proposed position size
            symbol: Trading symbol
            current_price: Current market price (optional)
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, rejection_reason)
        """
        
        # Check minimum size
        if units <= 0:
            return False, "Position size must be > 0"
        
        # Check maximum units
        if units > self.config.max_symbol_exposure_units:
            return False, f"Position size {units} exceeds max_symbol_exposure_units {self.config.max_symbol_exposure_units}"
        
        # Check maximum percentage exposure (if price provided)
        if current_price is not None and current_price > 0:
            position_value = units * current_price
            max_value = self.config.total_capital * self.config.max_symbol_exposure_pct
            
            if position_value > max_value:
                return False, f"Position value {position_value} exceeds max_symbol_exposure_pct limit {max_value}"
        
        return True, None
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
    ) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1=LONG, -1=SHORT)
            atr: Average True Range
        
        Returns:
            float: Stop loss price
        """
        stop_distance = atr * self.config.atr_multiplier
        
        if direction == 1:  # LONG
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
        
        return float(stop_price)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            direction: Trade direction (1=LONG, -1=SHORT)
            atr: Average True Range
            risk_reward_ratio: Risk-reward ratio (default 2:1)
        
        Returns:
            float: Take profit price
        """
        stop_distance = atr * self.config.atr_multiplier
        profit_distance = stop_distance * risk_reward_ratio
        
        if direction == 1:  # LONG
            tp_price = entry_price + profit_distance
        else:  # SHORT
            tp_price = entry_price - profit_distance
        
        return float(tp_price)
