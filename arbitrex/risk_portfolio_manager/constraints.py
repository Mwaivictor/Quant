"""
Portfolio Constraints Module

Enforces portfolio-level constraints on exposure, correlation, and position limits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .schemas import PortfolioState, Position


class PortfolioConstraints:
    """
    Portfolio constraint enforcer.
    
    Validates trades against:
    - Symbol exposure limits
    - Currency exposure limits  
    - Total exposure limits
    - Correlation limits
    - Position count limits
    """
    
    def __init__(self, config):
        """
        Initialize portfolio constraints.
        
        Args:
            config: RPMConfig instance
        """
        self.config = config
    
    def check_constraints(
        self,
        symbol: str,
        direction: int,
        proposed_units: float,
        portfolio_state: PortfolioState,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, List[str], float]:
        """
        Check all portfolio constraints for a proposed trade.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (1=LONG, -1=SHORT)
            proposed_units: Proposed position size
            portfolio_state: Current portfolio state
            current_price: Current market price (optional)
        
        Returns:
            Tuple[bool, List[str], float]: 
                - constraints_passed: True if all constraints passed
                - violations: List of constraint violations
                - adjusted_units: Position size after constraint adjustments
        """
        violations = []
        adjusted_units = proposed_units
        
        # Check 1: Position count limits
        passed, violation = self._check_position_count(portfolio_state)
        if not passed:
            violations.append(violation)
            return False, violations, 0.0
        
        # Check 2: Symbol position limit
        passed, violation = self._check_symbol_position_limit(symbol, portfolio_state)
        if not passed:
            violations.append(violation)
            return False, violations, 0.0
        
        # Check 3: Symbol exposure limits
        passed, violation, size_adjustment = self._check_symbol_exposure(
            symbol, proposed_units, portfolio_state, current_price
        )
        if not passed:
            violations.append(violation)
            if size_adjustment == 0.0:
                return False, violations, 0.0
        else:
            adjusted_units = min(adjusted_units, size_adjustment)
        
        # Check 4: Currency exposure limits
        passed, violation, size_adjustment = self._check_currency_exposure(
            symbol, direction, proposed_units, portfolio_state
        )
        if not passed:
            violations.append(violation)
            if size_adjustment == 0.0:
                return False, violations, 0.0
        else:
            adjusted_units = min(adjusted_units, size_adjustment)
        
        # Check 5: Total exposure limits
        passed, violation, size_adjustment = self._check_total_exposure(
            proposed_units, portfolio_state, current_price
        )
        if not passed:
            violations.append(violation)
            if size_adjustment == 0.0:
                return False, violations, 0.0
        else:
            adjusted_units = min(adjusted_units, size_adjustment)
        
        # Check 6: Correlation limits
        correlation_adjustment = self._check_correlation(
            symbol, direction, portfolio_state
        )
        if correlation_adjustment < 1.0:
            adjusted_units *= correlation_adjustment
            violations.append(
                f"High correlation detected - reduced size by {(1-correlation_adjustment)*100:.1f}%"
            )
        
        # Final validation
        if adjusted_units <= 0:
            violations.append("Adjusted position size is zero after constraint checks")
            return False, violations, 0.0
        
        constraints_passed = len(violations) == 0
        return constraints_passed, violations, adjusted_units
    
    def _check_position_count(
        self,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if we can add another position.
        
        Args:
            portfolio_state: Current portfolio state
        
        Returns:
            Tuple[bool, Optional[str]]: (passed, violation_message)
        """
        current_count = len(portfolio_state.open_positions)
        max_count = self.config.max_concurrent_positions
        
        if current_count >= max_count:
            return False, f"Max concurrent positions reached ({current_count}/{max_count})"
        
        return True, None
    
    def _check_symbol_position_limit(
        self,
        symbol: str,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol already has max positions.
        
        Args:
            symbol: Trading symbol
            portfolio_state: Current portfolio state
        
        Returns:
            Tuple[bool, Optional[str]]: (passed, violation_message)
        """
        # Count existing positions for this symbol
        symbol_positions = sum(
            1 for pos in portfolio_state.open_positions.values()
            if pos.symbol == symbol
        )
        
        if symbol_positions >= self.config.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached ({symbol_positions}/{self.config.max_positions_per_symbol})"
        
        return True, None
    
    def _check_symbol_exposure(
        self,
        symbol: str,
        proposed_units: float,
        portfolio_state: PortfolioState,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check symbol exposure limits.
        
        Args:
            symbol: Trading symbol
            proposed_units: Proposed position size
            portfolio_state: Current portfolio state
            current_price: Current market price (optional)
        
        Returns:
            Tuple[bool, Optional[str], float]: 
                (passed, violation_message, max_allowed_units)
        """
        # Get current exposure
        current_exposure = portfolio_state.symbol_exposure.get(symbol, 0.0)
        total_exposure = abs(current_exposure) + proposed_units
        
        # Check units limit
        if total_exposure > self.config.max_symbol_exposure_units:
            max_additional = max(0, self.config.max_symbol_exposure_units - abs(current_exposure))
            return False, f"Symbol exposure limit: {total_exposure:.2f} > {self.config.max_symbol_exposure_units}", max_additional
        
        # Check percentage limit (if price provided)
        if current_price is not None and current_price > 0:
            exposure_value = total_exposure * current_price
            max_value = self.config.total_capital * self.config.max_symbol_exposure_pct
            
            if exposure_value > max_value:
                max_additional_value = max(0, max_value - abs(current_exposure * current_price))
                max_additional_units = max_additional_value / current_price
                return False, f"Symbol exposure %: {exposure_value:.2f} > {max_value:.2f}", max_additional_units
        
        return True, None, proposed_units
    
    def _check_currency_exposure(
        self,
        symbol: str,
        direction: int,
        proposed_units: float,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check currency exposure limits.
        
        Decomposes FX pairs into base/quote currencies and checks net exposure.
        
        Args:
            symbol: Trading symbol (e.g., EURUSD)
            direction: Trade direction (1=LONG, -1=SHORT)
            proposed_units: Proposed position size
            portfolio_state: Current portfolio state
        
        Returns:
            Tuple[bool, Optional[str], float]: 
                (passed, violation_message, max_allowed_units)
        """
        # Decompose symbol into currencies (assumes format like EURUSD)
        if len(symbol) >= 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
        else:
            # Can't decompose, skip check
            return True, None, proposed_units
        
        # Calculate proposed currency exposures
        # LONG EURUSD = +EUR, -USD
        # SHORT EURUSD = -EUR, +USD
        base_delta = proposed_units * direction
        quote_delta = -proposed_units * direction
        
        # Check base currency exposure
        current_base = portfolio_state.currency_exposure.get(base_currency, 0.0)
        new_base = abs(current_base + base_delta)
        max_base = self.config.total_capital * self.config.max_currency_exposure_pct
        
        if new_base > max_base:
            max_additional = max(0, max_base - abs(current_base))
            return False, f"Currency exposure limit ({base_currency}): {new_base:.2f} > {max_base:.2f}", max_additional
        
        # Check quote currency exposure
        current_quote = portfolio_state.currency_exposure.get(quote_currency, 0.0)
        new_quote = abs(current_quote + quote_delta)
        max_quote = self.config.total_capital * self.config.max_currency_exposure_pct
        
        if new_quote > max_quote:
            max_additional = max(0, max_quote - abs(current_quote))
            return False, f"Currency exposure limit ({quote_currency}): {new_quote:.2f} > {max_quote:.2f}", max_additional
        
        return True, None, proposed_units
    
    def _check_total_exposure(
        self,
        proposed_units: float,
        portfolio_state: PortfolioState,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check total portfolio exposure limits.
        
        Args:
            proposed_units: Proposed position size
            portfolio_state: Current portfolio state
            current_price: Current market price (optional)
        
        Returns:
            Tuple[bool, Optional[str], float]: 
                (passed, violation_message, max_allowed_units)
        """
        if current_price is None or current_price <= 0:
            # Can't check without price
            return True, None, proposed_units
        
        # Calculate current gross exposure
        current_gross = sum(
            abs(pos.units * pos.entry_price) 
            for pos in portfolio_state.open_positions.values()
        )
        
        # Calculate proposed gross exposure
        proposed_value = proposed_units * current_price
        total_gross = current_gross + proposed_value
        max_gross = self.config.total_capital * self.config.max_total_exposure_pct
        
        if total_gross > max_gross:
            max_additional = max(0, max_gross - current_gross)
            max_additional_units = max_additional / current_price
            return False, f"Total gross exposure: {total_gross:.2f} > {max_gross:.2f}", max_additional_units
        
        return True, None, proposed_units
    
    def _check_correlation(
        self,
        symbol: str,
        direction: int,
        portfolio_state: PortfolioState,
    ) -> float:
        """
        Check correlation with existing positions.
        
        Applies penalty if new position is highly correlated with portfolio.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (1=LONG, -1=SHORT)
            portfolio_state: Current portfolio state
        
        Returns:
            float: Correlation adjustment multiplier [0-1]
        """
        # Simplified correlation logic
        # In production, would use actual correlation matrix
        
        if not portfolio_state.open_positions:
            return 1.0  # No correlation, no adjustment
        
        # Check for same-pair positions
        for pos in portfolio_state.open_positions.values():
            if pos.symbol == symbol:
                # Same symbol, same direction = high correlation
                if pos.direction == direction:
                    return self.config.correlation_penalty
        
        # Check for related pairs (simplified)
        # E.g., EURUSD and GBPUSD have some correlation
        base_currency = symbol[:3] if len(symbol) >= 6 else None
        
        if base_currency:
            related_positions = sum(
                1 for pos in portfolio_state.open_positions.values()
                if len(pos.symbol) >= 6 and pos.symbol[:3] == base_currency
                and pos.direction == direction
            )
            
            if related_positions >= 2:
                # Multiple related positions
                return self.config.correlation_penalty
        
        return 1.0  # No significant correlation
    
    def update_portfolio_exposure(
        self,
        portfolio_state: PortfolioState,
        symbol: str,
        direction: int,
        units: float,
        entry_price: float,
    ) -> None:
        """
        Update portfolio exposure tracking after trade execution.
        
        Args:
            portfolio_state: Portfolio state to update
            symbol: Trading symbol
            direction: Trade direction (1=LONG, -1=SHORT)
            units: Position size
            entry_price: Entry price
        """
        # Update symbol exposure
        current_symbol_exposure = portfolio_state.symbol_exposure.get(symbol, 0.0)
        portfolio_state.symbol_exposure[symbol] = current_symbol_exposure + (units * direction)
        
        # Update currency exposure
        if len(symbol) >= 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            # LONG EURUSD = +EUR, -USD
            current_base = portfolio_state.currency_exposure.get(base_currency, 0.0)
            current_quote = portfolio_state.currency_exposure.get(quote_currency, 0.0)
            
            portfolio_state.currency_exposure[base_currency] = current_base + (units * direction)
            portfolio_state.currency_exposure[quote_currency] = current_quote - (units * direction)
    
    def get_portfolio_summary(
        self,
        portfolio_state: PortfolioState,
    ) -> dict:
        """
        Get summary of portfolio constraints and utilization.
        
        Args:
            portfolio_state: Current portfolio state
        
        Returns:
            dict: Portfolio constraint summary
        """
        total_positions = len(portfolio_state.open_positions)
        
        # Symbol exposure summary
        symbol_exposures = {
            symbol: {
                'units': float(units),
                'pct_of_limit': float(abs(units) / self.config.max_symbol_exposure_units * 100),
            }
            for symbol, units in portfolio_state.symbol_exposure.items()
            if abs(units) > 0
        }
        
        # Currency exposure summary
        currency_exposures = {
            currency: {
                'units': float(units),
                'pct_of_limit': float(abs(units) / (self.config.total_capital * self.config.max_currency_exposure_pct) * 100),
            }
            for currency, units in portfolio_state.currency_exposure.items()
            if abs(units) > 0
        }
        
        return {
            'total_positions': total_positions,
            'max_positions': self.config.max_concurrent_positions,
            'position_utilization_pct': float(total_positions / self.config.max_concurrent_positions * 100),
            'symbol_exposures': symbol_exposures,
            'currency_exposures': currency_exposures,
        }
