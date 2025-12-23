"""
Risk & Portfolio Manager Engine

Core RPM engine - the gatekeeper with absolute veto authority over all trades.
"""

import time
import logging
from typing import Optional, Tuple, List, Dict
from datetime import datetime

LOG = logging.getLogger(__name__)

from .config import RPMConfig
from .schemas import (
    TradeDecision,
    ApprovedTrade,
    RejectedTrade,
    TradeApprovalStatus,
    RejectionReason,
    PortfolioState,
    RiskMetrics,
    RPMOutput,
)
from .position_sizing import PositionSizer
from .constraints import PortfolioConstraints
from .kill_switch import KillSwitchManager  # Unified kill-switch system
from .state_manager import StateManager, AutoSaveStateManager
from .order_manager import OrderManager, OrderType
from .correlation_risk import (
    CorrelationMatrix,
    PortfolioRiskCalculator,
    CorrelationAwareSizer,
)
from .mt5_sync import MT5AccountSync, create_mt5_synced_portfolio


class RiskPortfolioManager:
    """
    Risk & Portfolio Manager - THE GATEKEEPER.
    
    Absolute veto authority over all trades. No trade reaches execution without RPM approval.
    
    Processing flow:
    1. Kill switch check → HALT if triggered
    2. Regime restrictions → REJECT if violated
    3. Position sizing → Calculate volatility-scaled size
    4. Portfolio constraints → REJECT or ADJUST if violated
    5. Final validation → APPROVE or REJECT
    
    RPM is NON-BYPASSABLE. All trades MUST pass through this layer.
    """
    
    def __init__(
        self, 
        config: Optional[RPMConfig] = None, 
        enable_persistence: bool = True, 
        sync_with_mt5: bool = False,
        mt5_pool = None  # Optional[MT5ConnectionPool]
    ):
        """
        Initialize Risk & Portfolio Manager.
        
        Args:
            config: RPM configuration (uses defaults if not provided)
            enable_persistence: Enable state persistence (default: True)
            sync_with_mt5: Sync portfolio state with MT5 account on startup (default: False)
            mt5_pool: Existing MT5ConnectionPool to reuse (RECOMMENDED for production)
        """
        self.config = config if config is not None else RPMConfig()
        self.config.validate()
        
        # Core components
        self.position_sizer = PositionSizer(self.config)
        self.constraints = PortfolioConstraints(self.config)
        self.kill_switches = KillSwitchManager(config=self.config)  # Unified kill-switch manager
        
        # NEW: Order management
        self.order_manager = OrderManager()
        
        # NEW: Correlation-aware sizing
        self.correlation_matrix = CorrelationMatrix(default_correlation=0.3)
        self.correlation_matrix.set_fx_pair_correlations()  # Load FX correlations
        self.risk_calculator = PortfolioRiskCalculator(self.correlation_matrix)
        self.correlation_sizer = CorrelationAwareSizer(
            self.correlation_matrix,
            self.risk_calculator,
            max_portfolio_volatility=0.15,  # 15% annualized
        )
        
        # NEW: MT5 account synchronization
        self.mt5_sync = MT5AccountSync(
            mt5_pool=mt5_pool,  # Reuse existing pool if provided
            auto_initialize=(mt5_pool is None)  # Only auto-init if no pool
        )
        self.sync_with_mt5_enabled = sync_with_mt5
        
        # NEW: State persistence
        self.enable_persistence = enable_persistence
        if enable_persistence:
            self.state_manager = AutoSaveStateManager(
                storage_type="file",
                state_file_path="logs/rpm_state.json",
                auto_save_frequency=5,  # Save every 5 operations
            )
            # Try to load existing state
            self.portfolio_state = self.state_manager.load_state(
                default_capital=self.config.total_capital
            )
        else:
            self.state_manager = None
            self.portfolio_state = PortfolioState(total_capital=self.config.total_capital)
        
        # Optionally sync with MT5 on startup
        if sync_with_mt5 and self.mt5_sync.is_mt5_initialized():
            synced = self.mt5_sync.sync_portfolio_state(self.portfolio_state)
            if synced:
                LOG.info("RPM initialized with MT5 account data")
            else:
                LOG.warning("MT5 sync failed - using default/loaded state")
        
        # Risk metrics
        self.risk_metrics = RiskMetrics()
        
        # Metadata
        self.version = "1.1.0"  # Bumped version with new features
        self.config_hash = self.config.get_config_hash()
    
    def process_trade_intent(
        self,
        symbol: str,
        direction: int,
        confidence_score: float,
        regime: str,
        atr: float,
        vol_percentile: float,
        current_price: Optional[float] = None,
        # Kelly/expectancy stats (optional)
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        num_trades: Optional[int] = None,
        # Liquidity (optional)
        adv_units: Optional[float] = None,
        spread_pct: Optional[float] = None,
        daily_volatility: Optional[float] = None,
    ) -> RPMOutput:
        """
        Process trade intent from Signal Engine.
        
        THIS IS THE CRITICAL DECISION POINT - RPM exercises absolute veto authority here.
        
        Args:
            symbol: Trading symbol (e.g., EURUSD)
            direction: Trade direction (1=LONG, -1=SHORT)
            confidence_score: Signal confidence [0-1]
            regime: Market regime (TRENDING, RANGING, VOLATILE, STRESSED)
            atr: Average True Range for position sizing
            vol_percentile: Current volatility percentile [0-1]
            current_price: Current market price (optional, for exposure calculations)
            win_rate: Historical win rate [0-1] (optional, for Kelly)
            avg_win: Average win percentage (optional, for Kelly)
            avg_loss: Average loss percentage (optional, for Kelly)
            num_trades: Number of trades in sample (optional, for Kelly)
            adv_units: Average daily volume in units (optional, for liquidity)
            spread_pct: Bid-ask spread as decimal (optional, for liquidity)
            daily_volatility: Daily volatility (optional, for liquidity)
        
        Returns:
            RPMOutput: Complete decision with approval/rejection and full audit trail
        """
        start_time = time.time()
        
        # Initialize decision
        decision = TradeDecision(status=TradeApprovalStatus.REJECTED)
        
        # ========================================
        # STAGE 1: KILL SWITCH CHECK
        # ========================================
        
        kill_switch_triggered, reason, details = self.kill_switches.check_kill_switches(
            portfolio_state=self.portfolio_state,
            regime=regime,
            vol_percentile=vol_percentile,
            confidence_score=confidence_score,
        )
        
        if kill_switch_triggered:
            # REJECTED - Kill switch triggered
            decision.kill_switch_triggered = True
            decision.kill_switch_reason = details
            
            rejected_trade = RejectedTrade(
                symbol=symbol,
                direction=direction,
                confidence_score=confidence_score,
                rejection_reason=reason,
                rejection_details=details,
                current_drawdown=self.portfolio_state.current_drawdown,
                vol_percentile=vol_percentile,
            )
            
            decision.rejected_trade = rejected_trade
            
            self._update_rejection_metrics(reason)
            
            processing_time = (time.time() - start_time) * 1000
            decision.processing_time_ms = processing_time
            
            return self._create_output(decision)
        
        # ========================================
        # STAGE 2: POSITION SIZING
        # ========================================
        
        position_units, sizing_breakdown = self.position_sizer.calculate_position_size(
            symbol=symbol,
            atr=atr,
            confidence_score=confidence_score,
            regime=regime,
            vol_percentile=vol_percentile,
            current_price=current_price,
            # Kelly/expectancy stats (pass through)
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            # Liquidity (pass through)
            adv_units=adv_units,
            spread_pct=spread_pct,
            daily_volatility=daily_volatility,
        )
        
        if position_units <= 0:
            # REJECTED - Zero position size
            rejected_trade = RejectedTrade(
                symbol=symbol,
                direction=direction,
                confidence_score=confidence_score,
                rejection_reason=RejectionReason.ZERO_POSITION_SIZE,
                rejection_details=f"Position sizing resulted in zero units: {sizing_breakdown}",
                vol_percentile=vol_percentile,
            )
            
            decision.rejected_trade = rejected_trade
            
            self._update_rejection_metrics(RejectionReason.ZERO_POSITION_SIZE)
            
            processing_time = (time.time() - start_time) * 1000
            decision.processing_time_ms = processing_time
            
            return self._create_output(decision)
        
        decision.position_sizing_applied = True
        decision.sizing_adjustments = sizing_breakdown
        
        # Validate position size
        size_valid, size_rejection_reason = self.position_sizer.validate_position_size(
            units=position_units,
            symbol=symbol,
            current_price=current_price,
        )
        
        if not size_valid:
            # REJECTED - Invalid position size
            rejected_trade = RejectedTrade(
                symbol=symbol,
                direction=direction,
                confidence_score=confidence_score,
                rejection_reason=RejectionReason.SYMBOL_EXPOSURE_LIMIT,
                rejection_details=size_rejection_reason,
            )
            
            decision.rejected_trade = rejected_trade
            
            self._update_rejection_metrics(RejectionReason.SYMBOL_EXPOSURE_LIMIT)
            
            processing_time = (time.time() - start_time) * 1000
            decision.processing_time_ms = processing_time
            
            return self._create_output(decision)
        
        # ========================================
        # STAGE 3: PORTFOLIO CONSTRAINTS
        # ========================================
        
        constraints_passed, violations, adjusted_units = self.constraints.check_constraints(
            symbol=symbol,
            direction=direction,
            proposed_units=position_units,
            portfolio_state=self.portfolio_state,
            current_price=current_price,
        )
        
        if not constraints_passed:
            # REJECTED - Constraint violations
            decision.portfolio_constraints_passed = False
            decision.portfolio_constraint_violations = violations
            
            # Determine primary rejection reason
            if any('position' in v.lower() for v in violations):
                reason = RejectionReason.SYMBOL_EXPOSURE_LIMIT
            elif any('currency' in v.lower() for v in violations):
                reason = RejectionReason.CURRENCY_EXPOSURE_LIMIT
            elif any('correlation' in v.lower() for v in violations):
                reason = RejectionReason.CORRELATION_LIMIT
            else:
                reason = RejectionReason.SYMBOL_EXPOSURE_LIMIT
            
            rejected_trade = RejectedTrade(
                symbol=symbol,
                direction=direction,
                confidence_score=confidence_score,
                rejection_reason=reason,
                rejection_details="; ".join(violations),
                symbol_exposure=self.portfolio_state.symbol_exposure.get(symbol, 0.0),
            )
            
            decision.rejected_trade = rejected_trade
            
            self._update_rejection_metrics(reason)
            
            processing_time = (time.time() - start_time) * 1000
            decision.processing_time_ms = processing_time
            
            return self._create_output(decision)
        
        # Apply constraint adjustments
        final_units = adjusted_units
        
        # ========================================
        # STAGE 3.5: CORRELATION-AWARE ADJUSTMENT (NEW)
        # ========================================
        
        # Calculate symbol volatilities for portfolio risk calculation
        symbol_volatilities = {symbol: atr / (current_price if current_price else 1.0)}
        
        # Get current positions as list
        current_positions = list(self.portfolio_state.open_positions.values())
        
        # Apply correlation-aware sizing adjustment
        correlation_adjusted_units, corr_details = self.correlation_sizer.calculate_correlation_adjustment(
            symbol=symbol,
            proposed_units=final_units,
            current_positions=current_positions,
            symbol_volatilities=symbol_volatilities,
            regime=regime,
            entry_price=current_price if current_price else 1.0,
        )
        
        # Update final units with correlation adjustment
        final_units = correlation_adjusted_units
        correlation_adjustment = corr_details['adjustment_factor']
        
        # Add correlation details to decision
        decision.sizing_adjustments['correlation_adjustment'] = correlation_adjustment
        decision.sizing_adjustments['correlation_details'] = corr_details
        
        if len(violations) > 0:
            # Constraints passed but with adjustments
            decision.status = TradeApprovalStatus.ADJUSTED
            decision.portfolio_constraint_violations = violations
        else:
            # Clean approval
            decision.status = TradeApprovalStatus.APPROVED
        
        decision.portfolio_constraints_passed = True
        
        # ========================================
        # STAGE 4: FINAL APPROVAL
        # ========================================
        
        approved_trade = ApprovedTrade(
            symbol=symbol,
            direction=direction,
            position_units=final_units,
            confidence_score=confidence_score,
            regime=regime,
            base_units=sizing_breakdown['base_units'],
            confidence_adjustment=sizing_breakdown.get('confidence_multiplier', 1.0),
            regime_adjustment=sizing_breakdown.get('regime_multiplier', 1.0),
            correlation_adjustment=correlation_adjustment,  # NEW: Include correlation adjustment
            atr=atr,
            vol_percentile=vol_percentile,
            risk_per_trade=sizing_breakdown['risk_capital'],
        )
        
        decision.approved_trade = approved_trade
        
        # NEW: Create order for execution tracking
        order = self.order_manager.create_order(
            symbol=symbol,
            direction=direction,
            approved_units=final_units,
            order_type=OrderType.MARKET,
            rpm_config_hash=self.config_hash,
            confidence_score=confidence_score,
            regime=regime,
        )
        
        # Store order_id in decision for tracking
        decision.order_id = order.order_id
        
        # Update metrics
        self._update_approval_metrics(final_units)
        
        processing_time = (time.time() - start_time) * 1000
        decision.processing_time_ms = processing_time
        
        # NEW: Auto-save state if enabled
        if self.enable_persistence and self.state_manager:
            self.state_manager.maybe_save(self.portfolio_state)
        
        return self._create_output(decision)
    
    def _create_output(self, decision: TradeDecision) -> RPMOutput:
        """
        Create complete RPM output.
        
        Args:
            decision: Trade decision
        
        Returns:
            RPMOutput: Complete output with decision, state, and metrics
        """
        # Update portfolio state timestamp
        self.portfolio_state.last_update = datetime.utcnow()
        
        return RPMOutput(
            decision=decision,
            portfolio_state=self.portfolio_state,
            risk_metrics=self.risk_metrics,
            config_hash=self.config_hash,
            rpm_version=self.version,
        )
    
    def _update_approval_metrics(self, position_units: float) -> None:
        """Update metrics after trade approval"""
        self.risk_metrics.total_decisions += 1
        self.risk_metrics.trades_approved += 1
        
        # Update position size stats
        if self.risk_metrics.min_position_size == 0:
            self.risk_metrics.min_position_size = position_units
        else:
            self.risk_metrics.min_position_size = min(
                self.risk_metrics.min_position_size, position_units
            )
        
        self.risk_metrics.max_position_size = max(
            self.risk_metrics.max_position_size, position_units
        )
        
        # Update average
        n = self.risk_metrics.trades_approved
        self.risk_metrics.avg_position_size = (
            self.risk_metrics.avg_position_size * (n - 1) + position_units
        ) / n
        
        # Update approval rate
        self.risk_metrics.approval_rate = (
            self.risk_metrics.trades_approved / self.risk_metrics.total_decisions
        )
    
    def _update_rejection_metrics(self, reason: RejectionReason) -> None:
        """Update metrics after trade rejection"""
        self.risk_metrics.total_decisions += 1
        self.risk_metrics.trades_rejected += 1
        
        # Update rejection reasons
        if reason == RejectionReason.MAX_DRAWDOWN_EXCEEDED:
            self.risk_metrics.rejections_by_drawdown += 1
        elif reason == RejectionReason.DAILY_LOSS_LIMIT:
            self.risk_metrics.rejections_by_loss_limit += 1
        elif reason in (RejectionReason.SYMBOL_EXPOSURE_LIMIT, RejectionReason.CURRENCY_EXPOSURE_LIMIT):
            self.risk_metrics.rejections_by_exposure += 1
        elif reason == RejectionReason.EXTREME_VOLATILITY:
            self.risk_metrics.rejections_by_volatility += 1
        elif reason == RejectionReason.STRESSED_REGIME:
            self.risk_metrics.rejections_by_regime += 1
        elif reason == RejectionReason.LOW_MODEL_CONFIDENCE:
            self.risk_metrics.rejections_by_confidence += 1
        
        # Update approval rate
        if self.risk_metrics.total_decisions > 0:
            self.risk_metrics.approval_rate = (
                self.risk_metrics.trades_approved / self.risk_metrics.total_decisions
            )
    
    def get_health_status(self) -> dict:
        """
        Get RPM health status.
        
        Returns:
            dict: Complete health status
        """
        kill_switch_status = self.kill_switches.get_kill_switch_status(
            portfolio_state=self.portfolio_state,
            regime="UNKNOWN",  # Would come from latest market data
            vol_percentile=0.5,
            confidence_score=0.7,
        )
        
        portfolio_summary = self.constraints.get_portfolio_summary(
            portfolio_state=self.portfolio_state
        )
        
        return {
            'rpm_version': self.version,
            'config_hash': self.config_hash,
            'portfolio_state': self.portfolio_state.to_dict(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'kill_switches': kill_switch_status,
            'portfolio_constraints': portfolio_summary,
            'health': 'OPERATIONAL' if not self.portfolio_state.trading_halted else 'HALTED',
        }
    
    def reset_daily_metrics(self) -> None:
        """Reset daily PnL and metrics (call at start of new trading day)"""
        self.portfolio_state.daily_pnl = 0.0
    
    def reset_weekly_metrics(self) -> None:
        """Reset weekly PnL (call at start of new trading week)"""
        self.portfolio_state.weekly_pnl = 0.0
    
    # ========================================
    # NEW: ORDER MANAGEMENT METHODS
    # ========================================
    
    def update_order_fill(
        self,
        order_id: str,
        fill_units: float,
        fill_price: float,
        expected_price: Optional[float] = None,
    ) -> Optional['Order']:
        """
        Update order with fill information from execution engine.
        
        Args:
            order_id: Order identifier
            fill_units: Units filled in this execution
            fill_price: Execution price
            expected_price: Expected price (for slippage calculation)
        
        Returns:
            Order: Updated order, or None if not found
        """
        order = self.order_manager.update_order_fill(
            order_id, fill_units, fill_price, expected_price
        )
        
        # Auto-save after fill update
        if self.enable_persistence and self.state_manager:
            self.state_manager.maybe_save(self.portfolio_state)
        
        return order
    
    def get_pending_orders(self) -> List['Order']:
        """Get all active/pending orders"""
        return list(self.order_manager.active_orders.values())
    
    def get_slippage_stats(self) -> dict:
        """Get slippage statistics from recent orders"""
        return self.order_manager.get_slippage_statistics()
    
    def get_order_stats(self) -> dict:
        """Get order execution statistics"""
        return self.order_manager.get_order_statistics()
    
    # ========================================
    # NEW: PERSISTENCE METHODS
    # ========================================
    
    def save_state(self) -> bool:
        """Manually save current state"""
        if self.enable_persistence and self.state_manager:
            return self.state_manager.save_state(self.portfolio_state)
        return False
    
    def create_backup(self) -> bool:
        """Create timestamped backup of current state"""
        if self.enable_persistence and self.state_manager:
            return self.state_manager.create_backup()
        return False
    
    # ========================================
    # NEW: CORRELATION RISK METHODS
    # ========================================
    
    def get_portfolio_volatility(self, regime: str = 'RANGING') -> float:
        """
        Calculate current portfolio volatility considering correlations.
        
        Args:
            regime: Current market regime
        
        Returns:
            float: Portfolio volatility (annualized)
        """
        if not self.portfolio_state.open_positions:
            return 0.0
        
        positions = list(self.portfolio_state.open_positions.values())
        
        # Get symbol volatilities (simplified - would need actual ATR data)
        symbol_volatilities = {
            pos.symbol: 0.01  # Placeholder - should use actual ATR/price
            for pos in positions
        }
        
        return self.risk_calculator.calculate_portfolio_volatility(
            positions, symbol_volatilities, regime
        )
    
    def get_diversification_benefit(self, regime: str = 'RANGING') -> float:
        """
        Calculate diversification benefit of current portfolio.
        
        Args:
            regime: Current market regime
        
        Returns:
            float: Diversification benefit [0, 1]
        """
        if not self.portfolio_state.open_positions:
            return 1.0
        
        positions = list(self.portfolio_state.open_positions.values())
        
        # Get symbol volatilities (simplified)
        symbol_volatilities = {
            pos.symbol: 0.01
            for pos in positions
        }
        
        return self.risk_calculator.calculate_diversification_benefit(
            positions, symbol_volatilities, regime
        )
    
    def sync_with_mt5_account(self) -> bool:
        """
        Manually sync portfolio state with MT5 account.
        
        Updates:
        - Account balance (total_capital)
        - Account equity
        - Open positions from MT5
        - Unrealized PnL
        
        Returns:
            True if sync successful, False otherwise
        """
        return self.mt5_sync.sync_portfolio_state(self.portfolio_state)
    
    def get_mt5_sync_stats(self) -> Dict:
        """
        Get MT5 synchronization statistics.
        
        Returns dict with:
        - mt5_available: bool
        - mt5_initialized: bool
        - last_sync_time: str (ISO timestamp)
        """
        return self.mt5_sync.get_sync_stats()
