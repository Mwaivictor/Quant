"""
Strategy Intelligence & Performance Analytics - Enterprise Grade

Implements per-strategy performance tracking and adaptive intelligence:
1. Real-time expectancy and edge estimation
2. Strategy-specific risk profiles and limits
3. Performance attribution (alpha vs beta)
4. Drawdown contribution analysis
5. Automatic parameter adaptation based on regime
6. Strategy health scoring

Enables granular strategy-level risk management and optimization.

Version: 2.0.0 (Enterprise)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


class StrategyHealthStatus(Enum):
    """Strategy health classification"""
    EXCELLENT = "excellent"  # Consistent edge, low drawdown
    GOOD = "good"  # Positive edge, acceptable drawdown
    MARGINAL = "marginal"  # Minimal edge, needs monitoring
    POOR = "poor"  # No edge or high drawdown
    CRITICAL = "critical"  # Severe issues, should be disabled


@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics"""
    strategy_id: str
    
    # Basic statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # (total_wins / total_losses)
    expectancy: float  # E = p*W - (1-p)*L
    
    # Risk metrics
    max_drawdown: float
    current_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_losses: int
    
    # Risk-adjusted performance
    sharpe_ratio: float
    calmar_ratio: float  # Return / MaxDD
    
    # Recent performance (last 30 days)
    recent_win_rate: float
    recent_expectancy: float
    recent_sharpe: float
    
    # EWMA metrics (adaptive, non-stationary)
    ewma_win_rate: float  # Exponentially weighted win rate
    ewma_expectancy: float  # Exponentially weighted expectancy
    ewma_alpha: float  # Smoothing factor used
    
    # Edge confidence
    edge_confidence: float  # 0.0 to 1.0 (statistical significance)
    sample_size_adequate: bool
    
    # Health status
    health_status: StrategyHealthStatus
    health_score: float  # 0.0 to 1.0
    
    # Regime-specific performance
    regime_metrics: Dict[str, dict] = field(default_factory=dict)  # Per-regime stats
    current_regime: Optional[str] = None
    
    # Edge decay detection
    edge_is_decaying: bool = False
    edge_decay_pct: float = 0.0  # % change in recent vs full expectancy
    edge_decay_multiplier: float = 1.0  # Position size multiplier if decaying
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            'strategy_id': self.strategy_id,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'recent_win_rate': self.recent_win_rate,
            'recent_expectancy': self.recent_expectancy,
            'ewma_win_rate': self.ewma_win_rate,
            'ewma_expectancy': self.ewma_expectancy,
            'regime_metrics': self.regime_metrics,
            'current_regime': self.current_regime,
            'edge_is_decaying': self.edge_is_decaying,
            'edge_decay_pct': self.edge_decay_pct,
            'edge_decay_multiplier': self.edge_decay_multiplier,
            'edge_confidence': self.edge_confidence,
            'health_status': self.health_status.value,
            'health_score': self.health_score,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class TradeRecord:
    """Single trade record for analysis"""
    strategy_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    size: float
    commission: float = 0.0


class StrategyPerformanceTracker:
    """
    Tracks detailed performance for individual strategies.
    
    Maintains trade history and calculates comprehensive metrics.
    """
    
    def __init__(
        self,
        strategy_id: str,
        lookback_days: int = 90,
        recent_period_days: int = 30,
        min_trades_for_significance: int = 30,
        use_ewma: bool = True,
        ewma_alpha: float = 0.05,
        ewma_halflife_days: float = 30.0,
        track_regime_specific: bool = True
    ):
        self.strategy_id = strategy_id
        self.lookback_days = lookback_days
        self.recent_period_days = recent_period_days
        self.min_trades_for_significance = min_trades_for_significance
        self.use_ewma = use_ewma
        self.ewma_alpha = ewma_alpha
        self.ewma_halflife_days = ewma_halflife_days
        self.track_regime_specific = track_regime_specific
        
        # Trade history
        self.trade_history: List[TradeRecord] = []
        
        # Running statistics
        self.cumulative_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 0
        
        # EWMA state (exponentially weighted moving averages)
        self.ewma_win_rate: float = 0.5  # Initialize at 50%
        self.ewma_expectancy: float = 0.0
        self.ewma_initialized: bool = False
        
        # Regime-specific tracking
        self.regime_trades: Dict[str, List[TradeRecord]] = {}
        self.current_regime: Optional[str] = None
    
    def record_trade(self, trade: TradeRecord, regime: Optional[str] = None):
        """Record completed trade and update EWMA statistics"""
        self.trade_history.append(trade)
        self.current_regime = regime
        
        # Update running stats
        self.cumulative_pnl += trade.pnl
        self.peak_pnl = max(self.peak_pnl, self.cumulative_pnl)
        
        if trade.pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Update EWMA (exponentially weighted moving average)
        if self.use_ewma:
            is_win = 1.0 if trade.pnl > 0 else 0.0
            
            if not self.ewma_initialized:
                # First trade: initialize EWMA
                self.ewma_win_rate = is_win
                self.ewma_expectancy = trade.pnl
                self.ewma_initialized = True
            else:
                # Update EWMA: new_value = alpha * current + (1 - alpha) * old_value
                self.ewma_win_rate = self.ewma_alpha * is_win + (1 - self.ewma_alpha) * self.ewma_win_rate
                self.ewma_expectancy = self.ewma_alpha * trade.pnl + (1 - self.ewma_alpha) * self.ewma_expectancy
        
        # Track regime-specific performance
        if self.track_regime_specific and regime:
            if regime not in self.regime_trades:
                self.regime_trades[regime] = []
            self.regime_trades[regime].append(trade)
            
            # Trim old regime trades
            cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days)
            self.regime_trades[regime] = [t for t in self.regime_trades[regime] if t.exit_time >= cutoff_date]
        
        # Trim old trades
        cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days)
        self.trade_history = [t for t in self.trade_history if t.exit_time >= cutoff_date]
    
    def calculate_metrics(self) -> StrategyMetrics:
        """Calculate comprehensive strategy metrics with EWMA and regime-specific analysis"""
        if not self.trade_history:
            return self._empty_metrics()
        
        # Basic statistics
        total_trades = len(self.trade_history)
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trade_history)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Expectancy
        if avg_loss != 0:
            expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
        else:
            expectancy = win_rate * avg_win
        
        # Drawdown
        equity_curve = np.cumsum([t.pnl for t in self.trade_history])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown_curve = equity_curve - running_max
        max_drawdown = abs(np.min(drawdown_curve)) if len(drawdown_curve) > 0 else 0.0
        current_drawdown = abs(self.peak_pnl - self.cumulative_pnl)
        
        # Risk-adjusted metrics
        returns = np.array([t.return_pct for t in self.trade_history])
        sharpe_ratio = self._calculate_sharpe(returns)
        calmar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else np.inf
        
        # Recent performance
        recent_cutoff = datetime.utcnow() - timedelta(days=self.recent_period_days)
        recent_trades = [t for t in self.trade_history if t.exit_time >= recent_cutoff]
        
        if recent_trades:
            recent_wins = [t for t in recent_trades if t.pnl > 0]
            recent_win_rate = len(recent_wins) / len(recent_trades)
            recent_expectancy = self._calculate_expectancy(recent_trades)
            recent_returns = np.array([t.return_pct for t in recent_trades])
            recent_sharpe = self._calculate_sharpe(recent_returns)
        else:
            recent_win_rate = win_rate
            recent_expectancy = expectancy
            recent_sharpe = sharpe_ratio
        
        # EWMA metrics (use stored values)
        ewma_win_rate = self.ewma_win_rate if self.ewma_initialized else win_rate
        ewma_expectancy = self.ewma_expectancy if self.ewma_initialized else expectancy
        
        # Regime-specific metrics
        regime_metrics = {}
        for regime, trades in self.regime_trades.items():
            if len(trades) >= 5:  # Need minimum sample
                regime_wins = [t for t in trades if t.pnl > 0]
                regime_win_rate = len(regime_wins) / len(trades)
                regime_expectancy = self._calculate_expectancy(trades)
                regime_metrics[regime] = {
                    'trades': len(trades),
                    'win_rate': regime_win_rate,
                    'expectancy': regime_expectancy,
                    'avg_pnl': np.mean([t.pnl for t in trades])
                }
        
        # Edge decay detection
        edge_is_decaying = False
        edge_decay_pct = 0.0
        edge_decay_multiplier = 1.0
        
        if len(recent_trades) >= 20 and abs(expectancy) > 1e-6:
            # Compare recent expectancy vs full period
            edge_decay_pct = (recent_expectancy - expectancy) / abs(expectancy)
            
            # Flag as decaying if recent < full by 30%+
            if edge_decay_pct < -0.30:
                edge_is_decaying = True
                edge_decay_multiplier = 0.5  # Reduce position size to 50%
        
        # Edge confidence (statistical significance)
        edge_confidence = self._calculate_edge_confidence(win_rate, total_trades)
        sample_size_adequate = total_trades >= self.min_trades_for_significance
        
        # Health assessment
        health_status, health_score = self._assess_health(
            win_rate, expectancy, max_drawdown, current_drawdown,
            recent_win_rate, recent_expectancy, edge_confidence,
            edge_is_decaying  # Pass edge decay flag
        )
        
        return StrategyMetrics(
            strategy_id=self.strategy_id,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            consecutive_wins=self.consecutive_wins,
            consecutive_losses=self.consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            recent_win_rate=recent_win_rate,
            recent_expectancy=recent_expectancy,
            recent_sharpe=recent_sharpe,
            ewma_win_rate=ewma_win_rate,
            ewma_expectancy=ewma_expectancy,
            ewma_alpha=self.ewma_alpha,
            regime_metrics=regime_metrics,
            current_regime=self.current_regime,
            edge_is_decaying=edge_is_decaying,
            edge_decay_pct=edge_decay_pct,
            edge_decay_multiplier=edge_decay_multiplier,
            edge_confidence=edge_confidence,
            sample_size_adequate=sample_size_adequate,
            health_status=health_status,
            health_score=health_score,
            last_updated=datetime.utcnow()
        )
    
    def _calculate_expectancy(self, trades: List[TradeRecord]) -> float:
        """Calculate expectancy for trade list"""
        if not trades:
            return 0.0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if np.std(returns) == 0:
            return 0.0
        
        # Annualize assuming daily trades
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        return sharpe
    
    def _calculate_edge_confidence(self, win_rate: float, sample_size: int) -> float:
        """
        Calculate confidence in positive edge using binomial test.
        
        H0: win_rate = 0.5 (no edge)
        Returns p-value transformed to confidence (1 - p)
        """
        if sample_size < 10:
            return 0.0
        
        # Binomial test for win rate > 0.5
        from scipy.stats import binom
        
        wins = int(win_rate * sample_size)
        p_value = 1 - binom.cdf(wins - 1, sample_size, 0.5)
        
        # Transform to confidence
        confidence = 1 - p_value
        
        return confidence
    
    def _assess_health(
        self,
        win_rate: float,
        expectancy: float,
        max_drawdown: float,
        current_drawdown: float,
        recent_win_rate: float,
        recent_expectancy: float,
        edge_confidence: float,
        edge_is_decaying: bool = False
    ) -> Tuple[StrategyHealthStatus, float]:
        """
        Assess strategy health and assign score.
        
        Includes edge decay detection for non-stationary markets.
        
        Returns:
            (StrategyHealthStatus, health_score)
        """
        # Scoring components (each 0 to 1)
        score_components = []
        
        # 1. Win rate score
        if win_rate >= 0.60:
            score_components.append(1.0)
        elif win_rate >= 0.55:
            score_components.append(0.8)
        elif win_rate >= 0.52:
            score_components.append(0.6)
        elif win_rate >= 0.50:
            score_components.append(0.4)
        else:
            score_components.append(0.0)
        
        # 2. Expectancy score (normalized to -10 to +10 range)
        expectancy_normalized = np.clip(expectancy / 10.0, -1.0, 1.0)
        expectancy_score = (expectancy_normalized + 1.0) / 2.0
        score_components.append(expectancy_score)
        
        # 3. Drawdown score (penalize large drawdowns)
        if max_drawdown == 0:
            dd_score = 1.0
        else:
            dd_score = max(0.0, 1.0 - max_drawdown / 1000.0)  # Assume $1000 is severe
        score_components.append(dd_score)
        
        # 4. Current drawdown score
        if current_drawdown == 0:
            current_dd_score = 1.0
        else:
            current_dd_score = max(0.0, 1.0 - current_drawdown / (max_drawdown + 1e-6))
        score_components.append(current_dd_score)
        
        # 5. Recent performance score (is strategy improving or degrading?)
        performance_trend = (recent_expectancy - expectancy) / (abs(expectancy) + 1e-6)
        trend_score = np.clip((performance_trend + 0.5) * 0.5, 0.0, 1.0)
        score_components.append(trend_score)
        
        # 6. Edge confidence score
        score_components.append(edge_confidence)
        
        # Overall health score (weighted average)
        weights = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]  # Expectancy weighted highest
        health_score = sum(w * s for w, s in zip(weights, score_components))
        
        # Apply edge decay penalty (if edge is deteriorating)
        if edge_is_decaying:
            health_score *= 0.7  # 30% penalty for edge decay
        
        # Determine status
        if health_score >= 0.80:
            status = StrategyHealthStatus.EXCELLENT
        elif health_score >= 0.65:
            status = StrategyHealthStatus.GOOD
        elif health_score >= 0.50:
            status = StrategyHealthStatus.MARGINAL
        elif health_score >= 0.30:
            status = StrategyHealthStatus.POOR
        else:
            status = StrategyHealthStatus.CRITICAL
        
        return status, health_score
    
    def _empty_metrics(self) -> StrategyMetrics:
        """Return empty metrics"""
        return StrategyMetrics(
            strategy_id=self.strategy_id,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            max_consecutive_losses=0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            recent_win_rate=0.0,
            recent_expectancy=0.0,
            recent_sharpe=0.0,
            ewma_win_rate=0.5,
            ewma_expectancy=0.0,
            ewma_alpha=self.ewma_alpha,
            regime_metrics={},
            current_regime=None,
            edge_is_decaying=False,
            edge_decay_pct=0.0,
            edge_decay_multiplier=1.0,
            edge_confidence=0.0,
            sample_size_adequate=False,
            health_status=StrategyHealthStatus.CRITICAL,
            health_score=0.0,
            last_updated=datetime.utcnow()
        )


class StrategyIntelligenceEngine:
    """
    Orchestrates strategy intelligence across all strategies.
    
    Provides:
    - Multi-strategy performance comparison
    - Resource allocation recommendations
    - Automatic strategy disabling
    - Parameter adaptation suggestions
    """
    
    def __init__(
        self,
        min_health_score: float = 0.30,
        auto_disable_critical: bool = True
    ):
        self.min_health_score = min_health_score
        self.auto_disable_critical = auto_disable_critical
        
        # Strategy trackers: {strategy_id: StrategyPerformanceTracker}
        self.trackers: Dict[str, StrategyPerformanceTracker] = {}
        
        # Disabled strategies
        self.disabled_strategies: Set[str] = set()
    
    def register_strategy(self, strategy_id: str):
        """Register new strategy for tracking"""
        if strategy_id not in self.trackers:
            self.trackers[strategy_id] = StrategyPerformanceTracker(strategy_id)
    
    def record_trade(self, trade: TradeRecord):
        """Record trade for strategy"""
        if trade.strategy_id not in self.trackers:
            self.register_strategy(trade.strategy_id)
        
        self.trackers[trade.strategy_id].record_trade(trade)
        
        # Check if strategy should be disabled
        if self.auto_disable_critical:
            metrics = self.trackers[trade.strategy_id].calculate_metrics()
            if metrics.health_status == StrategyHealthStatus.CRITICAL:
                self.disabled_strategies.add(trade.strategy_id)
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get metrics for single strategy"""
        if strategy_id not in self.trackers:
            return None
        
        return self.trackers[strategy_id].calculate_metrics()
    
    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return {
            strategy_id: tracker.calculate_metrics()
            for strategy_id, tracker in self.trackers.items()
        }
    
    def is_strategy_enabled(self, strategy_id: str) -> bool:
        """Check if strategy is enabled"""
        if strategy_id in self.disabled_strategies:
            return False
        
        # Check health score
        metrics = self.get_strategy_metrics(strategy_id)
        if metrics and metrics.health_score < self.min_health_score:
            return False
        
        return True
    
    def get_capital_allocation_weights(self) -> Dict[str, float]:
        """
        Recommend capital allocation across strategies.
        
        Weights based on health scores (Kelly-like weighting).
        
        Returns:
            {strategy_id: weight} where sum(weights) = 1.0
        """
        all_metrics = self.get_all_strategy_metrics()
        
        # Filter enabled strategies
        enabled_metrics = {
            sid: m for sid, m in all_metrics.items()
            if sid not in self.disabled_strategies and m.health_score >= self.min_health_score
        }
        
        if not enabled_metrics:
            return {}
        
        # Weight by health score (could also use Sharpe, expectancy, etc.)
        total_score = sum(m.health_score for m in enabled_metrics.values())
        
        if total_score == 0:
            # Equal weight
            return {sid: 1.0 / len(enabled_metrics) for sid in enabled_metrics}
        
        weights = {
            sid: m.health_score / total_score
            for sid, m in enabled_metrics.items()
        }
        
        return weights
    
    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """
        Rank strategies by health score.
        
        Returns:
            List of (strategy_id, health_score) sorted descending
        """
        all_metrics = self.get_all_strategy_metrics()
        rankings = [(sid, m.health_score) for sid, m in all_metrics.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def suggest_parameter_adjustments(self, strategy_id: str) -> Dict[str, str]:
        """
        Suggest parameter adjustments based on performance.
        
        Returns:
            {parameter: suggestion_text}
        """
        metrics = self.get_strategy_metrics(strategy_id)
        if not metrics:
            return {}
        
        suggestions = {}
        
        # Low win rate - tighten entry criteria
        if metrics.win_rate < 0.50:
            suggestions['entry_criteria'] = "Tighten entry criteria (higher confirmation threshold)"
        
        # High drawdown - reduce position size
        if metrics.current_drawdown > metrics.max_drawdown * 0.5:
            suggestions['position_size'] = "Reduce position sizing by 30-50%"
        
        # Consecutive losses - pause trading
        if metrics.consecutive_losses >= 5:
            suggestions['trading_pause'] = "Consider pausing trading after 5 consecutive losses"
        
        # Declining recent performance
        if metrics.recent_expectancy < metrics.expectancy * 0.5:
            suggestions['regime_adaptation'] = "Recent performance degraded - check if market regime changed"
        
        # Low edge confidence - collect more data
        if not metrics.sample_size_adequate:
            suggestions['sample_size'] = f"Only {metrics.total_trades} trades - collect more data for statistical significance"
        
        return suggestions
