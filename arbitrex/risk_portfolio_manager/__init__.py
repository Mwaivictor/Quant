"""
Risk & Portfolio Manager (RPM)

Critical gatekeeper between Signal Generation Engine and Execution Engine.
Enforces capital preservation, portfolio constraints, and market-aware risk control.

Core Principle:
    "Signals propose trades. RPM decides survival."
    
Philosophy:
    - Capital preservation first
    - Portfolio-level risk management
    - Adaptive to volatility, confidence, and regime
    - Absolute veto authority over all trades
    - Profits are optional, survival is mandatory

Responsibilities:
    1. Position Sizing (volatility-scaled, confidence-weighted, regime-adjusted)
    2. Portfolio Constraints (exposure limits, correlation management)
    3. Loss Control (kill switches, drawdown protection)
    4. Final Trade Approval (execution-ready instructions)
    5. State Management (portfolio tracking, audit trail)

Flow:
    Signal Engine → Risk & Portfolio Manager → Execution Engine
    
Authority:
    RPM has absolute veto power. No trade may bypass it.

Version: 2.0.0 (Enterprise)
- Advanced kill switches (rejection-velocity, exposure-velocity, per-strategy)
- Adaptive thresholds (regime-aware, percentile-based, stress-adjusted)
- Portfolio variance targeting (VaR/CVaR, fat-tail modeling)
- Factor/sector exposure tracking
- Strategy intelligence & health scoring
- Enterprise observability (structured logging, metrics, alerting)
- Stress testing & crisis validation
"""

from arbitrex.risk_portfolio_manager.engine import RiskPortfolioManager
from arbitrex.risk_portfolio_manager.config import RPMConfig
from arbitrex.risk_portfolio_manager.schemas import (
    ApprovedTrade,
    RejectedTrade,
    TradeDecision,
    PortfolioState,
    RiskMetrics,
    RPMOutput,
    Position,
)
from arbitrex.risk_portfolio_manager.position_sizing import PositionSizer
from arbitrex.risk_portfolio_manager.constraints import PortfolioConstraints

# Unified Kill-Switch System (v3.0 - consolidates kill_switches.py + advanced_kill_switches.py)
from arbitrex.risk_portfolio_manager.kill_switch import (
    KillSwitchManager,
    KillSwitchLevel,
    ResponseAction,
    TriggerReason,
    AlertConfig,
    AlertManager,
    KillSwitchState,
)

# Legacy exports removed - all kill-switch functionality now in KillSwitchManager
# Previously: from kill_switches import KillSwitches
# Previously: from advanced_kill_switches import AdvancedKillSwitchManager, etc.
# Migration: Use KillSwitchManager for all kill-switch operations

from arbitrex.risk_portfolio_manager.state_manager import StateManager, AutoSaveStateManager
from arbitrex.risk_portfolio_manager.order_manager import OrderManager, Order, OrderStatus, OrderType
from arbitrex.risk_portfolio_manager.correlation_risk import (
    CorrelationMatrix,
    PortfolioRiskCalculator,
    CorrelationAwareSizer,
)

from arbitrex.risk_portfolio_manager.mt5_sync import (
    MT5AccountSync,
    create_mt5_synced_portfolio,
)

# v2.0.0 Enterprise Modules
from arbitrex.risk_portfolio_manager.adaptive_thresholds import (
    AdaptiveRiskManager,
    RegimeAwareRiskLimits,
    AdaptiveVolatilityThresholds,
    StressAdjustedLimits,
)
from arbitrex.risk_portfolio_manager.portfolio_risk import (
    PortfolioRiskEngine,
    PortfolioVarianceCalculator,
    VaRCalculator,
    CovarianceMatrixEstimator,
    PortfolioRiskMetrics,
)
from arbitrex.risk_portfolio_manager.factor_exposure import (
    FactorExposureCalculator,
    AssetFactorDatabase,
    PortfolioFactorExposure,
    EquityFactor,
    Sector,
)
from arbitrex.risk_portfolio_manager.strategy_intelligence import (
    StrategyIntelligenceEngine,
    StrategyPerformanceTracker,
    StrategyMetrics,
    StrategyHealthStatus,
)
from arbitrex.risk_portfolio_manager.observability import (
    ObservabilityManager,
    StructuredLogger,
    PrometheusMetrics,
    AlertingSystem,
    CorrelationContext,
    LogLevel,
    EventType,
    AlertSeverity,
)
from arbitrex.risk_portfolio_manager.stress_testing import (
    StressTestEngine,
    HistoricalCrisisLibrary,
    SyntheticStressGenerator,
    VaRBacktester,
    CrisisScenario,
    StressTestResult,
)

__version__ = "2.0.0"  # Enterprise upgrade

__all__ = [
    # Core v1.x components
    'RiskPortfolioManager',
    'RPMConfig',
    'ApprovedTrade',
    'RejectedTrade',
    'TradeDecision',
    'PortfolioState',
    'RiskMetrics',
    'RPMOutput',
    'Position',
    'PositionSizer',
    'PortfolioConstraints',
    
    # Unified Kill-Switch System v3.0 (RECOMMENDED)
    'KillSwitchManager',
    'KillSwitchLevel',
    'ResponseAction',
    'TriggerReason',
    'AlertConfig',
    'AlertManager',
    'KillSwitchState',
    
    # Legacy kill-switches removed (use KillSwitchManager instead)
    # Previously exported: KillSwitches, AdvancedKillSwitchManager, etc.
    
    'StateManager',
    'AutoSaveStateManager',
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'CorrelationMatrix',
    'PortfolioRiskCalculator',
    'CorrelationAwareSizer',
    'MT5AccountSync',
    'create_mt5_synced_portfolio',
    
    # v2.0.0 Enterprise components
    'AdaptiveRiskManager',
    'RegimeAwareRiskLimits',
    'AdaptiveVolatilityThresholds',
    'StressAdjustedLimits',
    
    # Portfolio Risk Math
    'PortfolioRiskEngine',
    'PortfolioVarianceCalculator',
    'VaRCalculator',
    'CovarianceMatrixEstimator',
    'PortfolioRiskMetrics',
    
    # Factor & Sector Exposure
    'FactorExposureCalculator',
    'AssetFactorDatabase',
    'PortfolioFactorExposure',
    'EquityFactor',
    'Sector',
    
    # Strategy Intelligence
    'StrategyIntelligenceEngine',
    'StrategyPerformanceTracker',
    'StrategyMetrics',
    'StrategyHealthStatus',
    
    # Observability
    'ObservabilityManager',
    'StructuredLogger',
    'PrometheusMetrics',
    'AlertingSystem',
    'CorrelationContext',
    'LogLevel',
    'EventType',
    'AlertSeverity',
    
    # Stress Testing
    'StressTestEngine',
    'HistoricalCrisisLibrary',
    'SyntheticStressGenerator',
    'VaRBacktester',
    'CrisisScenario',
    'StressTestResult',
]