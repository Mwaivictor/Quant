"""
Risk & Portfolio Manager Configuration

Defines risk thresholds, position sizing parameters, and kill switch limits.
"""

from dataclasses import dataclass, field
from typing import Dict
import hashlib
import json


@dataclass
class RPMConfig:
    """
    Risk & Portfolio Manager Configuration.
    
    Controls all risk parameters, position sizing, and portfolio constraints.
    """
    
    # ========================================
    # CAPITAL & RISK MANAGEMENT
    # ========================================
    
    total_capital: float = 100000.0  # Starting capital
    risk_per_trade: float = 0.01  # 1% of capital per trade
    
    # ========================================
    # KILL SWITCHES - Absolute Halt Triggers
    # ========================================
    
    # Drawdown limits
    max_drawdown: float = 0.10  # 10% drawdown triggers halt
    max_daily_drawdown: float = 0.03  # 3% daily drawdown triggers halt
    
    # Loss limits (as percentage of capital)
    daily_loss_limit: float = 0.02  # 2% of capital max daily loss before halt
    weekly_loss_limit: float = 0.05  # 5% of capital max weekly loss before halt
    
    # Volatility shocks
    extreme_volatility_threshold: float = 3.0  # 3x normal vol triggers halt
    volatility_spike_window: int = 10  # Bars to detect volatility spike
    
    # Model confidence floor
    min_confidence_threshold: float = 0.60  # Below this, halt trading
    
    # ========================================
    # POSITION SIZING - Volatility Scaled
    # ========================================
    
    # ATR-based sizing
    atr_window: int = 14  # ATR calculation window
    atr_multiplier: float = 1.5  # Stop distance = ATR * multiplier
    
    # Kelly Criterion (ADAPTIVE v2.0.0)
    kelly_enabled: bool = True
    kelly_safety_factor: float = 0.25  # Fractional Kelly (quarter Kelly)
    kelly_use_adaptive_cap: bool = True  # Enable regime-aware caps
    kelly_base_max_pct: float = 0.01  # 1% base cap for TRENDING regime
    kelly_min_win_rate: float = 0.51  # Require 51%+ win rate (positive edge)
    kelly_min_sample_size: int = 30  # Require 30+ trades for confidence
    
    # Kelly regime multipliers (applied to kelly_base_max_pct)
    kelly_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'TRENDING': 1.0,  # 1.0% (full Kelly - aggressive)
        'RANGING': 0.8,   # 0.8% (moderate)
        'VOLATILE': 0.5,  # 0.5% (conservative)
        'STRESSED': 0.2,  # 0.2% (defensive)
        'CRISIS': 0.1,    # 0.1% (survival mode)
    })
    
    # ========================================
    # ADAPTIVE EDGE TRACKING (Non-Stationary)
    # ========================================
    
    # EWMA parameters (exponentially weighted moving average)
    edge_use_ewma: bool = True  # Use EWMA instead of simple average
    edge_ewma_halflife_days: float = 30.0  # Half-life for edge decay (30 days)
    edge_ewma_alpha: float = 0.05  # Smoothing factor (0.05 = ~20 period EMA)
    
    # Regime-conditional tracking
    edge_regime_specific: bool = True  # Track edge separately per regime
    edge_min_trades_per_regime: int = 10  # Min trades to trust regime-specific edge
    
    # Edge decay detection
    edge_decay_threshold_pct: float = 0.30  # Alert if edge drops 30%
    edge_decay_lookback_trades: int = 20  # Compare last 20 vs full history
    edge_auto_reduce_on_decay: bool = True  # Reduce position size on edge decay
    edge_decay_multiplier: float = 0.5  # Reduce to 50% when edge decays
    
    # Volatility-adjusted expectancy
    edge_vol_adjusted: bool = True  # Adjust expectancy by volatility regime
    edge_vol_penalty_stressed: float = 0.7  # Reduce edge estimate by 30% in stress
    
    # Confidence weighting
    confidence_scaling: bool = True
    confidence_min_multiplier: float = 0.5  # Min size at low confidence
    confidence_max_multiplier: float = 1.5  # Max size at high confidence
    
    # Regime adjustments
    regime_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'TRENDING': 1.2,  # Increase size in trends
        'RANGING': 1.0,  # Normal size in ranges
        'VOLATILE': 0.7,  # Reduce size in volatility
        'STRESSED': 0.3,  # Minimal size in stress
    })
    
    # ========================================
    # LIQUIDITY CONSTRAINTS
    # ========================================
    
    # Market impact limits
    max_adv_pct: float = 0.01  # Max 1% of average daily volume
    max_spread_bps: float = 20.0  # Max 20 bps spread
    max_market_impact_pct: float = 0.005  # Max 0.5% market impact
    impact_coefficient: float = 0.1  # Almgren-Chriss impact coefficient
    min_adv_units: float = 10000.0  # Minimum liquidity threshold
    
    # ========================================
    # PORTFOLIO CONSTRAINTS
    # ========================================
    
    # Symbol exposure limits
    max_symbol_exposure_units: float = 200000.0  # Max units per symbol
    max_symbol_exposure_pct: float = 0.30  # Max 30% of capital per symbol
    
    # Currency exposure limits
    max_currency_exposure_pct: float = 0.50  # Max 50% net exposure per currency
    
    # Total exposure limits
    max_total_exposure_pct: float = 1.0  # Max 100% gross exposure
    max_net_exposure_pct: float = 0.60  # Max 60% net exposure
    
    # Correlation management
    max_correlation_exposure: float = 0.70  # Reduce size if corr > 0.70
    correlation_penalty: float = 0.5  # Reduce by 50% if highly correlated
    
    # Position limits
    max_concurrent_positions: int = 8  # Max open positions
    max_positions_per_symbol: int = 1  # Max positions per symbol
    
    # ========================================
    # REGIME-SPECIFIC CONSTRAINTS
    # ========================================
    
    stressed_regime_restrictions: Dict[str, any] = field(default_factory=lambda: {
        'max_new_positions': 2,  # Max new positions in stress
        'position_size_multiplier': 0.3,  # 30% of normal size
        'min_confidence': 0.75,  # Higher confidence required
    })
    
    volatile_regime_restrictions: Dict[str, any] = field(default_factory=lambda: {
        'max_new_positions': 4,
        'position_size_multiplier': 0.7,
        'min_confidence': 0.70,
    })
    
    # ========================================
    # TIMING & COOLDOWNS
    # ========================================
    
    # Trade spacing
    min_time_between_trades_seconds: int = 300  # 5 minutes between trades
    min_time_between_same_symbol_seconds: int = 3600  # 1 hour for same symbol
    
    # Halt recovery
    halt_cooldown_seconds: int = 3600  # 1 hour before resuming after halt
    
    # ========================================
    # MONITORING & HEALTH
    # ========================================
    
    # Health checks
    enable_health_monitoring: bool = True
    health_check_interval_seconds: int = 60
    
    # Metrics tracking
    track_rejection_reasons: bool = True
    track_sizing_adjustments: bool = True
    
    # Logging
    log_all_decisions: bool = True
    log_portfolio_state: bool = True
    
    # ========================================
    # ADVANCED FEATURES
    # ========================================
    
    # Risk override (manual intervention)
    allow_risk_override: bool = False
    
    # Dynamic sizing
    dynamic_sizing_enabled: bool = True
    
    # Adaptive thresholds
    adaptive_thresholds_enabled: bool = False
    
    # ========================================
    # VALIDATION & HASHING
    # ========================================
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        
        # Capital validation
        if self.total_capital <= 0:
            raise ValueError(f"total_capital must be > 0, got {self.total_capital}")
        
        if not (0 < self.risk_per_trade <= 0.05):
            raise ValueError(f"risk_per_trade must be in (0, 0.05], got {self.risk_per_trade}")
        
        # Drawdown validation
        if not (0 < self.max_drawdown <= 1.0):
            raise ValueError(f"max_drawdown must be in (0, 1.0], got {self.max_drawdown}")
        
        if not (0 < self.max_daily_drawdown <= self.max_drawdown):
            raise ValueError(
                f"max_daily_drawdown ({self.max_daily_drawdown}) must be <= max_drawdown ({self.max_drawdown})"
            )
        
        # Loss limits validation (now percentages)
        if not (0 < self.daily_loss_limit <= 1.0):
            raise ValueError(f"daily_loss_limit must be in (0, 1.0], got {self.daily_loss_limit}")
        
        if not (0 < self.weekly_loss_limit <= 1.0):
            raise ValueError(f"weekly_loss_limit must be in (0, 1.0], got {self.weekly_loss_limit}")
        
        # Confidence validation
        if not (0 < self.min_confidence_threshold <= 1.0):
            raise ValueError(
                f"min_confidence_threshold must be in (0, 1.0], got {self.min_confidence_threshold}"
            )
        
        # Sizing validation
        if self.atr_window <= 0:
            raise ValueError(f"atr_window must be > 0, got {self.atr_window}")
        
        if self.atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {self.atr_multiplier}")
        
        # Exposure validation
        if not (0 < self.max_symbol_exposure_pct <= 1.0):
            raise ValueError(
                f"max_symbol_exposure_pct must be in (0, 1.0], got {self.max_symbol_exposure_pct}"
            )
        
        if not (0 < self.max_currency_exposure_pct <= 1.0):
            raise ValueError(
                f"max_currency_exposure_pct must be in (0, 1.0], got {self.max_currency_exposure_pct}"
            )
        
        # Position limits validation
        if self.max_concurrent_positions <= 0:
            raise ValueError(
                f"max_concurrent_positions must be > 0, got {self.max_concurrent_positions}"
            )
        
        # Kelly Criterion validation
        if not (0 < self.kelly_safety_factor <= 1.0):
            raise ValueError(
                f"kelly_safety_factor must be in (0, 1.0], got {self.kelly_safety_factor}"
            )
        
        if not (0 < self.kelly_base_max_pct <= 0.05):
            raise ValueError(
                f"kelly_base_max_pct must be in (0, 0.05], got {self.kelly_base_max_pct}"
            )
        
        if not (0.5 <= self.kelly_min_win_rate < 1.0):
            raise ValueError(
                f"kelly_min_win_rate must be in [0.5, 1.0), got {self.kelly_min_win_rate}"
            )
        
        if self.kelly_min_sample_size < 10:
            raise ValueError(
                f"kelly_min_sample_size must be >= 10, got {self.kelly_min_sample_size}"
            )
        
        # Validate regime multipliers are between 0 and 1
        for regime, multiplier in self.kelly_regime_multipliers.items():
            if not (0 < multiplier <= 1.0):
                raise ValueError(
                    f"kelly_regime_multipliers[{regime}] must be in (0, 1.0], got {multiplier}"
                )
        
        # Adaptive edge tracking validation
        if self.edge_ewma_halflife_days <= 0:
            raise ValueError(
                f"edge_ewma_halflife_days must be > 0, got {self.edge_ewma_halflife_days}"
            )
        
        if not (0 < self.edge_ewma_alpha <= 1.0):
            raise ValueError(
                f"edge_ewma_alpha must be in (0, 1.0], got {self.edge_ewma_alpha}"
            )
        
        if not (0 < self.edge_decay_threshold_pct <= 1.0):
            raise ValueError(
                f"edge_decay_threshold_pct must be in (0, 1.0], got {self.edge_decay_threshold_pct}"
            )
        
        if not (0 < self.edge_decay_multiplier <= 1.0):
            raise ValueError(
                f"edge_decay_multiplier must be in (0, 1.0], got {self.edge_decay_multiplier}"
            )
        
        # Liquidity constraints validation
        if not (0 < self.max_adv_pct <= 0.1):
            raise ValueError(
                f"max_adv_pct must be in (0, 0.1], got {self.max_adv_pct}"
            )
        
        if self.max_spread_bps <= 0:
            raise ValueError(
                f"max_spread_bps must be > 0, got {self.max_spread_bps}"
            )
        
        return True
    
    def get_config_hash(self) -> str:
        """
        Generate deterministic hash of configuration.
        
        Used for versioning and validation.
        
        Returns:
            str: SHA256 hash of configuration
        """
        config_dict = {
            'total_capital': self.total_capital,
            'risk_per_trade': self.risk_per_trade,
            'max_drawdown': self.max_drawdown,
            'max_daily_drawdown': self.max_daily_drawdown,
            'daily_loss_limit': self.daily_loss_limit,
            'weekly_loss_limit': self.weekly_loss_limit,
            'extreme_volatility_threshold': self.extreme_volatility_threshold,
            'min_confidence_threshold': self.min_confidence_threshold,
            'atr_window': self.atr_window,
            'atr_multiplier': self.atr_multiplier,
            'confidence_scaling': self.confidence_scaling,
            'regime_adjustments': self.regime_adjustments,
            'kelly_safety_factor': self.kelly_safety_factor,
            'kelly_base_max_pct': self.kelly_base_max_pct,
            'kelly_regime_multipliers': self.kelly_regime_multipliers,
            'edge_use_ewma': self.edge_use_ewma,
            'edge_ewma_halflife_days': self.edge_ewma_halflife_days,
            'edge_regime_specific': self.edge_regime_specific,
            'max_adv_pct': self.max_adv_pct,
            'max_spread_bps': self.max_spread_bps,
            'max_symbol_exposure_pct': self.max_symbol_exposure_pct,
            'max_currency_exposure_pct': self.max_currency_exposure_pct,
            'max_concurrent_positions': self.max_concurrent_positions,
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'total_capital': float(self.total_capital),
            'risk_per_trade': float(self.risk_per_trade),
            'max_drawdown': float(self.max_drawdown),
            'max_daily_drawdown': float(self.max_daily_drawdown),
            'daily_loss_limit': float(self.daily_loss_limit),
            'weekly_loss_limit': float(self.weekly_loss_limit),
            'extreme_volatility_threshold': float(self.extreme_volatility_threshold),
            'volatility_spike_window': self.volatility_spike_window,
            'min_confidence_threshold': float(self.min_confidence_threshold),
            'atr_window': self.atr_window,
            'atr_multiplier': float(self.atr_multiplier),
            'confidence_scaling': bool(self.confidence_scaling),
            'kelly_enabled': bool(self.kelly_enabled),
            'kelly_safety_factor': float(self.kelly_safety_factor),
            'kelly_use_adaptive_cap': bool(self.kelly_use_adaptive_cap),
            'kelly_base_max_pct': float(self.kelly_base_max_pct),
            'kelly_min_win_rate': float(self.kelly_min_win_rate),
            'kelly_min_sample_size': self.kelly_min_sample_size,
            'kelly_regime_multipliers': {k: float(v) for k, v in self.kelly_regime_multipliers.items()},
            'edge_use_ewma': bool(self.edge_use_ewma),
            'edge_ewma_halflife_days': float(self.edge_ewma_halflife_days),
            'edge_ewma_alpha': float(self.edge_ewma_alpha),
            'edge_regime_specific': bool(self.edge_regime_specific),
            'edge_min_trades_per_regime': self.edge_min_trades_per_regime,
            'edge_decay_threshold_pct': float(self.edge_decay_threshold_pct),
            'edge_decay_lookback_trades': self.edge_decay_lookback_trades,
            'edge_auto_reduce_on_decay': bool(self.edge_auto_reduce_on_decay),
            'edge_decay_multiplier': float(self.edge_decay_multiplier),
            'edge_vol_adjusted': bool(self.edge_vol_adjusted),
            'edge_vol_penalty_stressed': float(self.edge_vol_penalty_stressed),
            'max_adv_pct': float(self.max_adv_pct),
            'max_spread_bps': float(self.max_spread_bps),
            'max_market_impact_pct': float(self.max_market_impact_pct),
            'impact_coefficient': float(self.impact_coefficient),
            'min_adv_units': float(self.min_adv_units),
            'confidence_min_multiplier': float(self.confidence_min_multiplier),
            'confidence_max_multiplier': float(self.confidence_max_multiplier),
            'regime_adjustments': {k: float(v) for k, v in self.regime_adjustments.items()},
            'max_symbol_exposure_units': float(self.max_symbol_exposure_units),
            'max_symbol_exposure_pct': float(self.max_symbol_exposure_pct),
            'max_currency_exposure_pct': float(self.max_currency_exposure_pct),
            'max_total_exposure_pct': float(self.max_total_exposure_pct),
            'max_net_exposure_pct': float(self.max_net_exposure_pct),
            'max_correlation_exposure': float(self.max_correlation_exposure),
            'correlation_penalty': float(self.correlation_penalty),
            'max_concurrent_positions': self.max_concurrent_positions,
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'stressed_regime_restrictions': self.stressed_regime_restrictions,
            'volatile_regime_restrictions': self.volatile_regime_restrictions,
            'min_time_between_trades_seconds': self.min_time_between_trades_seconds,
            'min_time_between_same_symbol_seconds': self.min_time_between_same_symbol_seconds,
            'halt_cooldown_seconds': self.halt_cooldown_seconds,
            'enable_health_monitoring': bool(self.enable_health_monitoring),
            'track_rejection_reasons': bool(self.track_rejection_reasons),
            'track_sizing_adjustments': bool(self.track_sizing_adjustments),
            'log_all_decisions': bool(self.log_all_decisions),
            'allow_risk_override': bool(self.allow_risk_override),
            'dynamic_sizing_enabled': bool(self.dynamic_sizing_enabled),
            'adaptive_thresholds_enabled': bool(self.adaptive_thresholds_enabled),
            'config_hash': self.get_config_hash(),
        }
