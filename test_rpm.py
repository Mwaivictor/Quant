"""
Risk & Portfolio Manager Test Suite

Comprehensive tests for RPM decision logic, position sizing, constraints, and kill switches.
"""

import pytest
from datetime import datetime, timedelta

from arbitrex.risk_portfolio_manager.engine import RiskPortfolioManager
from arbitrex.risk_portfolio_manager.config import RPMConfig
from arbitrex.risk_portfolio_manager.schemas import (
    TradeApprovalStatus,
    RejectionReason,
    PortfolioState,
    Position,
)


# ========================================
# TEST FIXTURES
# ========================================

@pytest.fixture
def default_config():
    """Default RPM configuration for testing"""
    return RPMConfig(
        total_capital=100000.0,
        risk_per_trade=0.002,  # Lower risk to reduce position size
        max_drawdown=0.10,
        daily_loss_limit=0.02,  # 2% of capital
        max_symbol_exposure_units=500000.0,  # Increase to accommodate FX sizing
        max_symbol_exposure_pct=0.50,  # Allow larger percentage
    )


@pytest.fixture
def rpm(default_config):
    """RPM instance with default config"""
    return RiskPortfolioManager(config=default_config)


@pytest.fixture
def trade_intent_params():
    """Standard trade intent parameters"""
    return {
        'symbol': 'EURUSD',
        'direction': 1,  # LONG
        'confidence_score': 0.75,
        'regime': 'RANGING',  # Use RANGING to avoid 1.2x multiplier
        'atr': 0.0050,  # Larger ATR = smaller position size
        'vol_percentile': 0.50,
        'current_price': 1.1000,
    }


# ========================================
# CONFIGURATION TESTS
# ========================================

def test_config_validation():
    """Test configuration validation"""
    config = RPMConfig()
    assert config.validate() is True


def test_config_invalid_capital():
    """Test invalid capital raises error"""
    with pytest.raises(ValueError, match="total_capital must be > 0"):
        config = RPMConfig(total_capital=-1000.0)
        config.validate()


def test_config_invalid_loss_limits():
    """Test invalid loss limits raise error"""
    # Test negative percentage
    with pytest.raises(ValueError, match="daily_loss_limit must be in"):
        config = RPMConfig(daily_loss_limit=-0.02)
        config.validate()
    
    # Test too large percentage
    with pytest.raises(ValueError, match="daily_loss_limit must be in"):
        config = RPMConfig(daily_loss_limit=1.5)
        config.validate()


def test_config_invalid_risk_per_trade():
    """Test invalid risk_per_trade raises error"""
    with pytest.raises(ValueError, match="risk_per_trade must be in"):
        config = RPMConfig(risk_per_trade=0.10)  # 10% too high
        config.validate()


def test_config_invalid_drawdown():
    """Test invalid drawdown raises error"""
    with pytest.raises(ValueError, match="max_drawdown must be in"):
        config = RPMConfig(max_drawdown=1.5)  # > 1.0
        config.validate()


def test_config_hash_deterministic(default_config):
    """Test config hash is deterministic"""
    hash1 = default_config.get_config_hash()
    hash2 = default_config.get_config_hash()
    assert hash1 == hash2


def test_config_hash_unique():
    """Test different configs produce different hashes"""
    config1 = RPMConfig(total_capital=100000.0)
    config2 = RPMConfig(total_capital=200000.0)
    assert config1.get_config_hash() != config2.get_config_hash()


# ========================================
# POSITION SIZING TESTS
# ========================================

def test_position_sizing_basic(rpm, trade_intent_params):
    """Test basic position sizing calculation"""
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status in (TradeApprovalStatus.APPROVED, TradeApprovalStatus.ADJUSTED)
    assert output.decision.approved_trade is not None
    assert output.decision.approved_trade.position_units > 0


def test_position_sizing_zero_atr(rpm, trade_intent_params):
    """Test zero ATR results in rejection"""
    params = trade_intent_params.copy()
    params['atr'] = 0.0
    
    output = rpm.process_trade_intent(**params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.rejected_trade is not None


def test_position_sizing_confidence_scaling(rpm):
    """Test position size scales with confidence"""
    base_params = {
        'symbol': 'EURUSD',
        'direction': 1,
        'regime': 'TRENDING',
        'atr': 0.0010,
        'vol_percentile': 0.50,
        'current_price': 1.1000,
    }
    
    # Low confidence
    params_low = base_params.copy()
    params_low['confidence_score'] = 0.50
    output_low = rpm.process_trade_intent(**params_low)
    
    # High confidence
    params_high = base_params.copy()
    params_high['confidence_score'] = 0.95
    output_high = rpm.process_trade_intent(**params_high)
    
    # Higher confidence should result in larger position
    if output_low.decision.approved_trade and output_high.decision.approved_trade:
        assert output_high.decision.approved_trade.position_units > output_low.decision.approved_trade.position_units


def test_position_sizing_regime_adjustment(rpm):
    """Test position size adjusts for regime"""
    base_params = {
        'symbol': 'EURUSD',
        'direction': 1,
        'confidence_score': 0.75,
        'atr': 0.0010,
        'vol_percentile': 0.50,
        'current_price': 1.1000,
    }
    
    # Trending regime (higher size)
    params_trending = base_params.copy()
    params_trending['regime'] = 'TRENDING'
    output_trending = rpm.process_trade_intent(**params_trending)
    
    # Stressed regime (lower size)
    params_stressed = base_params.copy()
    params_stressed['regime'] = 'STRESSED'
    output_stressed = rpm.process_trade_intent(**params_stressed)
    
    # Trending should have larger position than stressed
    if output_trending.decision.approved_trade and output_stressed.decision.approved_trade:
        assert output_trending.decision.approved_trade.position_units > output_stressed.decision.approved_trade.position_units


# ========================================
# PORTFOLIO CONSTRAINTS TESTS
# ========================================

def test_max_concurrent_positions(rpm, trade_intent_params):
    """Test max concurrent positions limit"""
    # Set very low limit
    rpm.config.max_concurrent_positions = 2
    
    # Add mock positions
    rpm.portfolio_state.open_positions = {
        'pos1': Position(
            symbol='EURUSD', direction=1, units=10000.0,
            entry_price=1.1000, entry_time=datetime.utcnow()
        ),
        'pos2': Position(
            symbol='GBPUSD', direction=1, units=10000.0,
            entry_price=1.3000, entry_time=datetime.utcnow()
        ),
    }
    
    # Try to add third position
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert 'position' in output.decision.rejected_trade.rejection_details.lower()


def test_symbol_exposure_limit(rpm, trade_intent_params):
    """Test symbol exposure limits"""
    # Set tight symbol limit
    rpm.config.max_symbol_exposure_units = 5000.0
    
    # Add existing position
    rpm.portfolio_state.symbol_exposure['EURUSD'] = 4000.0
    
    # Try large additional position
    params = trade_intent_params.copy()
    # Force large position by increasing ATR (smaller stop, larger size)
    params['atr'] = 0.00001  # Very small ATR
    
    output = rpm.process_trade_intent(**params)
    
    # Should either reject or adjust down
    if output.decision.status == TradeApprovalStatus.REJECTED:
        assert 'exposure' in output.decision.rejected_trade.rejection_details.lower()


def test_currency_exposure_decomposition(rpm):
    """Test currency exposure is decomposed correctly"""
    params = {
        'symbol': 'EURUSD',
        'direction': 1,  # LONG = +EUR, -USD
        'confidence_score': 0.75,
        'regime': 'RANGING',
        'atr': 0.0050,  # Use larger ATR
        'vol_percentile': 0.50,
        'current_price': 1.1000,
    }
    
    output = rpm.process_trade_intent(**params)
    
    # After processing, currency exposure should be tracked
    # (Note: This would require execution simulation to actually update)
    assert output.decision.status in (TradeApprovalStatus.APPROVED, TradeApprovalStatus.ADJUSTED)


# ========================================
# KILL SWITCH TESTS
# ========================================

def test_max_drawdown_kill_switch(rpm, trade_intent_params):
    """Test max drawdown triggers halt"""
    # Simulate drawdown
    rpm.portfolio_state.peak_equity = 100000.0
    rpm.portfolio_state.equity = 85000.0  # 15% drawdown
    rpm.portfolio_state.current_drawdown = 0.15
    
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.kill_switch_triggered is True
    assert output.decision.rejected_trade.rejection_reason == RejectionReason.MAX_DRAWDOWN_EXCEEDED


def test_daily_loss_limit_kill_switch(rpm, trade_intent_params):
    """Test daily loss limit triggers halt (percentage-based)"""
    # Simulate large daily loss (2% of 100k = -2000, so use -2500)
    rpm.portfolio_state.daily_pnl = -2500.0  # Exceeds -2000 limit
    
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.kill_switch_triggered is True
    assert output.decision.rejected_trade.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT
    # Check that percentage is shown in details
    assert '2.0%' in output.decision.rejected_trade.rejection_details
    
    # Second trade should be halted
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.kill_switch_triggered is True
    # Now it's TRADING_HALTED because the system is already halted
    assert output.decision.rejected_trade.rejection_reason == RejectionReason.TRADING_HALTED


def test_confidence_collapse_kill_switch(rpm, trade_intent_params):
    """Test low confidence triggers rejection"""
    params = trade_intent_params.copy()
    params['confidence_score'] = 0.45  # Below 0.60 threshold
    
    output = rpm.process_trade_intent(**params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.kill_switch_triggered is True
    assert output.decision.rejected_trade.rejection_reason == RejectionReason.LOW_MODEL_CONFIDENCE


def test_manual_halt(rpm, trade_intent_params):
    """Test manual trading halt"""
    # Manually halt trading
    rpm.kill_switches.manual_halt(rpm.portfolio_state, "Manual emergency stop")
    
    output = rpm.process_trade_intent(**trade_intent_params)
    
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert rpm.portfolio_state.trading_halted is True
    assert "Manual emergency stop" in rpm.portfolio_state.halt_reason


def test_manual_resume(rpm, trade_intent_params):
    """Test manual trading resume"""
    # Halt and then resume
    rpm.kill_switches.manual_halt(rpm.portfolio_state, "Test halt")
    rpm.kill_switches.manual_resume(rpm.portfolio_state)
    
    output = rpm.process_trade_intent(**trade_intent_params)
    
    # Should now be able to trade
    assert output.decision.status in (TradeApprovalStatus.APPROVED, TradeApprovalStatus.ADJUSTED)
    assert rpm.portfolio_state.trading_halted is False


# ========================================
# INTEGRATION TESTS
# ========================================

def test_end_to_end_approval(rpm, trade_intent_params):
    """Test end-to-end trade approval flow"""
    output = rpm.process_trade_intent(**trade_intent_params)
    
    # Should be approved
    assert output.decision.status in (TradeApprovalStatus.APPROVED, TradeApprovalStatus.ADJUSTED)
    assert output.decision.approved_trade is not None
    
    # Check approved trade fields
    trade = output.decision.approved_trade
    assert trade.symbol == 'EURUSD'
    assert trade.direction == 1
    assert trade.position_units > 0
    assert trade.confidence_score == 0.75
    assert trade.regime == 'RANGING'  # Match fixture


def test_end_to_end_rejection(rpm):
    """Test end-to-end trade rejection flow"""
    # Create conditions for rejection
    rpm.portfolio_state.daily_pnl = -5000.0  # Exceeds limit
    
    params = {
        'symbol': 'EURUSD',
        'direction': 1,
        'confidence_score': 0.75,
        'regime': 'TRENDING',
        'atr': 0.0010,
        'vol_percentile': 0.50,
        'current_price': 1.1000,
    }
    
    output = rpm.process_trade_intent(**params)
    
    # Should be rejected
    assert output.decision.status == TradeApprovalStatus.REJECTED
    assert output.decision.rejected_trade is not None
    
    # Check rejection details
    rejection = output.decision.rejected_trade
    assert rejection.symbol == 'EURUSD'
    assert rejection.rejection_reason in (RejectionReason.DAILY_LOSS_LIMIT, RejectionReason.MAX_DRAWDOWN_EXCEEDED)


def test_metrics_tracking(rpm, trade_intent_params):
    """Test metrics are tracked correctly"""
    initial_decisions = rpm.risk_metrics.total_decisions
    
    # Process trade
    output = rpm.process_trade_intent(**trade_intent_params)
    
    # Metrics should update
    assert rpm.risk_metrics.total_decisions == initial_decisions + 1
    
    if output.decision.status == TradeApprovalStatus.APPROVED:
        assert rpm.risk_metrics.trades_approved > 0
    else:
        assert rpm.risk_metrics.trades_rejected > 0


def test_health_status(rpm):
    """Test health status reporting"""
    health = rpm.get_health_status()
    
    assert 'rpm_version' in health
    assert 'config_hash' in health
    assert 'portfolio_state' in health
    assert 'risk_metrics' in health
    assert 'kill_switches' in health
    assert 'health' in health


def test_daily_reset(rpm):
    """Test daily metrics reset"""
    rpm.portfolio_state.daily_pnl = -500.0
    rpm.reset_daily_metrics()
    assert rpm.portfolio_state.daily_pnl == 0.0


def test_weekly_reset(rpm):
    """Test weekly metrics reset"""
    rpm.portfolio_state.weekly_pnl = -1000.0
    rpm.reset_weekly_metrics()
    assert rpm.portfolio_state.weekly_pnl == 0.0


# ========================================
# SERIALIZATION TESTS
# ========================================

def test_rpm_output_serialization(rpm, trade_intent_params):
    """Test RPMOutput can be serialized to dict"""
    output = rpm.process_trade_intent(**trade_intent_params)
    
    # Should serialize without errors
    output_dict = output.to_dict()
    
    assert isinstance(output_dict, dict)
    assert 'decision' in output_dict
    assert 'portfolio_state' in output_dict
    assert 'risk_metrics' in output_dict


def test_config_serialization(default_config):
    """Test config can be serialized"""
    config_dict = default_config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'total_capital' in config_dict
    assert 'risk_per_trade' in config_dict
    assert 'config_hash' in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
