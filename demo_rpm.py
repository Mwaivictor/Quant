"""
Risk & Portfolio Manager Demo

Interactive demonstration of RPM decision-making across various scenarios.
"""

from datetime import datetime
from arbitrex.risk_portfolio_manager import (
    RiskPortfolioManager,
    RPMConfig,
)


def print_header(title: str):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def print_decision(output):
    """Print RPM decision details"""
    decision = output.decision
    
    print(f"Status: {decision.status.value}")
    print(f"Processing Time: {decision.processing_time_ms:.2f}ms")
    
    if decision.kill_switch_triggered:
        print(f"\n🚨 KILL SWITCH TRIGGERED: {decision.kill_switch_reason}")
    
    if decision.approved_trade:
        trade = decision.approved_trade
        print(f"\n✅ TRADE APPROVED:")
        print(f"  Symbol: {trade.symbol}")
        print(f"  Direction: {'LONG' if trade.direction == 1 else 'SHORT'}")
        print(f"  Position Size: {trade.position_units:.2f} units")
        print(f"  Confidence: {trade.confidence_score:.2%}")
        print(f"  Regime: {trade.regime}")
        print(f"\n  Sizing Breakdown:")
        print(f"    Base Units: {trade.base_units:.2f}")
        print(f"    Confidence Adj: {trade.confidence_adjustment:.3f}x")
        print(f"    Regime Adj: {trade.regime_adjustment:.3f}x")
        print(f"    Risk/Trade: ${trade.risk_per_trade:.2f}")
    
    elif decision.rejected_trade:
        rejection = decision.rejected_trade
        print(f"\n❌ TRADE REJECTED:")
        print(f"  Symbol: {rejection.symbol}")
        print(f"  Direction: {'LONG' if rejection.direction == 1 else 'SHORT'}")
        print(f"  Reason: {rejection.rejection_reason.value}")
        print(f"  Details: {rejection.rejection_details}")
    
    if decision.portfolio_constraint_violations:
        print(f"\n⚠️  Constraint Violations:")
        for violation in decision.portfolio_constraint_violations:
            print(f"    - {violation}")


def demo_scenario_1_normal_approval():
    """Scenario 1: Normal trade approval in good conditions"""
    print_header("SCENARIO 1: Normal Trade Approval")
    
    config = RPMConfig(
        total_capital=100000.0,
        risk_per_trade=0.01,
    )
    rpm = RiskPortfolioManager(config=config)
    
    print("Conditions: Clean portfolio, trending market, high confidence")
    print()
    
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,  # LONG
        confidence_score=0.80,
        regime='TRENDING',
        atr=0.0010,
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    print_decision(output)
    
    print(f"\nPortfolio State:")
    print(f"  Equity: ${rpm.portfolio_state.equity:,.2f}")
    print(f"  Drawdown: {rpm.portfolio_state.current_drawdown:.2%}")
    print(f"  Open Positions: {len(rpm.portfolio_state.open_positions)}")


def demo_scenario_2_confidence_scaling():
    """Scenario 2: Position sizing scales with confidence"""
    print_header("SCENARIO 2: Confidence-Based Position Sizing")
    
    config = RPMConfig(total_capital=100000.0, risk_per_trade=0.01)
    rpm = RiskPortfolioManager(config=config)
    
    confidences = [0.50, 0.65, 0.80, 0.95]
    
    print("Comparing position sizes across confidence levels:\n")
    
    for conf in confidences:
        output = rpm.process_trade_intent(
            symbol='EURUSD',
            direction=1,
            confidence_score=conf,
            regime='TRENDING',
            atr=0.0010,
            vol_percentile=0.50,
            current_price=1.1000,
        )
        
        if output.decision.approved_trade:
            size = output.decision.approved_trade.position_units
            multiplier = output.decision.approved_trade.confidence_adjustment
            print(f"  Confidence {conf:.0%}: {size:>8.2f} units (multiplier: {multiplier:.3f}x)")


def demo_scenario_3_regime_adjustment():
    """Scenario 3: Position sizing adjusts for market regime"""
    print_header("SCENARIO 3: Regime-Based Position Adjustments")
    
    config = RPMConfig(total_capital=100000.0, risk_per_trade=0.01)
    rpm = RiskPortfolioManager(config=config)
    
    regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'STRESSED']
    
    print("Comparing position sizes across market regimes:\n")
    
    for regime in regimes:
        output = rpm.process_trade_intent(
            symbol='EURUSD',
            direction=1,
            confidence_score=0.75,
            regime=regime,
            atr=0.0010,
            vol_percentile=0.50,
            current_price=1.1000,
        )
        
        if output.decision.approved_trade:
            size = output.decision.approved_trade.position_units
            multiplier = output.decision.approved_trade.regime_adjustment
            print(f"  {regime:>12}: {size:>8.2f} units (multiplier: {multiplier:.3f}x)")
        else:
            print(f"  {regime:>12}: REJECTED - {output.decision.rejected_trade.rejection_reason.value}")


def demo_scenario_4_drawdown_halt():
    """Scenario 4: Maximum drawdown triggers halt"""
    print_header("SCENARIO 4: Drawdown Kill Switch")
    
    config = RPMConfig(
        total_capital=100000.0,
        max_drawdown=0.10,  # 10% max drawdown
    )
    rpm = RiskPortfolioManager(config=config)
    
    # Simulate drawdown
    rpm.portfolio_state.peak_equity = 100000.0
    rpm.portfolio_state.equity = 85000.0
    rpm.portfolio_state.current_drawdown = 0.15  # 15% drawdown
    
    print(f"Portfolio State:")
    print(f"  Peak Equity: ${rpm.portfolio_state.peak_equity:,.2f}")
    print(f"  Current Equity: ${rpm.portfolio_state.equity:,.2f}")
    print(f"  Drawdown: {rpm.portfolio_state.current_drawdown:.2%} (limit: {config.max_drawdown:.2%})")
    print()
    
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.85,
        regime='TRENDING',
        atr=0.0010,
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    print_decision(output)


def demo_scenario_5_daily_loss_limit():
    """Scenario 5: Daily loss limit triggers halt (percentage-based)"""
    print_header("SCENARIO 5: Daily Loss Limit Kill Switch (Percentage-Based)")
    
    config = RPMConfig(
        total_capital=100000.0,
        daily_loss_limit=0.02,  # Max 2% of capital loss per day ($2k for $100k)
    )
    rpm = RiskPortfolioManager(config=config)
    
    # Simulate daily losses exceeding 2% limit
    rpm.portfolio_state.daily_pnl = -2500.0  # Exceeds -2000 (2% of 100k)
    
    print(f"Portfolio State:")
    print(f"  Daily PnL: ${rpm.portfolio_state.daily_pnl:,.2f}")
    print(f"  Daily Loss Limit: {config.daily_loss_limit*100:.1f}% (${config.total_capital * config.daily_loss_limit:,.2f})")
    print()
    
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.85,
        regime='TRENDING',
        atr=0.0010,
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    print_decision(output)


def demo_scenario_6_confidence_collapse():
    """Scenario 6: Low model confidence triggers rejection"""
    print_header("SCENARIO 6: Model Confidence Collapse")
    
    config = RPMConfig(
        total_capital=100000.0,
        min_confidence_threshold=0.60,
    )
    rpm = RiskPortfolioManager(config=config)
    
    print(f"Min Confidence Threshold: {config.min_confidence_threshold:.2%}")
    print()
    
    # Try with low confidence
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.45,  # Below threshold
        regime='TRENDING',
        atr=0.0010,
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    print_decision(output)


def demo_scenario_7_exposure_limits():
    """Scenario 7: Symbol exposure limits"""
    print_header("SCENARIO 7: Symbol Exposure Limits")
    
    config = RPMConfig(
        total_capital=100000.0,
        max_symbol_exposure_units=10000.0,
    )
    rpm = RiskPortfolioManager(config=config)
    
    # Add existing exposure
    rpm.portfolio_state.symbol_exposure['EURUSD'] = 8000.0
    
    print(f"Existing EURUSD Exposure: 8,000 units")
    print(f"Max Symbol Exposure: {config.max_symbol_exposure_units:,.0f} units")
    print()
    
    # Try large position (will be adjusted or rejected)
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.80,
        regime='TRENDING',
        atr=0.00001,  # Very small ATR = large position
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    print_decision(output)


def demo_scenario_8_manual_halt_resume():
    """Scenario 8: Manual halt and resume"""
    print_header("SCENARIO 8: Manual Trading Halt & Resume")
    
    config = RPMConfig(total_capital=100000.0)
    rpm = RiskPortfolioManager(config=config)
    
    print("Step 1: Normal trading")
    output = rpm.process_trade_intent(
        symbol='EURUSD', direction=1, confidence_score=0.75,
        regime='TRENDING', atr=0.0010, vol_percentile=0.50, current_price=1.1000,
    )
    print(f"  Status: {output.decision.status.value}")
    
    print("\nStep 2: Manual halt triggered")
    rpm.kill_switches.manual_halt(rpm.portfolio_state, "Emergency market conditions")
    print(f"  Trading Halted: {rpm.portfolio_state.trading_halted}")
    print(f"  Halt Reason: {rpm.portfolio_state.halt_reason}")
    
    print("\nStep 3: Attempt trade during halt")
    output = rpm.process_trade_intent(
        symbol='GBPUSD', direction=1, confidence_score=0.80,
        regime='TRENDING', atr=0.0012, vol_percentile=0.50, current_price=1.3000,
    )
    print(f"  Status: {output.decision.status.value}")
    print(f"  Reason: {output.decision.rejected_trade.rejection_details if output.decision.rejected_trade else 'N/A'}")
    
    print("\nStep 4: Manual resume")
    rpm.kill_switches.manual_resume(rpm.portfolio_state)
    print(f"  Trading Halted: {rpm.portfolio_state.trading_halted}")
    
    print("\nStep 5: Trade after resume")
    output = rpm.process_trade_intent(
        symbol='GBPUSD', direction=1, confidence_score=0.80,
        regime='TRENDING', atr=0.0012, vol_percentile=0.50, current_price=1.3000,
    )
    print(f"  Status: {output.decision.status.value}")


def demo_scenario_9_health_monitoring():
    """Scenario 9: Health status monitoring"""
    print_header("SCENARIO 9: Health Status Monitoring")
    
    config = RPMConfig(total_capital=100000.0)
    rpm = RiskPortfolioManager(config=config)
    
    # Process some trades
    for i in range(5):
        rpm.process_trade_intent(
            symbol='EURUSD', direction=1, confidence_score=0.75,
            regime='TRENDING', atr=0.0010, vol_percentile=0.50, current_price=1.1000,
        )
    
    health = rpm.get_health_status()
    
    print(f"RPM Version: {health['rpm_version']}")
    print(f"Config Hash: {health['config_hash']}")
    print(f"Health Status: {health['health']}")
    print()
    
    print("Risk Metrics:")
    metrics = health['risk_metrics']
    print(f"  Total Decisions: {metrics['total_decisions']}")
    print(f"  Trades Approved: {metrics['trades_approved']}")
    print(f"  Trades Rejected: {metrics['trades_rejected']}")
    print(f"  Approval Rate: {metrics['approval_rate']:.2%}")
    print(f"  Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print()
    
    print("Kill Switch Status:")
    kill = health['kill_switches']
    print(f"  Trading Halted: {kill['trading_halted']}")
    print(f"  Drawdown: {kill['drawdown']['current_pct']:.2f}% / {kill['drawdown']['threshold_pct']:.2f}%")
    print(f"  Daily Loss: ${kill['daily_loss']['current']:.2f} / ${kill['daily_loss']['limit']:.2f}")


def main():
    """Run all demo scenarios"""
    print("\n" + "="*80)
    print(" RISK & PORTFOLIO MANAGER (RPM) - DEMONSTRATION")
    print(" The Gatekeeper with Absolute Veto Authority")
    print("="*80)
    
    scenarios = [
        demo_scenario_1_normal_approval,
        demo_scenario_2_confidence_scaling,
        demo_scenario_3_regime_adjustment,
        demo_scenario_4_drawdown_halt,
        demo_scenario_5_daily_loss_limit,
        demo_scenario_6_confidence_collapse,
        demo_scenario_7_exposure_limits,
        demo_scenario_8_manual_halt_resume,
        demo_scenario_9_health_monitoring,
    ]
    
    for scenario in scenarios:
        try:
            scenario()
        except Exception as e:
            print(f"\n❌ Error in scenario: {e}")
    
    print("\n" + "="*80)
    print(" Demo Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
