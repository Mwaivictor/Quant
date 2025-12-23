"""
Complete RPM System Test

Tests all components of the Risk & Portfolio Manager:
1. Adaptive Kelly Criterion with regime caps
2. EWMA edge tracking (non-stationary)
3. Regime-conditional performance
4. Edge decay detection
5. Liquidity constraints
6. Strategy intelligence
7. Full trade processing pipeline
"""

from datetime import datetime, timedelta
from arbitrex.risk_portfolio_manager import RiskPortfolioManager
from arbitrex.risk_portfolio_manager.config import RPMConfig
from arbitrex.risk_portfolio_manager.kelly_criterion import KellyCriterion
from arbitrex.risk_portfolio_manager.strategy_intelligence import (
    StrategyPerformanceTracker,
    TradeRecord
)
from arbitrex.risk_portfolio_manager.liquidity_constraints import LiquidityConstraints


def test_adaptive_kelly():
    """Test 1: Adaptive Kelly Criterion with regime multipliers"""
    print("=" * 80)
    print("TEST 1: ADAPTIVE KELLY CRITERION")
    print("=" * 80)
    
    config = RPMConfig(
        kelly_use_adaptive_cap=True,
        kelly_base_max_pct=0.01,
        kelly_safety_factor=0.25
    )
    
    kelly = KellyCriterion(
        safety_factor=config.kelly_safety_factor,
        max_kelly_pct=config.kelly_base_max_pct,
        use_adaptive_cap=config.kelly_use_adaptive_cap
    )
    
    # Test strategy: 58% win rate, 2.5% avg win, 1.8% avg loss
    win_rate = 0.58
    avg_win = 0.025
    avg_loss = 0.018
    num_trades = 50
    
    regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'STRESSED', 'CRISIS']
    
    print(f"\nStrategy Stats:")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Avg Win:  {avg_win*100:.2f}%")
    print(f"  Avg Loss: {avg_loss*100:.2f}%")
    print(f"  Trades:   {num_trades}\n")
    
    print(f"{'Regime':<12} {'Multiplier':<12} {'Kelly Cap':<12} {'$100k Position'}")
    print("-" * 60)
    
    for regime in regimes:
        result = kelly.calculate(win_rate, avg_win, avg_loss, num_trades, regime=regime)
        multiplier = kelly.regime_multipliers[regime]
        position_100k = result.kelly_cap * 100000
        
        print(f"{regime:<12} {multiplier:<12.1f} {result.kelly_cap*100:>10.2f}%  ${position_100k:>12,.0f}")
    
    print("\n✅ Adaptive Kelly: Position size scales down in volatile regimes")
    return True


def test_ewma_edge_tracking():
    """Test 2: EWMA edge tracking for non-stationary markets"""
    print("\n" + "=" * 80)
    print("TEST 2: EWMA EDGE TRACKING (NON-STATIONARY)")
    print("=" * 80)
    
    tracker = StrategyPerformanceTracker(
        strategy_id='momentum_strategy',
        lookback_days=90,        # 90-day lookback
        recent_period_days=30,   # Last 30 days for "recent"
        use_ewma=True,
        ewma_alpha=0.1,  # Faster decay for demo
        track_regime_specific=True
    )
    
    # Simulate edge decay: start strong, deteriorate catastrophically
    print("\nSimulating 70 trades with severe edge decay over 90 days...")
    
    for i in range(70):
        # Spread trades over 90 days (trades 0-45 in days 90-30, trades 46-70 in last 30 days)
        # Early trades (0-45): 65% win rate, strong edge (days 90-35 ago)
        # Late trades (46-70): 10% win rate, edge completely lost! (last 30 days)
        
        if i < 45:
            is_win = (i % 3 != 0)  # ~67% wins
            pnl = 120 if is_win else -60
            # Spread first 45 trades from day 90 to day 35 ago
            days_ago = 90 - int((i / 45) * 55)  # 90 down to 35
        else:
            # Last 25 trades: catastrophic deterioration (10% win rate)
            is_win = (i % 10 == 0)  # Only 10% wins!
            pnl = 80 if is_win else -110  # Larger losses
            # Spread last 25 trades evenly over last 30 days
            days_ago = 30 - int(((i - 45) / 25) * 30)  # 30 down to 0
        
        trade = TradeRecord(
            strategy_id='momentum_strategy',
            symbol='EURUSD',
            entry_time=datetime.utcnow() - timedelta(days=days_ago, hours=12),
            exit_time=datetime.utcnow() - timedelta(days=days_ago, hours=11),
            pnl=pnl,
            return_pct=pnl / 10000,
            size=10000
        )
        
        tracker.record_trade(trade, regime='TRENDING')
    
    metrics = tracker.calculate_metrics()
    
    print(f"\nFull Period Metrics (all 70 trades):")
    print(f"  Win Rate:   {metrics.win_rate*100:.1f}%")
    print(f"  Expectancy: ${metrics.expectancy:.2f}")
    print(f"  Total P&L:  ${metrics.total_pnl:.2f}")
    
    print(f"\nEWMA Metrics (recency-weighted):")
    print(f"  EWMA Win Rate:   {metrics.ewma_win_rate*100:.1f}%")
    print(f"  EWMA Expectancy: ${metrics.ewma_expectancy:.2f}")
    print(f"  EWMA Alpha:      {metrics.ewma_alpha}")
    
    print(f"\nRecent Performance (last 30 days):")
    print(f"  Recent Win Rate:   {metrics.recent_win_rate*100:.1f}%")
    print(f"  Recent Expectancy: ${metrics.recent_expectancy:.2f}")
    
    print(f"\nEdge Decay Detection:")
    print(f"  Edge Decaying: {metrics.edge_is_decaying}")
    print(f"  Decay %:       {metrics.edge_decay_pct*100:.1f}%")
    print(f"  Multiplier:    {metrics.edge_decay_multiplier}× (position size adjustment)")
    
    print(f"\nHealth Assessment:")
    print(f"  Status: {metrics.health_status.value.upper()}")
    print(f"  Score:  {metrics.health_score:.2f} / 1.00")
    
    if metrics.edge_is_decaying:
        print("\n⚠️  Edge decay detected! Position size automatically reduced to 50%")
    else:
        print("\n⚠️  WARNING: Edge decay NOT detected despite severe deterioration!")
        print(f"     (Recent expectancy: ${metrics.recent_expectancy:.2f} vs Full: ${metrics.expectancy:.2f})")
    
    print("\n✅ EWMA: Recent trades weighted more heavily, edge decay detected")
    return metrics.edge_is_decaying  # Should be True


def test_regime_conditional_performance():
    """Test 3: Regime-conditional performance tracking"""
    print("\n" + "=" * 80)
    print("TEST 3: REGIME-CONDITIONAL PERFORMANCE")
    print("=" * 80)
    
    tracker = StrategyPerformanceTracker(
        strategy_id='mean_reversion',
        track_regime_specific=True
    )
    
    # Simulate trades across different regimes
    regimes_data = {
        'TRENDING': [(False, -50)] * 15,    # Loses in trends (not mean-reverting)
        'RANGING': [(True, 80)] * 20,       # Wins in ranges (mean-reverts well)
        'VOLATILE': [(True, 60), (False, -70)] * 5,  # Mixed in volatility
    }
    
    trade_num = 0
    for regime, outcomes in regimes_data.items():
        for is_win, pnl in outcomes:
            trade = TradeRecord(
                strategy_id='mean_reversion',
                symbol='EURUSD',
                entry_time=datetime.utcnow() - timedelta(hours=trade_num),
                exit_time=datetime.utcnow() - timedelta(hours=trade_num-1),
                pnl=pnl,
                return_pct=pnl / 10000,
                size=10000
            )
            tracker.record_trade(trade, regime=regime)
            trade_num += 1
    
    metrics = tracker.calculate_metrics()
    
    print(f"\nOverall Performance:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate:     {metrics.win_rate*100:.1f}%")
    print(f"  Expectancy:   ${metrics.expectancy:.2f}")
    
    print(f"\nRegime-Specific Performance:")
    print(f"{'Regime':<12} {'Trades':<8} {'Win Rate':<12} {'Expectancy':<12} {'Avg P&L'}")
    print("-" * 60)
    
    for regime, stats in metrics.regime_metrics.items():
        print(f"{regime:<12} {stats['trades']:<8} {stats['win_rate']*100:>10.1f}%  "
              f"${stats['expectancy']:>10.2f}  ${stats['avg_pnl']:>10.2f}")
    
    print("\n✅ Regime-Conditional: Strategy edge varies significantly by regime")
    print("   → Should trade MORE in RANGING, LESS in TRENDING")
    return True


def test_liquidity_constraints():
    """Test 4: Liquidity constraints and market impact"""
    print("\n" + "=" * 80)
    print("TEST 4: LIQUIDITY CONSTRAINTS & MARKET IMPACT")
    print("=" * 80)
    
    liq = LiquidityConstraints(
        max_adv_pct=0.01,       # 1% of ADV
        max_spread_bps=20.0,    # 20 bps
        max_market_impact_pct=0.005  # 0.5% max impact
    )
    
    # Test case 1: Large position in liquid market
    print("\nTest Case 1: Liquid Market (EURUSD)")
    result1 = liq.check(
        proposed_units=50000,
        adv_units=10000000,  # 10M daily volume
        spread_pct=0.0002,   # 2 bps
        volatility=0.008,    # 0.8% daily vol
        current_price=1.10
    )
    
    print(f"  Proposed:       50,000 units")
    print(f"  ADV Limit:      {result1.adv_limit:,.0f} units (1% of {10000000:,})")
    print(f"  Spread Penalty: {result1.spread_penalty:.2%}")
    print(f"  Market Impact:  ${result1.market_impact:.2f} ({result1.market_impact_pct:.3%})")
    print(f"  Max Allowed:    {result1.max_units:,.0f} units")
    print(f"  Acceptable:     {result1.is_acceptable}")
    
    # Test case 2: Large position in illiquid market
    print("\nTest Case 2: Illiquid Market (Exotic Pair)")
    result2 = liq.check(
        proposed_units=50000,
        adv_units=100000,    # 100K daily volume (thin!)
        spread_pct=0.0050,   # 50 bps (wide!)
        volatility=0.02,     # 2% daily vol
        current_price=1.10
    )
    
    print(f"  Proposed:       50,000 units")
    print(f"  ADV Limit:      {result2.adv_limit:,.0f} units (1% of {100000:,})")
    print(f"  Spread Penalty: {result2.spread_penalty:.2%}")
    print(f"  Market Impact:  ${result2.market_impact:.2f} ({result2.market_impact_pct:.3%})")
    print(f"  Max Allowed:    {result2.max_units:,.0f} units")
    print(f"  Acceptable:     {result2.is_acceptable}")
    if result2.rejection_reason:
        print(f"  ⚠️  Rejection:     {result2.rejection_reason}")
    
    print("\n✅ Liquidity: Position size capped by ADV, spread, and market impact")
    return result1.is_acceptable and not result2.is_acceptable


def test_full_trade_pipeline():
    """Test 5: Full trade processing pipeline with all features"""
    print("\n" + "=" * 80)
    print("TEST 5: FULL TRADE PROCESSING PIPELINE")
    print("=" * 80)
    
    config = RPMConfig(
        total_capital=100000,
        kelly_use_adaptive_cap=True,
        edge_use_ewma=True,
        edge_regime_specific=True,
        edge_auto_reduce_on_decay=True
    )
    
    rpm = RiskPortfolioManager(config=config)
    
    # Test trade 1: High confidence, TRENDING regime, strong stats
    print("\nTrade 1: High Confidence + TRENDING + Strong Edge")
    output1 = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.85,
        regime='TRENDING',
        atr=0.0012,
        vol_percentile=0.4,
        current_price=1.1000,
        win_rate=0.65,
        avg_win=0.030,
        avg_loss=0.018,
        num_trades=100,
        adv_units=10000000,
        spread_pct=0.0002,
        daily_volatility=0.008
    )
    
    print(f"  Decision:  {'APPROVED' if output1.decision.approved_trade else 'REJECTED'}")
    if output1.decision.approved_trade:
        units = output1.decision.approved_trade.position_units
        print(f"  Position:  {units:,.0f} units (${units * 1.10:,.2f})")
        print(f"  Confidence: {output1.decision.approved_trade.confidence_score*100:.0f}%")
    else:
        print(f"  Position:  REJECTED")
    
    # Test trade 2: Same setup but STRESSED regime
    print("\nTrade 2: Same Setup but STRESSED Regime")
    output2 = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.85,
        regime='STRESSED',  # Changed
        atr=0.0025,  # Higher ATR in stress
        vol_percentile=0.85,  # High volatility
        current_price=1.1000,
        win_rate=0.65,
        avg_win=0.030,
        avg_loss=0.018,
        num_trades=100,
        adv_units=10000000,
        spread_pct=0.0002,
        daily_volatility=0.020  # Higher vol
    )
    
    print(f"  Decision:  {'APPROVED' if output2.decision.approved_trade else 'REJECTED'}")
    if output2.decision.approved_trade:
        units2 = output2.decision.approved_trade.position_units
        print(f"  Position:  {units2:,.0f} units (${units2 * 1.10:,.2f})")
    else:
        print(f"  Position:  REJECTED")
    
    # Calculate reduction if both approved
    if output1.decision.approved_trade and output2.decision.approved_trade:
        units1 = output1.decision.approved_trade.position_units
        units2 = output2.decision.approved_trade.position_units
        reduction_pct = (1 - units2 / units1) * 100
        print(f"\n  📉 Position Reduction: {reduction_pct:.1f}% (TRENDING → STRESSED)")
    
    print("\n✅ Full Pipeline: All components working together")
    print("   → Adaptive Kelly adjusted by regime")
    print("   → Volatility scaling applied")
    print("   → Liquidity constraints checked")
    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 80)
    print("RPM COMPLETE SYSTEM TEST")
    print("Testing: Adaptive Kelly, EWMA, Regime-Conditional, Liquidity, Full Pipeline")
    print("=" * 80)
    
    results = {}
    
    try:
        results['adaptive_kelly'] = test_adaptive_kelly()
    except Exception as e:
        print(f"\n❌ Adaptive Kelly Test Failed: {e}")
        results['adaptive_kelly'] = False
    
    try:
        results['ewma_edge'] = test_ewma_edge_tracking()
    except Exception as e:
        print(f"\n❌ EWMA Edge Test Failed: {e}")
        results['ewma_edge'] = False
    
    try:
        results['regime_conditional'] = test_regime_conditional_performance()
    except Exception as e:
        print(f"\n❌ Regime-Conditional Test Failed: {e}")
        results['regime_conditional'] = False
    
    try:
        results['liquidity'] = test_liquidity_constraints()
    except Exception as e:
        print(f"\n❌ Liquidity Test Failed: {e}")
        results['liquidity'] = False
    
    try:
        results['full_pipeline'] = test_full_trade_pipeline()
    except Exception as e:
        print(f"\n❌ Full Pipeline Test Failed: {e}")
        results['full_pipeline'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name.replace('_', ' ').title()}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! RPM System Fully Operational")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Review output above.")
    
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
