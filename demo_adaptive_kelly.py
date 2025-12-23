"""
Demo: Adaptive Kelly Criterion - Regime-Aware Position Sizing

Demonstrates how Kelly caps adapt automatically to market regimes,
eliminating the static 1% hard cap limitation.

Version: 2.0.0 (Enterprise)
"""

from arbitrex.risk_portfolio_manager.kelly_criterion import KellyCriterion


def demo_adaptive_kelly():
    """Show Kelly caps adapting across different market regimes"""
    
    print("=" * 80)
    print("ADAPTIVE KELLY CRITERION - Enterprise v2.0.0")
    print("=" * 80)
    print()
    
    # Initialize Kelly with adaptive caps enabled
    kelly = KellyCriterion(
        safety_factor=0.25,
        max_kelly_pct=0.01,  # Base cap for TRENDING
        use_adaptive_cap=True
    )
    
    # Trading statistics (same across all regimes for comparison)
    win_rate = 0.58  # 58% win rate
    avg_win = 0.025  # 2.5% average win
    avg_loss = 0.018  # 1.8% average loss
    num_trades = 50
    
    print("STRATEGY PERFORMANCE METRICS (Fixed):")
    print(f"  Win Rate:      {win_rate*100:.1f}%")
    print(f"  Average Win:   {avg_win*100:.2f}%")
    print(f"  Average Loss:  {avg_loss*100:.2f}%")
    print(f"  Sample Size:   {num_trades} trades")
    print()
    
    # Calculate raw Kelly
    p = win_rate
    W = avg_win
    L = avg_loss
    raw_kelly = ((p * W) - ((1 - p) * L)) / W
    fractional_kelly = 0.25 * raw_kelly
    
    print(f"KELLY MATHEMATICS:")
    print(f"  Raw Kelly:       {raw_kelly*100:.2f}%")
    print(f"  Fractional (λ=0.25): {fractional_kelly*100:.2f}%")
    print()
    print("=" * 80)
    print()
    
    # Test across all regimes
    regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'STRESSED', 'CRISIS']
    regime_descriptions = {
        'TRENDING': 'Low volatility, clear direction',
        'RANGING': 'Moderate volatility, choppy',
        'VOLATILE': 'High volatility, unpredictable',
        'STRESSED': 'Extreme volatility, risk-off',
        'CRISIS': 'Market crisis, liquidity freeze'
    }
    
    results = []
    
    for regime in regimes:
        result = kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            regime=regime
        )
        
        results.append((regime, result))
        
        print(f"REGIME: {regime}")
        print(f"  Description: {regime_descriptions[regime]}")
        print(f"  Kelly Cap:   {result.kelly_cap*100:.2f}% of capital")
        print(f"  Multiplier:  {kelly.regime_multipliers[regime]:.1f}×")
        
        # Show position size for $100k account
        account_size = 100_000
        position_size = account_size * result.kelly_cap
        print(f"  Max Position (on $100k): ${position_size:,.0f}")
        print()
    
    print("=" * 80)
    print()
    
    # Comparison table
    print("ADAPTIVE KELLY SUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Regime':<12} {'Multiplier':<12} {'Kelly Cap':<15} {'$100k Position':<20}")
    print("-" * 80)
    
    for regime, result in results:
        multiplier = kelly.regime_multipliers[regime]
        position = 100_000 * result.kelly_cap
        print(f"{regime:<12} {multiplier:<12.1f}× {result.kelly_cap*100:>6.2f}%        ${position:>15,.0f}")
    
    print("-" * 80)
    print()
    
    # Show reduction percentages
    base_cap = results[0][1].kelly_cap  # TRENDING cap
    
    print("RISK REDUCTION BY REGIME (vs TRENDING baseline):")
    print("-" * 80)
    
    for regime, result in results:
        reduction_pct = (1 - result.kelly_cap / base_cap) * 100
        if reduction_pct > 0:
            print(f"  {regime:<12}: -{reduction_pct:>5.1f}% position size")
        else:
            print(f"  {regime:<12}: Baseline (most aggressive)")
    
    print()
    print("=" * 80)
    print()
    
    # Demonstrate with no regime (backward compatible)
    print("BACKWARD COMPATIBILITY TEST:")
    print("-" * 80)
    
    result_no_regime = kelly.calculate(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=num_trades,
        regime=None  # No regime specified
    )
    
    print("When no regime is specified:")
    print(f"  Kelly Cap: {result_no_regime.kelly_cap*100:.2f}% (uses base max_kelly_pct)")
    print(f"  Equivalent to: TRENDING regime behavior")
    print()
    
    # Demonstrate with adaptive disabled
    kelly_static = KellyCriterion(
        safety_factor=0.25,
        max_kelly_pct=0.01,
        use_adaptive_cap=False  # Disable adaptation
    )
    
    result_static = kelly_static.calculate(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=num_trades,
        regime='STRESSED'  # Regime ignored when adaptive disabled
    )
    
    print("When adaptive caps are DISABLED:")
    print(f"  Kelly Cap: {result_static.kelly_cap*100:.2f}% (static, ignores regime)")
    print(f"  Regime parameter ignored - always uses base cap")
    print()
    print("=" * 80)
    print()
    
    # Real-world scenario
    print("REAL-WORLD SCENARIO:")
    print("-" * 80)
    print()
    print("Scenario: Strategy with 58% win rate, 2.5% avg win, 1.8% avg loss")
    print("Account: $1,000,000")
    print("Asset: EURUSD @ 1.1000")
    print()
    
    account = 1_000_000
    price = 1.1000
    
    for regime in ['TRENDING', 'VOLATILE', 'STRESSED']:
        result = kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            regime=regime
        )
        
        position_value = account * result.kelly_cap
        units = position_value / price
        
        print(f"{regime} Market:")
        print(f"  Max Position Value: ${position_value:,.0f}")
        print(f"  Max Units:          {units:,.0f}")
        print(f"  Risk Adjustment:    {kelly.regime_multipliers[regime]:.1f}× base cap")
        print()
    
    print("=" * 80)
    print()
    
    # Key insights
    print("KEY INSIGHTS:")
    print("-" * 80)
    print()
    print("1. ADAPTIVE BEHAVIOR:")
    print("   - In TRENDING markets: Full Kelly (1.0%) - maximize growth")
    print("   - In VOLATILE markets: Half Kelly (0.5%) - reduce risk")
    print("   - In STRESSED markets: 20% Kelly (0.2%) - capital preservation")
    print()
    print("2. AUTOMATIC RISK MANAGEMENT:")
    print("   - No manual intervention required")
    print("   - Position sizes adjust automatically to conditions")
    print("   - Prevents over-leverage in dangerous regimes")
    print()
    print("3. BACKWARD COMPATIBLE:")
    print("   - Can disable adaptive caps (use_adaptive_cap=False)")
    print("   - Works without regime parameter (uses base cap)")
    print("   - Maintains all v1.2.0 functionality")
    print()
    print("4. INTEGRATION:")
    print("   - Seamlessly integrates with AdaptiveRiskManager")
    print("   - Regime automatically detected and passed")
    print("   - No code changes needed in calling code")
    print()
    print("=" * 80)
    print()
    print("✅ Adaptive Kelly Criterion: Static cap limitation ELIMINATED")
    print("✅ Position sizing now responds dynamically to market conditions")
    print("✅ Enterprise-grade risk management achieved")
    print()


def demo_regime_transition():
    """Demonstrate Kelly cap changing as market regime shifts"""
    
    print("\n")
    print("=" * 80)
    print("REGIME TRANSITION DEMONSTRATION")
    print("=" * 80)
    print()
    print("Simulating a market regime shift from TRENDING to CRISIS...")
    print()
    
    kelly = KellyCriterion(use_adaptive_cap=True)
    
    # Same strategy performance throughout
    params = {
        'win_rate': 0.60,
        'avg_win': 0.03,
        'avg_loss': 0.02,
        'num_trades': 100
    }
    
    regime_sequence = [
        ('TRENDING', 'Day 1-10: Bull market, low vol'),
        ('RANGING', 'Day 11-15: Consolidation begins'),
        ('VOLATILE', 'Day 16-20: Sharp intraday swings'),
        ('STRESSED', 'Day 21-22: Risk-off selloff'),
        ('CRISIS', 'Day 23: Market crash'),
    ]
    
    print(f"{'Period':<30} {'Regime':<12} {'Kelly Cap':<12} {'Position ($100k)':<20}")
    print("-" * 80)
    
    for period, description in regime_sequence:
        result = kelly.calculate(**params, regime=period)
        position = 100_000 * result.kelly_cap
        
        print(f"{description:<30} {period:<12} {result.kelly_cap*100:>5.2f}%       ${position:>15,.0f}")
    
    print("-" * 80)
    print()
    print("OBSERVATION:")
    print("  As regime deteriorates, Kelly cap automatically reduces:")
    print("  - $1,000 → $500 → $200 = 80% risk reduction")
    print("  - No manual intervention required")
    print("  - Capital preserved for recovery phase")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_adaptive_kelly()
    demo_regime_transition()
    
    print("\n✅ Adaptive Kelly Criterion demonstration complete!")
    print("📊 Static 1% hard cap replaced with regime-aware dynamic caps")
    print("🎯 v2.0.0 Enterprise: Risk management that adapts to reality")
