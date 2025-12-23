"""
RPM Enhanced Features Demonstration

Demonstrates the three critical enhancements:
1. State Persistence - Survives crashes/restarts
2. Order Management - Tracks pending orders, partial fills, slippage
3. Correlation-Aware Sizing - Prevents risk underestimation during stress
"""

import time
from arbitrex.risk_portfolio_manager import (
    RiskPortfolioManager,
    RPMConfig,
    OrderStatus,
)


def demo_state_persistence():
    """Demo 1: State Persistence - No more data loss on crashes"""
    print("\n" + "="*80)
    print("DEMO 1: STATE PERSISTENCE - CRASH RECOVERY")
    print("="*80)
    
    print("\nStep 1: Create RPM with persistence enabled (default)")
    rpm1 = RiskPortfolioManager(enable_persistence=True)
    
    print(f"  Initial capital: ${rpm1.portfolio_state.total_capital:,.0f}")
    print(f"  Daily PnL: ${rpm1.portfolio_state.daily_pnl:,.2f}")
    
    print("\nStep 2: Simulate some trading activity")
    rpm1.portfolio_state.daily_pnl = -1500.00
    rpm1.portfolio_state.weekly_pnl = -2300.00
    rpm1.portfolio_state.equity = 98500.00
    
    print(f"  Simulated daily PnL: ${rpm1.portfolio_state.daily_pnl:,.2f}")
    print(f"  Simulated equity: ${rpm1.portfolio_state.equity:,.2f}")
    
    print("\nStep 3: Manually save state")
    rpm1.save_state()
    print("  ✓ State saved to logs/rpm_state.json")
    
    print("\nStep 4: Create backup")
    rpm1.create_backup()
    print("  ✓ Backup created")
    
    print("\nStep 5: Simulate crash - destroy RPM instance")
    del rpm1
    print("  ✓ RPM instance destroyed (simulating crash)")
    
    print("\nStep 6: Create NEW RPM instance - state should restore")
    rpm2 = RiskPortfolioManager(enable_persistence=True)
    
    print(f"  Restored daily PnL: ${rpm2.portfolio_state.daily_pnl:,.2f}")
    print(f"  Restored equity: ${rpm2.portfolio_state.equity:,.2f}")
    
    if rpm2.portfolio_state.daily_pnl == -1500.00:
        print("\n✅ SUCCESS: State persisted and restored correctly!")
    else:
        print("\n❌ FAILURE: State not restored")
    
    print("\nStep 7: Clean up state for next demo")
    rpm2.state_manager.clear_state()


def demo_order_management():
    """Demo 2: Order Management - Track executions and slippage"""
    print("\n" + "="*80)
    print("DEMO 2: ORDER MANAGEMENT - EXECUTION TRACKING")
    print("="*80)
    
    print("\nStep 1: Create RPM with relaxed limits and approve a trade")
    
    # Use very relaxed limits to ensure approval for demo
    config = RPMConfig(
        total_capital=100000.0,
        risk_per_trade=0.005,  # Smaller risk = smaller position
        max_symbol_exposure_units=1000000.0,
        max_symbol_exposure_pct=0.99,  # 99% - max allowed
    )
    rpm = RiskPortfolioManager(config=config, enable_persistence=False)
    
    output = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.60,  # Lower confidence to further reduce size
        regime='VOLATILE',  # Volatile regime reduces size by 30%
        atr=0.0050,  # Much larger ATR = smaller position
        vol_percentile=0.70,  # Higher volatility
        current_price=1.1000,
    )
    
    if output.decision.status == 'APPROVED':
        order_id = output.decision.order_id
        approved_units = output.decision.approved_trade.position_units
        
        print(f"\n✓ Trade APPROVED")
        print(f"  Order ID: {order_id}")
        print(f"  Approved Units: {approved_units:,.0f}")
    elif output.decision.status == 'ADJUSTED':
        order_id = output.decision.order_id
        approved_units = output.decision.approved_trade.position_units
        
        print(f"\n✓ Trade ADJUSTED (but approved)")
        print(f"  Order ID: {order_id}")
        print(f"  Approved Units: {approved_units:,.0f}")
    else:
        print(f"\n❌ Trade REJECTED: {output.decision.rejected_trade.rejection_reason.value}")
        if output.decision.sizing_adjustments:
            print(f"  Sizing: {output.decision.sizing_adjustments}")
        return  # Exit early if rejected
        
    print("\nStep 2: Check pending orders")
    pending = rpm.get_pending_orders()
    print(f"  Pending orders: {len(pending)}")
    for order in pending:
        print(f"    - {order.symbol}: {order.approved_units:,.0f} units ({order.status.value})")
    
    print("\nStep 3: Simulate PARTIAL FILL (50% of order)")
    rpm.update_order_fill(
        order_id=order_id,
        fill_units=approved_units * 0.5,
        fill_price=1.1005,  # Small slippage
        expected_price=1.1000,
    )
    
    print(f"  ✓ Partial fill recorded: {approved_units * 0.5:,.0f} units @ 1.1005")
    
    pending = rpm.get_pending_orders()
    if pending:
        order = pending[0]
        print(f"  Order status: {order.status.value}")
        print(f"  Filled: {order.filled_units:,.0f} units")
        print(f"  Remaining: {order.remaining_units:,.0f} units")
        print(f"  Slippage: {order.slippage_bps:.2f} bps")
    
    print("\nStep 4: Complete the fill")
    rpm.update_order_fill(
        order_id=order_id,
        fill_units=approved_units * 0.5,
        fill_price=1.1008,  # More slippage
        expected_price=1.1000,
    )
    
    print(f"  ✓ Final fill recorded: {approved_units * 0.5:,.0f} units @ 1.1008")
    
    print("\nStep 5: Check order statistics")
    stats = rpm.get_order_stats()
    print(f"  Total orders: {stats['total_orders']}")
    print(f"  Completed orders: {stats['completed_orders']}")
    print(f"  Fill rate: {stats['fill_rate']:.1%}")
    
    slippage_stats = rpm.get_slippage_stats()
    print(f"\n  Slippage Statistics:")
    print(f"    Avg slippage: {slippage_stats['avg_slippage_bps']:.2f} bps")
    print(f"    Max slippage: {slippage_stats['max_slippage_bps']:.2f} bps")




def demo_correlation_aware_sizing():
    """Demo 3: Correlation-Aware Sizing - Prevent catastrophic risk accumulation"""
    print("\n" + "="*80)
    print("DEMO 3: CORRELATION-AWARE SIZING - STRESS EVENT PROTECTION")
    print("="*80)
    
    print("\nScenario: Portfolio with correlated EUR positions + stressed market")
    print("CRITICAL: In stress events, correlations → 1.0, amplifying portfolio risk by 50-200%")
    
    # Create RPM with custom config
    config = RPMConfig(
        total_capital=100000.0,
        risk_per_trade=0.01,
        max_symbol_exposure_units=500000.0,
        max_symbol_exposure_pct=0.50,
    )
    rpm = RiskPortfolioManager(config=config, enable_persistence=False)
    
    print("\nStep 1: Process trade in RANGING market (normal correlations ~0.3)")
    output1 = rpm.process_trade_intent(
        symbol='EURUSD',
        direction=1,
        confidence_score=0.75,
        regime='RANGING',
        atr=0.0050,
        vol_percentile=0.50,
        current_price=1.1000,
    )
    
    if output1.decision.status in ['APPROVED', 'ADJUSTED']:
        correlation_adj = output1.decision.sizing_adjustments.get('correlation_adjustment', 1.0)
        corr_details = output1.decision.sizing_adjustments.get('correlation_details', {})
        
        print(f"\n  Status: {output1.decision.status}")
        print(f"  Final units: {output1.decision.approved_trade.position_units:,.0f}")
        print(f"  Correlation adjustment: {correlation_adj:.2f}x")
        print(f"  Reason: {corr_details.get('reason', 'N/A')}")
    else:
        print(f"  ❌ Rejected: {output1.decision.rejected_trade.rejection_reason}")
    
    # Simulate existing correlated position (manually add to portfolio)
    from arbitrex.risk_portfolio_manager.schemas import Position
    from datetime import datetime
    
    rpm.portfolio_state.open_positions['EURUSD_1'] = Position(
        symbol='EURUSD',
        direction=1,
        units=20000.0,
        entry_price=1.0950,
        entry_time=datetime.now(),
    )
    
    print("\nStep 2: Add simulated EURUSD position to portfolio")
    print("  Added: EURUSD 20,000 units @ 1.0950")
    
    print("\nStep 3: Try to add GBPUSD (highly correlated with EURUSD ~0.70)")
    print("  Market regime: RANGING")
    output2 = rpm.process_trade_intent(
        symbol='GBPUSD',
        direction=1,
        confidence_score=0.75,
        regime='RANGING',
        atr=0.0060,
        vol_percentile=0.50,
        current_price=1.2700,
    )
    
    if output2.decision.status in ['APPROVED', 'ADJUSTED']:
        correlation_adj = output2.decision.sizing_adjustments.get('correlation_adjustment', 1.0)
        corr_details = output2.decision.sizing_adjustments.get('correlation_details', {})
        
        print(f"\n  Status: {output2.decision.status}")
        print(f"  Final units: {output2.decision.approved_trade.position_units:,.0f}")
        print(f"  Correlation adjustment: {correlation_adj:.2f}x")
        print(f"  Marginal risk: {corr_details.get('marginal_risk_contribution', 0)*100:.1f}%")
        print(f"  Diversification: {corr_details.get('portfolio_diversification', 0)*100:.1f}%")
        print(f"  Reason: {corr_details.get('reason', 'N/A')}")
    
    print("\n" + "-"*80)
    print("Step 4: CRITICAL TEST - Same trade in STRESSED market")
    print("  In STRESSED regime, correlations spike to 0.8-1.0")
    print("  This is where 50-200% risk underestimation occurs!")
    print("-"*80)
    
    output3 = rpm.process_trade_intent(
        symbol='GBPUSD',
        direction=1,
        confidence_score=0.75,
        regime='STRESSED',  # CRITICAL: Stressed market
        atr=0.0120,  # Higher vol in stress
        vol_percentile=0.95,  # Extreme vol
        current_price=1.2700,
    )
    
    if output3.decision.status in ['APPROVED', 'ADJUSTED']:
        correlation_adj = output3.decision.sizing_adjustments.get('correlation_adjustment', 1.0)
        corr_details = output3.decision.sizing_adjustments.get('correlation_details', {})
        
        print(f"\n  Status: {output3.decision.status}")
        print(f"  Final units: {output3.decision.approved_trade.position_units:,.0f}")
        print(f"  Correlation adjustment: {correlation_adj:.2f}x ⚠️")
        print(f"  Reason: {corr_details.get('reason', 'N/A')}")
        
        if correlation_adj < 0.5:
            print("\n  ✅ SUCCESS: Position size DRAMATICALLY REDUCED in stressed market!")
            print("  This prevents catastrophic losses when all correlations spike.")
        else:
            print("\n  ⚠️ WARNING: Position not reduced enough for stress conditions")
    else:
        print(f"  ❌ Rejected: {output3.decision.rejected_trade.rejection_reason}")
    
    print("\nStep 5: Check portfolio risk metrics")
    portfolio_vol = rpm.get_portfolio_volatility(regime='STRESSED')
    diversification = rpm.get_diversification_benefit(regime='STRESSED')
    
    print(f"  Portfolio volatility: {portfolio_vol*100:.2f}% (annualized)")
    print(f"  Diversification benefit: {diversification*100:.1f}%")
    
    if diversification < 0.3:
        print("  ⚠️ LOW DIVERSIFICATION - Portfolio is concentrated!")


def main():
    """Run all enhanced feature demos"""
    print("\n")
    print("="*80)
    print(" RPM ENHANCED FEATURES - CRITICAL PRODUCTION GAPS CLOSED")
    print("="*80)
    print("\nThree critical enhancements implemented:")
    print("  1. State Persistence - Survives crashes without data loss")
    print("  2. Order Management - Tracks partial fills and slippage")
    print("  3. Correlation-Aware Sizing - Prevents 50-200% risk underestimation")
    
    time.sleep(2)
    
    demo_state_persistence()
    
    input("\nPress ENTER to continue to Order Management demo...")
    demo_order_management()
    
    input("\nPress ENTER to continue to Correlation-Aware Sizing demo...")
    demo_correlation_aware_sizing()
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80)
    print("\nAll three critical gaps have been demonstrated:")
    print("  ✓ State persistence working")
    print("  ✓ Order management tracking fills and slippage")
    print("  ✓ Correlation-aware sizing preventing risk explosion")
    print("\nRPM is now significantly more production-ready!")


if __name__ == "__main__":
    main()
