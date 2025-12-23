"""
Integration tests for Kill-Switch system with trading components
"""

import pytest
import time
import threading
from unittest.mock import Mock

from arbitrex.risk_portfolio_manager.kill_switch import (
    KillSwitchManager,
    KillSwitchLevel,
    ResponseAction,
    TriggerReason,
    AlertConfig
)

from arbitrex.execution_engine.parallel_executor import (
    ParallelExecutionEngine,
    ExecutionOrder,
    OrderSide,
    VenueConnector,
    OrderStatus
)


class MockVenue(VenueConnector):
    """Mock venue for testing"""
    
    def __init__(self, venue_id: str):
        super().__init__(venue_id=venue_id, venue_name=venue_id)
        self.submitted_orders = []
    
    def submit_order(self, order: ExecutionOrder) -> tuple[OrderStatus, str]:
        self.submitted_orders.append(order)
        return (OrderStatus.FILLED, f"fill_{order.order_id}")
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED


class TestKillSwitchExecutionIntegration:
    """Test kill-switch integration with execution engine"""
    
    def test_global_kill_switch_blocks_all_orders(self):
        """Test global kill-switch blocks all execution"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        # Register mock venue
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Normal operation - should work
        order1 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            price=1.1000,
            strategy_id="test_strategy"
        )
        
        future1 = engine.submit_order(order1)
        assert future1 is not None
        result1 = future1.result(timeout=2.0)
        assert result1 is not None
        assert len(venue.submitted_orders) == 1
        
        # Activate global kill-switch
        kill_switch.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.SUSPEND,
            TriggerReason.MARGIN_CALL
        )
        
        # Orders should now be blocked
        order2 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            price=1.1000,
            strategy_id="test_strategy"
        )
        
        future2 = engine.submit_order(order2)
        assert future2 is None  # Blocked by kill-switch
        assert len(venue.submitted_orders) == 1  # No new order
        
        # Cleanup
        engine.shutdown()
        kill_switch.stop()
    
    def test_strategy_level_kill_switch(self):
        """Test strategy-specific kill-switch"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Block specific strategy
        kill_switch.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "blocked_strategy",
            ResponseAction.SUSPEND,
            TriggerReason.CONSECUTIVE_LOSSES
        )
        
        # Blocked strategy should fail
        order1 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="blocked_strategy"
        )
        
        future1 = engine.submit_order(order1)
        assert future1 is None
        assert len(venue.submitted_orders) == 0
        
        # Other strategy should work
        order2 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="allowed_strategy"
        )
        
        future2 = engine.submit_order(order2)
        assert future2 is not None
        result2 = future2.result(timeout=2.0)
        assert result2 is not None
        assert len(venue.submitted_orders) == 1
        
        engine.shutdown()
        kill_switch.stop()
    
    def test_symbol_level_kill_switch(self):
        """Test symbol-specific kill-switch"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Block specific symbol
        kill_switch.activate_kill_switch(
            KillSwitchLevel.SYMBOL,
            "GBPUSD",
            ResponseAction.SUSPEND,
            TriggerReason.LIQUIDITY_CRISIS
        )
        
        # Blocked symbol should fail
        order1 = ExecutionOrder(
            symbol="GBPUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future1 = engine.submit_order(order1)
        assert future1 is None
        
        # Other symbol should work
        order2 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future2 = engine.submit_order(order2)
        assert future2 is not None
        result2 = future2.result(timeout=2.0)
        assert result2 is not None
        assert len(venue.submitted_orders) == 1
        
        engine.shutdown()
        kill_switch.stop()
    
    def test_throttle_allows_trading(self):
        """Test throttle mode still allows trading"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Activate throttle
        kill_switch.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        # Orders should still go through (but would be rate-limited in practice)
        order = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future = engine.submit_order(order)
        assert future is not None
        result = future.result(timeout=2.0)
        assert result is not None
        assert len(venue.submitted_orders) == 1
        
        engine.shutdown()
        kill_switch.stop()
    
    def test_hierarchical_override(self):
        """Test that global kill-switch overrides local ones"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Allow strategy-level throttle
        kill_switch.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        # Order should work
        order1 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future1 = engine.submit_order(order1)
        assert future1 is not None
        
        # Activate global shutdown
        kill_switch.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL
        )
        
        # Now order should be blocked despite strategy throttle
        order2 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future2 = engine.submit_order(order2)
        assert future2 is None
        
        engine.shutdown()
        kill_switch.stop()
    
    def test_concurrent_kill_switch_activations(self):
        """Test concurrent kill-switch activations don't break execution"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=10,
            workers_per_group=5,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        submitted_count = 0
        blocked_count = 0
        lock = threading.Lock()
        
        def submit_orders():
            nonlocal submitted_count, blocked_count
            for i in range(10):
                order = ExecutionOrder(
                    symbol="EURUSD",
                    side=OrderSide.BUY,
                    quantity=1.0,
                    strategy_id=f"strategy_{i % 3}"
                )
                
                future = engine.submit_order(order)
                with lock:
                    if future:
                        submitted_count += 1
                    else:
                        blocked_count += 1
                
                time.sleep(0.01)
        
        def activate_kill_switches():
            for i in range(5):
                kill_switch.activate_kill_switch(
                    KillSwitchLevel.STRATEGY,
                    f"strategy_{i % 3}",
                    ResponseAction.SUSPEND,
                    TriggerReason.LOSS_LIMIT
                )
                time.sleep(0.05)
        
        # Run concurrently
        thread1 = threading.Thread(target=submit_orders)
        thread2 = threading.Thread(target=submit_orders)
        thread3 = threading.Thread(target=activate_kill_switches)
        
        thread1.start()
        thread2.start()
        thread3.start()
        
        thread1.join()
        thread2.join()
        thread3.join()
        
        # Some orders should be blocked
        assert blocked_count > 0
        assert submitted_count > 0
        
        engine.shutdown()
        kill_switch.stop()
    
    def test_recovery_restores_trading(self):
        """Test that recovery allows trading again"""
        kill_switch = KillSwitchManager(enable_auto_recovery=False)
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=2,
            kill_switch_manager=kill_switch,
            emit_events=False
        )
        
        venue = MockVenue("test_venue")
        engine.venue_router.register_venue(venue, priority=1)
        
        # Block trading
        kill_switch.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SUSPEND,
            TriggerReason.LOSS_LIMIT
        )
        
        # Verify blocked
        order1 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future1 = engine.submit_order(order1)
        assert future1 is None
        
        # Recover
        kill_switch.deactivate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            deactivated_by="risk_manager"
        )
        
        # Should work now
        order2 = ExecutionOrder(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            strategy_id="test_strategy"
        )
        
        future2 = engine.submit_order(order2)
        assert future2 is not None
        result2 = future2.result(timeout=2.0)
        assert result2 is not None
        
        engine.shutdown()
        kill_switch.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
