"""
Tests for Parallel Execution Engine

Validates:
- 20 execution groups run concurrently
- Fills processed <1ms
- Multi-venue failover
- Backpressure handling
"""

import pytest
import time
import threading
from decimal import Decimal
from datetime import datetime

from arbitrex.execution_engine.parallel_executor import (
    ParallelExecutionEngine,
    ExecutionOrder,
    OrderSide,
    OrderStatus,
    VenueConnector,
    VenueStatus,
    FillEvent,
    FillProcessor,
    VenueRouter,
    BackpressureController
)


# ========================================
# Mock Venue Connector
# ========================================

class MockVenueConnector(VenueConnector):
    """Mock venue for testing"""
    
    def __init__(self, venue_id: str, venue_name: str, failure_rate: float = 0.0, latency_ms: float = 1.0):
        super().__init__(venue_id=venue_id, venue_name=venue_name)
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.submitted_orders = []
    
    def submit_order(self, order: ExecutionOrder) -> bool:
        """Mock order submission"""
        # Simulate latency
        time.sleep(self.latency_ms / 1000.0)
        
        # Simulate failures
        import random
        if random.random() < self.failure_rate:
            return False
        
        # Success
        self.submitted_orders.append(order.order_id)
        self.orders_submitted += 1
        self.orders_filled += 1
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED if order_id in self.submitted_orders else OrderStatus.PENDING


# ========================================
# Fill Processor Tests
# ========================================

class TestFillProcessor:
    """Test async fill processing"""
    
    def test_fill_processing_speed(self):
        """Test fills processed <1ms"""
        processor = FillProcessor()
        processor.start()
        
        # Track processing times
        processing_times = []
        
        def callback(fill: FillEvent):
            processing_times.append(fill.latency_ms)
        
        processor.register_callback(callback)
        
        # Submit 100 fills
        for i in range(100):
            fill = FillEvent(
                order_id=f"order_{i}",
                venue="test_venue",
                symbol="EURUSD",
                side=OrderSide.BUY,
                quantity=1.0,
                price=1.0800,
                latency_ms=0.5
            )
            processor.submit_fill(fill)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check metrics
        metrics = processor.get_metrics()
        assert metrics['fills_processed'] == 100
        assert metrics['fills_dropped'] == 0
        assert metrics['avg_processing_time_ms'] < 1.0  # <1ms processing
        
        processor.stop()
        
        print(f"\n✓ Processed 100 fills with avg time: {metrics['avg_processing_time_ms']:.4f}ms")
    
    def test_fill_queue_overflow(self):
        """Test backpressure when fill queue is full"""
        processor = FillProcessor(max_queue_size=10)
        processor.start()
        
        # Submit more fills than queue can hold (without processing)
        fills_submitted = 0
        fills_dropped = 0
        
        for i in range(20):
            fill = FillEvent(order_id=f"order_{i}", venue="test", symbol="EURUSD", side=OrderSide.BUY, quantity=1.0, price=1.08)
            if processor.submit_fill(fill):
                fills_submitted += 1
            else:
                fills_dropped += 1
        
        assert fills_submitted == 10  # Queue size
        assert fills_dropped == 10
        
        processor.stop()
        
        print(f"\n✓ Backpressure: {fills_submitted} submitted, {fills_dropped} dropped")
    
    def test_callback_execution(self):
        """Test fill callbacks are executed"""
        processor = FillProcessor()
        processor.start()
        
        received_fills = []
        
        def callback(fill: FillEvent):
            received_fills.append(fill.fill_id)
        
        processor.register_callback(callback)
        
        # Submit fills
        fill_ids = []
        for i in range(10):
            fill = FillEvent(order_id=f"order_{i}", venue="test", symbol="EURUSD", side=OrderSide.BUY, quantity=1.0, price=1.08)
            fill_ids.append(fill.fill_id)
            processor.submit_fill(fill)
        
        # Wait for processing
        time.sleep(0.2)
        
        # All fills should be received
        assert len(received_fills) == 10
        assert set(received_fills) == set(fill_ids)
        
        processor.stop()
        
        print(f"\n✓ All {len(received_fills)} fills received by callback")


# ========================================
# Venue Router Tests
# ========================================

class TestVenueRouter:
    """Test multi-venue routing with failover"""
    
    def test_venue_registration(self):
        """Test venue registration and priority"""
        router = VenueRouter()
        
        # Register venues with different priorities
        venue1 = MockVenueConnector("venue1", "Venue One")
        venue2 = MockVenueConnector("venue2", "Venue Two")
        venue3 = MockVenueConnector("venue3", "Venue Three")
        
        router.register_venue(venue1, priority=10)  # Highest priority
        router.register_venue(venue2, priority=20)
        router.register_venue(venue3, priority=30)  # Lowest priority
        
        # Check venue order
        health = router.get_venue_health()
        assert len(health) == 3
        assert 'venue1' in health
        
        print(f"\n✓ Registered {len(health)} venues")
    
    def test_venue_failover(self):
        """Test automatic failover on venue failure"""
        router = VenueRouter()
        
        # Register venues: first fails 100%, second succeeds
        venue1 = MockVenueConnector("venue1", "Failing Venue", failure_rate=1.0)
        venue2 = MockVenueConnector("venue2", "Working Venue", failure_rate=0.0)
        
        router.register_venue(venue1, priority=10)
        router.register_venue(venue2, priority=20)
        
        # Submit order
        order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
        success, venue_id, reason = router.route_order(order, failover_attempts=1)
        
        # Should succeed with venue2 after venue1 fails
        assert success
        assert venue_id == "venue2"
        assert venue1.orders_rejected > 0
        assert venue2.orders_filled > 0
        
        print(f"\n✓ Failover successful: venue1 rejected, routed to venue2")
    
    def test_preferred_venue(self):
        """Test preferred venue routing"""
        router = VenueRouter()
        
        venue1 = MockVenueConnector("venue1", "Venue One")
        venue2 = MockVenueConnector("venue2", "Venue Two")
        
        router.register_venue(venue1, priority=10)
        router.register_venue(venue2, priority=20)
        
        # Submit order with preferred venue
        order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0, preferred_venue="venue2")
        success, venue_id, reason = router.route_order(order)
        
        # Should route to preferred venue
        assert success
        assert venue_id == "venue2"
        
        print(f"\n✓ Routed to preferred venue: {venue_id}")
    
    def test_all_venues_fail(self):
        """Test behavior when all venues fail"""
        router = VenueRouter()
        
        # All venues fail
        venue1 = MockVenueConnector("venue1", "Venue One", failure_rate=1.0)
        venue2 = MockVenueConnector("venue2", "Venue Two", failure_rate=1.0)
        
        router.register_venue(venue1, priority=10)
        router.register_venue(venue2, priority=20)
        
        # Submit order
        order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
        success, venue_id, reason = router.route_order(order, failover_attempts=2)
        
        # Should fail
        assert not success
        assert reason == "all_venues_failed"
        
        print(f"\n✓ Correctly failed when all venues unavailable: {reason}")


# ========================================
# Backpressure Tests
# ========================================

class TestBackpressureController:
    """Test backpressure handling"""
    
    def test_pending_order_limit(self):
        """Test rejection when pending orders exceed limit"""
        controller = BackpressureController(max_pending_orders=10)
        
        # Add 10 pending orders
        for i in range(10):
            controller.register_submission(f"order_{i}")
        
        # Should reject new order
        assert not controller.can_accept_order()
        assert controller.orders_rejected_backpressure == 1
        
        # Complete one order
        controller.register_completion("order_0")
        
        # Should accept now
        assert controller.can_accept_order()
        
        print(f"\n✓ Backpressure limit enforced: {controller.get_queue_depth()} pending")
    
    def test_rate_limiting(self):
        """Test orders per second rate limiting"""
        controller = BackpressureController(max_orders_per_second=10)
        
        # Submit 10 orders (at limit)
        for i in range(10):
            assert controller.can_accept_order()
            controller.register_submission(f"order_{i}")
        
        # 11th order should be throttled
        assert not controller.can_accept_order()
        assert controller.orders_throttled == 1
        
        # Wait 1 second
        time.sleep(1.1)
        
        # Should accept now
        assert controller.can_accept_order()
        
        print(f"\n✓ Rate limiting enforced: {controller.orders_throttled} throttled")
    
    def test_saturation_detection(self):
        """Test saturation detection"""
        controller = BackpressureController(max_pending_orders=100, queue_warning_threshold=50)
        
        # Not saturated initially
        assert not controller.is_saturated()
        
        # Add orders until saturated
        for i in range(60):
            controller.register_submission(f"order_{i}")
        
        # Should be saturated
        assert controller.is_saturated()
        assert controller.get_queue_depth() == 60
        
        print(f"\n✓ Saturation detected at {controller.get_queue_depth()} orders")


# ========================================
# Execution Group Tests
# ========================================

class TestExecutionGroup:
    """Test single execution group"""
    
    def test_order_execution(self):
        """Test basic order execution in group"""
        from arbitrex.execution_engine.parallel_executor import ExecutionGroup
        
        # Setup
        venue_router = VenueRouter()
        venue = MockVenueConnector("venue1", "Test Venue")
        venue_router.register_venue(venue)
        
        fill_processor = FillProcessor()
        fill_processor.start()
        
        group = ExecutionGroup(
            group_id="test_group",
            max_workers=5,
            venue_router=venue_router,
            fill_processor=fill_processor
        )
        
        # Submit order
        order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
        future = group.submit_order(order)
        
        # Wait for completion
        result = future.result(timeout=5.0)
        
        # Check result
        assert result.status == OrderStatus.FILLED
        assert result.venue == "venue1"
        
        metrics = group.get_metrics()
        assert metrics['orders_submitted'] == 1
        assert metrics['orders_completed'] == 1
        
        group.shutdown()
        fill_processor.stop()
        
        print(f"\n✓ Order executed: {result.order_id} → {result.status.value}")
    
    def test_parallel_execution(self):
        """Test multiple orders execute in parallel"""
        from arbitrex.execution_engine.parallel_executor import ExecutionGroup
        
        venue_router = VenueRouter()
        venue = MockVenueConnector("venue1", "Test Venue", latency_ms=50)  # 50ms latency
        venue_router.register_venue(venue)
        
        group = ExecutionGroup(
            group_id="test_group",
            max_workers=10,
            venue_router=venue_router
        )
        
        # Submit 10 orders
        start_time = time.perf_counter()
        futures = []
        for i in range(10):
            order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
            future = group.submit_order(order)
            futures.append(future)
        
        # Wait for all
        for future in futures:
            future.result(timeout=5.0)
        
        elapsed = time.perf_counter() - start_time
        
        # Parallel execution should be much faster than sequential (500ms)
        # Allow 1s for thread pool overhead
        assert elapsed < 1.0
        
        group.shutdown()
        
        print(f"\n✓ 10 orders executed in parallel: {elapsed*1000:.2f}ms (sequential would be ~500ms)")


# ========================================
# Parallel Execution Engine Tests
# ========================================

class TestParallelExecutionEngine:
    """Test complete parallel execution engine"""
    
    def test_engine_initialization(self):
        """Test engine initializes with correct configuration"""
        engine = ParallelExecutionEngine(
            num_groups=20,
            workers_per_group=5,
            emit_events=False
        )
        
        metrics = engine.get_metrics()
        assert metrics['num_groups'] == 20
        assert metrics['workers_per_group'] == 5
        assert metrics['total_workers'] == 100
        
        engine.shutdown()
        
        print(f"\n✓ Engine initialized: {metrics['total_workers']} total workers")
    
    def test_20_groups_concurrent(self):
        """Test 20 execution groups run concurrently"""
        engine = ParallelExecutionEngine(
            num_groups=20,
            workers_per_group=5,
            enable_backpressure=False,
            emit_events=False
        )
        
        # Register venue
        venue = MockVenueConnector("venue1", "Test Venue", latency_ms=10)
        engine.venue_router.register_venue(venue)
        
        # Submit 100 orders (spread across 20 groups)
        start_time = time.perf_counter()
        futures = []
        for i in range(100):
            order = ExecutionOrder(
                symbol="EURUSD",
                side=OrderSide.BUY,
                quantity=1.0,
                signal_id=f"signal_{i}"
            )
            future = engine.submit_order(order)
            if future:
                futures.append(future)
        
        # Wait for all
        for future in futures:
            future.result(timeout=10.0)
        
        elapsed = time.perf_counter() - start_time
        
        # Check metrics
        metrics = engine.get_metrics()
        assert metrics['total_orders_completed'] == 100
        
        # Check distribution across groups (should be relatively even)
        group_counts = [g['orders_submitted'] for g in metrics['groups']]
        assert len([c for c in group_counts if c > 0]) >= 15  # At least 15 groups used
        
        engine.shutdown()
        
        print(f"\n✓ 100 orders across 20 groups in {elapsed*1000:.2f}ms")
        print(f"✓ Groups used: {len([c for c in group_counts if c > 0])}/20")
    
    def test_fill_processing_latency(self):
        """Test fills processed <1ms"""
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=5,
            emit_events=False
        )
        
        # Register venue
        venue = MockVenueConnector("venue1", "Test Venue", latency_ms=1)
        engine.venue_router.register_venue(venue)
        
        # Submit orders
        futures = []
        for i in range(50):
            order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
            future = engine.submit_order(order)
            if future:
                futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result(timeout=5.0)
        
        time.sleep(0.2)  # Allow fills to process
        
        # Check fill processing time
        fill_metrics = engine.fill_processor.get_metrics()
        assert fill_metrics['fills_processed'] == 50
        assert fill_metrics['avg_processing_time_ms'] < 1.0  # <1ms
        
        engine.shutdown()
        
        print(f"\n✓ 50 fills processed with avg latency: {fill_metrics['avg_processing_time_ms']:.4f}ms")
    
    def test_backpressure_handling(self):
        """Test backpressure prevents overload"""
        engine = ParallelExecutionEngine(
            num_groups=2,
            workers_per_group=2,
            enable_backpressure=True,
            emit_events=False
        )
        
        # Configure strict backpressure
        engine.backpressure.max_pending_orders = 10
        
        # Register slow venue
        venue = MockVenueConnector("venue1", "Slow Venue", latency_ms=100)
        engine.venue_router.register_venue(venue)
        
        # Submit many orders quickly
        accepted = 0
        rejected = 0
        
        for i in range(30):
            order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
            future = engine.submit_order(order)
            if future:
                accepted += 1
            else:
                rejected += 1
        
        # Some should be rejected due to backpressure
        assert rejected > 0
        assert engine.backpressure.orders_rejected_backpressure > 0
        
        engine.shutdown()
        
        print(f"\n✓ Backpressure: {accepted} accepted, {rejected} rejected")
    
    def test_venue_failover_integration(self):
        """Test multi-venue failover in full engine"""
        engine = ParallelExecutionEngine(
            num_groups=5,
            workers_per_group=3,
            emit_events=False
        )
        
        # Register venues: primary fails, backup succeeds
        venue1 = MockVenueConnector("venue1", "Primary (failing)", failure_rate=1.0)
        venue2 = MockVenueConnector("venue2", "Backup (working)", failure_rate=0.0)
        
        engine.venue_router.register_venue(venue1, priority=10)
        engine.venue_router.register_venue(venue2, priority=20)
        
        # Submit orders
        futures = []
        for i in range(20):
            order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
            future = engine.submit_order(order)
            if future:
                futures.append(future)
        
        # Wait
        completed = 0
        for future in futures:
            result = future.result(timeout=5.0)
            if result.status == OrderStatus.FILLED:
                completed += 1
                assert result.venue == "venue2"  # Should all be routed to backup
        
        assert completed == 20
        
        # Check venue health
        health = engine.venue_router.get_venue_health()
        assert health['venue2']['orders_filled'] == 20
        
        engine.shutdown()
        
        print(f"\n✓ Failover successful: {completed} orders routed to backup venue")
    
    def test_performance_benchmark(self):
        """Benchmark: 20 groups × 5 workers = 100 concurrent executions"""
        engine = ParallelExecutionEngine(
            num_groups=20,
            workers_per_group=5,
            enable_backpressure=False,
            emit_events=False
        )
        
        # Fast venue
        venue = MockVenueConnector("venue1", "Fast Venue", latency_ms=2)
        engine.venue_router.register_venue(venue)
        
        # Benchmark
        num_orders = 500
        start_time = time.perf_counter()
        
        futures = []
        for i in range(num_orders):
            order = ExecutionOrder(symbol="EURUSD", side=OrderSide.BUY, quantity=1.0)
            future = engine.submit_order(order)
            if future:
                futures.append(future)
        
        # Wait for all
        for future in futures:
            future.result(timeout=30.0)
        
        elapsed = time.perf_counter() - start_time
        throughput = num_orders / elapsed
        
        # Check metrics
        metrics = engine.get_metrics()
        fill_metrics = metrics['fill_processor']
        
        assert metrics['total_orders_completed'] == num_orders
        assert fill_metrics['avg_processing_time_ms'] < 1.0
        
        engine.shutdown()
        
        print(f"\n✓ Executed {num_orders} orders in {elapsed:.2f}s")
        print(f"✓ Throughput: {throughput:.1f} orders/sec")
        print(f"✓ Fill processing: {fill_metrics['avg_processing_time_ms']:.4f}ms avg")
        print(f"✓ Success rate: {metrics['overall_success_rate']*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
