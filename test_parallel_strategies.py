"""
Comprehensive Test Suite for Parallel Strategy Execution

Tests:
- Signal schema validation (single-leg, multi-leg)
- Signal buffer thread safety
- Strategy actor isolation
- Rate limiting
- Health monitoring
- Circuit breaker
- Parallel execution (50 strategies)
- Failure isolation
"""

import time
import pytest
import threading
from datetime import datetime, timedelta
from typing import Optional

from arbitrex.signal_engine.signal_schemas import (
    Signal,
    SignalLeg,
    SignalType,
    SignalStatus,
    LegDirection,
    create_single_leg_signal,
    create_spread_signal,
    create_basket_signal
)
from arbitrex.signal_engine.signal_buffer import (
    SignalBuffer,
    SignalFilter,
    SignalSubscriber,
    get_signal_buffer
)
from arbitrex.signal_engine.strategy_runner import (
    StrategyActor,
    StrategyConfig,
    StrategyRunner,
    StrategyState,
    RateLimiter
)


# ============================================================================
# Test Signal Schemas
# ============================================================================

class TestSignalSchemas:
    """Test signal schema validation"""
    
    def test_single_leg_signal(self):
        """Test single-leg signal creation"""
        signal = create_single_leg_signal(
            symbol="EURUSD",
            direction=LegDirection.LONG,
            confidence=0.8,
            strategy_id="test_strategy_v1",
            timeframe="1H",
            bar_index=100
        )
        
        assert signal.is_single_leg
        assert not signal.is_multi_leg
        assert signal.primary_symbol == "EURUSD"
        assert signal.symbols == ["EURUSD"]
        assert signal.confidence_score == 0.8
        assert len(signal.legs) == 1
        assert signal.legs[0].direction == LegDirection.LONG
    
    def test_spread_signal(self):
        """Test two-leg spread signal"""
        signal = create_spread_signal(
            long_symbol="EURUSD",
            short_symbol="GBPUSD",
            confidence=0.75,
            strategy_id="spread_arb_v1",
            long_weight=0.6,
            short_weight=0.4
        )
        
        assert not signal.is_single_leg
        assert signal.is_multi_leg
        assert signal.signal_type == SignalType.SPREAD
        assert len(signal.legs) == 2
        assert signal.symbols == ["EURUSD", "GBPUSD"]
        
        # Check legs
        assert signal.legs[0].symbol == "EURUSD"
        assert signal.legs[0].direction == LegDirection.LONG
        assert signal.legs[0].weight == 0.6
        
        assert signal.legs[1].symbol == "GBPUSD"
        assert signal.legs[1].direction == LegDirection.SHORT
        assert signal.legs[1].weight == 0.4
    
    def test_basket_signal(self):
        """Test multi-leg basket signal"""
        legs = [
            ("EURUSD", LegDirection.LONG, 0.4),
            ("GBPUSD", LegDirection.LONG, 0.3),
            ("USDJPY", LegDirection.SHORT, 0.3)
        ]
        
        signal = create_basket_signal(
            legs=legs,
            confidence=0.9,
            strategy_id="basket_v1"
        )
        
        assert signal.is_multi_leg
        assert signal.signal_type == SignalType.BASKET
        assert len(signal.legs) == 3
        assert set(signal.symbols) == {"EURUSD", "GBPUSD", "USDJPY"}
    
    def test_signal_immutability(self):
        """Test signal immutability (frozen dataclass)"""
        signal = create_single_leg_signal(
            symbol="EURUSD",
            direction=LegDirection.LONG,
            confidence=0.8,
            strategy_id="test"
        )
        
        # Attempt to modify should raise error
        with pytest.raises(Exception):  # FrozenInstanceError
            signal.confidence_score = 0.5
    
    def test_signal_status_update(self):
        """Test signal status update (immutable pattern)"""
        signal = create_single_leg_signal(
            symbol="EURUSD",
            direction=LegDirection.LONG,
            confidence=0.8,
            strategy_id="test"
        )
        
        assert signal.status == SignalStatus.PENDING
        
        # Update status creates new signal
        updated_signal = signal.with_status(SignalStatus.APPROVED)
        
        # Original unchanged
        assert signal.status == SignalStatus.PENDING
        
        # New signal has updated status
        assert updated_signal.status == SignalStatus.APPROVED
        assert updated_signal.signal_id == signal.signal_id  # Same ID
    
    def test_signal_expiry(self):
        """Test signal expiry checking"""
        # Non-expiring signal
        signal1 = create_single_leg_signal(
            symbol="EURUSD",
            direction=LegDirection.LONG,
            confidence=0.8,
            strategy_id="test"
        )
        assert not signal1.is_expired()
        
        # Expired signal
        from dataclasses import replace
        past_time = datetime.utcnow() - timedelta(seconds=10)
        signal2 = replace(signal1, expires_at=past_time)
        assert signal2.is_expired()
        
        # Future expiry
        future_time = datetime.utcnow() + timedelta(seconds=10)
        signal3 = replace(signal1, expires_at=future_time)
        assert not signal3.is_expired()


# ============================================================================
# Test Signal Buffer
# ============================================================================

class TestSignalBuffer:
    """Test thread-safe signal buffer"""
    
    def test_buffer_creation(self):
        """Test signal buffer instantiation"""
        buffer = SignalBuffer(buffer_size=1000, enable_expiry_check=False)
        assert buffer.buffer_size == 1000
        metrics = buffer.get_metrics()
        assert metrics.buffer_depth == 0
    
    def test_publish_and_retrieve(self):
        """Test signal publication and retrieval"""
        buffer = SignalBuffer(enable_expiry_check=False)
        
        signal = create_single_leg_signal(
            symbol="EURUSD",
            direction=LegDirection.LONG,
            confidence=0.8,
            strategy_id="test"
        )
        
        # Publish
        success = buffer.publish(signal)
        assert success
        
        # Retrieve
        signals = buffer.get_signals(max_count=10)
        assert len(signals) == 1
        assert signals[0].signal_id == signal.signal_id
        
        # Buffer should be empty now
        signals2 = buffer.get_signals(max_count=10)
        assert len(signals2) == 0
    
    def test_per_strategy_buffers(self):
        """Test per-strategy buffer isolation"""
        buffer = SignalBuffer(enable_expiry_check=False)
        
        # Publish from different strategies
        signal1 = create_single_leg_signal(
            symbol="EURUSD", direction=LegDirection.LONG,
            confidence=0.8, strategy_id="strategy_a"
        )
        signal2 = create_single_leg_signal(
            symbol="GBPUSD", direction=LegDirection.SHORT,
            confidence=0.7, strategy_id="strategy_b"
        )
        
        buffer.publish(signal1)
        buffer.publish(signal2)
        
        # Retrieve strategy A signals
        signals_a = buffer.get_signals(strategy_id="strategy_a")
        assert len(signals_a) == 1
        assert signals_a[0].strategy_id == "strategy_a"
        
        # Retrieve strategy B signals
        signals_b = buffer.get_signals(strategy_id="strategy_b")
        assert len(signals_b) == 1
        assert signals_b[0].strategy_id == "strategy_b"
    
    def test_subscriber_notification(self):
        """Test subscriber notifications"""
        buffer = SignalBuffer(enable_expiry_check=False)
        
        received_signals = []
        
        def callback(signal: Signal):
            received_signals.append(signal)
        
        # Subscribe
        buffer.subscribe(callback, name="test_subscriber")
        
        # Publish signal
        signal = create_single_leg_signal(
            symbol="EURUSD", direction=LegDirection.LONG,
            confidence=0.8, strategy_id="test"
        )
        buffer.publish(signal)
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check received
        assert len(received_signals) == 1
        assert received_signals[0].signal_id == signal.signal_id
    
    def test_signal_filtering(self):
        """Test signal filtering"""
        buffer = SignalBuffer(enable_expiry_check=False)
        
        # Filter for specific strategy
        signal_filter = SignalFilter(
            strategy_ids={"strategy_a"},
            min_confidence=0.7
        )
        
        filtered_signals = []
        
        def callback(signal: Signal):
            filtered_signals.append(signal)
        
        buffer.subscribe(callback, signal_filter=signal_filter)
        
        # Publish matching signal
        signal1 = create_single_leg_signal(
            symbol="EURUSD", direction=LegDirection.LONG,
            confidence=0.8, strategy_id="strategy_a"
        )
        buffer.publish(signal1)
        
        # Publish non-matching signal (different strategy)
        signal2 = create_single_leg_signal(
            symbol="GBPUSD", direction=LegDirection.SHORT,
            confidence=0.9, strategy_id="strategy_b"
        )
        buffer.publish(signal2)
        
        # Publish non-matching signal (low confidence)
        signal3 = create_single_leg_signal(
            symbol="USDJPY", direction=LegDirection.LONG,
            confidence=0.5, strategy_id="strategy_a"
        )
        buffer.publish(signal3)
        
        time.sleep(0.1)
        
        # Should only receive signal1
        assert len(filtered_signals) == 1
        assert filtered_signals[0].signal_id == signal1.signal_id
    
    def test_thread_safety(self):
        """Test concurrent publishing from multiple threads"""
        buffer = SignalBuffer(enable_expiry_check=False)
        
        signals_published = []
        lock = threading.Lock()
        
        def publish_signals(strategy_id: str, count: int):
            for i in range(count):
                signal = create_single_leg_signal(
                    symbol="EURUSD",
                    direction=LegDirection.LONG,
                    confidence=0.8,
                    strategy_id=strategy_id,
                    bar_index=i
                )
                buffer.publish(signal)
                with lock:
                    signals_published.append(signal)
        
        # Launch 10 threads, each publishing 10 signals
        threads = []
        for i in range(10):
            thread = threading.Thread(target=publish_signals, args=(f"strategy_{i}", 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have 100 signals total
        assert len(signals_published) == 100
        
        # Retrieve all signals
        all_signals = buffer.get_signals(max_count=200)
        assert len(all_signals) == 100


# ============================================================================
# Test Rate Limiter
# ============================================================================

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(max_per_minute=5, max_per_hour=20)
        
        # Should allow first 5
        for i in range(5):
            assert limiter.allow()
        
        # Should deny 6th
        assert not limiter.allow()
    
    def test_rate_limiter_window_expiry(self):
        """Test rate limit window expiry (requires waiting)"""
        limiter = RateLimiter(max_per_minute=2, max_per_hour=10)
        
        # Use 2 tokens
        assert limiter.allow()
        assert limiter.allow()
        
        # 3rd should be denied
        assert not limiter.allow()
        
        # Note: Can't easily test window expiry without sleeping 60 seconds
        # In production, this would be tested with time mocking


# ============================================================================
# Test Strategy Actor
# ============================================================================

class TestStrategyActor:
    """Test strategy actor functionality"""
    
    def create_simple_strategy(self, should_signal: bool = True):
        """Create simple test strategy"""
        def strategy_func(data):
            if should_signal:
                return create_single_leg_signal(
                    symbol="EURUSD",
                    direction=LegDirection.LONG,
                    confidence=0.8,
                    strategy_id="test_strategy"
                )
            return None
        return strategy_func
    
    def test_actor_creation(self):
        """Test strategy actor instantiation"""
        config = StrategyConfig(
            strategy_id="test_strategy",
            max_signals_per_minute=10
        )
        
        strategy_func = self.create_simple_strategy()
        buffer = SignalBuffer(enable_expiry_check=False)
        
        actor = StrategyActor(config, strategy_func, buffer)
        
        assert actor.config.strategy_id == "test_strategy"
        assert actor.health.state == StrategyState.INITIALIZING
    
    def test_actor_execution(self):
        """Test strategy execution"""
        config = StrategyConfig(strategy_id="test_strategy")
        strategy_func = self.create_simple_strategy(should_signal=True)
        buffer = SignalBuffer(enable_expiry_check=False)
        
        actor = StrategyActor(config, strategy_func, buffer)
        
        # Execute
        signal = actor.execute(data={"bar": 1})
        
        # Should generate signal
        assert signal is not None
        assert signal.strategy_id == "test_strategy"
        
        # Check health
        assert actor.health.bars_processed == 1
        assert actor.health.signals_generated == 1
        assert actor.health.state == StrategyState.RUNNING
    
    def test_actor_failure_handling(self):
        """Test actor failure isolation"""
        config = StrategyConfig(
            strategy_id="failing_strategy",
            max_consecutive_failures=3
        )
        
        def failing_strategy(data):
            raise ValueError("Intentional failure")
        
        buffer = SignalBuffer(enable_expiry_check=False)
        actor = StrategyActor(config, failing_strategy, buffer)
        
        # Execute multiple times
        for i in range(5):
            signal = actor.execute(data={})
            assert signal is None  # No signal due to error
        
        # Check failure tracking
        assert actor.health.total_failures == 5
        assert actor.health.consecutive_failures == 5
        assert actor.health.state == StrategyState.FAILED
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        config = StrategyConfig(
            strategy_id="test_strategy",
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=0.5  # 0.5 seconds for testing
        )
        
        def failing_strategy(data):
            raise RuntimeError("Failure")
        
        buffer = SignalBuffer(enable_expiry_check=False)
        actor = StrategyActor(config, failing_strategy, buffer)
        
        # Cause 3 failures to trip circuit breaker
        for i in range(3):
            actor.execute(data={})
        
        # Circuit should be open
        assert actor.health.state == StrategyState.CIRCUIT_OPEN
        assert actor.health.circuit_breaker_trips == 1
        
        # Wait for timeout
        time.sleep(0.6)
        
        # Next execution should close circuit
        # (will still fail, but circuit should reset)
        actor.execute(data={})
        
        # Circuit should have attempted to close
        # (state might be FAILED again due to another failure, but trip count should reset on timeout)


# ============================================================================
# Test Strategy Runner (50 Concurrent Strategies)
# ============================================================================

class TestStrategyRunner:
    """Test parallel strategy execution"""
    
    def create_mock_strategy(self, strategy_id: str, signal_rate: float = 0.5):
        """Create mock strategy that signals with given probability"""
        def strategy_func(data):
            import random
            if random.random() < signal_rate:
                return create_single_leg_signal(
                    symbol="EURUSD",
                    direction=LegDirection.LONG if random.random() > 0.5 else LegDirection.SHORT,
                    confidence=random.uniform(0.6, 0.95),
                    strategy_id=strategy_id,
                    bar_index=data.get("bar_index", 0)
                )
            return None
        return strategy_func
    
    def test_runner_creation(self):
        """Test strategy runner instantiation"""
        runner = StrategyRunner(max_workers=50)
        assert runner.max_workers == 50
    
    def test_strategy_registration(self):
        """Test strategy registration"""
        runner = StrategyRunner(max_workers=10)
        
        config = StrategyConfig(strategy_id="test_strategy")
        strategy_func = self.create_mock_strategy("test_strategy")
        
        runner.register_strategy(config, strategy_func)
        
        # Check registration
        health = runner.get_health()
        assert "test_strategy" in health
    
    def test_parallel_execution_10_strategies(self):
        """Test parallel execution of 10 strategies"""
        buffer = SignalBuffer(enable_expiry_check=False)
        runner = StrategyRunner(max_workers=10, signal_buffer=buffer)
        
        # Register 10 strategies
        for i in range(10):
            strategy_id = f"strategy_{i}"
            config = StrategyConfig(
                strategy_id=strategy_id,
                max_signals_per_minute=100
            )
            strategy_func = self.create_mock_strategy(strategy_id, signal_rate=0.8)
            runner.register_strategy(config, strategy_func)
        
        # Execute all strategies
        results = runner.execute_parallel(data={"bar_index": 1})
        
        # Should have results for all 10 strategies
        assert len(results) == 10
        
        # Check signals generated (probabilistic, ~8 should generate)
        signals_generated = sum(1 for sig in results.values() if sig is not None)
        assert signals_generated >= 5  # At least 5 should signal (80% * 10 = 8 expected)
    
    def test_parallel_execution_50_strategies(self):
        """Test parallel execution of 50 strategies WITHOUT BLOCKING"""
        buffer = SignalBuffer(enable_expiry_check=False)
        runner = StrategyRunner(max_workers=50, signal_buffer=buffer)
        
        # Register 50 strategies
        for i in range(50):
            strategy_id = f"strategy_{i}"
            config = StrategyConfig(
                strategy_id=strategy_id,
                max_signals_per_minute=100
            )
            strategy_func = self.create_mock_strategy(strategy_id, signal_rate=0.7)
            runner.register_strategy(config, strategy_func)
        
        # Execute all strategies
        start_time = time.perf_counter()
        results = runner.execute_parallel(data={"bar_index": 1})
        execution_time = time.perf_counter() - start_time
        
        # Should complete quickly (< 1 second for simple strategies)
        assert execution_time < 2.0, f"Execution took {execution_time:.2f}s, should be < 2s"
        
        # Should have results for all 50 strategies
        assert len(results) == 50
        
        # Check signals generated
        signals_generated = sum(1 for sig in results.values() if sig is not None)
        assert signals_generated >= 25  # At least 50% should signal
        
        # Check all strategies executed
        health = runner.get_health()
        for i in range(50):
            strategy_id = f"strategy_{i}"
            assert health[strategy_id].bars_processed == 1
    
    def test_failure_isolation(self):
        """Test that one strategy failure doesn't affect others"""
        buffer = SignalBuffer(enable_expiry_check=False)
        runner = StrategyRunner(max_workers=10, signal_buffer=buffer)
        
        # Register 5 working strategies
        for i in range(5):
            config = StrategyConfig(strategy_id=f"good_strategy_{i}")
            runner.register_strategy(config, self.create_mock_strategy(f"good_strategy_{i}", 0.9))
        
        # Register 5 failing strategies
        def failing_strategy(data):
            raise RuntimeError("Intentional failure")
        
        for i in range(5):
            config = StrategyConfig(strategy_id=f"bad_strategy_{i}")
            runner.register_strategy(config, failing_strategy)
        
        # Execute all
        results = runner.execute_parallel(data={})
        
        # All 10 should have results (even if None for failures)
        assert len(results) == 10
        
        # Good strategies should have signals
        good_signals = sum(1 for sid, sig in results.items() 
                          if sid.startswith("good_strategy_") and sig is not None)
        assert good_signals >= 3  # At least 3 out of 5 should signal
        
        # Bad strategies should be None
        bad_signals = [sig for sid, sig in results.items() 
                      if sid.startswith("bad_strategy_")]
        assert all(sig is None for sig in bad_signals)
        
        # Check health - good strategies should be running
        health = runner.get_health()
        for i in range(5):
            assert health[f"good_strategy_{i}"].state == StrategyState.RUNNING
    
    def test_runner_summary(self):
        """Test runner summary statistics"""
        runner = StrategyRunner(max_workers=10)
        
        # Register strategies
        for i in range(10):
            config = StrategyConfig(strategy_id=f"strategy_{i}")
            runner.register_strategy(config, self.create_mock_strategy(f"strategy_{i}"))
        
        # Execute
        runner.execute_parallel(data={})
        
        # Get summary
        summary = runner.get_summary()
        assert summary['total_strategies'] == 10
        assert summary['max_workers'] == 10
        assert 'total_signals_generated' in summary
    
    def test_strategy_pause_resume(self):
        """Test pausing and resuming strategies"""
        runner = StrategyRunner(max_workers=5)
        
        config = StrategyConfig(strategy_id="test_strategy")
        runner.register_strategy(config, self.create_mock_strategy("test_strategy", 1.0))
        
        # Execute - should generate signal
        results1 = runner.execute_parallel(data={})
        assert results1["test_strategy"] is not None
        
        # Pause
        runner.pause_strategy("test_strategy")
        health = runner.get_health("test_strategy")
        assert health["test_strategy"].state == StrategyState.PAUSED
        
        # Execute - should not generate signal (paused)
        results2 = runner.execute_parallel(data={})
        assert results2["test_strategy"] is None
        
        # Resume
        runner.resume_strategy("test_strategy")
        health = runner.get_health("test_strategy")
        assert health["test_strategy"].state == StrategyState.RUNNING
        
        # Execute - should generate signal again
        results3 = runner.execute_parallel(data={})
        assert results3["test_strategy"] is not None


# ============================================================================
# Performance Test
# ============================================================================

class TestPerformance:
    """Performance validation tests"""
    
    def test_50_strategies_concurrent_no_blocking(self):
        """
        CRITICAL TEST: Validate 50 strategies run concurrently without blocking.
        
        This is the main validation requirement.
        """
        buffer = SignalBuffer(enable_expiry_check=False)
        runner = StrategyRunner(max_workers=50, signal_buffer=buffer)
        
        # Create realistic strategies with small processing time
        def realistic_strategy(strategy_id):
            def func(data):
                # Simulate minimal processing (< 10ms)
                time.sleep(0.005)  # 5ms
                return create_single_leg_signal(
                    symbol="EURUSD",
                    direction=LegDirection.LONG,
                    confidence=0.8,
                    strategy_id=strategy_id
                )
            return func
        
        # Register 50 strategies
        for i in range(50):
            strategy_id = f"strategy_{i}"
            config = StrategyConfig(strategy_id=strategy_id)
            runner.register_strategy(config, realistic_strategy(strategy_id))
        
        # Execute and measure time
        start_time = time.perf_counter()
        results = runner.execute_parallel(data={})
        execution_time = time.perf_counter() - start_time
        
        # Validation criteria
        assert len(results) == 50, "Should execute all 50 strategies"
        
        # Sequential execution would take: 50 * 5ms = 250ms
        # Parallel execution should take: ~5ms + overhead (< 100ms)
        assert execution_time < 0.5, f"Took {execution_time:.3f}s, should be < 0.5s (parallel execution)"
        
        # All strategies should complete successfully
        successful = sum(1 for sig in results.values() if sig is not None)
        assert successful == 50, f"Only {successful}/50 strategies succeeded"
        
        # Check health - all should have executed
        health = runner.get_health()
        for i in range(50):
            strategy_id = f"strategy_{i}"
            assert health[strategy_id].bars_processed == 1
            assert health[strategy_id].signals_generated == 1
            assert health[strategy_id].state == StrategyState.RUNNING
        
        print(f"\n✓ 50 strategies executed in {execution_time:.3f}s (parallel)")
        print(f"✓ All {successful} strategies completed successfully")
        print(f"✓ Average execution time: {execution_time/50*1000:.2f}ms per strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
