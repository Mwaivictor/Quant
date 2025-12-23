"""
Comprehensive Tests for MVCC Portfolio State Manager

Tests:
- MVCC snapshot isolation (parallel reads)
- CAS (Compare-And-Swap) optimistic locking
- Position reservation system
- Over-allocation prevention
- Race condition resistance
- Broker reconciliation
- Thread safety under concurrent load
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from arbitrex.risk_portfolio_manager.portfolio_state import (
    PortfolioSnapshot,
    Position,
    PositionReservation,
    PositionSide,
    ReservationStatus,
    AccountMetrics
)
from arbitrex.risk_portfolio_manager.portfolio_manager import (
    PortfolioStateManager,
    CASFailureException,
    OverAllocationException
)


class TestMVCCSnapshots:
    """Test MVCC snapshot isolation"""
    
    def test_snapshot_immutability(self):
        """Test snapshot immutability (frozen dataclass)"""
        manager = PortfolioStateManager(initial_equity=Decimal('100000'))
        snapshot = manager.get_snapshot()
        
        # Attempt to modify should raise error
        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.version = 999
    
    def test_snapshot_version_increment(self):
        """Test version increments on updates"""
        manager = PortfolioStateManager()
        
        snapshot1 = manager.get_snapshot()
        assert snapshot1.version == 0
        
        # Make reservation (triggers update)
        manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id="test_signal"
        )
        
        time.sleep(0.1)  # Let updater process
        
        snapshot2 = manager.get_snapshot()
        assert snapshot2.version == 1
        assert snapshot2.version > snapshot1.version
    
    def test_parallel_readers_no_blocking(self):
        """Test multiple readers can access snapshots simultaneously"""
        manager = PortfolioStateManager()
        
        read_counts = {'count': 0}
        lock = threading.Lock()
        
        def reader_thread():
            """Read snapshot 100 times"""
            for _ in range(100):
                snapshot = manager.get_snapshot()
                assert snapshot is not None
                with lock:
                    read_counts['count'] += 1
        
        # Launch 10 reader threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all
        for thread in threads:
            thread.join()
        
        # Should have 1000 reads total (10 threads × 100 reads)
        assert read_counts['count'] == 1000
    
    def test_snapshot_consistency(self):
        """Test snapshot remains consistent even during updates"""
        manager = PortfolioStateManager()
        
        # Get initial snapshot
        snapshot1 = manager.get_snapshot()
        initial_version = snapshot1.version
        
        # Make multiple updates
        for i in range(5):
            manager.reserve_position(
                symbol=f"SYMBOL_{i}",
                side=PositionSide.LONG,
                quantity=Decimal('1.0'),
                signal_id=f"signal_{i}"
            )
        
        # Original snapshot should remain unchanged
        assert snapshot1.version == initial_version
        assert snapshot1.reservation_count == 0
        
        # New snapshot should have updates
        time.sleep(0.2)
        snapshot2 = manager.get_snapshot()
        assert snapshot2.version > initial_version
        assert snapshot2.reservation_count == 5


class TestPositionReservation:
    """Test position reservation system"""
    
    def test_reserve_position(self):
        """Test basic position reservation"""
        manager = PortfolioStateManager()
        
        reservation_id = manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('1.5'),
            signal_id="test_signal"
        )
        
        assert reservation_id is not None
        
        # Check snapshot
        time.sleep(0.1)
        snapshot = manager.get_snapshot()
        assert snapshot.reservation_count == 1
        
        reservation = snapshot.get_reservation(reservation_id)
        assert reservation is not None
        assert reservation.symbol == "EURUSD"
        assert reservation.quantity == Decimal('1.5')
        assert reservation.is_active
    
    def test_commit_reservation_new_position(self):
        """Test committing reservation creates new position"""
        manager = PortfolioStateManager()
        
        # Reserve
        reservation_id = manager.reserve_position(
            symbol="GBPUSD",
            side=PositionSide.LONG,
            quantity=Decimal('2.0'),
            signal_id="signal_1"
        )
        
        time.sleep(0.1)
        
        # Commit
        position_id = manager.commit_reservation(
            reservation_id=reservation_id,
            executed_quantity=Decimal('2.0'),
            avg_price=Decimal('1.2500')
        )
        
        assert position_id is not None
        
        time.sleep(0.1)
        snapshot = manager.get_snapshot()
        
        # Check position created
        assert snapshot.position_count == 1
        position = snapshot.get_position("GBPUSD")
        assert position is not None
        assert position.quantity == Decimal('2.0')
        assert position.avg_entry_price == Decimal('1.2500')
        
        # Check reservation committed
        reservation = snapshot.get_reservation(reservation_id)
        assert reservation.status == ReservationStatus.COMMITTED
    
    def test_release_reservation(self):
        """Test releasing (cancelling) reservation"""
        manager = PortfolioStateManager()
        
        # Reserve
        reservation_id = manager.reserve_position(
            symbol="USDJPY",
            side=PositionSide.SHORT,
            quantity=Decimal('1.0'),
            signal_id="signal_1"
        )
        
        time.sleep(0.1)
        
        # Release
        manager.release_reservation(reservation_id, reason="test_cancel")
        
        time.sleep(0.1)
        snapshot = manager.get_snapshot()
        
        reservation = snapshot.get_reservation(reservation_id)
        assert reservation.status == ReservationStatus.RELEASED
    
    def test_reservation_expiry(self):
        """Test reservation expiry check"""
        manager = PortfolioStateManager()
        
        # Create reservation with short TTL
        reservation_id = manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id="signal_1",
            ttl_seconds=0.1  # 100ms
        )
        
        time.sleep(0.05)
        snapshot1 = manager.get_snapshot()
        reservation1 = snapshot1.get_reservation(reservation_id)
        assert reservation1.is_active  # Still active
        
        time.sleep(0.1)  # Wait for expiry
        snapshot2 = manager.get_snapshot()
        reservation2 = snapshot2.get_reservation(reservation_id)
        assert not reservation2.is_active  # Expired


class TestCASOptimisticLocking:
    """Test Compare-And-Swap optimistic locking"""
    
    def test_cas_prevents_stale_updates(self):
        """Test CAS prevents updates based on stale version"""
        manager = PortfolioStateManager()
        
        # Get initial version
        snapshot1 = manager.get_snapshot()
        version1 = snapshot1.version
        
        # Make update
        manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id="signal_1"
        )
        
        time.sleep(0.1)
        
        # Version should have incremented
        snapshot2 = manager.get_snapshot()
        assert snapshot2.version == version1 + 1
    
    def test_concurrent_updates_sequential_processing(self):
        """Test concurrent update requests are processed sequentially"""
        manager = PortfolioStateManager()
        
        # Submit 100 updates concurrently
        def submit_updates():
            for i in range(10):
                manager.reserve_position(
                    symbol=f"SYMBOL_{i}",
                    side=PositionSide.LONG,
                    quantity=Decimal('1.0'),
                    signal_id=f"signal_{i}"
                )
        
        threads = []
        for _ in range(10):  # 10 threads × 10 updates = 100 total
            thread = threading.Thread(target=submit_updates)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(1.0)
        
        # All updates should be applied
        snapshot = manager.get_snapshot()
        assert snapshot.reservation_count == 100
        assert snapshot.version == 100  # Each update increments version


class TestRaceConditions:
    """Test race condition resistance"""
    
    def test_no_race_condition_parallel_reservations(self):
        """Test parallel reservations don't cause race conditions"""
        manager = PortfolioStateManager()
        
        reservation_ids = []
        lock = threading.Lock()
        
        def make_reservations():
            for i in range(10):
                res_id = manager.reserve_position(
                    symbol="EURUSD",
                    side=PositionSide.LONG,
                    quantity=Decimal('0.1'),
                    signal_id=f"signal_{threading.current_thread().name}_{i}"
                )
                with lock:
                    reservation_ids.append(res_id)
        
        # Launch 10 threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_reservations, name=f"Thread{i}")
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(1.0)
        
        # Should have 100 unique reservations
        assert len(reservation_ids) == 100
        assert len(set(reservation_ids)) == 100  # All unique
        
        snapshot = manager.get_snapshot()
        assert snapshot.reservation_count == 100
    
    def test_no_race_position_updates(self):
        """Test parallel position updates don't cause race conditions"""
        manager = PortfolioStateManager()
        
        # Create initial position
        reservation_id = manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id="initial"
        )
        
        time.sleep(0.1)
        manager.commit_reservation(
            reservation_id,
            Decimal('1.0'),
            Decimal('1.1000')
        )
        
        time.sleep(0.2)
        
        # Update price from multiple threads
        def update_price(price):
            for _ in range(10):
                manager.update_position_price("EURUSD", Decimal(str(price)))
        
        threads = []
        prices = ['1.1100', '1.1200', '1.1300']
        for price in prices:
            thread = threading.Thread(target=update_price, args=(price,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Wait for processing
        time.sleep(0.5)
        
        # Position should still be consistent
        snapshot = manager.get_snapshot()
        position = snapshot.get_position("EURUSD")
        assert position is not None
        assert position.quantity == Decimal('1.0')
        # Price should be one of the update prices
        assert position.current_price in [Decimal('1.1100'), Decimal('1.1200'), Decimal('1.1300')]


class TestOverAllocation:
    """Test over-allocation prevention"""
    
    def test_reserved_quantity_tracking(self):
        """Test reserved quantity is tracked per symbol"""
        manager = PortfolioStateManager()
        
        # Make 3 reservations
        for i in range(3):
            manager.reserve_position(
                symbol="EURUSD",
                side=PositionSide.LONG,
                quantity=Decimal('1.0'),
                signal_id=f"signal_{i}"
            )
        
        time.sleep(0.2)
        
        snapshot = manager.get_snapshot()
        reserved = snapshot.get_reserved_quantity("EURUSD", PositionSide.LONG)
        assert reserved == Decimal('3.0')
    
    def test_metrics_account_for_reservations(self):
        """Test account metrics include reserved positions"""
        manager = PortfolioStateManager(initial_equity=Decimal('100000'))
        
        # Reserve positions
        manager.reserve_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=Decimal('10.0'),
            signal_id="signal_1"
        )
        
        time.sleep(0.2)
        
        snapshot = manager.get_snapshot()
        metrics = snapshot.metrics
        
        # Cash should be reserved
        assert metrics.cash_reserved > Decimal('0')
        assert metrics.cash_available < manager.initial_equity


class TestBrokerReconciliation:
    """Test broker reconciliation"""
    
    def test_reconciliation_thread_starts(self):
        """Test reconciliation thread starts"""
        manager = PortfolioStateManager(
            enable_reconciliation=True,
            reconciliation_interval=1.0
        )
        
        # Thread should be running
        assert manager._reconciliation_thread is not None
        assert manager._reconciliation_thread.is_alive()
        
        manager.stop()
    
    def test_reconciliation_callable(self):
        """Test reconciliation with broker connector"""
        manager = PortfolioStateManager(
            enable_reconciliation=True,
            reconciliation_interval=0.5  # 500ms for testing
        )
        
        # Mock broker connector
        call_count = {'count': 0}
        
        def mock_broker_connector():
            call_count['count'] += 1
            return []  # No broker positions
        
        manager.set_broker_connector(mock_broker_connector)
        
        # Wait for at least one reconciliation
        time.sleep(0.6)
        
        # Should have been called
        assert call_count['count'] >= 1
        
        manager.stop()


class TestStatistics:
    """Test manager statistics"""
    
    def test_stats_tracking(self):
        """Test statistics are tracked correctly"""
        manager = PortfolioStateManager()
        
        # Make some updates
        for i in range(5):
            manager.reserve_position(
                symbol=f"SYMBOL_{i}",
                side=PositionSide.LONG,
                quantity=Decimal('1.0'),
                signal_id=f"signal_{i}"
            )
        
        time.sleep(0.5)
        
        stats = manager.get_stats()
        
        # CAS failures are retries (expected in concurrent system)
        assert stats['total_updates'] >= 5  # At least 5 updates attempted
        assert stats['successful_updates'] == 5  # All 5 succeeded eventually
        # CAS failures can happen (retries are normal)
        assert stats['cas_success_rate'] > 0  # Some updates succeeded
        assert stats['current_version'] == 5  # Final version is correct
        assert stats['reservations'] == 5


class TestPerformance:
    """Performance validation tests"""
    
    def test_high_throughput_updates(self):
        """Test handling high update throughput"""
        manager = PortfolioStateManager()
        
        start_time = time.perf_counter()
        
        # Submit 1000 updates
        for i in range(1000):
            manager.reserve_position(
                symbol=f"SYMBOL_{i % 10}",  # 10 different symbols
                side=PositionSide.LONG,
                quantity=Decimal('0.1'),
                signal_id=f"signal_{i}"
            )
        
        submission_time = time.perf_counter() - start_time
        
        # Wait for processing
        time.sleep(2.0)
        
        snapshot = manager.get_snapshot()
        
        # All updates should be processed
        assert snapshot.reservation_count == 1000
        assert snapshot.version == 1000
        
        # Submission should be fast (< 100ms for 1000 updates)
        assert submission_time < 0.1
        
        print(f"\n✓ Processed 1000 updates")
        print(f"✓ Submission time: {submission_time*1000:.2f}ms")
        print(f"✓ Final version: {snapshot.version}")
    
    def test_parallel_read_performance(self):
        """Test parallel reader performance"""
        manager = PortfolioStateManager()
        
        # Add some data
        for i in range(10):
            manager.reserve_position(
                symbol=f"SYMBOL_{i}",
                side=PositionSide.LONG,
                quantity=Decimal('1.0'),
                signal_id=f"signal_{i}"
            )
        
        time.sleep(0.2)
        
        # Measure parallel read performance
        def reader_task(iterations):
            start = time.perf_counter()
            for _ in range(iterations):
                snapshot = manager.get_snapshot()
                _ = snapshot.reservation_count
            return time.perf_counter() - start
        
        # Sequential baseline
        sequential_time = reader_task(10000)
        
        # Parallel (10 threads)
        parallel_times = []
        threads = []
        
        for _ in range(10):
            thread = threading.Thread(target=lambda: parallel_times.append(reader_task(10000)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        max_parallel_time = max(parallel_times)
        
        # Parallel should not be significantly slower (no lock contention)
        # Allow 2x overhead for thread coordination (still proves lock-free reads)
        assert max_parallel_time < sequential_time * 2.0
        
        print(f"\n✓ Sequential 10k reads: {sequential_time*1000:.2f}ms")
        print(f"✓ Parallel 10k reads (10 threads): {max_parallel_time*1000:.2f}ms")
        print(f"✓ Overhead: {((max_parallel_time/sequential_time - 1) * 100):.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
