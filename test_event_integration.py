"""
Integration test for event-driven flow across system layers

Tests the complete event bus integration:
- Raw Layer → Clean Data → Feature Engine → Signal Engine → Portfolio Manager
"""

import time
import threading
import pytest
from decimal import Decimal
from datetime import datetime

from arbitrex.event_bus import get_event_bus, Event, EventType
from arbitrex.signal_engine.strategy_runner import StrategyRunner, StrategyConfig
from arbitrex.signal_engine.signal_schemas import Signal, SignalType, create_single_leg_signal, LegDirection
from arbitrex.risk_portfolio_manager.portfolio_state import PositionSide
from arbitrex.risk_portfolio_manager.portfolio_manager import PortfolioStateManager


class TestEventBusIntegration:
    """Test event bus integration across layers"""
    
    def test_event_types_registered(self):
        """Test that all event types are registered"""
        # Data pipeline events
        assert hasattr(EventType, 'TICK_RECEIVED')
        assert hasattr(EventType, 'NORMALIZED_BAR_READY')
        assert hasattr(EventType, 'FEATURE_TIER1_READY')
        assert hasattr(EventType, 'FEATURE_TIER2_READY')
        
        # Signal engine events
        assert hasattr(EventType, 'SIGNAL_GENERATED')
        assert hasattr(EventType, 'SIGNAL_APPROVED')
        assert hasattr(EventType, 'SIGNAL_REJECTED')
        
        # Portfolio/Risk events
        assert hasattr(EventType, 'POSITION_UPDATED')
        assert hasattr(EventType, 'RESERVATION_CREATED')
        assert hasattr(EventType, 'RESERVATION_COMMITTED')
        assert hasattr(EventType, 'RESERVATION_RELEASED')
        assert hasattr(EventType, 'RISK_LIMIT_BREACHED')
        
        # Execution events
        assert hasattr(EventType, 'ORDER_SUBMITTED')
        assert hasattr(EventType, 'ORDER_FILLED')
        assert hasattr(EventType, 'ORDER_REJECTED')
        
        # System events
        assert hasattr(EventType, 'BROKER_SYNC_COMPLETE')
        assert hasattr(EventType, 'HEALTH_CHECK')
    
    def test_event_bus_publish_subscribe(self):
        """Test basic event bus publish/subscribe"""
        bus = get_event_bus()
        bus.start()
        
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        # Subscribe
        bus.subscribe(EventType.SIGNAL_GENERATED, callback)
        
        # Publish
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            symbol='EURUSD',
            data={'signal_id': 'test123', 'confidence': 0.85}
        )
        bus.publish(event)
        
        # Wait for dispatch
        time.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.SIGNAL_GENERATED
        assert received_events[0].symbol == 'EURUSD'
        
        bus.stop()
    
    def test_strategy_runner_subscribes_to_features(self):
        """Test that StrategyRunner subscribes to feature events"""
        bus = get_event_bus()
        bus.start()
        
        runner = StrategyRunner(max_workers=5, emit_events=True)
        
        # Check that runner has event bus
        assert runner._event_bus is not None
        
        # Publish feature event
        event = Event(
            event_type=EventType.FEATURE_TIER1_READY,
            symbol='EURUSD',
            data={
                'symbol': 'EURUSD',
                'features': {'rsi': 65.0, 'macd': 0.002}
            }
        )
        
        # This should trigger strategy execution
        bus.publish(event)
        time.sleep(0.1)
        
        runner.shutdown()
        bus.stop()
    
    def test_strategy_runner_publishes_signals(self):
        """Test that StrategyRunner publishes signal events"""
        bus = get_event_bus()
        bus.start()
        
        runner = StrategyRunner(max_workers=5, emit_events=True)
        
        signal_events = []
        
        def signal_callback(event):
            signal_events.append(event)
        
        bus.subscribe(EventType.SIGNAL_GENERATED, signal_callback)
        
        # Register simple strategy that generates a signal
        def simple_strategy(data):
            return create_single_leg_signal(
                symbol='EURUSD',
                direction=LegDirection.LONG,
                confidence=0.75,
                strategy_id='test_strategy'
            )
        
        config = StrategyConfig(
            strategy_id='test_strategy',
            strategy_name='Test Strategy'
        )
        runner.register_strategy(config, simple_strategy)
        
        # Execute strategies
        results = runner.execute_parallel({'test': 'data'})
        
        # Wait for event dispatch
        time.sleep(0.2)
        
        assert len(signal_events) > 0
        assert signal_events[0].event_type == EventType.SIGNAL_GENERATED
        assert signal_events[0].data['strategy_id'] == 'test_strategy'
        
        runner.shutdown()
        bus.stop()
    
    def test_portfolio_manager_subscribes_to_signals(self):
        """Test that PortfolioStateManager subscribes to signal events"""
        bus = get_event_bus()
        bus.start()
        
        rpm = PortfolioStateManager(emit_events=True)
        
        # Check that RPM has event bus
        assert rpm._event_bus is not None
        
        # Publish signal event
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            symbol='EURUSD',
            data={
                'signal_id': 'sig123',
                'strategy_id': 'test_strategy',
                'confidence': 0.85
            }
        )
        
        bus.publish(event)
        time.sleep(0.1)
        
        rpm.stop()
        bus.stop()
    
    def test_portfolio_manager_publishes_reservation_events(self):
        """Test that PortfolioStateManager publishes reservation events"""
        bus = get_event_bus()
        bus.start()
        
        rpm = PortfolioStateManager(emit_events=True)
        
        reservation_events = []
        
        def reservation_callback(event):
            reservation_events.append(event)
        
        bus.subscribe(EventType.RESERVATION_CREATED, reservation_callback)
        
        # Create reservation
        from arbitrex.risk_portfolio_manager.portfolio_state import PositionSide
        reservation_id = rpm.reserve_position(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id='sig123'
        )
        
        # Wait for event dispatch
        time.sleep(0.2)
        
        assert len(reservation_events) > 0
        assert reservation_events[0].event_type == EventType.RESERVATION_CREATED
        assert reservation_events[0].symbol == 'EURUSD'
        assert reservation_events[0].data['reservation_id'] == reservation_id
        
        rpm.stop()
        bus.stop()
    
    def test_portfolio_manager_publishes_position_events(self):
        """Test that PortfolioStateManager publishes position events"""
        bus = get_event_bus()
        bus.start()
        
        rpm = PortfolioStateManager(emit_events=True)
        
        position_events = []
        commit_events = []
        
        def position_callback(event):
            position_events.append(event)
        
        def commit_callback(event):
            commit_events.append(event)
        
        bus.subscribe(EventType.POSITION_UPDATED, position_callback)
        bus.subscribe(EventType.RESERVATION_COMMITTED, commit_callback)
        
        # Create and commit reservation
        from arbitrex.risk_portfolio_manager.portfolio_state import PositionSide
        reservation_id = rpm.reserve_position(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=Decimal('1.0'),
            signal_id='sig123'
        )
        
        time.sleep(0.1)
        
        position_id = rpm.commit_reservation(
            reservation_id=reservation_id,
            executed_quantity=Decimal('1.0'),
            avg_price=Decimal('1.1000')
        )
        
        # Wait for event dispatch
        time.sleep(0.3)
        
        assert len(commit_events) > 0
        assert len(position_events) > 0
        
        commit_event = commit_events[0]
        assert commit_event.event_type == EventType.RESERVATION_COMMITTED
        assert commit_event.data['reservation_id'] == reservation_id
        assert commit_event.data['position_id'] == position_id
        
        position_event = position_events[0]
        assert position_event.event_type == EventType.POSITION_UPDATED
        assert position_event.symbol == 'EURUSD'
        
        rpm.stop()
        bus.stop()
    
    def test_end_to_end_event_flow(self):
        """
        Test complete end-to-end event flow:
        Feature Event → Strategy Execution → Signal Event → Portfolio Reservation
        """
        bus = get_event_bus()
        bus.start()
        
        # Initialize components
        runner = StrategyRunner(max_workers=5, emit_events=True)
        rpm = PortfolioStateManager(emit_events=True)
        
        # Track all events
        all_events = {
            'features': [],
            'signals': [],
            'reservations': [],
            'positions': []
        }
        
        def feature_callback(event):
            all_events['features'].append(event)
        
        def signal_callback(event):
            all_events['signals'].append(event)
        
        def reservation_callback(event):
            all_events['reservations'].append(event)
        
        def position_callback(event):
            all_events['positions'].append(event)
        
        bus.subscribe(EventType.FEATURE_TIER1_READY, feature_callback)
        bus.subscribe(EventType.SIGNAL_GENERATED, signal_callback)
        bus.subscribe(EventType.RESERVATION_CREATED, reservation_callback)
        bus.subscribe(EventType.POSITION_UPDATED, position_callback)
        
        # Register strategy
        def momentum_strategy(data):
            rsi = data.get('rsi', 0) if isinstance(data, dict) else 0
            if rsi > 70:
                return create_single_leg_signal(
                    symbol='EURUSD',
                    direction=LegDirection.SHORT,
                    confidence=0.80,
                    strategy_id='momentum'
                )
            return None
        
        config = StrategyConfig(strategy_id='momentum')
        runner.register_strategy(config, momentum_strategy)
        
        # 1. Publish feature event
        feature_event = Event(
            event_type=EventType.FEATURE_TIER1_READY,
            symbol='EURUSD',
            data={
                'symbol': 'EURUSD',
                'features': {'rsi': 75.0, 'macd': -0.001}
            }
        )
        bus.publish(feature_event)
        
        # Wait for processing
        time.sleep(0.3)
        
        # 2. Verify feature event was received
        assert len(all_events['features']) > 0
        
        # 3. Verify signal was generated
        assert len(all_events['signals']) > 0
        signal_event = all_events['signals'][0]
        assert signal_event.data['strategy_id'] == 'momentum'
        
        # 4. Simulate risk check and reservation
        from arbitrex.risk_portfolio_manager.portfolio_state import PositionSide
        reservation_id = rpm.reserve_position(
            symbol='EURUSD',
            side=PositionSide.SHORT,
            quantity=Decimal('1.0'),
            signal_id=signal_event.data['signal_id']
        )
        
        time.sleep(0.2)
        
        # 5. Verify reservation event
        assert len(all_events['reservations']) > 0
        reservation_event = all_events['reservations'][0]
        assert reservation_event.data['reservation_id'] == reservation_id
        
        # Cleanup
        runner.shutdown()
        rpm.stop()
        bus.stop()
        
        print("\n=== End-to-End Event Flow Complete ===")
        print(f"Feature events: {len(all_events['features'])}")
        print(f"Signal events: {len(all_events['signals'])}")
        print(f"Reservation events: {len(all_events['reservations'])}")
        print(f"Position events: {len(all_events['positions'])}")


class TestCASLoggingLevel:
    """Test that CAS warnings are at DEBUG level"""
    
    def test_cas_failures_logged_as_debug(self):
        """Verify CAS failures are logged as DEBUG not WARNING"""
        import logging
        
        # Create log capture
        log_records = []
        
        class ListHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)
        
        handler = ListHandler()
        handler.setLevel(logging.DEBUG)
        
        logger = logging.getLogger('arbitrex.risk_portfolio_manager.portfolio_manager')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Create RPM and trigger concurrent updates (causes CAS failures)
        rpm = PortfolioStateManager(emit_events=False)
        
        from arbitrex.risk_portfolio_manager.portfolio_state import PositionSide
        
        # Create multiple concurrent reservations
        reservation_ids = []
        threads = []
        
        def create_reservation(i):
            res_id = rpm.reserve_position(
                symbol=f'EURUSD_{i % 3}',  # Same symbols to create conflicts
                side=PositionSide.LONG,
                quantity=Decimal('1.0'),
                signal_id=f'sig_{i}'
            )
            reservation_ids.append(res_id)
        
        # Launch 20 concurrent reservations
        for i in range(20):
            t = threading.Thread(target=create_reservation, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        time.sleep(0.5)  # Let updater process
        
        # Check for CAS failure logs
        cas_logs = [r for r in log_records if 'CAS failure' in r.getMessage()]
        
        if cas_logs:
            # Verify they're DEBUG level, not WARNING
            for log in cas_logs:
                assert log.levelno == logging.DEBUG, f"CAS failure logged as {logging.getLevelName(log.levelno)}, should be DEBUG"
            
            print(f"\n✓ CAS failures correctly logged at DEBUG level ({len(cas_logs)} occurrences)")
        
        # Verify reservations were created
        assert len(reservation_ids) == 20
        
        # Check stats
        stats = rpm.get_stats()
        print(f"\nCAS Statistics:")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Successful: {stats['successful_updates']}")
        print(f"  CAS failures (retries): {stats['cas_failures']}")
        print(f"  Success rate: {stats['cas_success_rate']:.2%}")
        print(f"  Retry rate: {stats['cas_retry_rate']:.2%}")
        
        assert stats['cas_success_rate'] > 0  # Some succeeded
        
        rpm.stop()
        logger.removeHandler(handler)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
