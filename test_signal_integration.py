"""
Integration Test: Old and New Signal Systems Working Together

Demonstrates:
1. SignalGenerationEngine (old) → TradeIntent → Signal → SignalBuffer
2. StrategyRunner (new) → Signal → SignalBuffer
3. Both systems coexisting and publishing to same buffer
4. Unified downstream consumption
"""

import pytest
from datetime import datetime

from arbitrex.signal_engine.schemas import TradeIntent, TradeDirection
from arbitrex.signal_engine.signal_schemas import (
    Signal,
    LegDirection,
    create_single_leg_signal,
    create_spread_signal
)
from arbitrex.signal_engine.signal_buffer import SignalBuffer
from arbitrex.signal_engine.signal_integration import (
    convert_trade_intent_to_signal,
    convert_signal_to_trade_intent,
    UnifiedSignalPublisher,
    HybridSignalRouter
)


class TestConversion:
    """Test conversion between TradeIntent and Signal"""
    
    def test_trade_intent_to_signal(self):
        """Test TradeIntent → Signal conversion"""
        # Create TradeIntent (old schema)
        trade_intent = TradeIntent(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            timeframe="1H",
            direction=TradeDirection.LONG,
            confidence_score=0.85,
            signal_source="momentum_v1",
            signal_version="abc123",
            bar_index=100
        )
        
        # Convert to Signal (new schema)
        signal = convert_trade_intent_to_signal(trade_intent, timeframe="1H")
        
        # Validate conversion
        assert signal.is_single_leg
        assert len(signal.legs) == 1
        assert signal.legs[0].symbol == "EURUSD"
        assert signal.legs[0].direction == LegDirection.LONG
        assert signal.confidence_score == 0.85
        assert signal.strategy_id == "momentum_v1"
        assert signal.strategy_version == "abc123"
        assert signal.bar_index == 100
        assert signal.metadata['source'] == 'SignalGenerationEngine'
    
    def test_signal_to_trade_intent_single_leg(self):
        """Test Signal → TradeIntent conversion (single-leg)"""
        # Create Signal (new schema)
        signal = create_single_leg_signal(
            symbol="GBPUSD",
            direction=LegDirection.SHORT,
            confidence=0.75,
            strategy_id="test_strategy",
            timeframe="4H",
            bar_index=50,
            strategy_version="xyz789"
        )
        
        # Convert to TradeIntent (old schema)
        trade_intent = convert_signal_to_trade_intent(signal)
        
        # Validate conversion
        assert trade_intent is not None
        assert trade_intent.symbol == "GBPUSD"
        assert trade_intent.direction == TradeDirection.SHORT
        assert trade_intent.confidence_score == 0.75
        assert trade_intent.signal_source == "test_strategy"
        assert trade_intent.signal_version == "xyz789"
        assert trade_intent.bar_index == 50
    
    def test_signal_to_trade_intent_multileg_fails(self):
        """Test multi-leg Signal cannot convert to TradeIntent"""
        # Create multi-leg spread signal
        signal = create_spread_signal(
            long_symbol="EURUSD",
            short_symbol="GBPUSD",
            confidence=0.8,
            strategy_id="spread_arb"
        )
        
        # Should raise error (multi-leg not compatible)
        with pytest.raises(ValueError):
            convert_signal_to_trade_intent(signal, raise_on_multileg=True)
        
        # Or return None if raise_on_multileg=False
        result = convert_signal_to_trade_intent(signal, raise_on_multileg=False)
        assert result is None


class TestUnifiedPublisher:
    """Test unified publisher accepting both schemas"""
    
    def test_publish_trade_intent(self):
        """Test publishing TradeIntent through unified publisher"""
        buffer = SignalBuffer(enable_expiry_check=False)
        publisher = UnifiedSignalPublisher(buffer)
        
        # Create TradeIntent
        trade_intent = TradeIntent(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            timeframe="1H",
            direction=TradeDirection.LONG,
            confidence_score=0.8,
            signal_source="old_engine",
            signal_version="v1",
            bar_index=1
        )
        
        # Publish
        success = publisher.publish_trade_intent(trade_intent)
        assert success
        
        # Retrieve from buffer as Signal
        signals = buffer.get_signals(max_count=10)
        assert len(signals) == 1
        assert signals[0].strategy_id == "old_engine"
        assert signals[0].is_single_leg
        
        # Check stats
        stats = publisher.get_stats()
        assert stats['trade_intents_published'] == 1
        assert stats['signals_published'] == 0
    
    def test_publish_signal(self):
        """Test publishing Signal through unified publisher"""
        buffer = SignalBuffer(enable_expiry_check=False)
        publisher = UnifiedSignalPublisher(buffer)
        
        # Create Signal
        signal = create_single_leg_signal(
            symbol="GBPUSD",
            direction=LegDirection.SHORT,
            confidence=0.9,
            strategy_id="new_strategy"
        )
        
        # Publish
        success = publisher.publish_signal(signal)
        assert success
        
        # Retrieve
        signals = buffer.get_signals(max_count=10)
        assert len(signals) == 1
        assert signals[0].strategy_id == "new_strategy"
        
        # Check stats
        stats = publisher.get_stats()
        assert stats['trade_intents_published'] == 0
        assert stats['signals_published'] == 1
    
    def test_auto_detect_publish(self):
        """Test auto-detect publish method"""
        buffer = SignalBuffer(enable_expiry_check=False)
        publisher = UnifiedSignalPublisher(buffer)
        
        # Publish TradeIntent
        trade_intent = TradeIntent(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            timeframe="1H",
            direction=TradeDirection.LONG,
            confidence_score=0.8,
            signal_source="old",
            signal_version="v1",
            bar_index=1
        )
        publisher.publish(trade_intent)
        
        # Publish Signal
        signal = create_single_leg_signal(
            symbol="GBPUSD",
            direction=LegDirection.SHORT,
            confidence=0.9,
            strategy_id="new"
        )
        publisher.publish(signal)
        
        # Both should be in buffer
        signals = buffer.get_signals(max_count=10)
        assert len(signals) == 2
        
        strategy_ids = {s.strategy_id for s in signals}
        assert "old" in strategy_ids
        assert "new" in strategy_ids


class TestHybridRouter:
    """Test hybrid router for both systems"""
    
    def test_route_from_old_system(self):
        """Test routing from SignalGenerationEngine (old)"""
        buffer = SignalBuffer(enable_expiry_check=False)
        router = HybridSignalRouter(buffer)
        
        # Create TradeIntent from "old" system
        trade_intent = TradeIntent(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            timeframe="1H",
            direction=TradeDirection.LONG,
            confidence_score=0.85,
            signal_source="SignalGenerationEngine",
            signal_version="v1",
            bar_index=100
        )
        
        # Route through old system path
        success = router.route_from_signal_engine(trade_intent, timeframe="1H")
        assert success
        
        # Check routing stats
        stats = router.get_routing_stats()
        assert stats['signals_from_old_system'] == 1
        assert stats['signals_from_new_system'] == 0
    
    def test_route_from_new_system(self):
        """Test routing from StrategyRunner (new)"""
        buffer = SignalBuffer(enable_expiry_check=False)
        router = HybridSignalRouter(buffer)
        
        # Create Signal from "new" system
        signal = create_single_leg_signal(
            symbol="GBPUSD",
            direction=LegDirection.SHORT,
            confidence=0.9,
            strategy_id="parallel_strategy_v2"
        )
        
        # Route through new system path
        success = router.route_from_strategy_runner(signal)
        assert success
        
        # Check routing stats
        stats = router.get_routing_stats()
        assert stats['signals_from_old_system'] == 0
        assert stats['signals_from_new_system'] == 1
    
    def test_hybrid_routing_both_systems(self):
        """Test routing from BOTH systems simultaneously"""
        buffer = SignalBuffer(enable_expiry_check=False)
        router = HybridSignalRouter(buffer)
        
        # Route 5 signals from old system
        for i in range(5):
            trade_intent = TradeIntent(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                timeframe="1H",
                direction=TradeDirection.LONG,
                confidence_score=0.8,
                signal_source=f"old_strategy_{i}",
                signal_version="v1",
                bar_index=i
            )
            router.route_from_signal_engine(trade_intent)
        
        # Route 5 signals from new system
        for i in range(5):
            signal = create_single_leg_signal(
                symbol="GBPUSD",
                direction=LegDirection.SHORT,
                confidence=0.9,
                strategy_id=f"new_strategy_{i}"
            )
            router.route_from_strategy_runner(signal)
        
        # Check all 10 signals in buffer
        signals = buffer.get_signals(max_count=20)
        assert len(signals) == 10
        
        # Check routing stats
        stats = router.get_routing_stats()
        assert stats['signals_from_old_system'] == 5
        assert stats['signals_from_new_system'] == 5
        assert stats['total_routed'] == 10
    
    def test_auto_route_mixed_signals(self):
        """Test auto-routing with mixed signal types"""
        buffer = SignalBuffer(enable_expiry_check=False)
        router = HybridSignalRouter(buffer)
        
        # Mix of TradeIntent and Signal objects
        signals_to_route = []
        
        # Add TradeIntents
        for i in range(3):
            signals_to_route.append(
                TradeIntent(
                    timestamp=datetime.utcnow(),
                    symbol="EURUSD",
                    timeframe="1H",
                    direction=TradeDirection.LONG,
                    confidence_score=0.8,
                    signal_source=f"old_{i}",
                    signal_version="v1",
                    bar_index=i
                )
            )
        
        # Add Signals
        for i in range(3):
            signals_to_route.append(
                create_single_leg_signal(
                    symbol="GBPUSD",
                    direction=LegDirection.SHORT,
                    confidence=0.9,
                    strategy_id=f"new_{i}"
                )
            )
        
        # Route all using auto-detect
        for sig in signals_to_route:
            router.route(sig, timeframe="1H")
        
        # All should be routed
        signals = buffer.get_signals(max_count=20)
        assert len(signals) == 6
        
        stats = router.get_routing_stats()
        assert stats['total_routed'] == 6


class TestEndToEndIntegration:
    """End-to-end integration test"""
    
    def test_complete_integration_flow(self):
        """
        Complete flow: Old engine + New strategies → Unified buffer → Consumer
        
        Simulates:
        1. SignalGenerationEngine producing TradeIntents
        2. StrategyRunner producing Signals (single and multi-leg)
        3. Both publishing to same SignalBuffer
        4. Unified consumer receiving all signals
        """
        buffer = SignalBuffer(enable_expiry_check=False)
        router = HybridSignalRouter(buffer)
        
        received_signals = []
        
        # Consumer subscribes to buffer
        def consumer_callback(signal: Signal):
            received_signals.append(signal)
        
        buffer.subscribe(consumer_callback, name="unified_consumer")
        
        # ===== OLD SYSTEM: SignalGenerationEngine =====
        # Produces 3 TradeIntents
        for i in range(3):
            trade_intent = TradeIntent(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                timeframe="1H",
                direction=TradeDirection.LONG if i % 2 == 0 else TradeDirection.SHORT,
                confidence_score=0.8,
                signal_source="SignalGenerationEngine",
                signal_version=f"v{i}",
                bar_index=i
            )
            router.route_from_signal_engine(trade_intent)
        
        # ===== NEW SYSTEM: StrategyRunner =====
        # Produces 2 single-leg signals
        for i in range(2):
            signal = create_single_leg_signal(
                symbol="GBPUSD",
                direction=LegDirection.LONG,
                confidence=0.9,
                strategy_id=f"momentum_strategy_{i}"
            )
            router.route_from_strategy_runner(signal)
        
        # Produces 1 multi-leg spread signal
        spread_signal = create_spread_signal(
            long_symbol="EURUSD",
            short_symbol="GBPUSD",
            confidence=0.85,
            strategy_id="spread_arbitrage"
        )
        router.route_from_strategy_runner(spread_signal)
        
        # ===== VERIFICATION =====
        # Total: 3 (old) + 2 (new single) + 1 (new multi) = 6 signals
        import time
        time.sleep(0.1)  # Let notifications dispatch
        
        assert len(received_signals) == 6
        
        # Check signal sources
        old_system_signals = [s for s in received_signals 
                             if s.metadata.get('source') == 'SignalGenerationEngine']
        assert len(old_system_signals) == 3
        
        new_system_signals = [s for s in received_signals 
                             if 'momentum_strategy' in s.strategy_id or 
                                s.strategy_id == 'spread_arbitrage']
        assert len(new_system_signals) == 3
        
        # Check multi-leg signal preserved
        multi_leg_signals = [s for s in received_signals if s.is_multi_leg]
        assert len(multi_leg_signals) == 1
        assert multi_leg_signals[0].strategy_id == "spread_arbitrage"
        
        # Check routing stats
        stats = router.get_routing_stats()
        assert stats['signals_from_old_system'] == 3
        assert stats['signals_from_new_system'] == 3
        assert stats['total_routed'] == 6
        
        print("\n✓ Integration test passed!")
        print(f"✓ Old system signals: {len(old_system_signals)}")
        print(f"✓ New system signals: {len(new_system_signals)}")
        print(f"✓ Multi-leg signals: {len(multi_leg_signals)}")
        print(f"✓ Total signals received by consumer: {len(received_signals)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
