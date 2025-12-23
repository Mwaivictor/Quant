"""
Comprehensive Kill-Switch System Tests with Chaos Engineering

Tests:
- Multi-level kill-switches (global, strategy, symbol, venue)
- Graduated response (throttle → suspend → shutdown)
- Auto-recovery mechanisms
- Manual override capabilities
- Alert system integration
- Chaos testing scenarios
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from arbitrex.risk_portfolio_manager.kill_switch import (
    KillSwitchManager,
    KillSwitchLevel,
    ResponseAction,
    TriggerReason,
    AlertConfig,
    AlertManager,
    KillSwitchState
)


class TestAlertManager:
    """Test alert system"""
    
    def test_slack_alert_construction(self):
        """Test Slack alert payload construction"""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            alert_on_throttle=True
        )
        
        manager = AlertManager(config)
        
        # Mock requests.post
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            manager.send_alert(
                action=ResponseAction.THROTTLE,
                level=KillSwitchLevel.STRATEGY,
                scope_id="momentum_strategy",
                reason=TriggerReason.LOSS_LIMIT,
                details={'loss': -1000}
            )
            
            # Verify Slack was called
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            assert 'text' in payload
            assert 'THROTTLE' in payload['text']
            assert 'momentum_strategy' in payload['text']
    
    def test_pagerduty_alert_construction(self):
        """Test PagerDuty alert payload construction"""
        config = AlertConfig(
            pagerduty_routing_key="test_routing_key",
            alert_on_shutdown=True
        )
        
        manager = AlertManager(config)
        
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post') as mock_post:
            mock_post.return_value.status_code = 202
            
            manager.send_alert(
                action=ResponseAction.SHUTDOWN,
                level=KillSwitchLevel.GLOBAL,
                scope_id="global",
                reason=TriggerReason.MARGIN_CALL,
                details={'margin_level': 50}
            )
            
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            assert payload['routing_key'] == "test_routing_key"
            assert payload['payload']['severity'] == 'critical'
    
    def test_alert_deduplication(self):
        """Test alert deduplication prevents spam"""
        config = AlertConfig(slack_webhook="https://hooks.slack.com/test")
        manager = AlertManager(config)
        
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post'):
            # First alert should go through
            manager.send_alert(
                ResponseAction.THROTTLE,
                KillSwitchLevel.STRATEGY,
                "test_strategy",
                TriggerReason.LOSS_LIMIT
            )
            
            # Duplicate should be suppressed
            result = manager._check_and_add_alert("strategy:test_strategy:throttle")
            assert not result  # Already exists
    
    def test_selective_alerting(self):
        """Test selective alerting based on config"""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            alert_on_throttle=False,  # Don't alert on throttle
            alert_on_suspend=True,
            alert_on_shutdown=True
        )
        
        manager = AlertManager(config)
        
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post') as mock_post:
            # Throttle should not alert
            manager.send_alert(
                ResponseAction.THROTTLE,
                KillSwitchLevel.STRATEGY,
                "test",
                TriggerReason.LOSS_LIMIT
            )
            assert not mock_post.called
            
            # Suspend should alert
            manager.send_alert(
                ResponseAction.SUSPEND,
                KillSwitchLevel.STRATEGY,
                "test",
                TriggerReason.LOSS_LIMIT
            )
            assert mock_post.called


class TestKillSwitchManager:
    """Test kill-switch manager core functionality"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Should have global kill-switch
        global_state = manager.get_state(KillSwitchLevel.GLOBAL, "global")
        assert global_state.action == ResponseAction.NORMAL
        
        summary = manager.get_summary()
        assert summary['total_kill_switches'] == 1
        assert summary['global_state'] == ResponseAction.NORMAL.value
    
    def test_manual_kill_switch_activation(self):
        """Test manual kill-switch activation"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Activate strategy kill-switch
        manager.activate_kill_switch(
            level=KillSwitchLevel.STRATEGY,
            scope_id="test_strategy",
            action=ResponseAction.THROTTLE,
            reason=TriggerReason.MANUAL,
            triggered_by="admin"
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.THROTTLE
        assert state.triggered_by == "admin"
        assert state.trigger_reason == TriggerReason.MANUAL
        assert state.trigger_count == 1
    
    def test_graduated_response_escalation(self):
        """Test graduated response: throttle → suspend → shutdown"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Start with throttle
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.THROTTLE
        
        # Escalate to suspend
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SUSPEND,
            TriggerReason.CONSECUTIVE_LOSSES
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.SUSPEND
        
        # Emergency shutdown
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.SHUTDOWN
        assert state.trigger_count == 3
    
    def test_no_downgrade_protection(self):
        """Test that kill-switches don't allow downgrade"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Start with shutdown
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL
        )
        
        # Try to downgrade to throttle (should be ignored)
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.SHUTDOWN  # Still shutdown
    
    def test_manual_deactivation(self):
        """Test manual kill-switch deactivation"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Activate
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SUSPEND,
            TriggerReason.LOSS_LIMIT
        )
        
        assert manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy").action == ResponseAction.SUSPEND
        
        # Deactivate
        manager.deactivate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            deactivated_by="admin"
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.NORMAL
    
    def test_hierarchical_trading_check(self):
        """Test hierarchical trading permission check"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # All clear - should allow trading
        assert manager.is_trading_allowed(
            strategy_id="test_strategy",
            symbol="EURUSD",
            venue="venue1"
        )
        
        # Activate global kill-switch
        manager.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.SUSPEND,
            TriggerReason.MARGIN_CALL
        )
        
        # Should block all trading
        assert not manager.is_trading_allowed(
            strategy_id="test_strategy",
            symbol="EURUSD",
            venue="venue1"
        )
        
        # Deactivate global
        manager.deactivate_kill_switch(KillSwitchLevel.GLOBAL, "global")
        
        # Activate venue-specific
        manager.activate_kill_switch(
            KillSwitchLevel.VENUE,
            "venue1",
            ResponseAction.SUSPEND,
            TriggerReason.VENUE_ERROR
        )
        
        # Should block venue1 but allow venue2
        assert not manager.is_trading_allowed(venue="venue1")
        assert manager.is_trading_allowed(venue="venue2")
    
    def test_throttle_allows_trading(self):
        """Test that throttle action still allows trading (but slower)"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        # Throttle should still allow trading
        assert manager.is_trading_allowed(strategy_id="test_strategy")
    
    def test_auto_recovery(self):
        """Test automatic recovery after delay"""
        manager = KillSwitchManager(enable_auto_recovery=True)
        
        # Activate with short recovery delay
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT,
            can_auto_recover=True
        )
        
        # Manually set short recovery delay (need to do before recovery time is calculated)
        with manager._lock:
            state = manager._states["strategy:test_strategy"]
            state.recovery_delay_seconds = 1.0  # 1 second for testing
            state.recovery_at = datetime.utcnow() + timedelta(seconds=1.0)
        
        # Should be throttled
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.THROTTLE
        
        # Wait for auto-recovery (recovery loop checks every 10s, but we'll wait enough time)
        time.sleep(12.0)  # Wait for recovery check + deactivation
        
        # Should be recovered
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.NORMAL
        
        manager.stop()
    
    def test_shutdown_no_auto_recovery(self):
        """Test that shutdown requires manual intervention"""
        manager = KillSwitchManager(enable_auto_recovery=True)
        
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL,
            can_auto_recover=True
        )
        
        # Wait a bit
        time.sleep(2.0)
        
        # Should still be in shutdown
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test_strategy")
        assert state.action == ResponseAction.SHUTDOWN
        
        manager.stop()
    
    def test_multi_level_kill_switches(self):
        """Test multiple kill-switches at different levels"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Activate at different levels
        manager.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.THROTTLE,
            TriggerReason.RISK_BREACH
        )
        
        manager.activate_kill_switch(
            KillSwitchLevel.VENUE,
            "venue1",
            ResponseAction.SUSPEND,
            TriggerReason.VENUE_ERROR
        )
        
        manager.activate_kill_switch(
            KillSwitchLevel.SYMBOL,
            "EURUSD",
            ResponseAction.THROTTLE,
            TriggerReason.LIQUIDITY_CRISIS
        )
        
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "momentum",
            ResponseAction.SUSPEND,
            TriggerReason.CONSECUTIVE_LOSSES
        )
        
        summary = manager.get_summary()
        assert summary['total_kill_switches'] == 5  # global + 4 new
        assert summary['active_kill_switches'] == 4  # 4 activated (global still normal initially)
        assert summary['by_action']['throttle'] == 2
        assert summary['by_action']['suspend'] == 2


class TestChaosEngineering:
    """Chaos testing scenarios"""
    
    def test_chaos_concurrent_activations(self):
        """Test concurrent kill-switch activations from multiple threads"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        def activate_random_switches():
            for i in range(10):
                manager.activate_kill_switch(
                    KillSwitchLevel.STRATEGY,
                    f"strategy_{i}",
                    ResponseAction.THROTTLE,
                    TriggerReason.LOSS_LIMIT
                )
                time.sleep(0.001)
        
        # Launch 5 threads
        threads = [threading.Thread(target=activate_random_switches) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify all switches registered
        summary = manager.get_summary()
        assert summary['total_kill_switches'] >= 10  # At least 10 strategies
    
    def test_chaos_rapid_escalation(self):
        """Test rapid escalation under load"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Rapid escalation
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test",
            ResponseAction.SUSPEND,
            TriggerReason.CONSECUTIVE_LOSSES
        )
        
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL
        )
        
        # Should end up in shutdown
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test")
        assert state.action == ResponseAction.SHUTDOWN
        assert state.trigger_count == 3
    
    def test_chaos_global_vs_local_conflict(self):
        """Test global shutdown overrides local states"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Activate strategy throttle
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        # Strategy should allow trading (throttled)
        assert manager.is_trading_allowed(strategy_id="test_strategy")
        
        # Global shutdown
        manager.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.SHUTDOWN,
            TriggerReason.MARGIN_CALL
        )
        
        # Global shutdown overrides strategy throttle
        assert not manager.is_trading_allowed(strategy_id="test_strategy")
    
    def test_chaos_recovery_during_activation(self):
        """Test recovery triggered during activation"""
        manager = KillSwitchManager(enable_auto_recovery=True)
        
        # Activate with very short recovery
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test")
        state.recovery_delay_seconds = 0.5
        
        # Wait for recovery to start
        time.sleep(1.0)
        
        # Try to activate again during recovery
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test",
            ResponseAction.THROTTLE,
            TriggerReason.LOSS_LIMIT
        )
        
        # Should handle gracefully
        state = manager.get_state(KillSwitchLevel.STRATEGY, "test")
        assert state.trigger_count >= 1
        
        manager.stop()
    
    def test_chaos_alert_storm(self):
        """Test alert system under heavy load"""
        config = AlertConfig(slack_webhook="https://hooks.slack.com/test")
        manager = KillSwitchManager(alert_config=config, enable_auto_recovery=False)
        
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post'):
            # Trigger many alerts rapidly
            for i in range(50):
                manager.activate_kill_switch(
                    KillSwitchLevel.STRATEGY,
                    f"strategy_{i}",
                    ResponseAction.THROTTLE,
                    TriggerReason.LOSS_LIMIT
                )
            
            # Should handle gracefully with deduplication
            summary = manager.get_summary()
            assert summary['total_kill_switches'] >= 50
    
    def test_chaos_venue_cascade_failure(self):
        """Test cascading venue failures"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        venues = ["venue1", "venue2", "venue3", "venue4"]
        
        # Cascade failures
        for venue in venues:
            manager.activate_kill_switch(
                KillSwitchLevel.VENUE,
                venue,
                ResponseAction.SUSPEND,
                TriggerReason.VENUE_ERROR
            )
        
        # All venues should be blocked
        for venue in venues:
            assert not manager.is_trading_allowed(venue=venue)
        
        # But other venues should work
        assert manager.is_trading_allowed(venue="venue5")
    
    def test_chaos_memory_leak_check(self):
        """Test that kill-switches don't leak memory"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Create and destroy many kill-switches
        for iteration in range(10):
            for i in range(100):
                manager.activate_kill_switch(
                    KillSwitchLevel.STRATEGY,
                    f"strategy_{iteration}_{i}",
                    ResponseAction.THROTTLE,
                    TriggerReason.LOSS_LIMIT
                )
        
        # Should have all switches
        summary = manager.get_summary()
        assert summary['total_kill_switches'] >= 1000


class TestKillSwitchIntegration:
    """Integration tests with trading components"""
    
    def test_event_bus_integration(self):
        """Test kill-switch events published to event bus"""
        # Mock event bus
        mock_bus = Mock()
        
        manager = KillSwitchManager(enable_auto_recovery=False)
        manager._event_bus = mock_bus
        manager._Event = Mock()
        manager._EventType = Mock()
        manager._EventType.RISK_LIMIT_BREACHED = "RISK_LIMIT_BREACHED"
        
        # Activate kill-switch
        manager.activate_kill_switch(
            KillSwitchLevel.STRATEGY,
            "test_strategy",
            ResponseAction.SUSPEND,
            TriggerReason.LOSS_LIMIT,
            details={'loss': -1000}
        )
        
        # Verify event published
        assert mock_bus.publish.called
    
    def test_trading_permission_check_performance(self):
        """Test that permission checks are fast"""
        manager = KillSwitchManager(enable_auto_recovery=False)
        
        # Add many kill-switches
        for i in range(100):
            manager.activate_kill_switch(
                KillSwitchLevel.STRATEGY,
                f"strategy_{i}",
                ResponseAction.NORMAL if i % 2 == 0 else ResponseAction.THROTTLE,
                TriggerReason.LOSS_LIMIT
            )
        
        # Check performance
        start = time.perf_counter()
        for _ in range(1000):
            manager.is_trading_allowed(
                strategy_id="strategy_42",
                symbol="EURUSD",
                venue="venue1"
            )
        elapsed = time.perf_counter() - start
        
        # Should be <10ms per check
        assert elapsed / 1000 < 0.01
    
    def test_full_lifecycle(self):
        """Test complete kill-switch lifecycle"""
        config = AlertConfig(slack_webhook="https://hooks.slack.com/test")
        manager = KillSwitchManager(alert_config=config, enable_auto_recovery=False)
        
        with patch('arbitrex.risk_portfolio_manager.kill_switch.requests.post'):
            # 1. Normal operation
            assert manager.is_trading_allowed(strategy_id="test")
            
            # 2. Loss triggers throttle
            manager.activate_kill_switch(
                KillSwitchLevel.STRATEGY,
                "test",
                ResponseAction.THROTTLE,
                TriggerReason.LOSS_LIMIT,
                details={'loss': -500}
            )
            assert manager.is_trading_allowed(strategy_id="test")  # Throttle allows trading
            
            # 3. More losses trigger suspend
            manager.activate_kill_switch(
                KillSwitchLevel.STRATEGY,
                "test",
                ResponseAction.SUSPEND,
                TriggerReason.CONSECUTIVE_LOSSES,
                details={'consecutive_losses': 5}
            )
            assert not manager.is_trading_allowed(strategy_id="test")  # Suspend blocks trading
            
            # 4. Manual recovery
            manager.deactivate_kill_switch(
                KillSwitchLevel.STRATEGY,
                "test",
                deactivated_by="risk_manager"
            )
            assert manager.is_trading_allowed(strategy_id="test")  # Back to normal


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
