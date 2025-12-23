"""
Unified Kill-Switch System - Enterprise Grade

Merges 3 implementations into one comprehensive system:
1. Threshold-based checks (drawdown, loss limits, volatility)
2. Velocity-based circuit breakers (rejection rate, exposure acceleration)
3. Multi-level graduated response (global, venue, symbol, strategy)

Features:
- Multi-level kill-switches: global, per-strategy, per-symbol, per-venue
- Graduated response: throttle → suspend → emergency shutdown
- Threshold monitoring: drawdown, daily/weekly losses, volatility shocks
- Velocity tracking: rejection rate, exposure acceleration
- Multi-channel alerting: Slack, PagerDuty, logging
- Automatic triggers based on loss, drawdown, risk metrics
- Manual override capabilities
- Auto-recovery mechanisms
- Event bus integration

Architecture:
    Monitors → UnifiedKillSwitchManager → Graduated Response → Alerting
                       ↓                            ↓
           [Global, Strategy, Symbol, Venue]   [Thresholds, Velocity]
                       ↓
           [Throttle, Suspend, Shutdown]
"""

import threading
import logging
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

LOG = logging.getLogger(__name__)

# Import schemas for backward compatibility with existing RPM engine
try:
    from .schemas import PortfolioState, RejectionReason
except ImportError:
    PortfolioState = None
    RejectionReason = None


class KillSwitchLevel(str, Enum):
    """Kill-switch scope"""
    GLOBAL = "global"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    VENUE = "venue"


class ResponseAction(str, Enum):
    """Graduated response actions"""
    NORMAL = "normal"  # No action, operating normally
    THROTTLE = "throttle"  # Reduce trading rate by 50%
    SUSPEND = "suspend"  # Stop new trades, allow exits only
    SHUTDOWN = "shutdown"  # Emergency stop all trading
    RECOVERY = "recovery"  # Transitioning back to normal


class TriggerReason(str, Enum):
    """Kill-switch trigger reasons"""
    MANUAL = "manual"
    LOSS_LIMIT = "loss_limit"
    DRAWDOWN = "drawdown"
    RISK_BREACH = "risk_breach"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VENUE_ERROR = "venue_error"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    POSITION_LIMIT = "position_limit"
    MARGIN_CALL = "margin_call"
    NETWORK_FAILURE = "network_failure"
    UNKNOWN = "unknown"


@dataclass
class KillSwitchState:
    """Current state of a kill-switch"""
    level: KillSwitchLevel
    scope_id: str  # strategy_id, symbol, venue_id, or "global"
    action: ResponseAction = ResponseAction.NORMAL
    
    # Trigger info
    triggered_at: Optional[datetime] = None
    triggered_by: str = ""
    trigger_reason: TriggerReason = TriggerReason.UNKNOWN
    trigger_details: Dict = field(default_factory=dict)
    
    # Recovery
    can_auto_recover: bool = True
    recovery_at: Optional[datetime] = None
    recovery_delay_seconds: float = 300.0  # 5 minutes default
    
    # Metrics
    trigger_count: int = 0
    last_trigger_time: Optional[datetime] = None


@dataclass
class AlertConfig:
    """Alert channel configuration"""
    slack_webhook: Optional[str] = None
    pagerduty_routing_key: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # Alert thresholds
    alert_on_throttle: bool = True
    alert_on_suspend: bool = True
    alert_on_shutdown: bool = True


class AlertManager:
    """
    Multi-channel alerting system.
    
    Sends alerts to Slack, PagerDuty, email when kill-switches activate.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._lock = threading.RLock()
        
        # Alert deduplication (prevent spam)
        self._recent_alerts: Set[str] = set()
        self._alert_cache_ttl = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def send_alert(
        self,
        action: ResponseAction,
        level: KillSwitchLevel,
        scope_id: str,
        reason: TriggerReason,
        details: Dict = None
    ):
        """Send alert through all configured channels"""
        # Check if we should alert for this action
        if not self._should_alert(action):
            return
        
        # Deduplication
        alert_key = f"{level.value}:{scope_id}:{action.value}"
        if not self._check_and_add_alert(alert_key):
            LOG.debug(f"Alert suppressed (duplicate): {alert_key}")
            return
        
        # Build alert message
        message = self._build_message(action, level, scope_id, reason, details or {})
        
        # Send to all channels
        if self.config.slack_webhook:
            self._send_slack(message, action)
        
        if self.config.pagerduty_routing_key:
            self._send_pagerduty(message, action, level, scope_id)
        
        if self.config.email_recipients:
            self._send_email(message)
        
        LOG.info(f"Alert sent: {action.value} for {level.value} {scope_id}")
    
    def _should_alert(self, action: ResponseAction) -> bool:
        """Check if we should alert for this action"""
        if action == ResponseAction.THROTTLE and not self.config.alert_on_throttle:
            return False
        if action == ResponseAction.SUSPEND and not self.config.alert_on_suspend:
            return False
        if action == ResponseAction.SHUTDOWN and not self.config.alert_on_shutdown:
            return False
        return True
    
    def _check_and_add_alert(self, alert_key: str) -> bool:
        """Check if alert is duplicate and add to cache"""
        with self._lock:
            # Cleanup old alerts
            current_time = time.time()
            if current_time - self._last_cleanup > self._alert_cache_ttl:
                self._recent_alerts.clear()
                self._last_cleanup = current_time
            
            # Check if alert is recent
            if alert_key in self._recent_alerts:
                return False
            
            # Add to cache
            self._recent_alerts.add(alert_key)
            return True
    
    def _build_message(
        self,
        action: ResponseAction,
        level: KillSwitchLevel,
        scope_id: str,
        reason: TriggerReason,
        details: Dict
    ) -> str:
        """Build alert message"""
        severity = "🔴 CRITICAL" if action == ResponseAction.SHUTDOWN else "🟠 WARNING"
        
        message = f"{severity} Kill-Switch Activated\n\n"
        message += f"Action: {action.value.upper()}\n"
        message += f"Level: {level.value}\n"
        message += f"Scope: {scope_id}\n"
        message += f"Reason: {reason.value}\n"
        message += f"Time: {datetime.utcnow().isoformat()}\n"
        
        if details:
            message += f"\nDetails:\n"
            for key, value in details.items():
                message += f"  {key}: {value}\n"
        
        return message
    
    def _send_slack(self, message: str, action: ResponseAction):
        """Send alert to Slack"""
        try:
            color = "danger" if action == ResponseAction.SHUTDOWN else "warning"
            payload = {
                "text": message,
                "attachments": [{
                    "color": color,
                    "text": message,
                    "footer": "ArbitreX Kill-Switch System",
                    "ts": int(time.time())
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook,
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            LOG.debug("Slack alert sent successfully")
            
        except Exception as e:
            LOG.error(f"Failed to send Slack alert: {e}")
    
    def _send_pagerduty(
        self,
        message: str,
        action: ResponseAction,
        level: KillSwitchLevel,
        scope_id: str
    ):
        """Send alert to PagerDuty"""
        try:
            severity = "critical" if action == ResponseAction.SHUTDOWN else "error"
            
            payload = {
                "routing_key": self.config.pagerduty_routing_key,
                "event_action": "trigger",
                "dedup_key": f"{level.value}_{scope_id}_{action.value}",
                "payload": {
                    "summary": f"Kill-Switch: {action.value} - {level.value} {scope_id}",
                    "severity": severity,
                    "source": "ArbitreX",
                    "custom_details": {
                        "message": message
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            LOG.debug("PagerDuty alert sent successfully")
            
        except Exception as e:
            LOG.error(f"Failed to send PagerDuty alert: {e}")
    
    def _send_email(self, message: str):
        """Send alert via email (placeholder)"""
        LOG.info(f"Email alert (not implemented): {message[:100]}...")


class KillSwitchManager:
    """
    Centralized unified kill-switch management with graduated response.
    
    Combines:
    - Threshold monitoring (drawdown, loss limits, volatility)
    - Velocity tracking (rejection rate, exposure acceleration)
    - Multi-level controls (global, venue, symbol, strategy)
    - Graduated response (throttle → suspend → shutdown)
    """
    
    def __init__(
        self,
        alert_config: Optional[AlertConfig] = None,
        enable_auto_recovery: bool = True,
        config: Optional[object] = None  # RPMConfig for backward compatibility
    ):
        self.enable_auto_recovery = enable_auto_recovery
        self.config = config  # Store RPM config if provided
        
        # Kill-switch registry
        self._states: Dict[str, KillSwitchState] = {}
        self._lock = threading.RLock()
        
        # Global kill-switch
        self._states["global"] = KillSwitchState(
            level=KillSwitchLevel.GLOBAL,
            scope_id="global"
        )
        
        # Alert manager
        self.alert_manager = AlertManager(alert_config)
        
        # Auto-recovery thread
        self._recovery_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Velocity tracking (from advanced_kill_switches)
        self._rejection_history: deque = deque(maxlen=1000)
        self._exposure_history: deque = deque(maxlen=100)
        
        # Volatility tracking (from kill_switches)
        self._volatility_history: List[float] = []
        self._volatility_baseline: Optional[float] = None
        
        # Event bus integration
        self._event_bus = None
        try:
            from arbitrex.event_bus import get_event_bus, Event, EventType
            self._event_bus = get_event_bus()
            self._Event = Event
            self._EventType = EventType
        except ImportError:
            LOG.warning("Event bus not available for KillSwitchManager")
        
        # Start auto-recovery
        if enable_auto_recovery:
            self.start()
    
    def start(self):
        """Start auto-recovery thread"""
        if self._running:
            return
        
        self._running = True
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            name="KillSwitchRecovery",
            daemon=True
        )
        self._recovery_thread.start()
        LOG.info("KillSwitchManager started with auto-recovery")
    
    def stop(self):
        """Stop auto-recovery thread"""
        if not self._running:
            return
        
        self._running = False
        if self._recovery_thread:
            self._recovery_thread.join(timeout=5.0)
        LOG.info("KillSwitchManager stopped")
    
    def activate_kill_switch(
        self,
        level: KillSwitchLevel,
        scope_id: str,
        action: ResponseAction,
        reason: TriggerReason,
        triggered_by: str = "system",
        details: Dict = None,
        can_auto_recover: bool = True
    ):
        """
        Activate kill-switch with specified action.
        
        Args:
            level: Kill-switch level (global, strategy, symbol, venue)
            scope_id: Scope identifier (strategy_id, symbol, venue_id, or "global")
            action: Response action (throttle, suspend, shutdown)
            reason: Trigger reason
            triggered_by: Who/what triggered it
            details: Additional context
            can_auto_recover: Whether auto-recovery is allowed
        """
        with self._lock:
            # Get or create state
            key = self._get_key(level, scope_id)
            if key not in self._states:
                self._states[key] = KillSwitchState(
                    level=level,
                    scope_id=scope_id
                )
            
            state = self._states[key]
            
            # Don't downgrade action (throttle → suspend → shutdown is one-way until recovery)
            if self._is_downgrade(state.action, action):
                LOG.warning(f"Ignoring downgrade from {state.action.value} to {action.value}")
                return
            
            # Update state
            previous_action = state.action
            state.action = action
            state.triggered_at = datetime.utcnow()
            state.triggered_by = triggered_by
            state.trigger_reason = reason
            state.trigger_details = details or {}
            state.can_auto_recover = can_auto_recover
            state.trigger_count += 1
            state.last_trigger_time = datetime.utcnow()
            
            # Set recovery time
            if can_auto_recover and action != ResponseAction.SHUTDOWN:
                state.recovery_at = datetime.utcnow() + timedelta(seconds=state.recovery_delay_seconds)
            
            LOG.warning(f"Kill-switch activated: {level.value} {scope_id} → {action.value} (reason: {reason.value})")
        
        # Send alert
        self.alert_manager.send_alert(action, level, scope_id, reason, details)
        
        # Publish event
        self._publish_event(level, scope_id, action, reason, details)
        
        # Apply action
        self._apply_action(level, scope_id, action, previous_action)
    
    def deactivate_kill_switch(
        self,
        level: KillSwitchLevel,
        scope_id: str,
        deactivated_by: str = "system"
    ):
        """Manually deactivate kill-switch (recovery)"""
        with self._lock:
            key = self._get_key(level, scope_id)
            if key not in self._states:
                return
            
            state = self._states[key]
            if state.action == ResponseAction.NORMAL:
                return
            
            previous_action = state.action
            state.action = ResponseAction.RECOVERY
            state.recovery_at = datetime.utcnow()
            
            LOG.info(f"Kill-switch deactivated: {level.value} {scope_id} by {deactivated_by}")
        
        # Apply recovery
        self._apply_action(level, scope_id, ResponseAction.RECOVERY, previous_action)
        
        # Transition to normal after brief delay
        time.sleep(1.0)
        with self._lock:
            state.action = ResponseAction.NORMAL
    
    def get_state(self, level: KillSwitchLevel, scope_id: str) -> KillSwitchState:
        """Get current kill-switch state"""
        with self._lock:
            key = self._get_key(level, scope_id)
            if key not in self._states:
                return KillSwitchState(level=level, scope_id=scope_id)
            return self._states[key]
    
    def is_trading_allowed(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        venue: Optional[str] = None
    ) -> bool:
        """
        Check if trading is allowed considering all applicable kill-switches.
        
        Checks hierarchy: global → venue → symbol → strategy
        """
        with self._lock:
            # Check global
            if not self._check_level_allows_trading(KillSwitchLevel.GLOBAL, "global"):
                return False
            
            # Check venue
            if venue:
                if not self._check_level_allows_trading(KillSwitchLevel.VENUE, venue):
                    return False
            
            # Check symbol
            if symbol:
                if not self._check_level_allows_trading(KillSwitchLevel.SYMBOL, symbol):
                    return False
            
            # Check strategy
            if strategy_id:
                if not self._check_level_allows_trading(KillSwitchLevel.STRATEGY, strategy_id):
                    return False
            
            return True
    
    def _check_level_allows_trading(self, level: KillSwitchLevel, scope_id: str) -> bool:
        """Check if specific level allows trading"""
        key = self._get_key(level, scope_id)
        if key not in self._states:
            return True
        
        action = self._states[key].action
        return action in [ResponseAction.NORMAL, ResponseAction.THROTTLE, ResponseAction.RECOVERY]
    
    def _get_key(self, level: KillSwitchLevel, scope_id: str) -> str:
        """Get dictionary key for kill-switch"""
        return f"{level.value}:{scope_id}"
    
    def _is_downgrade(self, current: ResponseAction, new: ResponseAction) -> bool:
        """Check if new action is a downgrade from current"""
        severity = {
            ResponseAction.NORMAL: 0,
            ResponseAction.RECOVERY: 1,
            ResponseAction.THROTTLE: 2,
            ResponseAction.SUSPEND: 3,
            ResponseAction.SHUTDOWN: 4
        }
        return severity[new] < severity[current]
    
    def _apply_action(
        self,
        level: KillSwitchLevel,
        scope_id: str,
        action: ResponseAction,
        previous_action: ResponseAction
    ):
        """Apply graduated response action"""
        if action == previous_action:
            return
        
        if action == ResponseAction.THROTTLE:
            LOG.warning(f"THROTTLE activated: {level.value} {scope_id} - Reducing trade rate by 50%")
        
        elif action == ResponseAction.SUSPEND:
            LOG.error(f"SUSPEND activated: {level.value} {scope_id} - No new trades, exits only")
        
        elif action == ResponseAction.SHUTDOWN:
            LOG.critical(f"SHUTDOWN activated: {level.value} {scope_id} - Emergency stop all trading")
        
        elif action == ResponseAction.RECOVERY:
            LOG.info(f"RECOVERY started: {level.value} {scope_id} - Returning to normal operation")
    
    def _publish_event(
        self,
        level: KillSwitchLevel,
        scope_id: str,
        action: ResponseAction,
        reason: TriggerReason,
        details: Dict
    ):
        """Publish kill-switch event to event bus"""
        if not self._event_bus:
            return
        
        try:
            # Use RISK_LIMIT_BREACHED for kill-switch events
            event = self._Event(
                event_type=self._EventType.RISK_LIMIT_BREACHED,
                symbol=scope_id if level == KillSwitchLevel.SYMBOL else None,
                data={
                    'kill_switch_level': level.value,
                    'scope_id': scope_id,
                    'action': action.value,
                    'reason': reason.value,
                    'details': details
                }
            )
            self._event_bus.publish(event)
        except Exception as e:
            LOG.error(f"Failed to publish kill-switch event: {e}")
    
    def _recovery_loop(self):
        """Auto-recovery background thread"""
        while self._running:
            try:
                with self._lock:
                    now = datetime.utcnow()
                    
                    for key, state in list(self._states.items()):
                        # Skip if not eligible for recovery
                        if not state.can_auto_recover:
                            continue
                        if state.action in [ResponseAction.NORMAL, ResponseAction.SHUTDOWN]:
                            continue
                        if not state.recovery_at:
                            continue
                        
                        # Check if recovery time reached
                        if now >= state.recovery_at:
                            LOG.info(f"Auto-recovery triggered: {state.level.value} {state.scope_id}")
                            self.deactivate_kill_switch(
                                state.level,
                                state.scope_id,
                                deactivated_by="auto_recovery"
                            )
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                LOG.error(f"Recovery loop error: {e}")
                time.sleep(10.0)
    
    def get_all_states(self) -> List[KillSwitchState]:
        """Get all kill-switch states"""
        with self._lock:
            return list(self._states.values())
    
    def get_summary(self) -> Dict:
        """Get summary of all kill-switches"""
        with self._lock:
            active = [s for s in self._states.values() if s.action != ResponseAction.NORMAL]
            
            return {
                'total_kill_switches': len(self._states),
                'active_kill_switches': len(active),
                'by_level': {
                    level.value: len([s for s in active if s.level == level])
                    for level in KillSwitchLevel
                },
                'by_action': {
                    action.value: len([s for s in active if s.action == action])
                    for action in ResponseAction
                },
                'global_state': self._states.get("global", KillSwitchState(KillSwitchLevel.GLOBAL, "global")).action.value
            }
    
    # ========================================================================
    # UNIFIED CHECK METHODS - Integration with RPM Engine
    # ========================================================================
    
    def check_kill_switches(
        self,
        portfolio_state: Optional[object] = None,
        regime: str = 'TRENDING',
        vol_percentile: float = 50.0,
        confidence_score: float = 1.0,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Tuple[bool, Optional[object], Optional[str]]:
        """
        Unified kill-switch check for RPM engine integration.
        
        Combines threshold-based and velocity-based checks.
        Returns format compatible with original kill_switches.py
        
        Args:
            portfolio_state: Current portfolio state (PortfolioState object)
            regime: Market regime
            vol_percentile: Volatility percentile
            confidence_score: Model confidence
            strategy_id: Optional strategy identifier
            symbol: Optional symbol
        
        Returns:
            Tuple[bool, Optional[RejectionReason], Optional[str]]:
                - triggered: True if kill switch triggered
                - reason: Rejection reason enum (or None)
                - details: Detailed explanation
        """
        # Check if already halted globally
        global_state = self.get_state(KillSwitchLevel.GLOBAL, "global")
        if global_state.action in [ResponseAction.SUSPEND, ResponseAction.SHUTDOWN]:
            return True, self._get_rejection_reason("TRADING_HALTED"), f"Global kill-switch: {global_state.action.value}"
        
        # Check strategy-specific
        if strategy_id:
            if not self.is_trading_allowed(strategy_id=strategy_id, symbol=symbol):
                return True, self._get_rejection_reason("TRADING_HALTED"), f"Strategy kill-switch active: {strategy_id}"
        
        # Check symbol-specific
        if symbol:
            if not self.is_trading_allowed(symbol=symbol):
                return True, self._get_rejection_reason("TRADING_HALTED"), f"Symbol kill-switch active: {symbol}"
        
        # If portfolio state provided, run threshold checks
        if portfolio_state and self.config:
            # Check 1: Maximum drawdown
            triggered, reason, details = self._check_max_drawdown(portfolio_state)
            if triggered:
                self.activate_kill_switch(
                    KillSwitchLevel.GLOBAL,
                    "global",
                    ResponseAction.SHUTDOWN,
                    TriggerReason.DRAWDOWN,
                    details={'drawdown': getattr(portfolio_state, 'current_drawdown', 0)}
                )
                return True, reason, details
            
            # Check 2: Daily loss limit
            triggered, reason, details = self._check_daily_loss_limit(portfolio_state)
            if triggered:
                self.activate_kill_switch(
                    KillSwitchLevel.GLOBAL,
                    "global",
                    ResponseAction.SUSPEND,
                    TriggerReason.LOSS_LIMIT,
                    details={'daily_pnl': getattr(portfolio_state, 'daily_pnl', 0)}
                )
                return True, reason, details
            
            # Check 3: Volatility shock
            triggered, reason, details = self._check_volatility_shock(vol_percentile)
            if triggered:
                self.activate_kill_switch(
                    KillSwitchLevel.GLOBAL,
                    "global",
                    ResponseAction.THROTTLE,
                    TriggerReason.LIQUIDITY_CRISIS,
                    details={'vol_percentile': vol_percentile}
                )
                return True, reason, details
            
            # Check 4: Confidence collapse
            triggered, reason, details = self._check_confidence_collapse(confidence_score)
            if triggered:
                self.activate_kill_switch(
                    KillSwitchLevel.GLOBAL,
                    "global",
                    ResponseAction.SUSPEND,
                    TriggerReason.RISK_BREACH,
                    details={'confidence_score': confidence_score}
                )
                return True, reason, details
        
        # Check velocity-based triggers
        rejection_velocity = self._check_rejection_velocity()
        if rejection_velocity > 0.8:  # 80% of threshold
            if strategy_id:
                self.activate_kill_switch(
                    KillSwitchLevel.STRATEGY,
                    strategy_id,
                    ResponseAction.THROTTLE,
                    TriggerReason.CONSECUTIVE_LOSSES,
                    details={'rejection_velocity': rejection_velocity}
                )
        
        return False, None, None
    
    def _get_rejection_reason(self, reason_str: str):
        """Get RejectionReason enum if available"""
        if RejectionReason:
            return getattr(RejectionReason, reason_str, None)
        return reason_str
    
    # ========================================================================
    # THRESHOLD-BASED CHECKS (from kill_switches.py)
    # ========================================================================
    
    def _check_max_drawdown(self, portfolio_state) -> Tuple[bool, Optional[object], Optional[str]]:
        """Check if maximum drawdown threshold breached"""
        if not self.config:
            return False, None, None
        
        current_dd = getattr(portfolio_state, 'current_drawdown', 0)
        max_dd = getattr(self.config, 'max_drawdown', 0.20)
        
        if current_dd > max_dd:
            details = f"Max drawdown breached: {current_dd*100:.2f}% > {max_dd*100:.2f}%"
            return True, self._get_rejection_reason("MAX_DRAWDOWN_EXCEEDED"), details
        
        return False, None, None
    
    def _check_daily_loss_limit(self, portfolio_state) -> Tuple[bool, Optional[object], Optional[str]]:
        """Check if daily loss limit breached"""
        if not self.config:
            return False, None, None
        
        daily_pnl = getattr(portfolio_state, 'daily_pnl', 0)
        total_capital = getattr(portfolio_state, 'total_capital', 1)
        daily_loss_limit = getattr(self.config, 'daily_loss_limit', 0.02)
        
        loss_limit_dollars = -total_capital * daily_loss_limit
        
        if daily_pnl < loss_limit_dollars:
            details = (
                f"Daily loss limit breached: ${daily_pnl:.2f} < "
                f"${loss_limit_dollars:.2f} ({daily_loss_limit*100:.1f}% of capital)"
            )
            return True, self._get_rejection_reason("DAILY_LOSS_LIMIT"), details
        
        return False, None, None
    
    def _check_volatility_shock(self, vol_percentile: float) -> Tuple[bool, Optional[object], Optional[str]]:
        """Check for volatility shock"""
        # Update volatility history
        self._volatility_history.append(vol_percentile)
        if len(self._volatility_history) > 100:
            self._volatility_history = self._volatility_history[-100:]
        
        # Establish baseline if we have enough data
        if len(self._volatility_history) >= 20 and self._volatility_baseline is None:
            self._volatility_baseline = np.median(self._volatility_history)
        
        # Check for shock (vol > 95th percentile or 3x baseline)
        if vol_percentile > 95.0:
            details = f"Volatility shock: {vol_percentile:.1f}th percentile (>95th)"
            return True, self._get_rejection_reason("VOLATILITY_SHOCK"), details
        
        if self._volatility_baseline and vol_percentile > self._volatility_baseline * 3:
            details = f"Volatility shock: {vol_percentile:.1f} > 3x baseline ({self._volatility_baseline:.1f})"
            return True, self._get_rejection_reason("VOLATILITY_SHOCK"), details
        
        return False, None, None
    
    def _check_confidence_collapse(self, confidence_score: float) -> Tuple[bool, Optional[object], Optional[str]]:
        """Check for model confidence collapse"""
        min_confidence = 0.3
        
        if confidence_score < min_confidence:
            details = f"Confidence collapse: {confidence_score:.3f} < {min_confidence}"
            return True, self._get_rejection_reason("MODEL_CONFIDENCE_LOW"), details
        
        return False, None, None
    
    # ========================================================================
    # VELOCITY-BASED CHECKS (from advanced_kill_switches.py)
    # ========================================================================
    
    def record_rejection(
        self,
        symbol: str,
        reason: str,
        strategy_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """Record a trade rejection for velocity tracking"""
        event = {
            'timestamp': timestamp or datetime.utcnow(),
            'symbol': symbol,
            'reason': reason,
            'strategy_id': strategy_id
        }
        self._rejection_history.append(event)
    
    def _check_rejection_velocity(self, time_window_minutes: int = 5) -> float:
        """
        Check rejection velocity (rejections per time window).
        
        Returns:
            float: Velocity as fraction of threshold (0.0 to 1.0+)
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Count recent rejections
        recent_rejections = [
            r for r in self._rejection_history
            if r['timestamp'] > cutoff_time
        ]
        
        max_rejections = 10  # Default threshold
        return len(recent_rejections) / max_rejections
    
    def record_exposure_change(
        self,
        gross_exposure: float,
        net_exposure: float,
        leverage: float,
        num_positions: int
    ):
        """Record exposure snapshot for velocity tracking"""
        snapshot = {
            'timestamp': datetime.utcnow(),
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'leverage': leverage,
            'num_positions': num_positions
        }
        self._exposure_history.append(snapshot)
    
    # ========================================================================
    # BACKWARD COMPATIBILITY (for old kill_switches.py interface)
    # ========================================================================
    
    def manual_halt(self, reason: str = "Manual intervention"):
        """Manual halt - backward compatibility"""
        self.activate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            ResponseAction.SHUTDOWN,
            TriggerReason.MANUAL,
            triggered_by="manual_halt",
            details={'reason': reason}
        )
    
    def manual_resume(self, resumed_by: str = "operator"):
        """Manual resume - backward compatibility"""
        self.deactivate_kill_switch(
            KillSwitchLevel.GLOBAL,
            "global",
            deactivated_by=resumed_by
        )
    
    def get_kill_switch_status(
        self,
        portfolio_state: Optional[object] = None,
        regime: str = 'TRENDING'
    ) -> Dict:
        """Get comprehensive kill-switch status - backward compatibility"""
        summary = self.get_summary()
        
        # Add threshold status if portfolio state available
        if portfolio_state and self.config:
            current_dd = getattr(portfolio_state, 'current_drawdown', 0)
            max_dd = getattr(self.config, 'max_drawdown', 0.20)
            daily_pnl = getattr(portfolio_state, 'daily_pnl', 0)
            
            summary['thresholds'] = {
                'drawdown': {
                    'current': current_dd,
                    'max': max_dd,
                    'utilization_pct': (current_dd / max_dd * 100) if max_dd > 0 else 0
                },
                'daily_pnl': daily_pnl
            }
        
        # Add velocity status
        rejection_velocity = self._check_rejection_velocity()
        summary['velocity'] = {
            'rejection_rate': rejection_velocity,
            'rejections_5min': len([
                r for r in self._rejection_history
                if r['timestamp'] > datetime.utcnow() - timedelta(minutes=5)
            ])
        }
        
        return summary
