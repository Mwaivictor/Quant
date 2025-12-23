"""
Signal Generation Engine

Conservative decision layer that converts quantitative validation and ML confidence
into actionable trade intents.

Core Principle:
    The Signal Engine decides WHETHER a trade should exist, NOT how it executes.
    
Philosophy:
    - Trade intents, not orders
    - Direction assignment from deterministic momentum
    - Confidence scoring for downstream risk sizing
    - Strict regime, statistics, and ML filtering gates
    - Fully deterministic and auditable
    - When in doubt, do nothing

Flow:
    Feature Engine → Quant Stats → ML Layer → Signal Engine → Risk Manager
    
Responsibilities:
    1. Regime gate enforcement (TRENDING only)
    2. Statistical robustness validation
    3. ML confidence filtering
    4. Direction assignment (deterministic)
    5. Confidence score computation
    6. Trade state management
    7. Trade intent emission

Output:
    Pure trade intent objects - no execution parameters
"""

from arbitrex.signal_engine.config import SignalEngineConfig
from arbitrex.signal_engine.schemas import (
    TradeIntent,
    TradeDirection,
    SignalState,
    SignalDecision,
    SignalEngineOutput
)
from arbitrex.signal_engine.engine import SignalGenerationEngine
from arbitrex.signal_engine.state_manager import SignalStateManager
from arbitrex.signal_engine.filters import (
    RegimeGate,
    QuantStatsGate,
    MLConfidenceGate
)

__version__ = "1.0.0"

__all__ = [
    'SignalEngineConfig',
    'SignalGenerationEngine',
    'SignalStateManager',
    'TradeIntent',
    'TradeDirection',
    'SignalState',
    'SignalDecision',
    'SignalEngineOutput',
    'RegimeGate',
    'QuantStatsGate',
    'MLConfidenceGate',
]
