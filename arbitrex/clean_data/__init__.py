"""
Arbitrex Clean Data Layer

Purpose:
    Transform raw OHLCV bars into analysis-safe bars without introducing
    lookahead bias, smoothing, or hidden assumptions.

Philosophy:
    - Deterministic and auditable
    - Flag issues, never fix them silently
    - Prefer no data over bad data
    - Explicit failure over silent corruption

Contract:
    Input:  Raw OHLCV bars (immutable, untrusted)
    Output: fx_ohlcv_clean (validated, flagged, or rejected)

Guarantees:
    - Raw OHLC values never altered
    - All timestamps aligned to UTC canonical grid
    - Missing bars explicitly flagged (never forward-filled)
    - Outliers detected and flagged (never corrected)
    - Invalid bars explicitly marked (valid_bar=False)
    - Complete auditability and reproducibility
"""

__version__ = "1.0.0"
__author__ = "Arbitrex Quantitative Research"

from arbitrex.clean_data.pipeline import CleanDataPipeline
from arbitrex.clean_data.schemas import CleanOHLCVSchema
from arbitrex.clean_data.config import CleanDataConfig
from arbitrex.clean_data.integration import RawToCleanBridge

__all__ = [
    "CleanDataPipeline",
    "CleanOHLCVSchema", 
    "CleanDataConfig",
    "RawToCleanBridge",
]
