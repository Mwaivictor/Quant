"""Market calendar abstraction.

This module tries to provide a production-grade calendar via `exchange_calendars` if
available. If not installed, it falls back to a conservative FX heuristic used elsewhere.

API:
- `is_market_open(symbol: str, ts: Optional[float] = None) -> bool`

Configuration:
- set `MARKET_CALENDAR=exchange_calendars` in env to enable exchange_calendars usage.
"""
from datetime import datetime, timezone
import os
import logging
import re
import json

LOG = logging.getLogger('arbitrex.raw.market_calendar')

try:
    import exchange_calendars as xcals
    XCALS_AVAILABLE = True
except Exception:
    XCALS_AVAILABLE = False

# Try loading a maintained symbol->calendar mapping file (JSON) if present.
_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'symbol_calendar_map.json')
_SYMBOL_CAL_MAP: dict[str, str] = {}
if os.path.exists(_MAPPING_PATH):
    try:
        with open(_MAPPING_PATH, 'r', encoding='utf8') as fh:
            _SYMBOL_CAL_MAP = json.load(fh)
            # normalize keys/values
            _SYMBOL_CAL_MAP = {k.strip().upper(): v.strip().upper() for k, v in _SYMBOL_CAL_MAP.items()}
            LOG.info('Loaded symbol->calendar mapping from %s (%d entries)', _MAPPING_PATH, len(_SYMBOL_CAL_MAP))
    except Exception as e:
        LOG.warning('Failed to load symbol_calendar_map.json: %s', e)


def _fx_heuristic(dt: datetime | None = None) -> bool:
    # FX generally open Sunday 22:00 UTC and close Friday 22:00 UTC
    now = dt if dt is not None else datetime.utcnow()
    wd = now.weekday()
    if wd == 4 and now.hour >= 22:
        return False
    if wd == 5:
        return False
    if wd == 6 and now.hour < 22:
        return False
    return True


_FX_RE = re.compile(r'^[A-Z]{6}$')


def _is_fx_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    s = symbol.strip().upper()
    # common FX pairs are 6 letters like EURUSD, GBPUSD
    if _FX_RE.match(s):
        return True
    # metals (XAU, XAG) often appear as XAUUSD etc.
    if s.startswith('XAU') or s.startswith('XAG'):
        return True
    return False


def map_symbol_to_calendar(symbol: str) -> str:
    """Return a calendar id or 'FX' for FX-like symbols.

    This uses simple heuristics. For production, provide a maintained mapping
    from symbols to exchange calendars (e.g., via config file).
    """
    s = (symbol or '').strip().upper()
    # First check the explicit mapping file if present
    if s and _SYMBOL_CAL_MAP:
        mapped = _SYMBOL_CAL_MAP.get(s)
        if mapped:
            return mapped
    if _is_fx_symbol(s):
        return 'FX'
    # ETFs/Indices heuristics: if symbol contains letters and length 3-5, assume exchange-listed
    if len(s) <= 5 and s.isalpha():
        return 'NYSE'
    # fallback
    return 'FX'


def is_market_open(symbol: str, ts: float | None = None) -> bool:
    """Return True if market for `symbol` is open at `ts` (epoch seconds) or now.

    Behavior:
    - FX symbols use a lightweight FX calendar (Sunday 22:00 UTC -> Friday 22:00 UTC).
    - If `exchange_calendars` is available and `MARKET_CALENDAR=exchange_calendars` is set,
      we attempt to use a mapped exchange calendar for non-FX symbols.
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    cal_id = map_symbol_to_calendar(symbol)

    if cal_id == 'FX':
        return _fx_heuristic(dt)

    if XCALS_AVAILABLE and os.environ.get('MARKET_CALENDAR', '').lower() == 'exchange_calendars':
        try:
            # Map to a best-guess exchange calendar id; default to NYSE for equities
            cal = xcals.get_calendar(cal_id)
            return cal.is_session(dt)
        except Exception as e:
            LOG.debug('exchange_calendars check failed for %s: %s', cal_id, e)

    # fallback: treat as FX
    return _fx_heuristic(dt)
