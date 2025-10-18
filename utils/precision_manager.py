"""
Central precision switch for the whole calculator stack.
REPL (or tests) may call set_dps(value) to bump precision; all
engines should only *read* the current value through get_dps().
"""
from typing import List
from mpmath import mp

_PRESETS: List[int] = [50, 1000, 7500, 15000, 100000, 1000000]  # default + 5 big ones
_CURRENT = mp.dps                                  # seed from mpmath import

def get_dps() -> int:
    """Return the active decimal-places setting."""
    return _CURRENT

def set_dps(value: int) -> None:
    """Set global precision if value is one of the approved presets."""
    global _CURRENT
    if value not in _PRESETS:
        raise ValueError(f"dps {value} not allowed; choose one of {_PRESETS}")
    _CURRENT = value
    mp.dps = value

def presets() -> List[int]:
    return _PRESETS.copy()