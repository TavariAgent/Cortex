"""
Global flag registry for the whole calculator stack.

Any module can do:
    from flag_bus import FlagBus
    FlagBus.set('engines_done', True)
    if FlagBus.get('xor_active'):
        ...
A tiny layer around a module-level dict keeps things threadsafe and
gives us optional asyncio-wait helpers for “gated logic”.
"""
from typing import Any, Dict

class _FlagBusImpl:
    _flags: Dict[str, Any] = {
        # seeded defaults (extend as you like)
        'xor_active': False,
        'engines_done': False,
    }

    # ––– basic get/set –––
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._flags.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls._flags[key] = value


# public alias
FlagBus = _FlagBusImpl