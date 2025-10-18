import asyncio

from abc_engines import BasicArithmeticEngine
from calculus_engine import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine
from elementary_engine import ElementaryEngine
from trigonometry_engine import TrigonometryEngine


class FlagManager:
    def __init__(self):
        # Example: boolean flags; initialize as you need
        self._flags = {
            'basic': False,
            'trig': False,
            'elem': False,
            'calc': False,
            'algebra': False,
        }
        # One event per flag
        self._events = {name: asyncio.Event() for name in self._flags}

    def set_flag(self, name: str, value: bool):
        # If set from non-async code running in the same thread as the loop, this is fine.
        if self._flags[name] != value:
            self._flags[name] = value
            self._events[name].set()  # wake any waiters

    async def wait_for_flag(self, name: str, *, value: bool = True):
        # Fast-path: already satisfied
        if self._flags[name] == value:
            return
        # Otherwise wait for a change signal, then check again
        while self._flags[name] != value:
            await self._events[name].wait()
            # Reset for future waits
            self._events[name].clear()

    async def monitor_flags(self):
        # Wait for any flag to change without polling
        while True:
            # Wait for any one event to be set
            waits = [evt.wait() for evt in self._events.values()]
            done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

            # Clear only those that fired and act on the changes
            for name, evt in self._events.items():
                if evt.is_set():
                    evt.clear()
                    new_value = self._flags[name]
                    # Handle the change for this flag here
                    # e.g., start/stop an engine, log, notify, etc.