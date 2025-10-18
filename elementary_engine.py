from abc import ABC, abstractmethod
from typing import List

from mpmath import mp
import sympy as sp

from slice_mixin import SliceMixin
from packing import convert_and_pack
from utils.precision_manager import get_dps
from utils.trace_helpers import add_traceback
from segment_manager import SegmentManager
from xor_string_compiler import XorStringCompiler

mp.dps = get_dps()


class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager, enable_injection=None):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager

        # Handle injection setup
        if enable_injection is None:
            enable_injection = get_dps() >= 1000

        self.enable_injection = enable_injection

        if enable_injection:
            self.xor_compiler = XorStringCompiler()
        else:
            self.xor_compiler = None

    @abstractmethod
    def compute(self, expr):
        """Parallel compute method: Calculate all available parts simultaneously, respecting primary level."""
        pass

    @staticmethod
    async def _compute_single_part(part):
        """Stub: Compute single part, respecting priorities."""
        if 'nest' in str(part):
            return f"nested_{part}"
        return part * 2


    def __add__(self, other):
        """Overload __add__: Send part orders to segment manager per-slice."""
        # Stub: Generate part orders based on engine logic
        part_order = [{'part': 'example', 'bytes': b'data'}]  # Mock
        slice_data = 'slice_1'  # From expression parsing
        self.segment_manager.receive_part_order(self.__class__.__name__, slice_data, part_order)
        return self  # Or metadata

    @staticmethod
    def _normalise_small(x, *, eps=None):
        eps = eps or mp.mpf(10) ** (-mp.dps + 2)  # ~100 ulps
        return mp.mpf(0) if mp.fabs(x) < eps else x

    @staticmethod
    def _convert_and_pack(parts, *, twos_complement=False):
        return convert_and_pack(parts, twos_complement=twos_complement)


class ElementaryEngine(SliceMixin, MathEngine):
    """
    Elementary functions: exp, log, log10, log2, log1p, sqrt.
    """

    def __init__(self, segment_mgr: SegmentManager):
        self.segment_manager = segment_mgr
        self.traceback_info: List[dict] = []
        self._cache: List[bytes] = []

    # -------------------------------------------------------------- #
    # Trace helper
    # -------------------------------------------------------------- #
    def _add_traceback(self, step: str, info: str):
        add_traceback(self, step, info)

    # -------------------------------------------------------------- #
    # Public helpers (direct calls)
    # -------------------------------------------------------------- #
    def _pack_and_send(self, tag: str, result):
        packed = convert_and_pack([result])
        self._cache.append(packed)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
        self._add_traceback(tag, f'Result = {result}')
        return str(result)

    # ---- elementary wrappers ------------------------------------- #
    def exp(self, x):      return self._pack_and_send('exp',   mp.exp  (mp.mpf(str(x))))
    def log(self, x):      return self._pack_and_send('log',   mp.log  (mp.mpf(str(x))))
    def log10(self, x):    return self._pack_and_send('log10', mp.log10(mp.mpf(str(x))))
    def log2(self, x):     return self._pack_and_send('log2',  mp.log2 (mp.mpf(str(x))))
    def log1p(self, x):    return self._pack_and_send('log1p', mp.log1p(mp.mpf(str(x))))
    def sqrt(self, x):     return self._pack_and_send('sqrt',  mp.sqrt (mp.mpf(str(x))))

    # -------------------------------------------------------------- #
    # compute() entry ‑- recognise textual functions
    # -------------------------------------------------------------- #
    def compute(self, expr: str):
        self._add_traceback('compute_start', expr)
        for name, fn in [('log10(', self.log10),
                         ('log2(',  self.log2),
                         ('log1p(', self.log1p),
                         ('log(',   self.log),
                         ('exp(',   self.exp),
                         ('sqrt(',  self.sqrt)]:
            if expr.startswith(name):
                inner = expr[len(name):-1]
                sym  = sp.sympify(inner).evalf(mp.dps)
                return fn(str(sym))
        raise ValueError(f'Unsupported elementary expression: {expr}')