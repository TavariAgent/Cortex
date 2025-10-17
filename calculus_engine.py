from abc import ABC, abstractmethod
import asyncio
from mpmath import mp
import sympy as sp

from packing_utils import convert_and_pack
from precision_manager import get_dps
from slice_mixin import SliceMixin

mp.dps = get_dps()


class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager

    @abstractmethod
    def compute(self, expr):
        """Parallel compute method: Calculate all available parts simultaneously, respecting primary level."""
        pass

    # Stub dunder methods for math ops (except __add__ which is handled by segment_manager)
    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __div__(self, other):
        pass

    @abstractmethod
    def __pow__(self, other):
        pass

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


class CalculusEngine(SliceMixin, MathEngine):
    """Handles derivatives, integrals, limits. Heavily relies on SymPy."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []
        self._value = "x"  # Default symbolic var

    def _add_traceback(self, step, info):
        """Add step-wise traceback for debugging."""
        try:
            loop = asyncio.get_running_loop()
            timestamp = loop.time()
        except RuntimeError:
            timestamp = 0

        self.traceback_info.append({
            'step': step,
            'info': info,
            'timestamp': timestamp
        })

    async def _compute_single_part(self, part):
        """Stub: Compute single part, respecting priorities."""
        if 'nest' in str(part):
            return f"nested_{part}"
        return part * 2

    async def _compute_parts_parallel(self, parts):
        """Helper: Compute all parts in parallel, using _set_part_order at start."""
        ordered_parts = self._set_part_order(parts, apply_at_start=True)
        tasks = [self._compute_single_part(part) for part in ordered_parts]
        results = await asyncio.gather(*tasks)
        # Apply again after return if needed
        final_results = self._set_part_order(results, apply_after_return=True)
        for i, result in enumerate(final_results):
            self.segment_manager.receive_part_order(
                self.__class__.__name__,
                f'part_{i}',
                [{'part': f'part_{i}', 'result': result}]
            )
        return final_results

    def compute(self, expr: str):
        """Compute a calculus-level expression and return NUMERIC text."""
        self._add_traceback('compute_start', expr)

        # 1. Parse; force evaluation of derivative/integral/limit
        expr_sym = sp.sympify(expr,
                              locals={
                                  'derivative': sp.diff,
                                  'diff'      : sp.diff,
                                  'integral'  : sp.integrate,
                                  'integrate' : sp.integrate,
                                  'limit'     : sp.limit,
                              },
                              evaluate=True)

        # 2. Resolve any remaining Derivative / Integral / Limit nodes
        expr_sym = expr_sym.doit().evalf(mp.dps)

        # 3. If the result is still symbolic try numerical eval
        try:
            num_val = expr_sym.evalf(mp.dps)       # type: ignore[attr-defined]
            result  = mp.mpf(str(num_val))
        except (TypeError, ValueError):
            # still symbolic ⇒ hand the unevaluated string upward
            result = expr_sym

        # 4. Pack & return
        packed = str(result).encode()
        self._cache.append(packed)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
        self._add_traceback('compute_end', f'{expr_sym} -> {result}')
        return str(result)


    def __sub__(self, other):
        """Calculus sub."""
        self._add_traceback('__sub__', f'Calculus sub: {other}')
        return self

    def __mul__(self, other):
        """Calculus mul."""
        self._add_traceback('__mul__', f'Calculus mul: {other}')
        return self

    def __div__(self, other):
        """Calculus div."""
        self._add_traceback('__div__', f'Calculus div: {other}')
        return self

    def __pow__(self, other):
        """Calculus pow (stub)."""
        self._add_traceback('__pow__', f'Calculus pow: {other}')
        return self