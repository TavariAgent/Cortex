from abc import ABC, abstractmethod
import asyncio
from mpmath import mp

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

class ComplexAlgebraEngine(SliceMixin, MathEngine):
    """Handles complex numbers: real + imag*j. Overloads for complex ops."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []
        self._value = "0+0j"  # Default complex

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

    @staticmethod
    async def _compute_single_part(part):
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

    def compute(self, expr):
        """Compute complex expressions, e.g., (1+2j)*(3+4j)."""
        self._add_traceback('compute_start', f'Expression: {expr}')

        # Simple: parse basic complex ops (expand as needed)
        if '+' in expr and 'j' in expr:
            parts = expr.split('+')
            real = mp.mpf(parts[0])
            imag = mp.mpf(parts[1].replace('j', ''))
            result = mp.mpc(real, imag)
        else:
            raise ValueError(f"Unsupported complex expr: {expr}")

        self._add_traceback('result', f'Complex result: {result}')
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        part_order = [{'part': 'complex_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(self.__class__.__name__, f'complex_{expr}', part_order)
        return str(result)

    def __sub__(self, other):
        """Complex sub."""
        self._add_traceback('__sub__', f'Complex sub: {other}')
        # Implement complex subtraction
        return self

    def __mul__(self, other):
        """Complex mul."""
        self._add_traceback('__mul__', f'Complex mul: {other}')
        # Implement complex multiplication
        return self

    def __div__(self, other):
        """Complex div (stub)."""
        self._add_traceback('__div__', f'Complex div: {other}')
        return self

    def __pow__(self, other):
        """Complex pow."""
        self._add_traceback('__pow__', f'Complex pow: {other}')
        # Implement complex exponentiation
        return self