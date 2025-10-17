from abc import ABC, abstractmethod
import asyncio
from mpmath import mp
import sympy as sp

from packing_utils import convert_and_pack
from precision_manager import get_dps
from slice_mixin import SliceMixin

mp.dps = get_dps()

# Helper to build the exact-angle lookup table

def _build_angle_table():
    table = {}
    dens = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    for d in dens:
        for n in range(0, 2*d + 1):  # cover 0 .. 2 inclusive
            r = sp.Rational(n, d)
            key = sp.Rational(0, 1) if r == 2 else r
            if key in table:
                continue
            angle = sp.pi * r
            s = sp.simplify(sp.sin(angle))
            c = sp.simplify(sp.cos(angle))
            t = sp.simplify(sp.tan(angle))
            if t is sp.zoo:
                t = sp.oo  # unify representation
            table[key] = {'sin': s, 'cos': c, 'tan': t}
    return table

class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager

    # ──────────────────────────────────────────────────────────────────
    #  Exact-angle lookup
    # ──────────────────────────────────────────────────────────────────
    # Build an exact-angle lookup for multiples of pi with improved readability
    # and broader coverage. See module-level _build_angle_table().
    _ANGLE_TABLE = _build_angle_table()

    @classmethod
    def _lookup_exact(cls, arg_mpf):
        """
        Try to convert arg (mp.mpf) into a SymPy Rational multiple of π.
        If found in table, return (True, sin_val, cos_val, tan_val),
        else (False, None, None, None).
        """
        # Convert mp.mpf -> SymPy rational multiple of pi
        try:
            ratio = sp.nsimplify(mp.mpf(arg_mpf) / mp.pi)
        except Exception:
            return False, None, None, None

        # Must be a rational multiple to match table
        if not isinstance(ratio, sp.Rational):
            return False, None, None, None

        # Reduce to principal range [0, 2)
        ratio = ratio % 2
        if ratio == 2:
            ratio = sp.Rational(0, 1)

        if ratio in cls._ANGLE_TABLE:
            entry = cls._ANGLE_TABLE[ratio]
            return True, entry['sin'], entry['cos'], entry['tan']
        return False, None, None, None


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


class TrigonometryEngine(SliceMixin, MathEngine):
    """Handles trig functions: sin, cos, tan, etc. Implements dunders where ops apply."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []
        self._value = "0"

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

    @staticmethod
    def snap_to_angle(arg, *, max_multiple: int = 4):
        """
        Return an *exact* multiple of π/4 (covers π/2, π) when the input
        is within an adaptive tolerance; otherwise return arg unchanged.

        The tolerance scales with current mp.dps so that, whether you run at
        50 dps or one million, “almost-π” will still be captured.
        """
        # 10 ulps worth of slack at the current precision
        tol = mp.mpf(10) ** (-mp.dps + 1)

        # Work in units of π/4                   (0, ¼π, ½π, ¾π, π, …)
        unit = mp.pi / 4
        ratio = arg / unit
        nearest = mp.nint(ratio)                # nearest integer

        # Only snap small multiples (avoids wrapping very large arguments)
        if abs(nearest) <= max_multiple * 4 and mp.fabs(ratio - nearest) < tol:
            return nearest * unit

        return arg

    def compute(self, expr):
        """Compute trig expressions with symbolic arg evaluation, pi recognition, and angle snapping."""
        self._add_traceback('compute_start', f'Expression: {expr}')

        # ── 1.  mixed expression with + – * / ? ──────────────────────
        if any(op in expr for op in ('+', '-', '*', '/', '^')):
            try:
                sy = sp.sympify(expr, locals={'sin': sp.sin,
                                              'cos': sp.cos,
                                              'tan': sp.tan})
                num = sy.evalf(mp.dps)  # evaluate at current precision
                self._add_traceback('sympy_eval', f'{sy} -> {num}')
                result = mp.mpf(str(num))
                packed = convert_and_pack([result])
                self._cache.append(packed)
                self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
                return str(result)
            except Exception as e:
                self._add_traceback('sympy_fail', str(e))
                # fall-through to single-call parser

        # ── 2.  single trig call  ─────────────────────────────────────
        if expr.startswith('sin('):
            arg_str = expr.split('sin(')[1].rstrip(')')
            try:
                sym_expr = sp.sympify(arg_str)
                num      = sym_expr.evalf(mp.dps)     # use active precision
                arg      = mp.mpf(str(num))
            except Exception:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.sin(arg)
        elif 'cos(' in expr:
            arg_str = expr.split('cos(')[1].rstrip(')')
            try:
                sym_expr = sp.sympify(arg_str)
                num      = sym_expr.evalf(mp.dps)
                arg      = mp.mpf(str(num))
            except Exception:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.cos(arg)
        elif 'tan(' in expr:
            arg_str = expr.split('tan(')[1].rstrip(')')
            try:
                sym_expr = sp.sympify(arg_str)
                num      = sym_expr.evalf(mp.dps)
                arg      = mp.mpf(str(num))
            except Exception:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.tan(arg)
        else:
            raise ValueError(f"Unsupported trig expression: {expr}")

        self._add_traceback('result', f'Trig result: {result}')
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        part_order = [{'part': 'trig_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(self.__class__.__name__, f'trig_{expr}', part_order)
        return str(result)

    def __sub__(self, other):
        """Trig-specific sub (stub)."""
        self._add_traceback('__sub__', f'Trig sub: {other}')
        return self

    def __mul__(self, other):
        """Trig-specific mul (stub)."""
        self._add_traceback('__mul__', f'Trig mul: {other}')
        return self

    def __div__(self, other):
        """Trig-specific div (stub)."""
        self._add_traceback('__div__', f'Trig div: {other}')
        return self

    def __pow__(self, other):
        """Trig-specific pow (stub)."""
        self._add_traceback('__pow__', f'Trig pow: {other}')
        return self

    def sin(self, x):
        """Compute sin using mpmath."""
        self._add_traceback('sin', f'sin({x})')
        result = mp.sin(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def cos(self, x):
        """Compute cos using mpmath."""
        self._add_traceback('cos', f'cos({x})')
        result = mp.cos(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def tan(self, x):
        """Compute tan using mpmath."""
        self._add_traceback('tan', f'tan({x})')
        result = mp.tan(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def sqrt(self, x):
        """Compute sqrt using mpmath."""
        self._add_traceback('sqrt', f'sqrt({x})')
        result = mp.sqrt(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def cbrt(self, x):
        """Compute cbrt using mpmath."""
        self._add_traceback('cbrt', f'cbrt({x})')
        result = mp.cbrt(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def atan(self, x):
        """Compute atan using mpmath."""
        self._add_traceback('atan', f'atan({x})')
        result = mp.atan(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def asin(self, x):
        """Compute asin using mpmath."""
        self._add_traceback('asin', f'asin({x})')
        result = mp.asin(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def acos(self, x):
        """Compute acos using mpmath."""
        self._add_traceback('acos', f'acos({x})')
        result = mp.acos(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def atan2(self, y, x):
        """Compute atan2 using mpmath."""
        self._add_traceback('atan2', f'atan2({y}, {x})')
        result = mp.atan2(mp.mpf(str(y)), mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def asinh(self, x):
        """Compute asinh using mpmath."""
        self._add_traceback('asinh', f'asinh({x})')
        result = mp.asinh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def acosh(self, x):
        """Compute acosh using mpmath."""
        self._add_traceback('acosh', f'acosh({x})')
        result = mp.acosh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def atanh(self, x):
        """Compute atanh using mpmath."""
        self._add_traceback('atanh', f'atanh({x})')
        result = mp.atanh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def sinh(self, x):
        """Compute sinh using mpmath."""
        self._add_traceback('sinh', f'sinh({x})')
        result = mp.sinh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def cosh(self, x):
        """Compute cosh using mpmath."""
        self._add_traceback('cosh', f'cosh({x})')
        result = mp.cosh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def tanh(self, x):
        """Compute tanh using mpmath."""
        self._add_traceback('tanh', f'tanh({x})')
        result = mp.tanh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result