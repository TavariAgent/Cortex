from abc import ABC, abstractmethod
import asyncio
from mpmath import mp
import sympy as sp

from packing_utils import convert_and_pack
from precision_manager import get_dps
from priority_rules import precedence_of
from slice_mixin import SliceMixin

mp.dps = get_dps()

class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager

    # ──────────────────────────────────────────────────────────────────
    #  Exact-angle lookup
    # ──────────────────────────────────────────────────────────────────
    _ANGLE_TABLE = {
        sp.Rational(0, 1):    {'sin': 0, 'cos': 1, 'tan': 0},
        sp.Rational(1, 6):    {'sin': sp.Rational(1, 2),
                               'cos': sp.sqrt(3)/2,
                               'tan': sp.sqrt(3)/3},
        sp.Rational(1, 4):    {'sin': sp.sqrt(2)/2,
                               'cos': sp.sqrt(2)/2,
                               'tan': 1},
        sp.Rational(1, 3):    {'sin': sp.sqrt(3)/2,
                               'cos': sp.Rational(1, 2),
                               'tan': sp.sqrt(3)},
        sp.Rational(1, 2):    {'sin': 1,
                               'cos': 0,
                               'tan': sp.oo},
        sp.Rational(2, 3):    {'sin': sp.sqrt(3)/2,
                               'cos': -sp.Rational(1, 2),
                               'tan': -sp.sqrt(3)},
        sp.Rational(3, 4):    {'sin':  sp.sqrt(2)/2,
                               'cos': -sp.sqrt(2)/2,
                               'tan': -1},
        sp.Rational(5, 6):    {'sin':  sp.Rational(1, 2),
                               'cos': -sp.sqrt(3)/2,
                               'tan': -sp.sqrt(3)/3},
        sp.Rational(1, 1):    {'sin': 0,
                               'cos': -1,
                               'tan': 0},
        # Add more multiples if desired
    }

    @classmethod
    def _lookup_exact(cls, arg_mpf):
        """
        Try to convert arg (mp.mpf) into a SymPy Rational multiple of π.
        If found in table, return (True, sin_val, cos_val, tan_val),
        else (False, None, None, None).
        """
        # Convert mp.mpf → SymPy Float with current precision
        sp_val = sp.nsimplify(arg_mpf, [sp.pi])
        if isinstance(sp_val, sp.Mul) and sp.pi in sp_val.args:
            ratio = sp_val / sp.pi
        elif sp_val == 0:
            ratio = sp.Rational(0, 1)
        else:
            return False, None, None, None

        # Reduce modulo 2π (ratio modulo 2)
        ratio = sp.fraction(ratio)[0] % 2  # numerator mod 2
        if ratio > 1:
            ratio -= 2  # map to (-1,1]
        ratio = sp.Rational(ratio).limit_denominator()

        if ratio in cls._ANGLE_TABLE:
            entry = cls._ANGLE_TABLE[ratio]
            return True, entry['sin'], entry['cos'], entry['tan']
        return False, None, None, None


    @abstractmethod
    def compute(self, expr):
        """Parallel compute method: Calculate all available parts simultaneously, respecting primary level."""
        pass

    @staticmethod
    async def _compute_single_part(self, part):
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


class TrigonometryEngine(SliceMixin, MathEngine):
    """Handles trig functions: sin, cos, tan, etc. Implements dunders where ops apply."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []
        self._value = "0"

    # ------------------------------------------------------------------
    # SliceMixin requirement: how to actually evaluate *one* slice
    # ------------------------------------------------------------------
    def _evaluate_atom(self, slice_text: str):
        """
        Evaluate a slice that now contains *no parentheses*.
        Handle ^, *, /, +, - with mpmath high precision.
        """
        tokens = self._linear_tokenize(slice_text)      # NEW helper below
        # Apply operator precedence using priority_rules.py
        for op_level in [4, 3, 2]:                      # ^  then */  then +-
            i = 0
            while i < len(tokens):
                if precedence_of(tokens[i]) == op_level:
                    left = mp.mpf(tokens[i-1]); right = mp.mpf(tokens[i+1])
                    if tokens[i] == '^':
                        val = mp.power(left, right)
                    elif tokens[i] == '*':
                        val = left * right
                    elif tokens[i] == '/':
                        val = left / right
                    elif tokens[i] == '+':
                        val = left + right
                    else:
                        val = left - right
                    tokens = tokens[:i-1] + [str(val)] + tokens[i+2:]
                else:
                    i += 1
        if len(tokens) != 1:
            raise ValueError(f'Could not resolve slice: {tokens}')
        return mp.mpf(tokens[0])


    def _linear_tokenize(self, flat_expr: str):
        """Simple left-to-right tokenizer for numbers and operators (no parens)."""
        out, cur = [], ''
        for ch in flat_expr:
            if ch.isdigit() or ch == '.':
                cur += ch
            elif ch in '+-*/^':
                if cur:
                    out.append(cur); cur = ''
                out.append(ch)
            elif ch == ' ':
                continue
            else:
                raise ValueError(f'Unexpected char {ch}')
        if cur:
            out.append(cur)
        return out

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
    def _set_part_order(parts, apply_at_start=True, apply_after_return=False):
        """Helper: Set part order via priority + left-to-right association. Flows around PEMDAS by respecting priorities within parts.

        - Priority mapping: ^ (highest), * /, + - (lowest). Sums/limits prioritize slice-level.
        - Applies at start (before computation) and/or after return (post-result).
        - Returns ordered list for dunder method requests.
        """
        priority_map = {
            '^': 4,  # Exponentiation highest
            '*': 3, '/': 3,  # Mul/div mid
            '+': 2, '-': 2,  # Add/sub lowest
            'sum': 1, 'limit': 1, 'other': 0  # Sums/limits lower, slice-focused
        }

        def get_priority(part):
            # Extract op from part (stub: assume part is dict or string with op)
            op = getattr(part, 'op', str(part).split()[0] if ' ' in str(part) else 'other')
            return priority_map.get(op, 0)

        if apply_at_start or apply_after_return:
            # Sort by priority desc, then left-to-right (by original index as tiebreaker)
            ordered = sorted(enumerate(parts), key=lambda x: (-get_priority(x[1]), x[0]))
            return [part for _, part in ordered]
        return parts  # No change if not applying

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
    def _convert_and_pack(parts, *, twos_complement=False):
        return convert_and_pack(parts, twos_complement=twos_complement)

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
        if 'sin(' in expr:
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