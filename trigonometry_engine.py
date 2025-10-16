from abc import ABC, abstractmethod
import asyncio
import mpmath as mp
import sympy as sp

from packing_utils import convert_and_pack
from priority_rules import precedence_of
from slice_mixin import SliceMixin


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
    def snap_to_angle(arg):
        """Snap arg to common angles for exact trig results (avoids mpmath approximations)."""
        if abs(arg - 0) < 1e-10:
            return 0
        if abs(arg - mp.pi) < 1e-10:
            return mp.pi
        if abs(arg - mp.pi/2) < 1e-10:
            return mp.pi/2
        if abs(arg - mp.pi/4) < 1e-10:
            return mp.pi/4
        # Add more as needed (e.g., 2*pi, etc.)
        return arg

    def compute(self, expr):
        """Compute trig expressions with symbolic arg evaluation, pi recognition, and angle snapping."""
        self._add_traceback('compute_start', f'Expression: {expr}')
        mp.dps = 50

        # Simple: assume expr is like 'cos(pi/2)'
        if 'sin(' in expr:
            arg_str = expr.split('sin(')[1].rstrip(')')
            try:
                arg = mp.mpf(str(sp.N(sp.sympify(arg_str))))  # Evaluate symbolic expressions numerically
            except:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.sin(arg)
        elif 'cos(' in expr:
            arg_str = expr.split('cos(')[1].rstrip(')')
            try:
                arg = mp.mpf(str(sp.N(sp.sympify(arg_str))))
            except:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.cos(arg)
        elif 'tan(' in expr:
            arg_str = expr.split('tan(')[1].rstrip(')')
            try:
                arg = mp.mpf(str(sp.N(sp.sympify(arg_str))))
            except:
                raise ValueError(f"Invalid argument: {arg_str}")
            arg = self.snap_to_angle(arg)
            result = mp.tan(arg)
        else:
            raise ValueError(f"Unsupported trig expr: {expr}")

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