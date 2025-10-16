import functools
from abc import ABC, abstractmethod
import asyncio
import numpy as np
from mpmath import mp
from decimal import Decimal
from sympy import sympify, nsimplify, srepr

from packing_utils import convert_and_pack
from precision_manager import get_dps
from priority_rules import precedence_of
from segment_manager import SegmentManager
from slice_mixin import SliceMixin

mp.dps = get_dps()

class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager
        # Set high precision

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

    @staticmethod
    def _convert_and_pack(parts, *, twos_complement=False):
        return convert_and_pack(parts, twos_complement=twos_complement)


class BasicArithmeticEngine(SliceMixin, MathEngine):
    """Handles basic arithmetic: sub, mul, div, pow. Uses built-in math where possible, falls back to Decimal/mpmath."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []  # For step-wise debug info
        self._value = "0"  # Default value for chaining
        # _cache is inherited from MathEngine

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
            # No running event loop
            timestamp = 0

        self.traceback_info.append({
            'step': step,
            'info': info,
            'timestamp': timestamp
        })

    def _validate_number(self, num_str):
        """Validate a number string has at most one decimal point."""
        if num_str.count('.') > 1:
            raise ValueError(f"Invalid number format: {num_str}")

    def call_helper(self, expr):
        """Dynamic helper: Route mixed expressions to appropriate engines and send results to segment_pools."""
        # Determine engine based on keywords
        if 'sin' in expr or 'cos' in expr or 'tan' in expr:
            from trigonometry_engine import TrigonometryEngine
            engine = TrigonometryEngine(self.segment_manager)
        elif 'derivative' in expr or 'integral' in expr:
            from calculus_engine import CalculusEngine
            engine = CalculusEngine(self.segment_manager)
        elif 'j' in expr or ('+' in expr and 'j' in expr):
            from complex_algebra_engine import ComplexAlgebraEngine
            engine = ComplexAlgebraEngine(self.segment_manager)
        else:
            engine = BasicArithmeticEngine(self.segment_manager)

        # Compute result
        result = engine.compute(expr)

        # Pack as mixed result (e.g., 'mixed:result') and send to segment_pools
        mixed_packed = f"mixed:{result}".encode('utf-8')
        self.segment_manager.receive_packed_segment('CallHelper', mixed_packed)

        # Optional: Send part_order for deeper control
        part_order = [{'part': 'mixed_result', 'value': str(result), 'bytes': mixed_packed}]
        self.segment_manager.receive_part_order('CallHelper', f'mixed_{expr}', part_order)

        return result

    @staticmethod
    def _detect_functions(self, tokens):
        """Detect and mark functions in tokens."""
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] in ['sin', 'cos', 'tan', 'log', 'sqrt']:  # Add more as needed
                if i + 2 < len(tokens) and tokens[i + 1] == '(' and tokens[i + 3] == ')':
                    # Function with argument
                    arg = tokens[i + 2]
                    result.append(f"{tokens[i]}({arg})")  # Mark function call
                    i += 4
                else:
                    result.append(tokens[i])
                    i += 1
            else:
                result.append(tokens[i])
                i += 1
        return result

    @staticmethod
    def _is_pure_symbolic_arithmetic(expr: str) -> bool:
        """Return True iff expr is made only of '+-*/^', numbers,
        SymPy named constants, or symbols – i.e. no functions."""
        try:
            tree = sympify(expr, convert_xor=True, evaluate=False)
        except Exception:
            return False
        return all(node.is_Atom or node.is_Pow or node.is_Add or node.is_Mul
                   for node in tree.args + (tree,))

    def compute(self, expr: str):
        """Parse and compute arithmetic expressions with full PEMDAS support.

        Supports: +, -, *, / with proper order of operations.
        Uses mpmath for high precision computation, avoiding floats and Decimals as per guidelines.
        Integrates with parallel flow and segment manager.
        """
        self._add_traceback('compute_start', expr)

        # Parse the expression
        expr = expr.strip()
        if self._is_pure_symbolic_arithmetic(expr):
            sym_res = sympify(expr).evalf(mp.dps)
            self._add_traceback('sym_eval', f'SymPy direct: {sym_res}')
            packed = self._convert_and_pack([sym_res])
            self._cache.append(packed)
            self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
            return str(sym_res)

        # Evaluate with PEMDAS
        result = self._compute_slice_parallel(expr)
        packed = self._convert_and_pack([result])
        self._cache.append(packed)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)

        # Pack result as bytes and accumulate in cache
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)

        # Send packed bytes to segment manager
        self.segment_manager.receive_packed_segment(
            self.__class__.__name__,
            packed_bytes
        )

        # Send to segment manager (original behavior for compatibility)
        part_order = [{'part': 'result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'compute_{expr.replace(" ", "")}',
            part_order
        )
        return str(result)

    def __add__(self, other):
        """Overload __add__: Send part orders to segment manager per-slice."""
        # Stub: Generate part orders based on engine logic
        part_order = [{'part': 'example', 'bytes': b'data'}]  # Mock
        slice_data = 'slice_1'  # From expression parsing
        self.segment_manager.receive_part_order(self.__class__.__name__, slice_data, part_order)
        return self  # Or metadata

    @functools.lru_cache(maxsize=128)  # Cache up to 128 recent additions
    def _cached_add(self, self_val_str, other_val_str):
        """Cached addition using mpmath (hashable strings for lru_cache).

        Returns result string; cache diverts repeated ops.
        """
        self_val = mp.mpf(self_val_str)
        other_val = mp.mpf(other_val_str)
        result = self_val + other_val
        return str(result)

    def add_arithmetic(self, other):
        """Dedicated arithmetic addition using mpmath with caching (no pool management).

        Call this for pure arithmetic to leverage lru_cache.
        """
        self._add_traceback('add_arithmetic', f'Adding {self._value} and {other}')

        # Use cached add
        result_str = self._cached_add(self._value, str(other))

        self._add_traceback('result', f'Addition result (cached): {result_str}')

        # Check cache info (optional debug)
        cache_info = self._cached_add.cache_info()
        self._add_traceback('cache_status', f'Hits: {cache_info.hits}, Misses: {cache_info.misses}')

        # Return new engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = result_str
        new_engine.traceback_info = self.traceback_info.copy()
        return new_engine

    def __mul__(self, other):
        """Perform multiplication using mpmath for high precision.

        Uses mpmath to avoid floats and Decimals as per guidelines.
        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__mul__', f'Multiplying {self._value} and {other}')

        # Convert to mpmath
        self_val = mp.mpf(self._value)
        other_val = mp.mpf(str(other))

        self._add_traceback('conversion', f'Converted: self={self_val}, other={other_val}')

        # Perform multiplication using mpmath
        result = self_val * other_val

        self._add_traceback('result', f'Multiplication result: {result}')

        # Pack result as bytes and accumulate in cache
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        
        # Send packed bytes to segment manager
        self.segment_manager.receive_packed_segment(
            self.__class__.__name__,
            packed_bytes
        )

        # Send to segment manager (original behavior for compatibility)
        part_order = [{'part': 'mul_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'mul_op',
            part_order
        )

        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        new_engine._cache = self._cache.copy()  # Propagate cache
        return new_engine

    def __sub__(self, other):
        """Perform subtraction using mpmath for high precision.

        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__sub__', f'Subtracting {other} from {self._value}')

        # Convert to mpmath
        self_val = mp.mpf(self._value)
        other_val = mp.mpf(str(other))

        self._add_traceback('conversion', f'Converted: self={self_val}, other={other_val}')

        # Perform subtraction using mpmath
        result = self_val - other_val

        self._add_traceback('result', f'Subtraction result: {result}')

        # Pack result as bytes and accumulate in cache
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)

        # Send packed bytes to segment manager
        self.segment_manager.receive_packed_segment(
            self.__class__.__name__,
            packed_bytes
        )

        # Send to segment manager
        part_order = [{'part': 'sub_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'sub_op',
            part_order
        )

        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        new_engine._cache = self._cache.copy()  # Propagate cache
        return new_engine

    def __pow__(self, other):
        """Perform exponentiation using mpmath for high precision.

        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__pow__', f'Raising {self._value} to power {other}')

        # Convert to mpmath
        self_val = mp.mpf(self._value)
        other_val = mp.mpf(str(other))

        self._add_traceback('conversion', f'Converted: self={self_val}, other={other_val}')

        # Perform exponentiation using mpmath
        result = mp.power(self_val, other_val)

        self._add_traceback('result', f'Power result: {result}')

        # Pack result as bytes and accumulate in cache
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)

        # Send packed bytes to segment manager
        self.segment_manager.receive_packed_segment(
            self.__class__.__name__,
            packed_bytes
        )

        # Send to segment manager
        part_order = [{'part': 'pow_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'pow_op',
            part_order
        )

        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        new_engine._cache = self._cache.copy()  # Propagate cache
        return new_engine

    def __div__(self, other):
        """Custom division using mpmath for high precision (deprecated but precise).

        Avoids float fallbacks by using mpmath directly.
        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__div__', f'Dividing {self._value} by {other}')

        # Convert to mpmath
        self_val = mp.mpf(self._value)
        other_val = mp.mpf(str(other))

        if other_val == 0:
            raise ValueError("Division by zero")

        # Perform division using mpmath (direct operator for precision)
        result = self_val / other_val

        self._add_traceback('result', f'Division result: {result}')

        # Send to segment manager
        part_order = [{'part': 'div_result', 'value': str(result), 'bytes': str(result).encode('utf-8')}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'div_op',
            part_order
        )

        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        return new_engine

    @staticmethod
    def _spawn_div(numerator, denominator):
        """Dynamic spawner for division: Generates a fresh division function per request.

        This modularizes division logic, avoiding fixed abstracts and potential float contamination.
        Returns a lambda that performs mpmath division.
        """

        def div_func():
            num_mp = mp.mpf(str(numerator))
            den_mp = mp.mpf(str(denominator))
            if den_mp == 0:
                raise ValueError("Division by zero in spawned div")
            return str(num_mp / den_mp)

        return div_func