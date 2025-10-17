import functools
from abc import ABC, abstractmethod
import asyncio
from typing import List

import numpy as np
from mpmath import mp
from decimal import Decimal
from sympy import sympify, nsimplify, srepr
from utils.trace_helpers import add_traceback
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
        # Set high precision

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
    def _normalise_small(x, *, eps=None):
        eps = eps or mp.mpf(10) ** (-mp.dps + 2)  # ~100 ulps
        return mp.mpf(0) if mp.fabs(x) < eps else x

    @staticmethod
    def _convert_and_pack(parts, *, twos_complement=False):
        return convert_and_pack(parts, twos_complement=twos_complement)


class BasicArithmeticEngine(SliceMixin, MathEngine):
    """Handles basic arithmetic: sub, mul, div, pow. Uses built-in math where possible, falls back to Decimal/mpmath."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []  # For step-wise debug info
        self._value = "0"  # Default value for chaining
        self.traceback_info: List[dict] = []
        # _cache is inherited from MathEngine

    def _add_traceback(self, step, info):
        add_traceback(self, step, info)

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

    @staticmethod
    def _validate_number(num_str):
        """Validate a number string has at most one decimal point."""
        if num_str.count('.') > 1:
            raise ValueError(f"Invalid number format: {num_str}")

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
        add_traceback(self, 'compute_start', f'Expr: {expr}', with_stack=False)

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