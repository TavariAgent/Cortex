from abc import ABC, abstractmethod
import asyncio
import math
import sympy as sp
import numpy as np
import mpmath as mp
from decimal import Decimal
from segment_manager import SegmentManager

class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager):
        self.segment_manager = segment_manager
        self.parallel_tasks = []

    @abstractmethod
    def compute(self, expr):
        """Parallel compute method: Calculate all available parts simultaneously, respecting primary level."""
        pass

    def _set_part_order(self, parts, apply_at_start=True, apply_after_return=False):
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
        """Stub: Compute a single part, respecting priorities."""
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
    def __truediv__(self, other):
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

    def _convert_and_pack(self, parts):
        """Helper method: Convert multi-part inputs to byte arrays, pre-pack for intra-engine ops, and prepare for __add__ to segment manager.

        This handles conversions during expression stages, producing new values to pre-pack.
        Engines can override for specific logic (e.g., numerical vs. symbolic).
        Returns a pre-packed value ready for XOR sub-directory segment handling.
        """
        # Stub: Default conversion logic
        byte_arrays = []
        for part in parts:
            if isinstance(part, (int, float)):
                # Convert to byte string (big-endian, with liberal allocation)
                byte_str = str(part).encode('utf-8')
                byte_arrays.append(bytearray(byte_str))
            elif isinstance(part, str):
                byte_arrays.append(bytearray(part.encode('utf-8')))
            elif isinstance(part, complex):
                # For complex: pack real and imag separately
                real_bytes = bytearray(str(part.real).encode('utf-8'))
                imag_bytes = bytearray(str(part.imag).encode('utf-8'))
                byte_arrays.extend([real_bytes, imag_bytes])
            else:
                # Fallback: assume bytes or convertible
                byte_arrays.append(bytearray(str(part).encode('utf-8')))

        # Pre-pack: Combine into a single bytearray (simple concatenation; engines can customize)
        packed = bytearray()
        for ba in byte_arrays:
            packed.extend(ba)

        # Return pre-packed value for __add__ to segment manager (via XOR sub-dir)
        return packed


class BasicArithmeticEngine(MathEngine):
    """Handles basic arithmetic: sub, mul, div, pow. Uses built-in math where possible, falls back to Decimal/mpmath."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []  # For step-wise debug info

    def _add_traceback(self, step, info):
        """Add step-wise traceback for debugging."""
        self.traceback_info.append({
            'step': step,
            'info': info,
            'timestamp': asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        })

    def compute(self, expr):
        """Parse and compute simple arithmetic expressions like '2+3' or '3*4'.
        
        Uses mpmath for high precision computation, avoiding floats and Decimals as per guidelines.
        Integrates with parallel flow and segment manager.
        """
        self._add_traceback('compute_start', f'Expression: {expr}')
        
        # Parse the expression
        expr = expr.strip()
        
        # Check for addition
        if '+' in expr:
            parts = expr.split('+')
            self._add_traceback('parse', f'Addition detected: {parts}')
            left = parts[0].strip()
            right = parts[1].strip()
            
            # Convert to mpmath for high precision
            left_mp = mp.mpf(left)
            right_mp = mp.mpf(right)
            
            self._add_traceback('conversion', f'Converted to mpmath: {left_mp}, {right_mp}')
            
            # Perform addition using mpmath
            result = mp.fadd(left_mp, right_mp)
            
            self._add_traceback('computation', f'Result: {result}')
            
            # Send to segment manager
            part_order = [{'part': 'result', 'value': str(result), 'bytes': str(result).encode('utf-8')}]
            self.segment_manager.receive_part_order(
                self.__class__.__name__,
                f'add_{left}_{right}',
                part_order
            )
            
            return result
            
        # Check for multiplication
        elif '*' in expr:
            parts = expr.split('*')
            self._add_traceback('parse', f'Multiplication detected: {parts}')
            left = parts[0].strip()
            right = parts[1].strip()
            
            # Convert to mpmath for high precision
            left_mp = mp.mpf(left)
            right_mp = mp.mpf(right)
            
            self._add_traceback('conversion', f'Converted to mpmath: {left_mp}, {right_mp}')
            
            # Perform multiplication using mpmath
            result = mp.fmul(left_mp, right_mp)
            
            self._add_traceback('computation', f'Result: {result}')
            
            # Send to segment manager
            part_order = [{'part': 'result', 'value': str(result), 'bytes': str(result).encode('utf-8')}]
            self.segment_manager.receive_part_order(
                self.__class__.__name__,
                f'mul_{left}_{right}',
                part_order
            )
            
            return result
        else:
            # Just a number
            self._add_traceback('parse', 'Single number')
            result = mp.mpf(expr)
            self._add_traceback('computation', f'Result: {result}')
            return result

    def __add__(self, other):
        """Perform addition using mpmath for high precision.
        
        Overrides the base class to use mpmath instead of floats/Decimals.
        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__add__', f'Adding {self} and {other}')
        
        # Convert to mpmath
        if hasattr(self, '_value'):
            self_val = mp.mpf(self._value)
        else:
            self_val = mp.mpf(0)
            
        if isinstance(other, (int, float, str)):
            other_val = mp.mpf(other)
        elif hasattr(other, '_value'):
            other_val = mp.mpf(other._value)
        else:
            other_val = mp.mpf(0)
        
        self._add_traceback('conversion', f'Converted: self={self_val}, other={other_val}')
        
        # Perform addition using mpmath
        result = mp.fadd(self_val, other_val)
        
        self._add_traceback('result', f'Addition result: {result}')
        
        # Send to segment manager
        part_order = [{'part': 'add_result', 'value': str(result), 'bytes': str(result).encode('utf-8')}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'add_op',
            part_order
        )
        
        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        return new_engine

    def __mul__(self, other):
        """Perform multiplication using mpmath for high precision.
        
        Uses mpmath to avoid floats and Decimals as per guidelines.
        Integrates with segment manager for parallel flow.
        """
        self._add_traceback('__mul__', f'Multiplying {self} and {other}')
        
        # Convert to mpmath
        if hasattr(self, '_value'):
            self_val = mp.mpf(self._value)
        else:
            self_val = mp.mpf(1)
            
        if isinstance(other, (int, float, str)):
            other_val = mp.mpf(other)
        elif hasattr(other, '_value'):
            other_val = mp.mpf(other._value)
        else:
            other_val = mp.mpf(1)
        
        self._add_traceback('conversion', f'Converted: self={self_val}, other={other_val}')
        
        # Perform multiplication using mpmath
        result = mp.fmul(self_val, other_val)
        
        self._add_traceback('result', f'Multiplication result: {result}')
        
        # Send to segment manager
        part_order = [{'part': 'mul_result', 'value': str(result), 'bytes': str(result).encode('utf-8')}]
        self.segment_manager.receive_part_order(
            self.__class__.__name__,
            f'mul_op',
            part_order
        )
        
        # Return result wrapped in engine instance for chaining
        new_engine = BasicArithmeticEngine(self.segment_manager)
        new_engine._value = str(result)
        new_engine.traceback_info = self.traceback_info.copy()
        return new_engine

    def __sub__(self, other):
        """Stub implementation for subtraction."""
        self._add_traceback('__sub__', f'Subtracting {other} from {self}')
        # TODO: Implement using mpmath
        return self

    def __truediv__(self, other):
        """Stub implementation for division."""
        self._add_traceback('__truediv__', f'Dividing {self} by {other}')
        # TODO: Implement using mpmath
        return self

    def __pow__(self, other):
        """Stub implementation for power."""
        self._add_traceback('__pow__', f'Raising {self} to power {other}')
        # TODO: Implement using mpmath
        return self

    def _convert_and_pack(self, parts):
        """Override: Use Decimal for high precision in conversions."""
        packed = super()._convert_and_pack(parts)
        # Additional: Ensure Decimal packing
        decimal_parts = [Decimal(str(p)) for p in parts if isinstance(p, (int, float))]
        if decimal_parts:
            packed.extend(b'DECIMAL_PACK:' + str(sum(decimal_parts)).encode('utf-8'))
        return packed


class TrigonometryEngine(MathEngine):
    """Handles trig functions: sin, cos, tan, etc. Implements dunders where ops apply."""

    def compute(self, expr):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def sin(self, x):
        pass

    def cos(self, x):
        pass

    def _convert_and_pack(self, parts):
        """Override: Handle angle/radian conversions for trig."""
        packed = super()._convert_and_pack(parts)
        # Additional: Pre-pack trig-specific (e.g., convert to radians)
        for part in parts:
            if 'deg' in str(part):  # Assume degrees marker
                rad = math.radians(float(str(part).replace('deg', '')))
                packed.extend(bytearray(str(rad).encode('utf-8')))
        return packed


class ComplexAlgebraEngine(MathEngine):
    """Handles complex numbers: real + imag*j. Overloads for complex ops."""

    def compute(self, expr):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass

    def _convert_and_pack(self, parts):
        """Override: Pack complex parts efficiently."""
        # Use default but ensure complex handling
        return super()._convert_and_pack(parts)


class CalculusEngine(MathEngine):
    """Handles derivatives, integrals, limits. Heavily relies on SymPy."""

    def compute(self, expr):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def derivative(self, f, var):
        pass

    def integral(self, f, var):
        pass

    def _convert_and_pack(self, parts):
        """Override: Handle symbolic expressions for packing."""
        packed = bytearray()
        for part in parts:
            if isinstance(part, sp.Expr):
                # Convert SymPy expr to string bytes
                packed.extend(bytearray(str(part).encode('utf-8')))
            else:
                packed.extend(super()._convert_and_pack([part]))
        return packed