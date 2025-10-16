from abc import ABC, abstractmethod
import asyncio
import mpmath as mp
import sympy as sp
from segment_manager import SegmentManager


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
            if isinstance(part, int):
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


class CalculusEngine(MathEngine):
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

    def compute(self, expr):
        """Compute calculus expressions, supporting dual actions like derivative(x**2, x), 2*x."""
        self._add_traceback('compute_start', f'Expression: {expr}')

        # Check for dual action (comma-separated)
        if ',' in expr:
            parts = [p.strip() for p in expr.split(',')]
            if len(parts) == 2:
                left_expr, right_expr = parts
                left_result = self._compute_single(left_expr)
                right_result = self._compute_single(right_expr)
                # For verification: if left is derivative/integral, compare to right
                if 'derivative(' in left_expr or 'integral(' in left_expr:
                    comparison = "Equal" if str(
                        left_result) == right_expr else f"Difference: {left_result} vs {right_expr}"
                    self._add_traceback('dual_comparison',
                                        f'{left_expr} = {left_result}, expected {right_expr} -> {comparison}')
                    result = f"{left_result}, {comparison}"
                else:
                    result = f"{left_result}, {right_result}"
            else:
                raise ValueError("Dual actions support exactly two parts separated by ','")
        else:
            result = self._compute_single(expr)

        self._add_traceback('result', f'Final result: {result}')
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        part_order = [{'part': 'calculus_result', 'value': str(result), 'bytes': packed_bytes}]
        self.segment_manager.receive_part_order(self.__class__.__name__, f'calculus_{expr}', part_order)
        return str(result)

    def _compute_single(self, expr):
        """Helper: Compute a single calculus expression."""
        if 'derivative(' in expr:
            inner = expr.split('derivative(')[1].rstrip(')')
            func_str, var_str = inner.split(', ')
            func = sp.sympify(func_str)
            var = sp.symbols(var_str)
            return sp.diff(func, var)
        elif 'integral(' in expr:
            inner = expr.split('integral(')[1].rstrip(')')
            func_str, var_str = inner.split(', ')
            func = sp.sympify(func_str)
            var = sp.symbols(var_str)
            return sp.integrate(func, var)
        elif 'second_derivative(' in expr:
            inner = expr.split('second_derivative(')[1].rstrip(')')
            func_str, var_str = inner.split(', ')
            func = sp.sympify(func_str)
            var = sp.symbols(var_str)
            return sp.diff(func, var, 2)
        elif 'limit(' in expr:
            inner = expr.split('limit(')[1].rstrip(')')
            func_str, var_str, point_str = inner.split(', ')
            func = sp.sympify(func_str)
            var = sp.symbols(var_str)
            if point_str == 'infinity':
                return sp.limit(func, var, sp.oo)
            elif point_str == '-infinity':
                return sp.limit(func, var, -sp.oo)
            else:
                return sp.limit(func, var, sp.sympify(point_str))
        elif 'definite_integral(' in expr:
            inner = expr.split('definite_integral(')[1].rstrip(')')
            func_str, var_str, a_str, b_str = inner.split(', ')
            func = sp.sympify(func_str)
            var = sp.symbols(var_str)
            a = sp.sympify(a_str)
            b = sp.sympify(b_str)
            return sp.integrate(func, (var, a, b))
        elif 'product_rule(' in expr:
            inner = expr.split('product_rule(')[1].rstrip(')')
            u_str, v_str, var_str = inner.split(', ')
            u = sp.sympify(u_str)
            v = sp.sympify(v_str)
            var = sp.symbols(var_str)
            return sp.diff(u, var) * v + u * sp.diff(v, var)
        elif 'quotient_rule(' in expr:
            inner = expr.split('quotient_rule(')[1].rstrip(')')
            u_str, v_str, var_str = inner.split(', ')
            u = sp.sympify(u_str)
            v = sp.sympify(v_str)
            var = sp.symbols(var_str)
            numerator = sp.diff(u, var) * v - u * sp.diff(v, var)
            return numerator / (v ** 2)
        elif 'chain_rule(' in expr:
            inner = expr.split('chain_rule(')[1].rstrip(')')
            outer_str, inner_str, var_str = inner.split(', ')
            outer = sp.sympify(outer_str)
            inner_func = sp.sympify(inner_str)
            var = sp.symbols(var_str)
            u = sp.symbols('u')
            return sp.diff(outer.subs(u, inner_func), var).subs(u, inner_func)
        elif 'integration_by_parts(' in expr:
            inner = expr.split('integration_by_parts(')[1].rstrip(')')
            u_str, dv_str, var_str = inner.split(', ')
            u = sp.sympify(u_str)
            dv = sp.sympify(dv_str)
            var = sp.symbols(var_str)
            v = sp.integrate(dv, var)
            return u * v - sp.integrate(sp.diff(u, var) * v, var)
        elif 'u_substitution(' in expr:
            inner = expr.split('u_substitution(')[1].rstrip(')')
            expr_str, u_str, du_str, var_str = inner.split(', ')
            expr_func = sp.sympify(expr_str)
            u = sp.sympify(u_str)
            du = sp.sympify(du_str)
            var = sp.symbols(var_str)
            return sp.integrate(expr_func.subs(var, u), du)
        else:
            raise ValueError(f"Unsupported calculus expr: {expr}")

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

    def derivative(self, f, var):
        """Compute derivative using SymPy."""
        self._add_traceback('derivative', f'd/d{var} {f}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        result = sp.diff(func, v)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def integral(self, f, var):
        """Compute integral using SymPy."""
        self._add_traceback('integral', f'∫ {f} d{var}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        result = sp.integrate(func, v)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def limit(self, expr, var, point):
        """Compute limit of expr as var approaches point (supports infinity)."""
        self._add_traceback('limit', f'lim_{var}->{point} {expr}')
        func = sp.sympify(expr)
        v = sp.symbols(var)
        if point == 'infinity':
            result = sp.limit(func, v, sp.oo)
        elif point == '-infinity':
            result = sp.limit(func, v, -sp.oo)
        else:
            result = sp.limit(func, v, sp.sympify(point))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def second_derivative(self, f, var):
        """Compute second derivative (curvature/concavity)."""
        self._add_traceback('second_derivative', f'd²/d{var}² {f}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        result = sp.diff(func, v, 2)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def critical_points(self, f, var):
        """Find critical points (where f'(x)=0 or undefined)."""
        self._add_traceback('critical_points', f'Critical points of {f} w.r.t {var}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        deriv = sp.diff(func, v)
        solutions = sp.solve(deriv, v)
        packed_bytes = str(solutions).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return solutions

    def definite_integral(self, f, var, a, b):
        """Compute definite integral from a to b."""
        self._add_traceback('definite_integral', f'∫_{a}^{b} {f} d{var}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        result = sp.integrate(func, (v, a, b))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def u_substitution(self, expr, u_expr, du_expr, var):
        """Perform u-substitution for integration."""
        self._add_traceback('u_substitution', f'u-sub: {expr}, u={u_expr}, du={du_expr}')
        # Simplified: Assume expr can be rewritten
        func = sp.sympify(expr)
        u = sp.sympify(u_expr)
        du = sp.sympify(du_expr)
        v = sp.symbols(var)
        # Basic substitution (expand for full logic)
        result = sp.integrate(func.subs(v, u), du)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def chain_rule(self, outer, inner, var):
        """Apply chain rule for differentiation."""
        self._add_traceback('chain_rule', f'Chain: {outer}({inner}) w.r.t {var}')
        outer_func = sp.sympify(outer)
        inner_func = sp.sympify(inner)
        v = sp.symbols(var)
        u = sp.symbols('u')
        result = sp.diff(outer_func.subs(u, inner_func), v).subs(u, inner_func)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def product_rule(self, u, v, var):
        """Apply product rule for differentiation."""
        self._add_traceback('product_rule', f'Product: ({u})({v}) w.r.t {var}')
        u_func = sp.sympify(u)
        v_func = sp.sympify(v)
        w = sp.symbols(var)
        result = sp.diff(u_func, w) * v_func + u_func * sp.diff(v_func, w)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def quotient_rule(self, u, v, var):
        """Apply quotient rule for differentiation."""
        self._add_traceback('quotient_rule', f'Quotient: ({u})/({v}) w.r.t {var}')
        u_func = sp.sympify(u)
        v_func = sp.sympify(v)
        w = sp.symbols(var)
        numerator = sp.diff(u_func, w) * v_func - u_func * sp.diff(v_func, w)
        denominator = v_func ** 2
        result = numerator / denominator
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def integration_by_parts(self, u, dv, var):
        """Apply integration by parts."""
        self._add_traceback('integration_by_parts', f'IBP: u={u}, dv={dv} w.r.t {var}')
        u_func = sp.sympify(u)
        dv_func = sp.sympify(dv)
        w = sp.symbols(var)
        v = sp.integrate(dv_func, w)
        result = u_func * v - sp.integrate(sp.diff(u_func, w) * v, w)
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

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