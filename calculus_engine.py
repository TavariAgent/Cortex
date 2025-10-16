from abc import ABC, abstractmethod
import asyncio
import mpmath as mp
import sympy as sp

from packing_utils import convert_and_pack
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
    def _convert_and_pack(parts, *, twos_complement=False):
        return convert_and_pack(parts, twos_complement=twos_complement)


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
        """Compute calculus expressions."""
        self._add_traceback('compute_start', f'Expression: {expr}')
        # Map lowercase aliases → SymPy classes so ‘derivative(...)’ etc. are recognised.
        expr = sp.sympify(
            expr,
            locals={
                'derivative': sp.Derivative,
                'integral': sp.Integral,
                'limit': sp.Limit,
            },  # type: ignore[arg-type]
        )
        parts = []

        # ------------------------------------------------------------------
        # 1️⃣ Fast path ─── the whole expression *is* a Derivative
        # ------------------------------------------------------------------
        if isinstance(expr, sp.Derivative):
            # Evaluate it directly and return; no extra diff-of-diff.
            result = expr.doit()
            self._add_traceback(
                'compute_derivative_direct',
                f'{expr} -> {result}'
            )
            packed_bytes = str(result).encode('utf-8')
            self._cache.append(packed_bytes)
            self.segment_manager.receive_packed_segment(self.__class__.__name__,
                                                        packed_bytes)
            return result

        # ------------------------------------------------------------------
        # 2️⃣ Mixed expression that *contains* derivatives
        # ------------------------------------------------------------------
        if expr.has(sp.Derivative):
            # Evaluate only the inner Derivative nodes, keep outer structure.
            expr = expr.doit(deep=False)
            self._add_traceback(
                'compute_inner_derivatives',
                f'Inner derivatives evaluated -> {expr}'
            )
        if expr.is_Symbol:
            self._value = str(expr)
            # Calculate integral
            parts = expr.find(lambda x: x.is_Integral)
            self._add_traceback(
                'compute_integral',
                f'Calculus expression: {expr}, Integral: {parts}'
            )
            if parts:
                return self.integral(str(expr), self._value)
                # Calculate limit
            parts = expr.find(lambda x: x.is_Limit)
            self._add_traceback(
                'compute_limit',
                f'Calculus expression: {expr}, Limit: {parts}'
            )
            if parts:
                return self.limit(str(expr), self._value, 'infinity')
                # Calculate Critical Points
            parts = expr.find(lambda x: x.is_Function)
            self._add_traceback(
                'compute_critical_points',
                f'Calculus expression: {expr}, Critical Points: {parts}'
            )
            if parts:
                return self.critical_points(str(expr), self._value)
                # Calculate definite integral
            parts = expr.find(lambda x: x.is_Integral)
            self._add_traceback(
                'compute_definite_integral',
                f'Calculus expression: {expr}, Definite Integral: {parts}'
            )
            if parts:
                return self.definite_integral(str(expr), self._value, 0, 1)
                # Calculate u-substitution
            parts = expr.find(lambda x: x.is_Function)
            self._add_traceback(
                'compute_u_substitution',
                f'Calculus expression: {expr}, u-substitution: {parts}'
            )
            if parts:
                return self.u_substitution(str(expr), 'u', 'du', self._value)
                # Calculate chain rule
            parts = expr.find(lambda x: x.is_Function)
            self._add_traceback(
                'compute_chain_rule',
                f'Calculus expression: {expr}, Chain Rule: {parts}'
            )
            if parts:
                return self.chain_rule(str(expr), 'f', self._value)
                # Calculate product rule
            parts = expr.find(lambda x: x.is_Mul)
            self._add_traceback(
                'compute_product_rule',
                f'Calculus expression: {expr}, Product Rule: {parts}'
            )
            if parts:
                return self.product_rule(str(expr), 'u', self._value)
                # Calculate quotient rule
            parts = expr.find(lambda x: x.is_Mul)
            self._add_traceback(
                'compute_quotient_rule',
                f'Calculus expression: {expr}, Quotient Rule: {parts}'
            )
            if parts:
                return self.quotient_rule(str(expr), 'u', self._value)
                # Calculate integration by parts
            parts = expr.find(lambda x: x.is_Integral)
            self._add_traceback(
                'compute_integration_by_parts',
                f'Calculus expression: {expr}, Integration by Parts: {parts}'
            )
            if parts:
                return self.integration_by_parts(str(expr), 'u', self._value)
                # Calculate Taylor series expansion
            parts = expr.find(lambda x: x.is_Function)
            self._add_traceback(
                'compute_taylor_series_expansion',
                f'Calculus expression: {expr}, Taylor Series Expansion: {parts}'
            )
            if parts:
                return self.taylor_series_expansion(str(expr), 'f', self._value)

        ordered_parts = self._set_part_order(parts)
        results = asyncio.run(self._compute_parts_parallel(ordered_parts))
        ordered = self._set_part_order(results)
        self._add_traceback('compute_end', f'Results: {results}')
        return ordered[0] if len(ordered) == 1 else ordered

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

    def taylor_series_expansion(self, f, var, n):
        """Compute Taylor series expansion of f up to order n."""
        self._add_traceback('taylor_series_expansion', f'Taylor series expansion of {f} up to order {n}')
        func = sp.sympify(f)
        v = sp.symbols(var)
        result = sp.series(func, v, n)