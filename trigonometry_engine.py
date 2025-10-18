from abc import ABC, abstractmethod
import asyncio
from mpmath import mp
import sympy as sp

from packing import convert_and_pack
from utils.precision_manager import get_dps
from slice_mixin import SliceMixin
from xor_string_compiler import XorStringCompiler

mp.dps = get_dps()


class MathEngine(ABC):
    """Abstract base class for all math engines. Enables parallel computation with priority-flow helpers."""

    def __init__(self, segment_manager, enable_injection=None):
        self.segment_manager = segment_manager
        self.parallel_tasks = []
        self._cache = []  # Cache for packed bytes before sending to segment_manager

        # Handle injection setup
        if enable_injection is None:
            enable_injection = get_dps() >= 1000

        self.enable_injection = enable_injection

        if enable_injection:
            self.xor_compiler = XorStringCompiler()
        else:
            self.xor_compiler = None

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

    def __init__(self, segment_manager, enable_injection=None):
        # Initialize MathEngine first
        MathEngine.__init__(self, segment_manager, enable_injection)
        # Initialize SliceMixin (no args)
        SliceMixin.__init__(self)

        self.traceback_info = []
        self._value = "0"

        # Build extended constant table
        self._CONSTANT_TABLE = self._build_constant_table(include_e=True)

    def compute(self, expr):
        """
        Implement abstract compute method from MathEngine.
        Routes trig function calls to appropriate methods.
        """
        self._add_traceback('compute', f'Processing: {expr}')

        # Parse the expression to determine which trig function to call
        expr_str = str(expr).strip()

        # Extract function name and argument
        if '(' in expr_str and ')' in expr_str:
            func_name = expr_str[:expr_str.index('(')].strip()
            arg_str = expr_str[expr_str.index('(') + 1:expr_str.rindex(')')].strip()

            try:
                # Convert argument to mpmath number
                arg = mp.mpf(arg_str) if arg_str else 0

                # Route to appropriate function
                if func_name == 'sin':
                    return self.sin(arg)
                elif func_name == 'cos':
                    return self.cos(arg)
                elif func_name == 'tan':
                    return self.tan(arg)
                elif func_name == 'sinh':
                    return self.sinh(arg)
                elif func_name == 'cosh':
                    return self.cosh(arg)
                elif func_name == 'tanh':
                    return self.tanh(arg)
                elif func_name in ['asin', 'arcsin']:
                    return self.asin(arg)
                elif func_name in ['acos', 'arccos']:
                    return self.acos(arg)
                elif func_name in ['atan', 'arctan']:
                    return self.atan(arg)
                else:
                    raise ValueError(f"Unknown trig function: {func_name}")

            except (ValueError, TypeError) as e:
                self._add_traceback('compute_error', str(e))
                raise

        # If not a function call, try to evaluate as a numeric value
        try:
            result = mp.mpf(str(expr))
            return result
        except:
            raise ValueError(f"Cannot compute expression: {expr}")

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

    def sympy_snapping(self, arg, constant_type='pi', snap=True, inject_precision=None):
        # Normalize constant type
        if constant_type in ['E', 'e']:
            constant_type = 'e'  # Use lowercase internally

        if snap:
            # SymPy snapping mode
            if constant_type == 'pi':
                return self._snap_to_pi_multiple(arg)
            elif constant_type == 'e':
                return self._snap_to_e_multiple(arg)
        else:
            # Constant injection mode
            if inject_precision is None:
                inject_precision = min(mp.dps, 100000)

            if self.xor_compiler:
                const_value = self.xor_compiler.get_constant(constant_type, inject_precision)
                if const_value:
                    return mp.mpf(const_value)

            # Fallback to mpmath constants
            return mp.pi if constant_type == 'pi' else mp.e

    @staticmethod
    def _snap_to_pi_multiple(arg, max_multiple=4):
        """
        Snap to exact multiples of π using SymPy symbolic math.
        """
        tol = mp.mpf(10) ** (-mp.dps + 1)
        unit = mp.pi / 4
        ratio = arg / unit
        nearest = mp.nint(ratio)

        if abs(nearest) <= max_multiple * 4 and mp.fabs(ratio - nearest) < tol:
            # Return symbolic representation for exact evaluation
            return sp.pi * (nearest / 4)
        return arg

    @staticmethod
    def _snap_to_e_multiple(arg, max_power=3):
        """
        Snap to exact powers of e using SymPy symbolic math.
        """
        tol = mp.mpf(10) ** (-mp.dps + 1)

        # Check for powers of e (e^0, e^1, e^2, etc.)
        log_val = mp.log(mp.fabs(arg)) if arg != 0 else float('-inf')
        nearest_power = mp.nint(log_val)

        if abs(nearest_power) <= max_power:
            expected = mp.exp(nearest_power)
            if mp.fabs(arg - expected) < tol * mp.fabs(expected):
                # Return symbolic representation
                return sp.exp(nearest_power)

        # Check for multiples of e
        ratio = arg / mp.e
        nearest_mult = mp.nint(ratio)

        if abs(nearest_mult) <= max_power and mp.fabs(ratio - nearest_mult) < tol:
            return sp.E * nearest_mult

        return arg

    @staticmethod
    def _build_constant_table(include_e=True):
        """
        Build lookup table for both pi and e based exact angles/values.
        """
        table = {}

        # Pi-based angles (existing functionality)
        dens = [1, 2, 3, 4, 5, 6, 8, 10, 12]
        for d in dens:
            for n in range(0, 2 * d + 1):
                r = sp.Rational(n, d)
                key = ('pi', sp.Rational(0, 1) if r == 2 else r)
                if key not in table:
                    angle = sp.pi * r
                    s = sp.simplify(sp.sin(angle))
                    c = sp.simplify(sp.cos(angle))
                    t = sp.simplify(sp.tan(angle))
                    if t is sp.zoo:
                        t = sp.oo
                    table[key] = {'sin': s, 'cos': c, 'tan': t}

        if include_e:
            # E-based values for hyperbolic functions
            for power in range(-3, 4):  # e^-3 to e^3
                key = ('e', power)
                e_val = sp.exp(power)
                sinh_val = (e_val - 1 / e_val) / 2
                cosh_val = (e_val + 1 / e_val) / 2
                tanh_val = sinh_val / cosh_val
                table[key] = {
                    'sinh': sp.simplify(sinh_val),
                    'cosh': sp.simplify(cosh_val),
                    'tanh': sp.simplify(tanh_val)
                }

        return table

    def compute_with_injection(self, expr, use_injection=None):
        """
        Enhanced compute method with optional constant injection.

        Args:
            expr: Expression to compute
            use_injection: If True, use injected constants; if False, use SymPy;
                          if None, auto-detect based on precision
        """
        # Auto-detect injection mode based on precision
        if use_injection is None:
            use_injection = mp.dps >= 1000

        self._add_traceback('compute_start', f'Expression: {expr}, Injection: {use_injection}')

        # Initialize XorStringCompiler if needed and not present
        if use_injection and not hasattr(self, 'xor_compiler'):
            self.xor_compiler = XorStringCompiler()

        # Pre-process expression for constant injection if enabled
        if use_injection and hasattr(self, 'xor_compiler'):
            try:
                xor_result = self.xor_compiler ^ expr
                if xor_result.get('constants_injected'):
                    expr = xor_result['modified_expr']
                    self._add_traceback('constant_injection',
                                        f'Injected: {xor_result.get("detected_constants", [])}')
            except Exception as e:
                self._add_traceback('injection_skip', str(e))

    def sin(self, x, use_sympy=None):
        """
        Compute sin using mpmath or SymPy for exact values.

        Args:
            x: Input value
            use_sympy: If True, try SymPy first; if None, auto-detect
        """
        self._add_traceback('sin', f'sin({x})')

        # Auto-detect if we should try SymPy
        if use_sympy is None:
            use_sympy = mp.dps >= 50  # Use SymPy for high precision

        if use_sympy:
            # Check if x is a special angle
            snapped = self.sympy_snapping(x, constant_type='pi', snap=True)
            if snapped != x:  # Was snapped to a pi multiple
                # Look up exact value from constant table
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'pi' and abs(float(sp.pi * key[1]) - float(x)) < 1e-10:
                        exact_val = values['sin']
                        # Convert SymPy exact to mpmath
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        # Standard mpmath computation
        result = mp.sin(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def cos(self, x, use_sympy=None):
        """Compute cos with SymPy fallback for exact values."""
        self._add_traceback('cos', f'cos({x})')

        if use_sympy is None:
            use_sympy = mp.dps >= 50

        if use_sympy:
            snapped = self.sympy_snapping(x, constant_type='pi', snap=True)
            if snapped != x:
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'pi' and abs(float(sp.pi * key[1]) - float(x)) < 1e-10:
                        exact_val = values['cos']
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        result = mp.cos(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def tan(self, x, use_sympy=None):
        """Compute tan with SymPy fallback and infinity handling."""
        self._add_traceback('tan', f'tan({x})')

        if use_sympy is None:
            use_sympy = mp.dps >= 50

        if use_sympy:
            snapped = self.sympy_snapping(x, constant_type='pi', snap=True)
            if snapped != x:
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'pi' and abs(float(sp.pi * key[1]) - float(x)) < 1e-10:
                        exact_val = values['tan']
                        if exact_val is sp.oo:
                            return mp.inf
                        elif exact_val is -sp.oo:
                            return -mp.inf
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        result = mp.tan(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    # For hyperbolic functions with e-based exact values:

    def sinh(self, x, use_sympy=None):
        """Compute sinh with SymPy fallback for e-based exact values."""
        self._add_traceback('sinh', f'sinh({x})')

        if use_sympy is None:
            use_sympy = mp.dps >= 50

        if use_sympy:
            # Check for e-based special values
            snapped = self.sympy_snapping(x, constant_type='e', snap=True)
            if snapped != x:
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'e' and abs(key[1] - mp.log(abs(x))) < 1e-10:
                        exact_val = values['sinh']
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        result = mp.sinh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def cosh(self, x, use_sympy=None):
        """Compute cosh with SymPy fallback for e-based exact values."""
        self._add_traceback('cosh', f'cosh({x})')

        if use_sympy is None:
            use_sympy = mp.dps >= 50

        if use_sympy:
            snapped = self.sympy_snapping(x, constant_type='e', snap=True)
            if snapped != x:
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'e' and abs(key[1] - mp.log(abs(x))) < 1e-10:
                        exact_val = values['cosh']
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        result = mp.cosh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def tanh(self, x, use_sympy=None):
        """Compute tanh with SymPy fallback for e-based exact values."""
        self._add_traceback('tanh', f'tanh({x})')

        if use_sympy is None:
            use_sympy = mp.dps >= 50

        if use_sympy:
            snapped = self.sympy_snapping(x, constant_type='e', snap=True)
            if snapped != x:
                for key, values in self._CONSTANT_TABLE.items():
                    if key[0] == 'e' and abs(key[1] - mp.log(abs(x))) < 1e-10:
                        exact_val = values['tanh']
                        return mp.mpf(str(exact_val.evalf(mp.dps)))

        result = mp.tanh(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def asin(self, x, use_sympy=None):
        """Compute arcsin."""
        self._add_traceback('asin', f'asin({x})')

        result = mp.asin(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def acos(self, x, use_sympy=None):
        """Compute arccos."""
        self._add_traceback('acos', f'acos({x})')

        result = mp.acos(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result

    def atan(self, x, use_sympy=None):
        """Compute arctan."""
        self._add_traceback('atan', f'atan({x})')

        result = mp.atan(mp.mpf(str(x)))
        packed_bytes = str(result).encode('utf-8')
        self._cache.append(packed_bytes)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed_bytes)
        return result