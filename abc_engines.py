import functools
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
        self._cache = []  # Cache for packed bytes before sending to segment_manager

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


class BasicArithmeticEngine(MathEngine):
    """Handles basic arithmetic: sub, mul, div, pow. Uses built-in math where possible, falls back to Decimal/mpmath."""

    def __init__(self, segment_manager):
        super().__init__(segment_manager)
        self.traceback_info = []  # For step-wise debug info
        self._value = "0"  # Default value for chaining
        # _cache is inherited from MathEngine

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

    def _tokenize(self, expr):
        """Tokenize into slices and functions, grouping numbers/operators per slice."""
        tokens = []
        i = 0

        while i < len(expr):
            char = expr[i]

            if char.isalpha():
                # Function detection (same as before)
                func_start = i
                while i < len(expr) and expr[i].isalpha():
                    i += 1
                func_name = expr[func_start:i]

                if i < len(expr) and expr[i] == '(':
                    paren_count = 1
                    arg_start = i + 1
                    i += 1
                    while i < len(expr) and paren_count > 0:
                        if expr[i] == '(':
                            paren_count += 1
                        elif expr[i] == ')':
                            paren_count -= 1
                        i += 1
                    func_token = expr[func_start:i]
                    tokens.append(func_token)
                else:
                    tokens.append(func_name)
            elif char == '(':
                # Slice: tokenize entire nest as one token (deepest first)
                slice_token = self._tokenize_slice(expr, i)
                tokens.append(slice_token)  # e.g., "(3*4-5)"
                i += len(slice_token)
            elif char.isdigit() or char == '.' or char in '+-*/':
                # For top-level without parens: group as a slice token
                slice_start = i
                while i < len(expr) and (expr[i].isdigit() or expr[i] in '+-*/.' or expr[i] == ' '):
                    i += 1
                slice_content = expr[slice_start:i].replace(' ', '')  # Remove spaces
                tokens.append(f"({slice_content})")  # Wrap as slice for consistency
            elif char == ' ':
                i += 1
            else:
                raise ValueError(f"Invalid character: {char}")

        # Treat the whole expression as one slice to avoid splitting on trailing operators
        tokens = [f"({expr.replace(' ', '')})"]
        return tokens

    def _tokenize_slice(self, expr, start_idx):
        """Tokenize a paren slice as one token, recursing for nests."""
        paren_count = 1
        i = start_idx + 1
        content = ""

        while i < len(expr) and paren_count > 0:
            if expr[i] == '(':
                # Nested slice
                nested = self._tokenize_slice(expr, i)
                content += nested
                i += len(nested)
            elif expr[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    break
            else:
                content += expr[i]
                i += 1

        return f"({content})"

    def _evaluate_tokens(self, tokens):
        """Process tokens, evaluating slices and functions recursively."""
        processed = []

        for token in tokens:
            if token.startswith('(') and token.endswith(')'):
                # Slice: strip parens and compute inner arithmetic
                inner_expr = token[1:-1]
                if inner_expr:  # Not empty
                    result = self._evaluate_slice(inner_expr)
                    processed.append(str(result))
                else:
                    processed.append(token)
            elif any(func in token for func in ['sin(', 'cos(', 'tan(']):
                # Function: quick unpack
                unpacked = self.quick_unpack_function(token)
                processed.append(unpacked)
            else:
                processed.append(token)

        self._add_traceback('processed_tokens', f'Processed tokens: {processed}')
        return processed

    def _evaluate_slice(self, slice_expr):
        """Evaluate a slice's inner expression with functions, parens, and PEMDAS."""
        # First, handle functions by replacing with computed values
        for func in ['sin', 'cos', 'tan', 'log', 'sqrt']:
            func_call = f"{func}("
            while func_call in slice_expr:
                start = slice_expr.find(func_call)
                paren_count = 1
                j = start + len(func_call)
                while j < len(slice_expr) and paren_count > 0:
                    if slice_expr[j] == '(':
                        paren_count += 1
                    elif slice_expr[j] == ')':
                        paren_count -= 1
                    j += 1
                if paren_count > 0:
                    raise ValueError("Mismatched parens in function")
                func_token = slice_expr[start:j]
                replaced = self.quick_unpack_function(func_token)
                slice_expr = slice_expr[:start] + replaced + slice_expr[j:]

        # Then handle parentheses recursively
        while '(' in slice_expr:
            start = slice_expr.find('(')
            end = start + 1
            count = 1
            while end < len(slice_expr) and count > 0:
                if slice_expr[end] == '(':
                    count += 1
                elif slice_expr[end] == ')':
                    count -= 1
                end += 1
            if count > 0:
                raise ValueError("Mismatched parentheses in slice")
            sub_expr = slice_expr[start + 1:end - 1]
            sub_result = self._evaluate_slice(sub_expr)  # Recursive call
            slice_expr = slice_expr[:start] + str(sub_result) + slice_expr[end:]

        # Now tokenize the flat expression (no parens or functions left)
        sub_tokens = []
        current = ""
        for char in slice_expr:
            if char.isdigit() or char == '.':
                current += char
            elif char in '+-*/':
                if current:
                    sub_tokens.append(current)
                    current = ""
                sub_tokens.append(char)
            elif char != ' ':
                raise ValueError(f"Invalid char in slice: {char}")
        if current:
            sub_tokens.append(current)

        # Apply PEMDAS: first * /, then + -
        tokens = sub_tokens[:]
        i = 0
        while i < len(tokens):
            if i > 0 and tokens[i] in ['*', '/'] and i + 1 < len(tokens):
                left = mp.mpf(tokens[i - 1])
                right = mp.mpf(tokens[i + 1])
                op = tokens[i]
                result = left * right if op == '*' else left / right
                tokens = tokens[:i - 1] + [str(result)] + tokens[i + 2:]
            else:
                i += 1

        i = 0
        while i < len(tokens):
            if i > 0 and tokens[i] in ['+', '-'] and i + 1 < len(tokens):
                left = mp.mpf(tokens[i - 1])
                right = mp.mpf(tokens[i + 1])
                op = tokens[i]
                result = left + right if op == '+' else left - right
                tokens = tokens[:i - 1] + [str(result)] + tokens[i + 2:]
            else:
                i += 1

        if len(tokens) != 1:
            raise ValueError(f"Slice evaluation failed: {tokens}")

        result = mp.mpf(tokens[0])
        return result

    def quick_unpack_function(self, func_token):
        """Quick-unpack function: Calculate inner numerics, fold result.

        E.g., 'sin(3+4)' -> compute inner '3+4' = 7, then sin(7).
        Uses mpmath for precision, delegates to engines if complex.
        Returns folded string result.
        """
        # Parse function name and argument
        if '(' not in func_token or ')' not in func_token:
            raise ValueError(f"Invalid function token: {func_token}")

        func_name = func_token.split('(')[0]
        inner_expr = func_token.split('(')[1].rstrip(')')

        # Compute inner expression (simple for now; delegate to compute if nested)
        if '+' in inner_expr or '-' in inner_expr or '*' in inner_expr or '/' in inner_expr:
            # Use basic compute for inner arithmetic
            inner_result = self.compute(inner_expr)
        else:
            inner_result = inner_expr

        # Apply function using mpmath
        arg = mp.mpf(inner_result)
        if func_name == 'sin':
            result = mp.sin(arg)
        elif func_name == 'cos':
            result = mp.cos(arg)
        elif func_name == 'tan':
            result = mp.tan(arg)
        else:
            raise ValueError(f"Unsupported function: {func_name}")

        self._add_traceback('function_unpack', f'{func_token} -> {func_name}({inner_result}) = {result}')
        return str(result)

    def _process_parentheses(self, tokens):
        """Recursively process parentheses to isolate nests."""
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] == '(':
                paren_count = 1
                j = i + 1
                while j < len(tokens) and paren_count > 0:
                    if tokens[j] == '(':
                        paren_count += 1
                    elif tokens[j] == ')':
                        paren_count -= 1
                    j += 1
                if paren_count > 0:
                    raise ValueError("Mismatched parentheses")
                # Recursively process the subexpression
                subexpr = tokens[i + 1:j - 1]
                processed_sub = self._process_parentheses(subexpr)
                result.append(f"({','.join(processed_sub)})")  # Mark as processed nest
                i = j
            elif tokens[i] == ')':
                raise ValueError("Mismatched parentheses")
            else:
                result.append(tokens[i])
                i += 1
        return result

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

    def compute(self, expr):
        """Parse and compute arithmetic expressions with full PEMDAS support.

        Supports: +, -, *, / with proper order of operations.
        Uses mpmath for high precision computation, avoiding floats and Decimals as per guidelines.
        Integrates with parallel flow and segment manager.
        """
        self._add_traceback('compute_start', f'Expression: {expr}')

        # Set high precision
        mp.dps = 50

        # Parse the expression
        expr = expr.strip()

        # Tokenize the expression
        tokens = self._tokenize(expr)
        self._add_traceback('tokenize', f'Tokens: {tokens}')

        # Evaluate with PEMDAS
        result = self._evaluate_tokens(tokens)
        processed = self._evaluate_tokens(tokens)
        self._add_traceback('computation', f'Final result: {processed}')

        # Strip list if single result
        final_result = processed[0] if isinstance(processed, list) and len(processed) == 1 else processed

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

        return str(final_result)

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
        """Stub implementation for subtraction."""
        self._add_traceback('__sub__', f'Subtracting {other} from {self._value}')
        # TODO: Implement using mpmath
        return self

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

    def __pow__(self, other):
        """Stub implementation for power."""
        self._add_traceback('__pow__', f'Raising {self._value} to power {other}')
        # TODO: Implement using mpmath
        return self

    def _convert_and_pack(self, parts):
        """Override: Use str conversions to avoid floats."""
        packed = super()._convert_and_pack(parts)
        # Additional: Ensure str packing (no Decimal/float here)
        str_parts = [str(p) for p in parts if isinstance(p, int)]
        if str_parts:
            packed.extend(b'STR_PACK:' + ','.join(str_parts).encode('utf-8'))
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
                rad = mp.radians(mp.mpf(str(part).replace('deg', '')))
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