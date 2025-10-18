import asyncio
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from engine_worker import EngineWorker


class ThreadedEngineManager:
    """Manages threaded execution of engines."""

    def __init__(self):
        self.engine_workers = {}  # Dict of engine_name: worker_instance
        self.tasks = []  # For async tasks

    def add_engine(self, engine_name, engine_class, *args):
        """Add an engine worker."""
        worker = EngineWorker(engine_class, *args)
        self.engine_workers[engine_name] = worker

    async def run_engines(self, tasks):
        """Run engines in threads, await completion."""
        for name, worker in self.engine_workers.items():
            task = asyncio.create_task(worker.run(tasks.get(name, {})))
            self.tasks.append(task)
        await asyncio.gather(*self.tasks)


class ConstantCache:
    """Manages pre-computed high-precision constants."""

    def __init__(self, cache_path: str = "constants_cache.pkl"):
        self.cache_path = Path(cache_path)
        self.constants: Dict[str, str] = {}
        self.precision_map: Dict[str, int] = {}
        self.active = False
        self.constant_cache = {}
        self._load_cache()


    def _load_cache(self):
        """Load pre-computed constants from file."""
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.constants = data.get('constants', {})
                self.precision_map = data.get('precision', {})

    def get_constant(self, name, precision=None):
        """Get constant with support for both 'e' and 'E' aliases."""
        # Normalize E to e for internal storage
        if not self.active:
            return None

        if name == 'E':
            name = 'e'

        key = f"{name}_{precision}"
        if key in self.constants:
            return self.constants[key]

        # Fallback to highest available precision
        available = [k for k in self.constants if k.startswith(f"{name}_")]
        if available:
            return self.constants[max(available, key=lambda x: int(x.split('_')[1]))]
        return None


class XorStringCompiler:
    def __init__(self):
        self.byte_strings: list[bytes] = []
        self.packed_integers: list[bytearray] = []
        self.segment_pools: dict[str, dict[str, bytes]] = defaultdict(dict)
        self.flag_pool: dict[str, bool] = {}
        self.constant_cache = ConstantCache()
        self.detected_constants: set = set()
        self.constant_types: set = set()

    def __xor__(self, other):
        """
        XOR operator as constant injection controller.
        Activates when multiple constant types are detected.
        """
        # Parse expression for constants
        if isinstance(other, str):
            self._detect_constants(other)

        # Check activation conditions
        multiple_constant_types = len(self.constant_types) > 1
        requires_high_precision = any(
            c in self.detected_constants
            for c in ['pi', 'e', 'phi', 'sqrt2', 'ln2']
        )

        if multiple_constant_types or requires_high_precision:
            # Activate constant cache
            self.constant_cache.active = True
            self.flag_pool['constant_injection'] = True
            self.flag_pool['cache_active'] = True

            # Inject constants into expression
            if isinstance(other, str):
                other = self._inject_constants(other)

            return {
                'op': 'xor_constant_injection',
                'constants_detected': list(self.detected_constants),
                'types': list(self.constant_types),
                'cache_active': True,
                'modified_expr': other
            }
        else:
            # Standard XOR behavior
            self.flag_pool.update({'xor_active': True, 'engines_done': False})
            return [{'op': 'xor', 'count': len(self.byte_strings)}]

    def _detect_constants(self, expr: str):
        """Detect mathematical constants in expression."""
        self.detected_constants.clear()
        self.constant_types.clear()

        # Mathematical constants
        math_constants = {
            'pi': 'transcendental',
            'e': 'transcendental',
            'phi': 'algebraic',
            'sqrt2': 'algebraic',
            'ln2': 'logarithmic',
            'euler': 'special'
        }

        for const, const_type in math_constants.items():
            if const in expr.lower():
                self.detected_constants.add(const)
                self.constant_types.add(const_type)

    def _inject_constants(self, expr: str) -> str:
        """Replace constant placeholders with high-precision values."""
        modified = expr

        for const in self.detected_constants:
            # Get high-precision value from cache
            value = self.constant_cache.get_constant(const, precision=10000)
            if value:
                # Replace placeholder with actual value
                modified = modified.replace(const, value)

        return modified

    @staticmethod
    def _pack_with_precision(value: str) -> bytes:
        """Pack high-precision decimal as bytes using 2's complement for negative values."""
        is_negative = value.startswith('-')
        if is_negative:
            value = value[1:]

        # Split into integer and decimal parts
        parts = value.split('.')
        integer_part = int(parts[0]) if parts[0] else 0
        decimal_part = parts[1] if len(parts) > 1 else ''

        # Pack integer part
        int_bytes = integer_part.to_bytes((integer_part.bit_length() + 7) // 8 or 1, 'little')

        # Pack decimal part (store as string for precision)
        dec_bytes = decimal_part.encode('utf-8')

        # Combine with length headers
        result = bytearray()
        result.extend(len(int_bytes).to_bytes(2, 'little'))
        result.extend(int_bytes)
        result.extend(len(dec_bytes).to_bytes(4, 'little'))
        result.extend(dec_bytes)

        # Apply 2's complement if negative
        if is_negative:
            result = bytearray(~b & 0xFF for b in result)
            result[-1] = (result[-1] + 1) & 0xFF

        return bytes(result)
