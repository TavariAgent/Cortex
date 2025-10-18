import asyncio
from collections import defaultdict

from abc_engines import BasicArithmeticEngine
from calculus_engine import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine
from elementary_engine import ElementaryEngine
from flag_bus import FlagBus
from trigonometry_engine import TrigonometryEngine


class StageTracer:
    """Class for tracing the execution of the expression."""

    def __init__(self):
        self.trace_data = []

    def __bool__(self):
        return len(self.trace_data) > 0

    def _check(self):
        if self.trace_data:
            print(f"Trace active with {len(self.trace_data)} events")
        else:
            print("No trace data")


class Structure:
    """Class for determining the structure of the expression and returning a snapshot of all slices."""

    def __init__(self):
        self.flags = {}
        self._initialize_flags()

    def _initialize_flags(self):
        self.flags = {
            'addition': False,
            'multiplication': False,
            'xor_op': True,
            'engines_done': False, #Flag for engine completion
        }
        self.flags.update({
            'packing': FlagBus.get('packing', False),
            'engine_done': FlagBus.get('engine_done', False)
        })

    # Async to update structure flags
    async def _inform_structure(self, expr):
        slices = expr.split()
        slice_map = {i: slice for i, slice in enumerate(slices)}
        parts_per_slice = {slice: len(slice) for slice in slices}
        # Arithmetic
        if '+' in expr:
            self.flags['addition'] = True
        if '*' in expr:
            self.flags['multiplication'] = True
        if '-' in expr:
            self.flags['subtraction'] = True
        if '/' in expr:
            self.flags['division'] = True
        if '^' in expr:
            self.flags['exponentiation'] = True
        # Calculus
        if 'derivative(' in expr:
            self.flags['derivative'] = True
        if 'diff(' in expr:
            self.flags['derivative'] = True
        # Elementary
        if 'abs(' in expr:
            self.flags['absolute'] = True
        if 'log(' in expr:
            self.flags['logarithm'] = True
        if 'ln(' in expr:
            self.flags['natural_log'] = True
        if 'log10(' in expr:
            self.flags['logarithm_10'] = True
        if 'log2(' in expr:
            self.flags['logarithm_2'] = True
        if 'sqrt(' in expr:
            self.flags['square_root'] = True
        if 'exp(' in expr:
            self.flags['exponential'] = True
        # Trigonometry
        if 'sin(' in expr:
            self.flags['sine'] = True
        if 'cos(' in expr:
            self.flags['cosine'] = True
        if 'tan(' in expr:
            self.flags['tangent'] = True
        if 'asin(' in expr:
            self.flags['arc_sine'] = True
        if 'acos(' in expr:
            self.flags['arc_cosine'] = True
        if 'atan(' in expr:
            self.flags['arc_tangent'] = True

        snapshot = {
            'interval_slice_map': slice_map,
            'parts_per_slice': parts_per_slice,
            'flags': self.flags.copy()
        }
        return snapshot


class EngineWorker:
    """Worker class for running engines in async context."""
    
    def __init__(self, engine_class, *args):
        self.engine_class = engine_class
        self.args = args
        self.engine_instance = None
    
    async def run(self, task_config):
        """Run the engine with given task configuration."""
        if self.engine_instance is None:
            if self.args:
                self.engine_instance = self.engine_class(*self.args)
            else:
                from segment_manager import SegmentManager
                struct = Structure()
                seg_mgr = SegmentManager(struct)
                self.engine_instance = self.engine_class(seg_mgr)
        
        # Run the task
        expr = task_config.get('expr', '')
        if expr:
            result = self.engine_instance.compute(expr)
            return result
        return None


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


class XorStringCompiler:
    def __init__(self):
        self.byte_strings: list[bytes] = []
        self.packed_integers: list[bytearray] = []
        self.segment_pools: dict[str, dict[str, bytes]] = defaultdict(dict)
        self.flag_pool: dict[str, bool] = {}

    def finalize_pack(self) -> bytes:
        """Pack byte_strings into one final artifact (example: concatenate)."""
        return b"".join(self.byte_strings)

    def __str__(self) -> str:
        # Return a textual summary
        return f"XorStringCompiler(byte_strings={len(self.byte_strings)}, segments={sum(len(p) for p in self.segment_pools.values())})"

    def __xor__(self, other):
        # Establish XOR operation metadata; example placeholder
        self.flag_pool.update({'xor_active': True, 'engines_done': False})
        return [{'op': 'xor', 'count': len(self.byte_strings)}]

    def __add__(self, other):
        # Expect other to expose segments
        segments = getattr(other, 'segments', [])
        for seg in segments:
            engine_name = seg.get('engine', 'basic')  # single default only
            seg_id = seg.get('id')
            byte_seg = seg.get('bytes', b'')
            if seg_id is None:
                continue
            self.segment_pools[engine_name][seg_id] = byte_seg
        return {'segments_added': len(segments)}

    def _pack_integers(self, results: list[bytes]) -> bytearray:
        packed = bytearray()
        for result in results:
            try:
                num = int(result.decode('utf-8'))
                packed.extend(num.to_bytes((num.bit_length() + 7) // 8 or 1, 'little'))
            except (ValueError, AttributeError):
                packed.extend(result if isinstance(result, (bytes, bytearray)) else bytes(result))
        self.packed_integers.append(packed)
        return packed


class ComputedResultAsync:
    def __init__(self, compiler, engines_done_event: asyncio.Event):
        self.compiler = compiler
        self._engines_done = engines_done_event

    async def wait_and_compile(self, expr: str = '') -> str:
        # await without polling/sleeping
        await self._engines_done.wait()
        result = self.compile_results(expr)
        return result or expr

    def compile_results(self, expr: str = '') -> str | None:
        # same logic as the sync version above
        if not self.compiler.flag_pool.get('engine_done', False):
            return None
        assembled_any = False
        for engine_name, pool in self.compiler.segment_pools.items():
            for seg_id, byte_seg in pool.items():
                self.compiler.byte_strings.append(
                    byte_seg if isinstance(byte_seg, (bytes, bytearray)) else str(byte_seg).encode('utf-8')
                )
                assembled_any = True
        return expr if assembled_any else None


class Diagnostic:
    """Class for reporting diagnostics and errors."""

    def __init__(self):
        self.errors = []

    def log_error(self, msg):
        self.errors.append(msg)
        print(f"Error: {msg}")


if __name__ == '__main__':
    struct = Structure()
    compiler = XorStringCompiler()
    manager = ThreadedEngineManager()
    compute = ComputedResultAsync()
    diag = Diagnostic()

    # Add engines to manager
    manager.add_engine('basic', BasicArithmeticEngine)
    manager.add_engine('trig', TrigonometryEngine)
    manager.add_engine('algebra', ComplexAlgebraEngine)
    manager.add_engine('elem', ElementaryEngine)
    manager.add_engine('calc', CalculusEngine)