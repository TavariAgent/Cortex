import asyncio
from collections import defaultdict

from abc_engines import BasicArithmeticEngine
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
            'packing': False,
            'engines_done': False, #Flag for engine completion
        }

    async def _inform_structure(self):
        # Simulate async flag updates
        await asyncio.sleep(0.01)
        expression = "2 + 3 * 4"
        slices = expression.split()
        slice_map = {i: slice for i, slice in enumerate(slices)}
        parts_per_slice = {slice: len(slice) for slice in slices}
        # Update flags
        if '+' in expression:
            self.flags['addition'] = True
        if '*' in expression:
            self.flags['multiplication'] = True
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
        # Set engines_done flag
        # self.structure.flags['engines_done'] = True  # Link to Structure

class XorStringCompiler:
    """Class for compiling XOR strings and their XOR operations as well as integer packing"""

    def __init__(self):
        self.byte_strings = []
        self.packed_integers = []
        self.segment_pools = defaultdict(dict)  # XOR-created dict pools: engine_name -> {segment_id: byte_segment}
        self.flag_pool = {}  # Pool for flags

    def __str__(self):
        """Formalize objects to byte strings. Dual role: initial conversion after expression detection, final return."""
        formalized = []
        for obj in self.byte_strings:
            if isinstance(obj, str):
                formalized.append(obj.encode('utf-8'))
            elif isinstance(obj, int):
                formalized.append(str(obj).encode('utf-8'))
            else:
                formalized.append(bytes(obj))
        xor_result = self._compute_xor(formalized)
        packed = self._pack_integers(xor_result)
        return packed.decode('utf-8', errors='ignore')

    def __xor__(self, other):
        """Create XOR dict pools for engines and flags."""
        xor_dirs = [{'op': 'xor', 'targets': self.byte_strings}]
        math_engines = {'add': lambda x, y: x + y, 'mul': lambda x, y: x * y}
        self.flag_pool = {'xor_active': True, 'engines_done': False}
        # Create pools for each engine (stub: assume engine names)
        for engine_name in ['basic', 'trig', 'complex']:
            self.segment_pools[engine_name] = {}
        return xor_dirs

    def __add__(self, other):
        """Overload __add__ with mid-step: Add segment byte strings to dict pools, update flags."""
        # Get segments from other (e.g., from engines)
        segments = getattr(other, 'segments', [])
        for seg in segments:
            engine_name = seg.get('engine', 'basic')  # Assume segment has engine metadata
            seg_id = seg.get('id', 'default')
            byte_seg = seg.get('bytes', b'')
            # Add to pool dict
            self.segment_pools[engine_name][seg_id] = byte_seg
            # Update flags async (simulate)
            asyncio.create_task(self._update_flags_async())
        # Return self for chaining or metadata
        return {'segments_added': len(segments), 'pools': dict(self.segment_pools)}

    async def _update_flags_async(self):
        """Async update flags during __add__."""
        await asyncio.sleep(0.01)
        self.flag_pool['addition'] = True  # Example

    async def _compute_xor(self, byte_strings=None):
        if byte_strings is None:
            byte_strings = self.byte_strings
        result = []
        for bs in byte_strings:
            xor_bs = bytes(b ^ 0xAA for b in bs)
            result.append(xor_bs)
        return result

    def _pack_integers(self, results):
        packed = bytearray()
        for result in results:
            try:
                num = int(result.decode('utf-8'))
                packed.extend(num.to_bytes((num.bit_length() + 7) // 8, 'big'))
            except ValueError:
                packed.extend(result)
        self.packed_integers.append(packed)
        return packed

class Compute:
    """Class for executing the final byte string compilation and execution."""

    def __init__(self, compiler, manager):
        self.compiler = compiler
        self.manager = manager

    async def _compute_expression(self):
        """Enabler finalization: Trigger when engines done, enable __str__ for formalization."""
        # Wait for engines_done flag
        while not self.compiler.flag_pool.get('engines_done', False):
            await asyncio.sleep(0.1)
        # Trigger full expression to __str__ again
        formalized = str(self.compiler)  # Formalize pools into object
        # Compile segments into result
        compiled = self._compile_expression()
        return compiled

    def _compile_expression(self):
        """Compile segments from pools into final result."""
        final_bytes = b''
        for pool in self.segment_pools.values():
            for seg in pool.values():
                final_bytes += seg
        try:
            final_int = int.from_bytes(final_bytes, 'big')
            return str(final_int)
        except:
            return final_bytes.decode('utf-8', errors='ignore')

class Diagnostic:
    """Class for reporting diagnostics and errors."""

    def __init__(self):
        self.errors = []

    def log_error(self, msg):
        self.errors.append(msg)
        print(f"Error: {msg}")

if __name__ == '__main__':
    # Example setup
    tracer = StageTracer()
    struct = Structure()
    compiler = XorStringCompiler()
    manager = ThreadedEngineManager()
    compute = Compute(compiler, manager)
    diag = Diagnostic()

    # Add engines to manager
    manager.add_engine('basic', BasicArithmeticEngine)
    manager.add_engine('trig', TrigonometryEngine)