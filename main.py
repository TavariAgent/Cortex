import asyncio

from abc_engines import BasicArithmeticEngine
from calculus_engine import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine
from elementary_engine import ElementaryEngine
from engine_worker import EngineWorker
from segment_manager import Structure
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
    manager = ThreadedEngineManager()
    compute = ComputedResultAsync()
    diag = Diagnostic()

    # Add engines to manager
    manager.add_engine('basic', BasicArithmeticEngine)
    manager.add_engine('trig', TrigonometryEngine)
    manager.add_engine('algebra', ComplexAlgebraEngine)
    manager.add_engine('elem', ElementaryEngine)
    manager.add_engine('calc', CalculusEngine)