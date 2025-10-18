from segment_manager import Structure

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