from flag_bus import FlagBus
from priority_rules import precedence_of, PRIORITY


class SegmentManager:
    """Manages segments with parallel part assembly, left-to-right on same level, and structure-defined ordering."""

    def __init__(self, structure_instance):
        self.structure = structure_instance
        self.part_orders = {}  # Dict for parallel parts: engine -> [orders]
        self.finalized_segments = []
        self.primary_level_locked = True
        self.segment_pools = {}  # Dict: engine_name -> list of packed byte arrays
        self.completion_flags = {}  # Dict: engine_name -> bool

    def receive_part_order(self, engine_name, slice_data, part_order):
        if engine_name not in self.part_orders:
            self.part_orders[engine_name] = []
        self.part_orders[engine_name].append({
            'slice': slice_data,
            'parts': part_order,
            'level': self._determine_level(slice_data)  # Add level info
        })
        if not self.primary_level_locked:
            # For async context, we should schedule this
            pass

    def receive_packed_segment(self, engine_name, packed_bytes):
        """Accumulate packed byte arrays from engines."""
        if engine_name not in self.segment_pools:
            FlagBus.set('last_pack_from', engine_name)
            self.segment_pools[engine_name] = []
        self.segment_pools[engine_name].append(packed_bytes)

    @staticmethod
    def _determine_level(slice_data):
        """Stub: Determine structural level of slice (e.g., from expression nesting)."""
        # Example: Parse slice_data for nesting depth
        return len(slice_data.split('_'))  # Simple proxy for level

    def unlock_primary_level(self):
        self.primary_level_locked = False
        # _dropdown_assemble should be called from async context
        # For now, we'll just mark it as unlocked

    async def _dropdown_assemble(self):
        snapshot = await self.structure._inform_structure()
        slice_map = snapshot['interval_slice_map']
        assembled = {}
        level_groups = {}  # Group by level for left-to-right sorting

        for engine, orders in self.part_orders.items():
            for order in orders:
                slice_id = order['slice']
                level = order['level']
                if level not in level_groups:
                    level_groups[level] = []
                # Extract operator from slice_id for priority ordering
                op = None
                for op_char in PRIORITY.keys():
                    if op_char in slice_id:
                        op = op_char
                        break
                priority = PRIORITY.get(op, 0)
                level_groups[level].append((slice_id, order['parts'], priority))

        # For each level, sort by priority (desc) then left-to-right
        for level in sorted(level_groups.keys()):
            # Sort by priority (descending), then by slice_id (left-to-right)
            level_groups[level].sort(key=lambda x: (-x[2], x[0]))
            for slice_id, parts, priority in level_groups[level]:
                if slice_id in slice_map:
                    location = slice_map[slice_id]
                    assembled[location] = parts

        # Finalize to main's __add__ when all parts are ready
        if assembled:
            # Late import to avoid circular dependency
            from xor_string_compiler import XorStringCompiler
            compiler = XorStringCompiler()
            result = compiler + type('AssembledData', (), {'segments': list(assembled.values())})()
            self.finalized_segments.append(result)
            return result

    async def flag_engine_done(self, engine_name):
        """Flag engine as done."""
        self.completion_flags[engine_name] = True
        if all(self.completion_flags.values()):
            await self._dropdown_assemble()

    def get_finalized(self):
        """Return finalized segments."""
        return self.finalized_segments


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


