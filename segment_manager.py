from flag_bus import FlagBus


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

    def _determine_level(self, slice_data):
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
        
        # Priority map for operator precedence (matches MathEngine._set_part_order)
        priority_map = {
            '^': 4,  # Exponentiation highest
            '*': 3, '/': 3,  # Mul/div mid
            '+': 2, '-': 2,  # Add/sub lowest
            'sum': 1, 'limit': 1, 'other': 0  # Sums/limits lower, slice-focused
        }

        for engine, orders in self.part_orders.items():
            for order in orders:
                slice_id = order['slice']
                level = order['level']
                if level not in level_groups:
                    level_groups[level] = []
                # Extract operator from slice_id for priority ordering
                op = None
                for op_char in priority_map.keys():
                    if op_char in slice_id:
                        op = op_char
                        break
                priority = priority_map.get(op, 0)
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
            from main import XorStringCompiler
            compiler = XorStringCompiler()
            result = compiler + type('AssembledData', (), {'segments': list(assembled.values())})()
            self.finalized_segments.append(result)
            return result

    def get_finalized(self):
        """Return finalized segments."""
        return self.finalized_segments

    def all_complete(self):
        """Check if all engines have completed their work."""
        if not self.completion_flags:
            return False
        return all(self.completion_flags.values())