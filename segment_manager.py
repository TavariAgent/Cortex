class SegmentManager:
    """Manages segments with parallel part assembly, left-to-right on same level, and structure-defined ordering."""

    def __init__(self, structure_instance):
        self.structure = structure_instance
        self.part_orders = {}  # Dict for parallel parts: engine -> [orders]
        self.finalized_segments = []
        self.primary_level_locked = True

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

        for engine, orders in self.part_orders.items():
            for order in orders:
                slice_id = order['slice']
                level = order['level']
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append((slice_id, order['parts']))

        # For each level, sort same-level parts left-to-right
        for level in sorted(level_groups.keys()):
            level_groups[level].sort(key=lambda x: x[0])  # Left-to-right by slice_id
            for slice_id, parts in level_groups[level]:
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