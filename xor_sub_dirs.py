# Example XOR sub-directory for segment management (could live in main.py or a separate xor_dirs.py)
class XorSubDirectory:
    """Manages segments per engine via XOR directories."""

    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.segments = []

    def add_segment(self, packed_value):
        """Add pre-packed value from engine's __add__ (preserves __add__ for XOR flow)."""
        # XOR flag logic here (stub)
        xor_applied = bytes(b ^ 0xFF for b in packed_value)  # Example XOR
        self.segments.append(xor_applied)

    def finalize_segments(self):
        """Finalize for main XOR flow."""
        return b''.join(self.segments)