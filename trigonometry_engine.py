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