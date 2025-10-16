"""
convert_and_pack(parts, *, twos_complement=False)  → bytes

• Accepts an iterable of “parts” (int | str | bytes | complex | mp.mpf | SymPy).
• Performs *lossless* least-significant-byte concatenation.
• Optionally emits two’s-complement for signed ints when twos_complement=True.
• Always sets FlagBus.set('packing', True) on first invocation per event loop
  tick so Structure / SegmentManager can gate on it.
"""
from __future__ import annotations
from typing import Iterable, Union
import numbers
import mpmath as mp
import sympy as sp
from flag_bus import FlagBus

def _int_to_bytes(n: int, twos_complement: bool) -> bytes:
    if n == 0:
        return b'\x00'
    if twos_complement:
        # find minimal width that fits n in two’s-compl.
        width = (n.bit_length() + 8) // 8
        return n.to_bytes(width, byteorder='little', signed=True)
    return n.to_bytes((n.bit_length() + 7) // 8, 'little', signed=False)

def convert_and_pack(parts: Iterable[object],
                     *,
                     twos_complement: bool = False) -> bytes:
    FlagBus.set('packing', True)          # unified side-channel flag
    packed = bytearray()

    for p in parts:
        # 1.  ints
        if isinstance(p, numbers.Integral) and not isinstance(p, bool):
            packed.extend(_int_to_bytes(int(p), twos_complement))
        # 2.  floats / mp.mpf – serialise via str to avoid binary fp
        elif isinstance(p, (numbers.Real, mp.mpf)):
            packed.extend(str(p).encode())
        # 3.  complex →  "a+bi" canonical ascii
        elif isinstance(p, (complex, mp.mpc)):
            packed.extend(f"{p.real}+{p.imag}j".encode())
        # 4.  SymPy objects
        elif isinstance(p, sp.Basic):
            packed.extend(str(p).encode())
        # 5.  bytes already
        elif isinstance(p, (bytes, bytearray)):
            packed.extend(p)
        # 6.  fallback to str()
        else:
            packed.extend(str(p).encode())

    return bytes(packed)