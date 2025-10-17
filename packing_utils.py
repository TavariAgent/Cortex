"""
packing_utils – serialise arbitrary numeric / symbolic objects to bytes.

Rules
-----
1.  Pure integers are packed as big-endian byte strings, optionally in two’s
    complement form (for bit-exact XOR workflows).
2.  Every other object is converted to its *string* representation (ASCII)
    and then encoded with UTF-8.  This avoids binary float formats.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from typing import Any, Union, overload

import sympy as sp
from mpmath import mp, mpf, mpc

from flag_bus import FlagBus


# ────────────────────────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────────────────────────
def _int_to_bytes(n: int, twos: bool) -> bytes:
    """
    Serialise integer *n* as big-endian.
    If *twos* is True, encode negative numbers using two’s complement with
    the minimal number of whole bytes; otherwise use sign-magnitude.
    """
    if n == 0:
        return b'\x00'

    if not twos:
        # sign-magnitude: prepend '-' for negative, ASCII only
        return (str(n)).encode()

    # two’s complement
    if n > 0:
        bitlen = n.bit_length()
        width  = (bitlen + 7) // 8
        return n.to_bytes(width, byteorder='big', signed=False)

    # negative – two’s complement
    bitlen = (-n).bit_length()
    width  = (bitlen + 7) // 8
    # Python's int.to_bytes supports signed=True
    return n.to_bytes(width, byteorder='big', signed=True)


def _as_ascii(x: Any) -> bytes:
    """Return ASCII/UTF-8 bytes for *x* using str()."""
    return str(x).encode()


# ────────────────────────────────────────────────────────────────────
#  public API
# ────────────────────────────────────────────────────────────────────
def convert_and_pack(parts: Iterable[Any], *, twos_complement: bool = False) -> bytes:
    """
    Convert every element of *parts* into a deterministic byte sequence.

    Supported element types
    -----------------------
    int, Decimal, Fraction, mpmath (mpf, mpc), complex, SymPy Basic,
    bytes/bytearray, and anything supporting `str()`.
    """
    FlagBus.set('packing', True)                 # unified side-channel flag
    packed = bytearray()

    for p in parts:
        # 1. exact integers (Python int, SymPy Integer)
        if isinstance(p, numbers.Integral) and not isinstance(p, bool):
            packed.extend(_int_to_bytes(int(p), twos_complement))

        # 2. high-precision rationals
        elif isinstance(p, (Fraction, sp.Rational)) and p.q != 1:
            packed.extend(_as_ascii(p))

        # 3. Decimals, mpf, other real numbers
        elif isinstance(p, (Decimal, mpf, numbers.Real)) and not isinstance(p, bool):
            packed.extend(_as_ascii(p))

        # 4. complex numbers (built-in complex or mpmath mpc)
        elif hasattr(p, "real") and hasattr(p, "imag"):
            packed.extend(f"{p.real}+{p.imag}j".encode())

        # 5. SymPy symbolic objects (pi, sqrt(2), etc.)
        elif isinstance(p, sp.Basic):
            packed.extend(_as_ascii(p))

        # 6. raw bytes already
        elif isinstance(p, (bytes, bytearray)):
            packed.extend(p)

        # 7. None sentinel → empty marker
        elif p is None:
            packed.extend(b'NULL')

        # 8. fallback
        else:
            packed.extend(_as_ascii(p))

    return bytes(packed)