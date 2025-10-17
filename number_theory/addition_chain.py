"""
number_theory.addition_chain
============================

Minimal-length addition chains, the Landau function *l(n)*, and
the inequality

    l(2·n − 1) ≤ (n − 1) + l(n)

Each chain is pure integer arithmetic, so unlimited precision is
automatic.  Results are memoised and—if the surrounding Cortex
machinery is present—packed and forwarded to SegmentManager so XOR /
byte-stream consumers can replay them.

Public API
----------

chain(n)   -> list[int]          one optimal chain  (1 … n)
l(n)       -> int                length  (= len(chain(n)) − 1)
verify(k)  -> list[int]          n (2…k) that break the inequality
"""

from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Tuple

# Optional integrations -------------------------------------------------
try:
    from packing_utils import convert_and_pack
except ModuleNotFoundError:          # running stand-alone tests
    convert_and_pack = lambda x, **k: b''

try:
    from segment_manager import SegmentManager
    from main import Structure
except ModuleNotFoundError:
    SegmentManager = None            # type: ignore[assignment]
    Structure = None                 # type: ignore[assignment]

try:
    from flag_bus import FlagBus
except ModuleNotFoundError:
    class _MockFlagBus(dict):        # minimal stub
        def set(self, k, v): self[k] = v
    FlagBus = _MockFlagBus()         # type: ignore[assignment]


# ----------------------------------------------------------------------
#  Core algorithm (Brauer–Scholz reflection, BFS with pruning)
# ----------------------------------------------------------------------

@lru_cache(maxsize=None)
def _minimal_chain(n: int) -> Tuple[int, ...]:
    """
    Return one minimal addition chain for *n*.

    The search is breadth-first; it stops as soon as the first minimal
    chain is found, so run-time stays practical for n < 2·10⁴.
    """
    if n == 1:
        return (1,)

    # Fast path – pure powers of two: 1, 2, 4, …, n
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        return tuple(1 << i for i in range(k + 1))

    best: Tuple[int, ...] | None = None
    frontier: List[Tuple[int, ...]] = [(1,)]

    while frontier:
        new_frontier: List[Tuple[int, ...]] = []
        for chain in frontier:
            highest = chain[-1]
            # Generate candidates in *reverse* to bias BFS toward large jumps
            for a in reversed(chain):
                nxt = highest + a
                if nxt > n:
                    continue
                new_chain = chain + (nxt,)
                if nxt == n:
                    if best is None or len(new_chain) < len(best):
                        best = new_chain
                else:
                    # prune aggressively: keep candidates shorter than current best −1
                    if best is None or len(new_chain) < len(best) - 1:
                        new_frontier.append(new_chain)
        if best is not None:
            break
        frontier = new_frontier

    # mypy: we know best is set
    return best if best is not None else (1, n)  # type: ignore[return-value]


# ----------------------------------------------------------------------
#  Public helpers
# ----------------------------------------------------------------------

def chain(n: int) -> List[int]:
    """Return *one* optimal addition chain for *n* (including 1 and n)."""
    if n < 1:
        raise ValueError("n must be positive")
    seq = list(_minimal_chain(n))

    # Flag & optional byte-packing for Cortex observers
    FlagBus.set('addition_chain_last', n)
    if SegmentManager and Structure:
        seg_mgr = SegmentManager(Structure())
        seg_mgr.receive_packed_segment('addition_chain',
                                       convert_and_pack(seq))

    return seq


def l(n: int) -> int:
    """Length of the shortest addition chain producing *n*."""
    return len(_minimal_chain(n)) - 1


def verify(max_n: int = 1000) -> List[int]:
    """
    Check inequality l(2n−1) ≤ n−1 + l(n) for 2 ≤ n ≤ *max_n*.
    Returns a list of n that violate the claim (expected to be empty).
    """
    bad: List[int] = []
    for n in range(2, max_n + 1):
        if l(2 * n - 1) > n - 1 + l(n):
            bad.append(n)
    return bad


# ----------------------------------------------------------------------
#  Quick CLI test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(" l(1 … 20):", [l(i) for i in range(1, 21)])
    print(" inequality holds up to 2000?", "YES" if not verify(2000) else "NO")