"""
Cross–engine part inspector & orchestrator.

Usage
-----
    inspector = PartInspector(segment_mgr)
    result = await inspector.evaluate("sin(pi/6) + derivative(x**2, x).subs(x, 3)")
"""

import asyncio
from typing import Any

import sympy as sp
from mpmath import mp

from abc_engines import BasicArithmeticEngine
from elementary_engine import ElementaryEngine
from trigonometry_engine import TrigonometryEngine
from calculus_engine     import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine   # placeholder

_ENGINE_MAP = {
    # trigonometry
    sp.sin:  'trig',
    sp.cos:  'trig',
    sp.tan:  'trig',
    sp.asin: 'trig',
    sp.acos: 'trig',
    sp.atan: 'trig',
    sp.sinh: 'trig',
    sp.cosh: 'trig',
    sp.tanh: 'trig',
    sp.asinh: 'trig',
    sp.acosh: 'trig',
    sp.atanh: 'trig',
    # elementary
    sp.exp:  'elem',
    sp.log:  'elem',
    sp.sqrt: 'elem',
    sp.Abs:   'elem',
    # calculus
    sp.Derivative: 'calc',
    sp.Integral: 'calc',
    sp.Limit: 'calc',
    # extend as needed
}

class PartInspector:
    def __init__(self, segment_mgr):
        self.seg_mgr = segment_mgr
        # create engine singletons
        self._engines = {
            'arith': BasicArithmeticEngine(segment_mgr),
            'trig':  TrigonometryEngine(segment_mgr),
            'calc':  CalculusEngine(segment_mgr),
            'elem':  ElementaryEngine(segment_mgr),
            'complex': ComplexAlgebraEngine(segment_mgr),
        }

    async def evaluate(self, expr_txt: str):
        """
        Master entry: return numeric / mp.mpf / mp.mpc result of expr_txt.
        """
        raw_tree = sp.sympify(expr_txt)
        substituted = await self._resolve_tree(raw_tree)
        # At this point the tree should contain only +−*/^ and literals
        final = self._engines['arith'].compute(str(substituted))
        return final

    # ──────────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────────
    async def _resolve_tree(self, node) -> Any:
        """
        Recursively compute non-arithmetic leaves, substitute numeric
        values, return new SymPy tree.
        """
        # 1. if node maps to a specialist engine → compute directly
        eng_key = _ENGINE_MAP.get(type(node.func) if isinstance(node, sp.Function) else type(node))
        if eng_key == 'trig':
            txt = str(node)
            val_str = self._engines['trig'].compute(txt)
            return mp.mpf(val_str)

        if eng_key == 'calc':
            txt = str(node)
            val_str = self._engines['calc'].compute(txt)
            # If the result is still symbolic, force numeric evaluation
            try:
                return mp.mpf(str(val_str))
            except Exception:
                return sp.sympify(val_str)

        if eng_key == 'elem':
            txt = str(node)
            val_str = self._engines['elem'].compute(txt)
            return mp.mpf(val_str)

        if eng_key == 'complex':
            txt = str(node)
            val_str = self._engines['complex'].compute(txt)
            return mp.mpc(val_str)

        # 2. otherwise recurse on args
        if isinstance(node, sp.Atom):
            return node
        new_args = [await self._resolve_tree(arg) for arg in node.args]
        return node.func(*new_args)


# Async wrapper for REPL convenience
async def compute(expr):
    from segment_manager import SegmentManager
    from main import Structure
    seg_mgr = SegmentManager(Structure())
    ins = PartInspector(seg_mgr)
    return await ins.evaluate(expr)

# quick self-test
if __name__ == "__main__":
    mp.dps = 50
    out = asyncio.run(compute("sin(pi/6) + sin(pi/6)"))
    print(out)      # 1.0
    out2 = asyncio.run(compute("sin(pi/4) + derivative(x**3,x).subs(x,2)"))
    print(out2)     # 0.7071… + 12