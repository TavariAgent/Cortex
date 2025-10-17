import asyncio
from typing import List

from mpmath import mp
import sympy as sp

from slice_mixin import SliceMixin
from packing_utils import convert_and_pack
from precision_manager import get_dps
from utils.trace_helpers import add_traceback
from segment_manager import SegmentManager

mp.dps = get_dps()

class ElementaryEngine(SliceMixin):
    """
    Elementary functions: exp, log, log10, log2, log1p, sqrt.
    """

    def __init__(self, segment_mgr: SegmentManager):
        self.segment_manager = segment_mgr
        self.traceback_info: List[dict] = []
        self._cache: List[bytes] = []

    # -------------------------------------------------------------- #
    # Trace helper
    # -------------------------------------------------------------- #
    def _add_traceback(self, step: str, info: str):
        add_traceback(self, step, info)

    # -------------------------------------------------------------- #
    # SliceMixin atom evaluator (needed but rarely used here)
    # -------------------------------------------------------------- #
    def _evaluate_atom(self, slice_text: str):
        # Defer to BasicArithmeticEngine style evaluation if someone
        # sends an arithmetic slice to us.
        from priority_rules import precedence_of
        tokens = self._linear_tokenize(slice_text)
        for op_level in [4, 3, 2]:
            i = 0
            while i < len(tokens):
                if precedence_of(tokens[i]) == op_level:
                    a = mp.mpf(tokens[i - 1]); b = mp.mpf(tokens[i + 1])
                    val = (mp.power if tokens[i] == '^'
                           else a.__mul__ if tokens[i] == '*'
                           else a.__truediv__ if tokens[i] == '/'
                           else a.__add__ if tokens[i] == '+'
                           else a.__sub__)(b)
                    tokens = tokens[:i-1] + [str(val)] + tokens[i+2:]
                else:
                    i += 1
        if len(tokens) != 1:
            raise ValueError(f'Could not resolve slice {tokens}')
        return mp.mpf(tokens[0])

    # Same tokenizer as other engines
    def _linear_tokenize(self, expr: str):
        out, cur = [], ''
        for ch in expr:
            if ch.isdigit() or ch == '.':
                cur += ch
            elif ch in '+-*/^':
                if cur:
                    out.append(cur); cur = ''
                out.append(ch)
            elif ch == ' ':
                continue
            else:
                raise ValueError(f'Unexpected char {ch}')
        if cur:
            out.append(cur)
        return out

    # -------------------------------------------------------------- #
    # Public helpers (direct calls)
    # -------------------------------------------------------------- #
    def _pack_and_send(self, tag: str, result):
        packed = convert_and_pack([result])
        self._cache.append(packed)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
        self._add_traceback(tag, f'Result = {result}')
        return str(result)

    # ---- elementary wrappers ------------------------------------- #
    def exp(self, x):      return self._pack_and_send('exp',   mp.exp  (mp.mpf(str(x))))
    def log(self, x):      return self._pack_and_send('log',   mp.log  (mp.mpf(str(x))))
    def log10(self, x):    return self._pack_and_send('log10', mp.log10(mp.mpf(str(x))))
    def log2(self, x):     return self._pack_and_send('log2',  mp.log2 (mp.mpf(str(x))))
    def log1p(self, x):    return self._pack_and_send('log1p', mp.log1p(mp.mpf(str(x))))
    def sqrt(self, x):     return self._pack_and_send('sqrt',  mp.sqrt (mp.mpf(str(x))))

    # -------------------------------------------------------------- #
    # compute() entry ‑- recognise textual functions
    # -------------------------------------------------------------- #
    def compute(self, expr: str):
        self._add_traceback('compute_start', expr)
        for name, fn in [('log10(', self.log10),
                         ('log2(',  self.log2),
                         ('log1p(', self.log1p),
                         ('log(',   self.log),
                         ('exp(',   self.exp),
                         ('sqrt(',  self.sqrt)]:
            if expr.startswith(name):
                inner = expr[len(name):-1]
                sym  = sp.sympify(inner).evalf(mp.dps)
                return fn(str(sym))
        raise ValueError(f'Unsupported elementary expression: {expr}')