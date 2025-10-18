from mpmath import mp
from slice_mixin import SliceMixin
from packing import convert_and_pack
from utils.precision_manager import get_dps
from abc import ABC, abstractmethod

mp.dps = get_dps()

class MathEngine(ABC):
    def __init__(self, seg_mgr):
        self.segment_manager = seg_mgr
        self._cache = []

    @abstractmethod
    def compute(self, expr):    ...

class VectorEngine(SliceMixin, MathEngine):
    def __init__(self, seg_mgr):
        super().__init__(seg_mgr)
        self.trace = []

    def _add_tb(self, step, info):
        self.trace.append((step, info))

    # atom evaluator isn’t used; everything comes through helpers

    # ----------------------------------------------------------
    def fsum(self, iterable):
        self._add_tb('fsum', f'{iterable}')
        vals = [mp.mpf(str(x)) for x in iterable]
        res  = mp.fsum(vals)
        self._cache_send(res, 'fsum')
        return res

    def fdot(self, xs, ys):
        self._add_tb('fdot', f'{xs} · {ys}')
        x = [mp.mpf(str(a)) for a in xs]
        y = [mp.mpf(str(b)) for b in ys]
        res = mp.fdot(x, y)
        self._cache_send(res, 'fdot')
        return res

    # ----------------------------------------------------------
    def _cache_send(self, res, tag):
        packed = convert_and_pack([res])
        self._cache.append(packed)
        self.segment_manager.receive_packed_segment(self.__class__.__name__, packed)
        self.segment_manager.receive_part_order(
            self.__class__.__name__, tag,
            [{'part': tag, 'value': str(res), 'bytes': packed}]
        )

    # keep compute() stub for compatibility
    def compute(self, expr):
        raise NotImplementedError("Use fsum()/fdot() directly")