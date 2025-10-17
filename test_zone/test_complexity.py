"""
Simple cross-engine complexity smoke–tests for Cortex.

To run:
    $ python -m unittest test_complexity.py
(or “pytest” will also detect them).

The tests instantiate *one* SegmentManager/Structure pair and go
through BasicArithmeticEngine.call_helper(), which routes every
expression to the appropriate specialised engine (trig / calculus /
complex).  We compare the returned string with a high-precision mpmath /
SymPy evaluation.

If any assertion fails you will see which engine / expression broke.
"""

import unittest
import math
from mpmath import mp
import sympy as sp

# Cortex imports – adjust relative path if you vendored differently.
from segment_manager import SegmentManager
from main import Structure
from abc_engines import BasicArithmeticEngine


mp.dps = 50  # generous precision for the numeric comparisons


class CrossEngineComplexityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        struct = Structure()
        cls.seg_mgr = SegmentManager(struct)
        cls.engine  = BasicArithmeticEngine(cls.seg_mgr)

    # helper: compare numeric strings with 1 ulp tolerance at current dps
    def _assertAlmostEqualMP(self, result_str, expected):
        res = mp.mpf(result_str)
        exp = mp.mpf(str(expected))
        self.assertAlmostEqual(res, exp, delta=mp.mpf(10) ** (-mp.dps + 2))

    # 1.  trig-only, repeated value, mixed “+”
    def test_double_sin(self):
        expr     = "sin(pi/6) + sin(pi/6)"
        expected = mp.sin(mp.pi/6) + mp.sin(mp.pi/6)   # exactly 1
        res_str  = self.engine.call_helper(expr)
        self._assertAlmostEqualMP(res_str, expected)

    # 2.  trig * calculus mix
    def test_trig_times_derivative(self):
        expr = "cos(pi/3)*2 + derivative(x**2, x).subs(x, 3)"
        # cos 60° = 0.5; derivative 2x at 3 = 6
        expected = mp.cos(mp.pi/3) * 2 + 6
        res_str  = self.engine.call_helper(expr)
        self._assertAlmostEqualMP(res_str, expected)

    # 3.  pure complex algebra via ComplexAlgebraEngine
    def test_complex_product(self):
        expr     = "(1+2j)*(3+4j)"
        expected = complex(1+2j) * complex(3+4j)   # ⇒ (-5+10j)
        res_str  = self.engine.call_helper(expr)
        # Basic string equality works for Python complex repr
        self.assertEqual(res_str.replace(' ', ''), str(expected))

    # 4.  everything at once: trig + calc + complex addition
    def test_grand_mix(self):
        expr = "sin(pi/4) + derivative(x**3, x).subs(x, 2) + (1+0j)"
        expected = (
            mp.sin(mp.pi/4) +
            3 * (2**2) +        # derivative 3x^2 at x=2
            (1+0j)
        )
        res_str = self.engine.call_helper(expr)
        # Split possible complex result “a+bj”
        real_part, imag_part = map(str.strip, res_str.replace('j','').split('+'))
        self._assertAlmostEqualMP(real_part, expected.real)
        self._assertAlmostEqualMP(imag_part, expected.imag)


if __name__ == "__main__":
    unittest.main()