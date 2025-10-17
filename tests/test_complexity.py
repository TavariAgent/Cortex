"""
Cross-engine regression / complexity tests for the Cortex calculator.

Each test feeds one textual expression to the *async* part-inspector
(`part_inspector.compute`) which in turn dispatches work to the various
domain engines (trigonometry, calculus, complex, elementary, arithmetic).

The goal is to make sure that:

• engines can cooperate inside a single expression;
• numeric accuracy agrees with mpmath / Python’s built-ins;
• complex-valued results are returned in a canonical text form that we
  can still evaluate back to Python’s complex type.

Run with:

    python -m unittest test_complexity.py
    # or:  pytest -q

If a test fails, the traceback printed by every engine (when the REPL
`trace` flag is ON) should reveal the faulty stage.
"""
import asyncio
import unittest
from typing import Union

import sympy as sp
from mpmath import mp

# ----------------------------------------------------------------------
#  System under test
# ----------------------------------------------------------------------
from part_inspector import compute as inspect_compute


mp.dps = 50  # generous precision for all numeric comparisons


def eval_async(expr: str) -> str:
    """Utility: synchronously run the async inspector for unit tests."""
    return asyncio.run(inspect_compute(expr))


def _as_mpf(s: Union[str, float, int]) -> mp.mpf:
    """Convert string → mp.mpf loss-lessly for ≈ comparisons."""
    return mp.mpf(str(s))


class ComplexitySuite(unittest.TestCase):
    # 1 ──────────────────────────────────────────────────────────────
    def test_double_sine(self):
        expr = "sin(pi/6) + sin(pi/6)"
        result = _as_mpf(eval_async(expr))
        expected = mp.sin(mp.pi / 6) * 2     # exactly 1
        self.assertAlmostEqual(result, expected, delta=mp.mpf('1e-45'))

    # 2 ──────────────────────────────────────────────────────────────
    def test_trig_plus_derivative(self):
        expr = "cos(pi/3)*2 + derivative(x**2, x).subs(x, 3)"
        result = _as_mpf(eval_async(expr))
        expected = mp.cos(mp.pi / 3) * 2 + 2 * 3     # (= 1 + 6)
        self.assertAlmostEqual(result, expected, delta=mp.mpf('1e-45'))

    # 3 ──────────────────────────────────────────────────────────────
    def test_complex_product(self):
        expr = "(1+2j)*(3+4j)"
        out_str = eval_async(expr).replace(' ', '')
        out_str = out_str.replace('*I', 'j')
        result = complex(out_str)
        expected = (1 + 2j) * (3 + 4j)       # (-5+10j)
        self.assertAlmostEqual(result.real, expected.real, delta=1e-12)
        self.assertAlmostEqual(result.imag, expected.imag, delta=1e-12)

    # 4 ──────────────────────────────────────────────────────────────
    def test_grand_mix(self):
        expr = "sin(pi/4) + derivative(x**3, x).subs(x, 2) + (1+0j)"
        out_str = eval_async(expr)
        out_str = out_str.replace(' ', '').replace('*I', 'j')
        result = complex(out_str) if 'j' in out_str else complex(float(out_str))
        expected = mp.sin(mp.pi / 4) + 3 * 2 ** 2 + 1  # ≈ 13.70710678
        self.assertAlmostEqual(result.real, float(expected), delta=1e-10)
        self.assertAlmostEqual(result.imag, 0.0, delta=1e-12)

    # 5 ──────────────────────────────────────────────────────────────
    def test_exp_log_mix(self):
        expr = "exp(1) + log(10) - sin(pi/2)"
        result = _as_mpf(eval_async(expr))
        expected = mp.e + mp.log(10) - 1      # sin(pi/2) = 1
        self.assertAlmostEqual(result, expected, delta=mp.mpf('1e-45'))


if __name__ == "__main__":
    unittest.main()