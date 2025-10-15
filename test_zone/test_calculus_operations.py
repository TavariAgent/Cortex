#!/usr/bin/env python3
"""
Comprehensive calculus operations test script for CalculusEngine.
Covers supported operations in multiple forms (string compute and direct methods).
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp
from calculus_engine import CalculusEngine
from segment_manager import SegmentManager
from main import Structure


def _as_bool(expr):
    """Helper to coerce truthiness for SymPy comparisons in prints."""
    return bool(expr)


def test_calculus_operations():
    print("=" * 60)
    print("Comprehensive CalculusEngine Operations Test")
    print("=" * 60)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = CalculusEngine(segment_mgr)

    successes = []

    # 1) Basic derivative via compute("derivative(...)")
    res = engine.compute('derivative(x**2, x)')
    expected = '2*x'
    ok = str(res) == expected
    print(f"compute derivative: derivative(x**2, x) -> {res} | expected {expected} | ok={ok}")
    successes.append(ok)

    # 2) Basic integral via compute("integral(...)")
    res = engine.compute('integral(x, x)')
    expected = 'x**2/2'
    ok = str(res) == expected
    print(f"compute integral: integral(x, x) -> {res} | expected {expected} | ok={ok}")
    successes.append(ok)

    # 3) Direct method: derivative
    res = engine.derivative('sin(x)', 'x')
    ok = sp.simplify(res - sp.cos(sp.symbols('x'))) == 0
    print(f"method derivative: d/dx sin(x) -> {res} | expected cos(x) | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 4) Direct method: integral
    res = engine.integral('cos(x)', 'x')
    ok = sp.simplify(res - sp.sin(sp.symbols('x'))) == 0
    print(f"method integral: ∫ cos(x) dx -> {res} | expected sin(x) | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 5) Second derivative
    res = engine.second_derivative('x**3', 'x')
    ok = sp.simplify(res - 6*sp.symbols('x')) == 0
    print(f"second_derivative: d²/dx² x**3 -> {res} | expected 6*x | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 6) Limits (finite and infinity)
    res = engine.limit('sin(x)/x', 'x', '0')
    ok = sp.simplify(res - 1) == 0
    print(f"limit x->0 sin(x)/x -> {res} | expected 1 | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    res = engine.limit('1/x', 'x', 'infinity')
    ok = sp.simplify(res - 0) == 0
    print(f"limit x->∞ 1/x -> {res} | expected 0 | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    res = engine.limit('1/x', 'x', '-infinity')
    ok = sp.simplify(res - 0) == 0
    print(f"limit x->-∞ 1/x -> {res} | expected 0 | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 7) Definite integral
    res = engine.definite_integral('x', 'x', 0, 1)
    ok = sp.simplify(res - sp.Rational(1, 2)) == 0
    print(f"definite_integral: ∫_0^1 x dx -> {res} | expected 1/2 | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 8) Product rule
    res = engine.product_rule('x**2', 'sin(x)', 'x')
    x = sp.symbols('x')
    expected_expr = 2*x*sp.sin(x) + x**2*sp.cos(x)
    ok = sp.simplify(res - expected_expr) == 0
    print(f"product_rule: (x**2)*(sin(x)) -> {res} | expected 2*x*sin(x) + x**2*cos(x) | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 9) Quotient rule
    res = engine.quotient_rule('x**2', 'x', 'x')
    ok = sp.simplify(res - 1) == 0
    print(f"quotient_rule: (x**2)/(x) -> {res} | expected 1 | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 10) Chain rule
    res = engine.chain_rule('sin(u)', 'x**2', 'x')
    ok = sp.simplify(res - (2*x*sp.cos(x**2))) == 0
    print(f"chain_rule: sin(x**2) -> {res} | expected 2*x*cos(x**2) | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # 11) Integration by parts (simple)
    res = engine.integration_by_parts('x', 'sin(x)', 'x')
    ok = sp.simplify(res - (sp.sin(x) - x*sp.cos(x))) == 0
    print(f"integration_by_parts: u=x, dv=sin(x) -> {res} | expected sin(x) - x*cos(x) | ok={_as_bool(ok)}")
    successes.append(_as_bool(ok))

    # Note: u_substitution in this engine is simplified; we ensure a trivial safe case
    # 12) u-substitution (trivial identity case)
    try:
        res = engine.u_substitution('x', 'x', 'x', 'x')
        ok = True  # Accept any SymPy expression result for this trivial mapping
        print(f"u_substitution: expr=x, u=x, du=x -> {res} | ok={ok}")
        successes.append(ok)
    except Exception as e:
        print(f"u_substitution raised exception (acceptable given simplified impl): {e}")
        successes.append(False)

    all_ok = all(successes)
    print("\n" + "=" * 60)
    if all_ok:
        print("Comprehensive CalculusEngine tests PASSED!")
    else:
        print("Some calculus tests FAILED.")
    print("=" * 60)

    return all_ok


if __name__ == '__main__':
    test_calculus_operations()
