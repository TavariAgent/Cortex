#!/usr/bin/env python3
"""
Test script for CalculusEngine.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus_engine import CalculusEngine
from segment_manager import SegmentManager
from main import Structure


def test_calculus():
    """Test CalculusEngine with derivatives and integrals."""
    print("=" * 60)
    print("Testing CalculusEngine")
    print("=" * 60)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = CalculusEngine(segment_mgr)

    # Test derivative
    result_deriv = engine.compute('derivative(x**2, x)')
    expected_deriv = "2*x"  # SymPy result
    success_deriv = str(result_deriv) == expected_deriv
    print(f"derivative(x**2, x): {result_deriv} (Expected: {expected_deriv}) - Success: {success_deriv}")

    # Overall
    all_success = success_deriv
    print("\n" + "=" * 60)
    if all_success:
        print("CalculusEngine tests PASSED!")
    else:
        print("Some tests FAILED.")
    print("=" * 60)

    return all_success


if __name__ == '__main__':
    test_calculus()