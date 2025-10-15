#!/usr/bin/env python3
"""
Test script for TrigonometryEngine.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trigonometry_engine import TrigonometryEngine
from segment_manager import SegmentManager
from main import Structure


def test_trigonometry():
    """Test TrigonometryEngine with sin, cos functions."""
    print("=" * 60)
    print("Testing TrigonometryEngine")
    print("=" * 60)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = TrigonometryEngine(segment_mgr)

    # Test sin
    result_sin = engine.compute('sin(3.14159)')
    expected_sin = "0.0"  # Approx sin(pi) ≈ 0
    success_sin = abs(float(result_sin) - float(expected_sin)) < 1e-5
    print(f"sin(3.14159): {result_sin} (Expected: ~{expected_sin}) - Success: {success_sin}")

    # Test cos
    result_cos = engine.compute('cos(0)')
    expected_cos = "1.0"
    success_cos = abs(float(result_cos) - float(expected_cos)) < 1e-10
    print(f"cos(0): {result_cos} (Expected: {expected_cos}) - Success: {success_cos}")

    # Overall
    all_success = success_sin and success_cos
    print("\n" + "=" * 60)
    if all_success:
        print("TrigonometryEngine tests PASSED!")
    else:
        print("Some tests FAILED.")
    print("=" * 60)

    return all_success


if __name__ == '__main__':
    test_trigonometry()