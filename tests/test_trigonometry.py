#!/usr/bin/env python3
"""
Test script for TrigonometryEngine.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_trigonometry():
    """Test TrigonometryEngine with sin, cos functions."""
    print("=" * 60)
    print("Testing TrigonometryEngine")
    print("=" * 60)

    # Test sin
    result_sin = engine.compute('sin(pi/2)')
    expected_sin = "1.0"
    success_sin = abs(float(result_sin) - float(expected_sin)) < 1e-10
    print(f"sin(pi/2): {result_sin} (Expected: {expected_sin}) - Success: {success_sin}")

    # Test cos
    result_cos = engine.compute('cos(0)')
    expected_cos = "1.0"
    success_cos = abs(float(result_cos) - float(expected_cos)) < 1e-10
    print(f"cos(0): {result_cos} (Expected: {expected_cos}) - Success: {success_cos}")

    # Test tan
    result_tan = engine.compute('tan(pi/4)')
    expected_tan = "1.0"
    success_tan = abs(float(result_tan) - float(expected_tan)) < 1e-10
    print(f"tan(pi/4): {result_tan} (Expected: {expected_tan}) - Success: {success_tan}")

    # Test trigonometry angles
    result_sin = engine.compute('sin(pi/6)')
    expected_sin = "0.5"
    success_sin = abs(float(result_sin) - float(expected_sin)) < 1e-10
    print(f"sin(pi/6): {result_sin} (Expected: {expected_sin}) - Success: {success_sin}")

    result_cos = engine.compute('cos(pi/3)')
    expected_cos = "0.5"
    success_cos = abs(float(result_cos) - float(expected_cos)) < 1e-10
    print(f"cos(pi/3): {result_cos} (Expected: {expected_cos}) - Success: {success_cos}")


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