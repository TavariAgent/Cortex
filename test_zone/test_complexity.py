#!/usr/bin/env python3
"""
Test script for cross-engine complexity: Mixed expressions via call_helper.
Tests routing, computation, and segment_pools for systematic control.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


def test_complexity():
    """Test call_helper with expressions across engines."""
    print("=" * 80)
    print("Testing Cross-Engine Complexity via call_helper")
    print("=" * 80)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)  # Base engine for call_helper

    tests = [
        ("cos(pi)", "-1.0", "Trigonometry"),
        ("derivative(x**2, x)", "2*x", "Calculus"),
        ("integral(2*x, x)", "x**2", "Calculus"),
        ("1+2j", "(1.0 + 2.0j)", "Complex"),
        ("2+3", "5.0", "Arithmetic"),
        ("((2**3)-5)+1", "4.0", "Arithmetic with powers"),
    ]

    all_success = True
    for expr, expected, category in tests:
        print(f"\nTesting {category}: {expr}")
        print("-" * 40)
        try:
            result = engine.call_helper(expr)
            success = abs(float(result) - float(expected)) < 1e-5 if expected.replace('.', '').isdigit() else str(
                result) == expected
            print(f"Result: {result}")
            print(f"Expected: {expected}")
            print(f"Success: {success}")
            if not success:
                all_success = False
        except Exception as e:
            print(f"Error: {e}")
            all_success = False

    # Check segment_pools for mixed results
    print(f"\nSegment pools after tests: {list(segment_mgr.segment_pools.keys())}")
    if 'CallHelper' in segment_mgr.segment_pools:
        print(f"Mixed segments: {len(segment_mgr.segment_pools['CallHelper'])}")

    print("\n" + "=" * 80)
    if all_success:
        print("Cross-engine complexity tests PASSED!")
    else:
        print("Some tests FAILED - check routing or computations.")
    print("=" * 80)

    return all_success


if __name__ == '__main__':
    test_complexity()