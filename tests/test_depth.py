#!/usr/bin/env python3
"""
Test script for deep nested expressions with up to four layers of parentheses and mixed operators.
Tests the calculator's ability to handle complex PEMDAS with nesting.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


def test_deep_expressions():
    """Test three deep nested expressions with mixed operators."""
    print("=" * 80)
    print("Testing Deep Nested Expressions (Up to 4 Layers)")
    print("=" * 80)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)

    # Test 1: 4-layer nested expression with mixed ops
    expr1 = "(((2+3)*4)-5)/2"
    expected1 = 7.5  # ((5*4)-5)/2 = (20-5)/2 = 15/2 = 7.5
    print(f"\nTest 1: {expr1}")
    print("-" * 40)
    result1 = engine.compute(expr1)
    success1 = abs(float(result1) - expected1) < 1e-10
    print(f"Result: {result1}")
    print(f"Expected: {expected1}")
    print(f"Success: {success1}")

    if not success1:
        print("Traceback:")
        for trace in engine.traceback_info[-10:]:  # Last 10 traces
            print(f"  {trace['step']}: {trace['info']}")

    # Test 2: 4-layer with division and multiplication
    expr2 = "(((10/2)*3)+4)*2"
    expected2 = 38.0  # (((5)*3)+4)*2 = (15+4)*2 = 19*2 = 38
    print(f"\nTest 2: {expr2}")
    print("-" * 40)
    result2 = engine.compute(expr2)
    success2 = abs(float(result2) - expected2) < 1e-10
    print(f"Result: {result2}")
    print(f"Expected: {expected2}")
    print(f"Success: {success2}")

    if not success2:
        print("Traceback:")
        for trace in engine.traceback_info[-10:]:
            print(f"  {trace['step']}: {trace['info']}")

    # Test 3: 4-layer with subtraction and addition
    expr3 = "(((8-3)+2)*4)-6"
    expected3 = 22.0  # (((5)+2)*4)-6 = (7*4)-6 = 28-6 = 22
    print(f"\nTest 3: {expr3}")
    print("-" * 40)
    result3 = engine.compute(expr3)
    success3 = abs(float(result3) - expected3) < 1e-10
    print(f"Result: {result3}")
    print(f"Expected: {expected3}")
    print(f"Success: {success3}")

    if not success3:
        print("Traceback:")
        for trace in engine.traceback_info[-10:]:
            print(f"  {trace['step']}: {trace['info']}")

    #Test 4: 4-layer with exponents and addition
    expr4 = "(((2**3)-5)+1)**2"
    expected4 = 16.0  # ((8-5)+1)**2 = 4**2 = 16
    print(f"\nTest 4: {expr4}")
    print("-" * 40)
    result4 = engine.compute(expr4)
    success4 = abs(float(result4) - expected4) < 1e-10
    print(f"Result: {result4}")
    print(f"Expected: {expected4}")
    print(f"Success: {success4}")

    if not success4:
        print("Traceback:")
        for trace in engine.traceback_info[-10:]:
            print(f"  {trace['step']}: {trace['info']}")

    # Overall success
    all_success = success1 and success2 and success3 and success4
    print("\n" + "=" * 80)
    if all_success:
        print("All deep expression tests PASSED!")
    else:
        print("Some tests FAILED - check tracebacks above.")
    print("=" * 80)

    return all_success

if __name__ == '__main__':
    test_deep_expressions()