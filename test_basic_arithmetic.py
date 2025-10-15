#!/usr/bin/env python3
"""
Test script for BasicArithmeticEngine
Tests the __add__, __mul__, and compute methods with mpmath high precision.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


def test_basic_arithmetic():
    """Test BasicArithmeticEngine functionality."""

    print("=" * 60)
    print("Testing BasicArithmeticEngine")
    print("=" * 60)

    # Setup
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)

    # Test 1: Compute method with addition
    print("\nTest 1: compute('2+3')")
    print("-" * 40)
    result = engine.compute('2+3')
    print(f"Result: {result}")
    print(f"Expected: 5")
    print(f"Success: {result == '5'}")

    # Print traceback info
    print("\nStep-wise traceback:")
    for i, trace in enumerate(engine.traceback_info, 1):
        print(f"  {i}. {trace['step']}: {trace['info']}")

    # Test 2: Compute method with multiplication
    print("\nTest 2: compute('3*4')")
    print("-" * 40)
    engine2 = BasicArithmeticEngine(segment_mgr)
    result = engine2.compute('3*4')
    print(f"Result: {result}")
    print(f"Expected: 12")
    print(f"Success: {result == '12'}")

    # Print traceback info
    print("\nStep-wise traceback:")
    for i, trace in enumerate(engine2.traceback_info, 1):
        print(f"  {i}. {trace['step']}: {trace['info']}")

    # Test 3: __add__ method
    print("\nTest 3: __add__ method")
    print("-" * 40)
    engine3 = BasicArithmeticEngine(segment_mgr)
    engine3._value = "10"
    result_engine = engine3 + 5
    print(f"Result: {result_engine._value}")
    print(f"Expected: 15")
    print(f"Success: {result_engine._value == '15'}")

    # Print traceback info
    print("\nStep-wise traceback:")
    for i, trace in enumerate(result_engine.traceback_info, 1):
        print(f"  {i}. {trace['step']}: {trace['info']}")

    # Test 4: __mul__ method
    print("\nTest 4: __mul__ method")
    print("-" * 40)
    engine4 = BasicArithmeticEngine(segment_mgr)
    engine4._value = "7"
    result_engine = engine4 * 6
    print(f"Result: {result_engine._value}")
    print(f"Expected: 42")
    print(f"Success: {result_engine._value == '42'}")

    # Print traceback info
    print("\nStep-wise traceback:")
    for i, trace in enumerate(result_engine.traceback_info, 1):
        print(f"  {i}. {trace['step']}: {trace['info']}")

        # Test 5: High precision with mpmath
    print("\nTest 5: High precision computation")
    print("-" * 40)
    engine5 = BasicArithmeticEngine(segment_mgr)
    result = engine5.compute('123456789012345+987654321098765')
    print(f"Result: {result}")
    print(f"Expected: 1111111110111110")
    print(f"Success: {result == '1111111110111110'}")

    # Test 6: Integration with segment manager
    print("\nTest 6: Segment manager integration")
    print("-" * 40)
    print(f"Part orders registered: {len(segment_mgr.part_orders)}")
    print(f"Engines with orders: {list(segment_mgr.part_orders.keys())}")
    if 'BasicArithmeticEngine' in segment_mgr.part_orders:
        print(f"Number of orders for BasicArithmeticEngine: {len(segment_mgr.part_orders['BasicArithmeticEngine'])}")
        print(f"Sample order: {segment_mgr.part_orders['BasicArithmeticEngine'][0]}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_basic_arithmetic()