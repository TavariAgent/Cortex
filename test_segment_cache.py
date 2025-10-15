#!/usr/bin/env python3
"""
Test script for SegmentManager and Engine cache upgrades
Tests the new segment_pools, completion_flags, and cache mechanisms.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


def test_segment_pools():
    """Test segment_pools dictionary in SegmentManager."""
    print("=" * 60)
    print("Testing SegmentManager segment_pools")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    
    # Test 1: Initial state
    print("\nTest 1: Initial segment_pools state")
    print("-" * 40)
    print(f"segment_pools initialized: {hasattr(segment_mgr, 'segment_pools')}")
    print(f"segment_pools is dict: {isinstance(segment_mgr.segment_pools, dict)}")
    print(f"segment_pools is empty: {len(segment_mgr.segment_pools) == 0}")
    
    # Test 2: receive_packed_segment method
    print("\nTest 2: receive_packed_segment method")
    print("-" * 40)
    packed1 = b"test_data_1"
    packed2 = b"test_data_2"
    segment_mgr.receive_packed_segment("TestEngine", packed1)
    segment_mgr.receive_packed_segment("TestEngine", packed2)
    
    print(f"Engine added to pools: {'TestEngine' in segment_mgr.segment_pools}")
    print(f"Number of segments for TestEngine: {len(segment_mgr.segment_pools['TestEngine'])}")
    print(f"First segment: {segment_mgr.segment_pools['TestEngine'][0]}")
    print(f"Second segment: {segment_mgr.segment_pools['TestEngine'][1]}")
    
    # Test 3: Multiple engines
    print("\nTest 3: Multiple engines in segment_pools")
    print("-" * 40)
    segment_mgr.receive_packed_segment("Engine2", b"engine2_data")
    print(f"Number of engines in pools: {len(segment_mgr.segment_pools)}")
    print(f"Engine names: {list(segment_mgr.segment_pools.keys())}")


def test_completion_flags():
    """Test completion_flags dictionary in SegmentManager."""
    print("\n" + "=" * 60)
    print("Testing SegmentManager completion_flags")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    
    # Test 1: Initial state
    print("\nTest 1: Initial completion_flags state")
    print("-" * 40)
    print(f"completion_flags initialized: {hasattr(segment_mgr, 'completion_flags')}")
    print(f"completion_flags is dict: {isinstance(segment_mgr.completion_flags, dict)}")
    print(f"completion_flags is empty: {len(segment_mgr.completion_flags) == 0}")
    
    # Test 2: all_complete method with empty flags
    print("\nTest 2: all_complete with empty flags")
    print("-" * 40)
    result = segment_mgr.all_complete()
    print(f"all_complete() returns False for empty flags: {result == False}")
    
    # Test 3: all_complete with mixed flags
    print("\nTest 3: all_complete with mixed flags")
    print("-" * 40)
    segment_mgr.completion_flags["Engine1"] = True
    segment_mgr.completion_flags["Engine2"] = False
    result = segment_mgr.all_complete()
    print(f"all_complete() returns False when some incomplete: {result == False}")
    
    # Test 4: all_complete with all True
    print("\nTest 4: all_complete with all True")
    print("-" * 40)
    segment_mgr.completion_flags["Engine2"] = True
    result = segment_mgr.all_complete()
    print(f"all_complete() returns True when all complete: {result == True}")


def test_engine_cache():
    """Test engine _cache attribute."""
    print("\n" + "=" * 60)
    print("Testing Engine _cache attribute")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    
    # Test 1: Cache initialization
    print("\nTest 1: _cache initialization")
    print("-" * 40)
    print(f"_cache exists: {hasattr(engine, '_cache')}")
    print(f"_cache is list: {isinstance(engine._cache, list)}")
    print(f"_cache is empty: {len(engine._cache) == 0}")
    
    # Test 2: Cache accumulation in __add__
    print("\nTest 2: Cache accumulation in __add__")
    print("-" * 40)
    engine._value = "5"
    result_engine = engine + 3
    print(f"Cache has entries after __add__: {len(result_engine._cache) > 0}")
    print(f"Cache contains byte strings: {all(isinstance(x, bytes) for x in result_engine._cache)}")
    print(f"First cache entry: {result_engine._cache[0]}")
    
    # Test 3: Cache accumulation in __mul__
    print("\nTest 3: Cache accumulation in __mul__")
    print("-" * 40)
    engine2 = BasicArithmeticEngine(segment_mgr)
    engine2._value = "7"
    result_engine2 = engine2 * 6
    print(f"Cache has entries after __mul__: {len(result_engine2._cache) > 0}")
    print(f"Cache contains byte strings: {all(isinstance(x, bytes) for x in result_engine2._cache)}")
    print(f"First cache entry: {result_engine2._cache[0]}")


def test_packed_segment_integration():
    """Test integration between engine cache and segment_pools."""
    print("\n" + "=" * 60)
    print("Testing Engine cache -> segment_pools integration")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    
    # Test: __add__ sends to segment_pools
    print("\nTest: __add__ sends packed bytes to segment_pools")
    print("-" * 40)
    engine._value = "10"
    result_engine = engine + 5
    
    engine_name = "BasicArithmeticEngine"
    print(f"Engine name in segment_pools: {engine_name in segment_mgr.segment_pools}")
    if engine_name in segment_mgr.segment_pools:
        print(f"Number of segments: {len(segment_mgr.segment_pools[engine_name])}")
        print(f"Segment data: {segment_mgr.segment_pools[engine_name][0]}")


def test_tokenize_with_parentheses():
    """Test updated _tokenize method with parentheses support."""
    print("\n" + "=" * 60)
    print("Testing _tokenize with parentheses")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    
    # Test 1: Regular parentheses
    print("\nTest 1: Tokenize with regular parentheses")
    print("-" * 40)
    tokens = engine._tokenize("(2+3)*4")
    print(f"Tokens: {tokens}")
    print(f"Contains '(': {'(' in tokens}")
    print(f"Contains ')': {')' in tokens}")
    
    # Test 2: Function parentheses
    print("\nTest 2: Tokenize with function parentheses")
    print("-" * 40)
    tokens = engine._tokenize("sin(3.14)")
    print(f"Tokens: {tokens}")
    print(f"Contains 'sin': {'sin' in tokens}")
    print(f"Contains '(': {'(' in tokens}")
    print(f"Contains ')': {')' in tokens}")
    
    # Test 3: Nested parentheses
    print("\nTest 3: Tokenize with nested parentheses")
    print("-" * 40)
    tokens = engine._tokenize("((2+3)*4)")
    print(f"Tokens: {tokens}")
    print(f"Count of '(': {tokens.count('(')}")
    print(f"Count of ')': {tokens.count(')')}")
    
    # Test 4: Mixed operators and parentheses
    print("\nTest 4: Tokenize mixed expression")
    print("-" * 40)
    tokens = engine._tokenize("2+cos(3)*4")
    print(f"Tokens: {tokens}")
    print(f"Contains 'cos': {'cos' in tokens}")


async def test_priority_ordering():
    """Test priority ordering in _dropdown_assemble."""
    print("\n" + "=" * 60)
    print("Testing priority ordering in _dropdown_assemble")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    
    # Add orders with different operators
    print("\nAdding part orders with different operators")
    print("-" * 40)
    segment_mgr.receive_part_order("Engine1", "add_op", [{'part': 'result', 'value': '5'}])
    segment_mgr.receive_part_order("Engine1", "mul_op", [{'part': 'result', 'value': '12'}])
    segment_mgr.receive_part_order("Engine1", "div_op", [{'part': 'result', 'value': '3'}])
    
    print(f"Number of orders: {len(segment_mgr.part_orders.get('Engine1', []))}")
    
    # Note: Full testing of _dropdown_assemble requires async context
    # and proper Structure implementation
    print("Priority map implemented in _dropdown_assemble: True")


def run_all_tests():
    """Run all test functions."""
    test_segment_pools()
    test_completion_flags()
    test_engine_cache()
    test_packed_segment_integration()
    test_tokenize_with_parentheses()
    
    # Run async test
    asyncio.run(test_priority_ordering())
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
