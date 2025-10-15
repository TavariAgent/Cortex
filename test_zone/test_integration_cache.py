#!/usr/bin/env python3
"""
Integration test for segment_pools and cache mechanisms
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


def test_integration():
    """Test full integration of cache and segment_pools."""
    print("=" * 60)
    print("Integration Test: Cache and SegmentPools")
    print("=" * 60)
    
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    
    # Test complex expression
    print("\nTest 1: Complex expression with caching")
    print("-" * 40)
    result = engine.compute('3*4-5+1+6/2+2')
    print(f"Result: {result}")
    
    # Check segment_pools
    print("\nSegment pools populated:")
    for engine_name, segments in segment_mgr.segment_pools.items():
        print(f"  {engine_name}: {len(segments)} segments")
        for i, seg in enumerate(segments):
            print(f"    Segment {i+1}: {seg}")
    
    # Test chaining with cache propagation
    print("\nTest 2: Chained operations with cache propagation")
    print("-" * 40)
    engine2 = BasicArithmeticEngine(segment_mgr)
    engine2._value = "5"
    result_engine = (engine2 + 3) * 2
    
    print(f"Final value: {result_engine._value}")
    print(f"Cache entries: {len(result_engine._cache)}")
    print(f"Cache contents: {result_engine._cache}")
    
    # Check completion flags functionality
    print("\nTest 3: Completion flags")
    print("-" * 40)
    print(f"Initial all_complete(): {segment_mgr.all_complete()}")
    
    segment_mgr.completion_flags['BasicArithmeticEngine'] = True
    print(f"After setting BasicArithmeticEngine=True: {segment_mgr.all_complete()}")
    
    segment_mgr.completion_flags['OtherEngine'] = False
    print(f"After adding OtherEngine=False: {segment_mgr.all_complete()}")
    
    segment_mgr.completion_flags['OtherEngine'] = True
    print(f"After setting OtherEngine=True: {segment_mgr.all_complete()}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_integration()
