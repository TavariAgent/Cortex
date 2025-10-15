#!/usr/bin/env python3
"""
Test script for ComplexAlgebraEngine.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complex_algebra_engine import ComplexAlgebraEngine
from segment_manager import SegmentManager
from main import Structure


def test_complex_algebra():
    """Test ComplexAlgebraEngine with basic complex ops."""
    print("=" * 60)
    print("Testing ComplexAlgebraEngine")
    print("=" * 60)

    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = ComplexAlgebraEngine(segment_mgr)

    # Test simple complex
    result = engine.compute('1+2j')
    expected = "(1.0 + 2.0j)"  # mpmath format
    success = result == expected
    print(f"1+2j: {result} (Expected: {expected}) - Success: {success}")

    # Overall
    print("\n" + "=" * 60)
    if success:
        print("ComplexAlgebraEngine tests PASSED!")
    else:
        print("Some tests FAILED.")
    print("=" * 60)

    return success


if __name__ == '__main__':
    test_complex_algebra()