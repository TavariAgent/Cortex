#!/usr/bin/env python3
"""
Integration test demonstrating the complete flow with BasicArithmeticEngine.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Structure, EngineWorker
from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager

async def test_integration():
    """Test the complete integration flow."""
    
    print("="*60)
    print("Integration Test: BasicArithmeticEngine Flow")
    print("="*60)
    
    # Setup
    struct = Structure()
    compiler = XorStringCompiler()
    segment_mgr = SegmentManager(struct)
    
    # Create engine
    engine = BasicArithmeticEngine(segment_mgr)
    
    # Test 1: Simple addition
    print("\nTest 1: Expression '2+3'")
    print("-"*40)
    result1 = engine.compute('2+3')
    print(f"Result: {result1}")
    print(f"Traceback steps: {len(engine.traceback_info)}")
    
    # Test 2: Simple multiplication
    print("\nTest 2: Expression '3*4'")
    print("-"*40)
    engine2 = BasicArithmeticEngine(segment_mgr)
    result2 = engine2.compute('3*4')
    print(f"Result: {result2}")
    print(f"Traceback steps: {len(engine2.traceback_info)}")
    
    # Test 3: Structure analysis
    print("\nTest 3: Structure Analysis")
    print("-"*40)
    snapshot = await struct._inform_structure()
    print(f"Flags: {snapshot['flags']}")
    print(f"Addition flag: {snapshot['flags']['addition']}")
    print(f"Multiplication flag: {snapshot['flags']['multiplication']}")
    
    # Test 4: Segment manager integration
    print("\nTest 4: Segment Manager")
    print("-"*40)
    print(f"Registered engines: {list(segment_mgr.part_orders.keys())}")
    print(f"Total part orders: {sum(len(orders) for orders in segment_mgr.part_orders.values())}")
    
    # Test 5: Using EngineWorker
    print("\nTest 5: EngineWorker")
    print("-"*40)
    worker = EngineWorker(BasicArithmeticEngine, segment_mgr)
    result3 = await worker.run({'expr': '10+20'})
    print(f"Result from worker: {result3}")
    
    # Test 6: XOR String Compiler
    print("\nTest 6: XOR String Compiler")
    print("-"*40)
    xor_dirs = compiler ^ None
    print(f"XOR pools created: {list(compiler.segment_pools.keys())}")
    print(f"Flag pool: {compiler.flag_pool}")
    
    print("\n" + "="*60)
    print("Integration test completed successfully!")
    print("="*60)

if __name__ == '__main__':
    asyncio.run(test_integration())
