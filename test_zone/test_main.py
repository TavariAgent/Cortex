import asyncio
from main import Structure, XorStringCompiler, Compute, Diagnostic
from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager


async def test_cortex():
    """Basic test for Cortex: Run a simple expression through the parallel flow."""

    # Setup instances
    struct = Structure()
    compiler = XorStringCompiler()
    segment_mgr = SegmentManager(struct)
    compute = Compute(compiler, segment_mgr)
    diag = Diagnostic()

    # Mock engine with segment manager
    engine = BasicArithmeticEngine(segment_mgr)

    # Sample expression
    expr = "2 + 3 * 4"

    # Simulate flow
    print("Testing Cortex flow...")

    # 1. Structure analysis
    snapshot = await struct._inform_structure()
    print(f"Structure snapshot: {snapshot}")

    # 2. Parallel computation in engine
    parts = [2, '+', 3, '*', 4]  # Mock parts
    results = await engine._compute_parts_parallel(parts)
    print(f"Parallel results: {results}")

    # 3. Unlock primary level and assemble
    segment_mgr.unlock_primary_level()
    finalized = segment_mgr.get_finalized()
    print(f"Finalized segments: {finalized}")

    # 4. Compute and finalize
    final = await compute._compute_expression()
    print(f"Final result: {final}")

    # Check for errors
    diag.log_error("Test completed - no errors")  # Mock success


if __name__ == '__main__':
    asyncio.run(test_cortex())