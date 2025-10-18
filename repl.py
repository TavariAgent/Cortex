#!/usr/bin/env python3
"""
Interactive REPL for Cortex Calculator.
Allows real-time testing of expressions across engines (arithmetic, trig, calculus, complex).
Type expressions like '2+3', 'sin(pi)', or 'derivative(x**2, x)'. Type 'quit' to exit.
"""
import asyncio
import sys
import os

from calculus_engine import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine
from precision_manager import presets, get_dps, set_dps
from trigonometry_engine import TrigonometryEngine
from part_inspector import compute as inspect_compute
from xor_string_compiler import XorStringCompiler  # Add this import

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager, Structure


def main():
    """Run the interactive REPL."""
    print("=" * 80)
    print("Welcome to Cortex Calculator REPL")
    print("Type expressions (e.g., '2+3', 'sin(pi)', 'derivative(x**2, x)')")
    print("Type 'trace' to toggle traceback display")
    print("Type 'precision' or 'p' to show current precision")
    print("Type 'precision N' where N is one of", presets(), "to change precision")
    print("Type 'quit' to exit")
    print("=" * 80)

    # Setup
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    xor_compiler = XorStringCompiler()  # Initialize XOR compiler
    show_trace = False
    auto_inject_threshold = 1000  # Auto-inject constants when precision >= 1000

    while True:
        try:
            user_input = input("cortex> ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            #  precision commands  ------------------------------------------------
            if user_input.lower().startswith(('precision', 'p')):
                parts = user_input.split()
                if len(parts) == 1:
                    current_dps = get_dps()
                    print(f"Current precision: {current_dps} dps")
                    if current_dps >= auto_inject_threshold:
                        print(f"  (High-precision constant injection: ACTIVE)")
                elif len(parts) == 2:
                    try:
                        new_dps = int(parts[1])
                        set_dps(new_dps)
                        # refresh engine instances with new mp.dps
                        engine = BasicArithmeticEngine(segment_mgr)
                        engine = CalculusEngine(segment_mgr)
                        engine = TrigonometryEngine(segment_mgr)
                        engine = ComplexAlgebraEngine(segment_mgr)
                        print(f"Precision set to {new_dps} dps")
                        if new_dps >= auto_inject_threshold:
                            print(
                                f"  High-precision constant injection enabled (up to {min(new_dps, 100000)} decimals)")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("Usage: precision [N]")
                continue

            if user_input.lower() == 'trace':
                show_trace = not show_trace
                print(f"Traceback display: {'ON' if show_trace else 'OFF'}")
                continue

            # Process expression - apply XOR injection in high-precision mode
            expression = user_input
            current_precision = get_dps()

            if current_precision >= auto_inject_threshold:
                # Apply XOR operator for constant injection
                try:
                    # Set the XOR compiler to use current precision (capped at 100k)
                    effective_precision = min(current_precision, 100000)
                    xor_result = xor_compiler ^ expression

                    if xor_result.get('constants_injected'):
                        expression = xor_result['modified_expr']
                        if show_trace:
                            print(f"  Constants injected at {effective_precision} decimal precision")
                            for const in xor_result.get('detected_constants', []):
                                print(f"    - {const}")
                except Exception as e:
                    if show_trace:
                        print(f"  Constant injection skipped: {e}")

            # Compute expression via call_helper (routes to appropriate engine)
            result = asyncio.run(inspect_compute(expression))
            print(f"Result: {result}")

            if show_trace:
                print("\nTracebacks:")
                for trace in engine.traceback_info[-5:]:  # Last 5 for brevity
                    print(f"  {trace['step']}: {trace['info']}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            if show_trace:
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
