#!/usr/bin/env python3
"""
Interactive REPL for Cortex Calculator.
Allows real-time testing of expressions across engines (arithmetic, trig, calculus, complex).
Type expressions like '2+3', 'sin(pi)', or 'derivative(x**2, x)'. Type 'quit' to exit.
"""

import sys
import os

from calculus_engine import CalculusEngine
from complex_algebra_engine import ComplexAlgebraEngine
from precision_manager import presets, get_dps, set_dps
from trigonometry_engine import TrigonometryEngine

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure


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
    show_trace = False

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
                    print(f"Current precision: {get_dps()} dps")
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
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("Usage: precision [N]")
                continue

            if user_input.lower() == 'trace':
                show_trace = not show_trace
                print(f"Traceback display: {'ON' if show_trace else 'OFF'}")
                continue

            # Compute expression via call_helper (routes to appropriate engine)
            result = engine.call_helper(user_input)
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