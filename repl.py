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
from utils.precision_manager import presets, get_dps, set_dps
from trigonometry_engine import TrigonometryEngine
from part_inspector import compute as inspect_compute
from xor_string_compiler import XorStringCompiler
from utils.constants_jsonl_generator import ConstantsJSONLGenerator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager, Structure


def sync_int_str_limit(dps):
    """
    Synchronize Python's integer string conversion limit with current precision.

    Args:
        dps: Decimal places of precision
    """

    # Calculate required digits (add buffer for intermediate calculations)
    # Use 2x the precision as buffer for intermediate values
    required_digits = max(dps * 2, 15000)  # Minimum 15000 as default

    # Cap at a reasonable maximum (e.g., 1 million)
    required_digits = min(required_digits, 1000000)

    try:
        sys.set_int_max_str_digits(required_digits)
        return required_digits
    except ValueError:
        # Fallback to a safe value if setting fails
        sys.set_int_max_str_digits(15000)
        return 15000


def main():
    """Run the interactive REPL."""
    print("=" * 80)
    print("Welcome to Cortex Calculator REPL")
    print("Type expressions (e.g., '2+3', 'sin(pi)', 'derivative(x**2, x)')")
    print("Type 'trace' to toggle traceback display")
    print("Type 'precision' or 'p' to show current precision")
    print("Type 'precision N' where N is one of", presets(), "to change precision")
    print("Type 'generate' for constant generation commands")
    print("Type 'quit' to exit")

    # Initialize integer string limit based on current precision
    initial_dps = get_dps()
    sync_int_str_limit(initial_dps)

    # Setup
    struct = Structure()
    segment_mgr = SegmentManager(struct)
    engine = BasicArithmeticEngine(segment_mgr)
    xor_compiler = XorStringCompiler()  # Initialize XOR compiler
    show_trace = False
    auto_inject_threshold = 1000  # Auto-inject constants when precision >= 1000

    constants_gen = ConstantsJSONLGenerator()

    while True:
        try:
            user_input = input("cortex> ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower().startswith('generate'):
                parts = user_input.split()
                if len(parts) == 1:
                    # Use the reusable help text
                    print(constants_gen.get_help_text(for_repl=True))
                elif parts[1].lower() == 'list':
                    print(f"Available constants: {', '.join(constants_gen.list_available())}")
                elif parts[1].lower() == 'all':
                    dps = get_dps()
                    print(f"Generating all constants at {dps} dps...")
                    results = constants_gen.generate_all(dps)
                    successful = sum(1 for v in results.values() if v)
                    print(f"Generated {successful}/{len(results)} constants in 'constants/' folder")
                else:
                    constant_name = parts[1].lower()
                    try:
                        filepath = constants_gen.generate_and_save(constant_name)
                        print(f"Generated {constant_name} at {get_dps()} dps")
                        print(f"Saved to: {filepath}")
                    except ValueError as e:
                        print(f"Error: {e}")
                        print(f"Available constants: {', '.join(constants_gen.list_available())}")
                continue

            #  precision commands  ------------------------------------------------
            if user_input.lower().startswith(('precision', 'p')):
                parts = user_input.split()
                if len(parts) == 1:
                    current_dps = get_dps()
                    current_int_limit = sys.get_int_max_str_digits()
                    print(f"Current precision: {current_dps} dps")
                    print(f"  Integer string limit: {current_int_limit} digits")
                    if current_dps >= auto_inject_threshold:
                        print(f"  High-precision constant injection: ACTIVE")
                elif len(parts) == 2:
                    try:
                        new_dps = int(parts[1])
                        set_dps(new_dps)

                        # Sync integer string digit limit
                        int_limit = sync_int_str_limit(new_dps)

                        # refresh engine instances with new mp.dps
                        engine = BasicArithmeticEngine(segment_mgr)
                        engine = CalculusEngine(segment_mgr)
                        engine = TrigonometryEngine(segment_mgr)
                        engine = ComplexAlgebraEngine(segment_mgr)

                        print(f"Precision set to {new_dps} dps")
                        print(f"  Integer string limit: {int_limit} digits")
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
