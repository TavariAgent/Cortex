#!/usr/bin/env python3
"""
Constants JSONL Generator for Cortex Calculator.
Generates high-precision mathematical constants and saves them as JSONL files.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.precision_manager import get_dps, set_dps
import json
from pathlib import Path
from typing import Dict, Any
import mpmath as mp
from datetime import datetime


class ConstantsJSONLGenerator:
    """Generate and save mathematical constants to JSONL format."""

    @staticmethod
    def get_help_text(for_repl=False):
        """Get formatted help text for constant generation.

        Args:
            for_repl: If True, formats for REPL usage. If False, for CLI usage.

        Returns:
            Formatted help string
        """
        if for_repl:
            return """Usage: generate <command> [args]
Commands:
  generate list              - List available constants  
  generate <name>            - Generate specific constant at current precision
  generate all               - Generate all constants at current precision

Examples:
  generate pi                - Generate pi at current precision
  generate all               - Generate all constants"""
        else:
            return """Usage: python constants_jsonl_generator.py <command> [args]
Commands:
  list                    - List available constants
  generate <name> [dps]   - Generate specific constant
  generate-all [dps]      - Generate all constants

Examples:
  python constants_jsonl_generator.py generate pi 1000000
  python constants_jsonl_generator.py generate-all 50000"""

    def __init__(self, output_dir: str = "constants"):
        """Initialize generator with output directory."""
        # Get the project root (parent of utils directory)
        project_root = Path(__file__).parent.parent
        self.output_dir = project_root / output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Map of constant names to mpmath generators
        self.generators = {
            'pi': lambda: mp.pi,
            'e': lambda: mp.e,
            'phi': lambda: mp.phi,  # Golden ratio
            'euler': lambda: mp.euler,  # Euler-Mascheroni constant
            'ln2': lambda: mp.ln2,
            'ln10': lambda: mp.ln10,
            'sqrt2': lambda: mp.sqrt(2),
            'sqrt3': lambda: mp.sqrt(3),
            'sqrt5': lambda: mp.sqrt(5),
            'catalan': lambda: mp.catalan,
            'khinchin': lambda: mp.khinchin,
            'glaisher': lambda: mp.glaisher,
            'apery': lambda: mp.apery,  # Apéry's constant ζ(3)
        }

    def generate_constant(self, name: str, dps: int = None) -> Dict[str, Any]:
        """
        Generate a single constant at specified precision.

        Args:
            name: Name of the constant (e.g., 'pi', 'e')
            dps: Decimal places (uses current if not specified)

        Returns:
            Dictionary with constant data
        """
        if name not in self.generators:
            raise ValueError(f"Unknown constant: {name}. Available: {list(self.generators.keys())}")

        # Set precision if specified
        original_dps = mp.mp.dps  # Access dps through mp.mp
        dps = dps or mp.mp.dps
        mp.mp.dps = dps

        try:
            # Generate constant with mpmath
            value = self.generators[name]()

            # Convert to string with full precision
            value_str = mp.nstr(value, dps)

            # Create metadata
            result = {
                'name': name,
                'precision': dps,
                'value': value_str,
                'length': len(value_str.replace('.', '').replace('-', '')),
                'timestamp': datetime.utcnow().isoformat(),
                'generator': 'mpmath',
                'mpmath_version': mp.__version__ if hasattr(mp, '__version__') else 'unknown'
            }

            return result

        finally:
            # Restore original precision
            mp.mp.dps = original_dps

    def save_constant(self, constant_data: Dict[str, Any]) -> str:
        """
        Save constant data to JSONL file.

        Args:
            constant_data: Dictionary with constant information

        Returns:
            Path to saved file
        """
        name = constant_data['name']
        dps = constant_data['precision']

        # Create filename: constant_name_dps.jsonl
        filename = f"{name}_{dps}dps.jsonl"
        filepath = self.output_dir / filename

        # Write as JSONL (one JSON object per line)
        with open(filepath, 'w') as f:
            json.dump(constant_data, f)
            f.write('\n')

        return str(filepath)

    def generate_and_save(self, name: str, dps: int = None) -> str:
        """
        Generate constant and save to JSONL.

        Args:
            name: Constant name
            dps: Precision (uses current if None)

        Returns:
            Path to saved file
        """
        constant_data = self.generate_constant(name, dps)
        filepath = self.save_constant(constant_data)
        return filepath

    def generate_all(self, dps: int = None) -> Dict[str, str]:
        """
        Generate all available constants.

        Args:
            dps: Precision for all constants

        Returns:
            Dictionary mapping constant names to file paths
        """
        results = {}
        for name in self.generators:
            try:
                filepath = self.generate_and_save(name, dps)
                results[name] = filepath
                print(f"Generated {name} at {dps or mp.mp.dps} dps -> {filepath}")  # Changed here
            except Exception as e:
                print(f"Failed to generate {name}: {e}")
                results[name] = None
        return results

    def list_available(self) -> list:
        """Get list of available constants."""
        return list(self.generators.keys())


def main():
    """CLI interface for constant generation."""

    generator = ConstantsJSONLGenerator()

    if len(sys.argv) < 2:
        print("Usage: python constants_jsonl_generator.py <command> [args]")
        print("\nCommands:")
        print("  list                    - List available constants")
        print("  generate <name> [dps]   - Generate specific constant")
        print("  generate-all [dps]      - Generate all constants")
        print("\nExamples:")
        print("  python constants_jsonl_generator.py generate pi 1000000")
        print("  python constants_jsonl_generator.py generate-all 50000")
        return

    command = sys.argv[1].lower()

    if command == 'list':
        print("Available constants:")
        for name in generator.list_available():
            print(f"  - {name}")

    elif command == 'generate':
        if len(sys.argv) < 3:
            print("Error: Specify constant name")
            print(f"Available: {', '.join(generator.list_available())}")
            return

        name = sys.argv[2].lower()
        dps = int(sys.argv[3]) if len(sys.argv) > 3 else get_dps()

        try:
            filepath = generator.generate_and_save(name, dps)
            print(f"Successfully generated {name} at {dps} dps")
            print(f"Saved to: {filepath}")
        except Exception as e:
            print(f"Error: {e}")

    elif command == 'generate-all':
        dps = int(sys.argv[2]) if len(sys.argv) > 2 else get_dps()
        print(f"Generating all constants at {dps} dps...")
        results = generator.generate_all(dps)

        successful = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]

        print(f"\nGenerated {len(successful)}/{len(results)} constants")
        if failed:
            print(f"Failed: {', '.join(failed)}")

    else:
        print(f"Unknown command: {command}")


if __name__ == '__main__':
    main()
