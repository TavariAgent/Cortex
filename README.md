# Cortex Calculator

A modular, high-precision math calculator built with Python, supporting arithmetic, trigonometry, calculus, and complex algebra through specialized engines.

## Architecture

- **Modular Engines**: Separate classes for different math domains (BasicArithmetic, Trigonometry, Calculus, ComplexAlgebra).
- **Precision Flow**: Avoids floats via Decimal → str → mpmath conversions.
- **Debugging**: Step-wise tracebacks for all operations.
- **Caching**: XOR-gated LRU for dynamic, ephemeral non-static objects.
- **Cross-Engine Communication**: call_helper for mixed symbolic/numeric expressions.

## Practices

- Double underscores (__add__, __mul__, etc.) for operator overloading.
- Tracebacks integrated into all compute methods.
- Modularization: New logic splits into dedicated engines.

## Credits

Built with GitHub Copilot. Code by BoneKEY (TavariAgent).

## Usage

```python
from abc_engines import BasicArithmeticEngine
engine = BasicArithmeticEngine(segment_manager)
result = engine.compute('sin(3.14) + 2**3')
```

## Roadmap
- Precision Flow for ...
- Debugging for ...
- Caching for ...
- Cross-Engine Communication for ...
- Complex Algebra
- Complex Numbers
- Complex Trigonometry
- Complex Calculus
- Complex Logarithms
- Complex Exponentials
- Complex Power
- Complex Roots

Built with AI assistance—let's compute for the future!