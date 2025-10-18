# Cortex Calculator

A modular, high-precision math calculator built with Python, supporting arithmetic, trigonometry, calculus, and complex algebra through specialized engines.

## Architecture

- **Modular Engines**: Separate classes for different math domains (BasicArithmetic, Trigonometry, Calculus, ComplexAlgebra).
- **Precision Flow**: Avoids floats via conversions.
- **Debugging**: Step-wise tracebacks for all operations.
- **Caching**: XOR-gated constants for faster execution.
- **Cross-Engine Communication**: Async and parrallel execution.

## High-Precision Constants & "Constant Injection"

Cortex uses pre-computed, ultra-high-precision mathematical constants (π, e, φ, etc.) stored locally for unprecedented accuracy in calculations.

### How It Works

1. **Constant Generation**: On first run, type "generate" in the REPL to compute and store constants to 1 million decimal places
2. **Constant Injection**: When precision ≥ 15,000 dps, these pre-computed constants are automatically injected into calculations
3. **Precision Cascade**: High-precision constants propagate through the entire calculation chain

### Why Results Differ from Other Calculators

Most calculators use IEEE 754 double precision (~15-17 decimal digits). Cortex can compute with arbitrary precision, revealing:

- **Hidden Digits**: Values beyond standard precision that other calculators can't compute
- **Compound Precision Loss**: Standard calculators lose precision at each operation; Cortex maintains it
- **True Irrationals**: While others approximate π as 3.141592653589793, Cortex can use millions of verified digits

Example:

- Standard Calculator: sin(π/e) ≈ 0.4485279604341046 (15 digits) 

- Cortex: (1M dps): sin(π/e) = 0.4485279604341046... (~1,000,000 digits)

## Deeper Truth

The fundamental insight: 

**All digital computation is approximation**, but Cortex pushes the approximation boundary so far that it reveals mathematical truths invisible to standard precision.

You're not getting "different" answers - you're getting **more complete** answers.


## Practices

- Double underscores (__add__, __mul__, etc.) for operator overloading.
- Tracebacks integrated into all compute methods.
- Modularization: New logic splits into dedicated engines.

## Usage

```python
python repl.py
```

## Roadmap Near-Term to Long-Term
- Elementary Engine (e.g. exponents, logarithms, roots, etc.)
- Calculus Engine (derivatives, integrals, etc.)
- Complex Algebra Engine (complex numbers, complex roots, etc.)
- Trigonometry Engine (more angles and functions, etc.)

**- More Operators** 
- More Operators (e.g. Greek letters, etc.)
- More Operators (e.g. Bitwise, etc.)
- More Operators (e.g. Matrix, etc.)
- More Operators (e.g. Set, etc.)
- More Operators (e.g. String, etc.)
- More Operators (e.g. Function, etc.)
- More Operators (e.g. Lambda, etc.)
- More Operators (e.g. Trigonometric, etc.)
- More Operators (e.g. Logarithmic, etc.)
- More Operators (e.g. Exponential, etc.)

**- Support for Units**
- Unit Conversion and assembly
- More Units (SI, Imperial, etc.)
- More Units (Temperature, Pressure, etc.)
- More Units (Time, Distance, etc.)
- More Units (Mass, Volume, etc.)
- More Units (Energy, Power, etc.)
- More Units (Data, Data Rate, etc.)
- More Units (Frequency, etc.)

**- More Engines**
- More Engines (e.g. Physics, Chemistry, etc.)
- More Engines (e.g. Finance, Economics, etc.)
- More Engines (e.g. Biology, Chemistry, etc.)
- More Engines (e.g. Astronomy, Physics, etc.)
- More Engines (e.g. Robotics, Mechanics, etc.)
- More Engines (e.g. Computer Science, Math, etc.)
- More Engines (e.g. Engineering, Mechanical, etc.)
- More Engines (e.g. Bioengineering, Biomechanics, etc.)
- More Engines (e.g. Biomedical, Medical, etc.)
- More Engines (e.g. Atomic, etc.)

## Credits

Built with GitHub Copilot. Code by BoneKEY. 

"Think smart, compute smarter."

Built with AI assistance—let's compute for the future!