# BasicArithmeticEngine Implementation

This document describes the implementation of the `__add__`, `__mul__`, and `compute` methods in the `BasicArithmeticEngine` class.

## Overview

The `BasicArithmeticEngine` has been enhanced to support high-precision arithmetic operations using `mpmath`, avoiding the use of floats and Decimals as per the project guidelines.

## Features

### 1. High-Precision Arithmetic
- Uses `mpmath` library for all numerical computations
- Avoids floating-point precision issues
- Supports arbitrary precision calculations

### 2. Step-wise Debug Tracebacks
Each operation records detailed debug information including:
- Operation type (parse, conversion, computation, etc.)
- Input values and results
- Timestamps (when in async context)

Access tracebacks via: `engine.traceback_info`

### 3. Segment Manager Integration
All operations integrate seamlessly with the existing parallel flow:
- Results are sent to `segment_manager` via `receive_part_order()`
- Supports modular, parallel computation architecture
- Compatible with XOR sub-directory segments

## Methods Implemented

### `compute(expr: str)`
Parses and evaluates simple arithmetic expressions.

**Supported Operations:**
- Addition: `"2+3"` → `5.0`
- Multiplication: `"3*4"` → `12.0`

**Example:**
```python
from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure

struct = Structure()
segment_mgr = SegmentManager(struct)
engine = BasicArithmeticEngine(segment_mgr)

result = engine.compute('2+3')
print(f"Result: {result}")  # Output: Result: 5.0

# View traceback
for trace in engine.traceback_info:
    print(f"{trace['step']}: {trace['info']}")
```

### `__add__(other)`
Performs high-precision addition using mpmath.

**Example:**
```python
engine1 = BasicArithmeticEngine(segment_mgr)
engine1._value = "10"

result_engine = engine1 + 5
print(result_engine._value)  # Output: 15.0
```

### `__mul__(other)`
Performs high-precision multiplication using mpmath.

**Example:**
```python
engine2 = BasicArithmeticEngine(segment_mgr)
engine2._value = "7"

result_engine = engine2 * 6
print(result_engine._value)  # Output: 42.0
```

## Error Handling

The implementation includes robust error handling:
- Validates expression structure before parsing
- Checks for empty operands
- Provides meaningful error messages

**Example:**
```python
try:
    engine.compute('2+')  # Missing right operand
except ValueError as e:
    print(e)  # Output: Empty operands in expression: 2+
```

## Testing

Two comprehensive test suites are provided:

### `test_basic_arithmetic.py`
Tests core functionality of all methods:
- Addition computation
- Multiplication computation
- __add__ operator
- __mul__ operator
- High-precision calculations
- Segment manager integration

Run: `python test_basic_arithmetic.py`

### `test_integration.py`
Tests integration with the complete Cortex flow:
- Structure analysis
- Segment manager
- EngineWorker
- XOR String Compiler

Run: `python test_integration.py`

## Architecture Integration

### Parallel Flow
The engine integrates with the parallel computation flow:
1. Expression is parsed
2. Operations are computed using mpmath
3. Results are sent to segment manager
4. Segment manager handles parallel assembly

### Segment Manager
Each operation creates part orders:
```python
part_order = [{
    'part': 'result',
    'value': str(result),
    'bytes': str(result).encode('utf-8')
}]
```

These are registered with the segment manager for parallel processing.

### Debug Tracebacks
Step-wise information is recorded:
```python
{
    'step': 'computation',
    'info': 'Result: 5.0',
    'timestamp': 123.456
}
```

## Limitations

Current implementation:
- Supports only simple binary operations (one operator per expression)
- Handles addition (+) and multiplication (*) only
- Does not support complex expressions with multiple operators
- No operator precedence handling (by design - uses modular approach)

Future enhancements could add:
- Subtraction, division, and power operations
- Support for parentheses
- Multiple operators in single expression

## Performance

Using `mpmath` provides:
- Arbitrary precision (configurable)
- Consistent results across platforms
- Avoidance of floating-point errors

Trade-off: Slightly slower than native float operations for simple calculations, but much more accurate for high-precision work.

## Dependencies

- `mpmath`: High-precision arithmetic
- `asyncio`: Async operation support
- `segment_manager`: Integration with Cortex architecture

## Notes

- All numeric operations use `mpmath.mpf()` for conversion
- Results are returned as mpmath objects (can be converted to string/int as needed)
- The engine maintains state via `_value` attribute for chaining operations
- Traceback info accumulates across operations on the same instance
