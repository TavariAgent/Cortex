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
Parses and evaluates arithmetic expressions with full PEMDAS support.

**Supported Operations:**
- Addition: `"2+3"` → `5.0`
- Subtraction: `"10-5"` → `5.0`
- Multiplication: `"3*4"` → `12.0`
- Division: `"10/2"` → `5.0`
- Complex expressions with proper order of operations: `"3*4-5+1+6/2+2"` → `13.0`

**Order of Operations:**
- Multiplication and division are evaluated first (left-to-right)
- Addition and subtraction are evaluated second (left-to-right)
- Follows standard PEMDAS rules

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

# Complex expression with PEMDAS
result = engine.compute('3*4-5+1+6/2+2')
print(f"Result: {result}")  # Output: Result: 13.0

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

## Implementation Details

### Tokenizer
The `_tokenize()` method breaks down arithmetic expressions into tokens:
- Numbers (including decimals)
- Operators (+, -, *, /)
- Handles whitespace gracefully

### Evaluator
The `_evaluate_tokens()` method processes tokens following PEMDAS:
1. **First pass**: Processes multiplication and division left-to-right
2. **Second pass**: Processes addition and subtraction left-to-right
3. All calculations use `mpmath.mpf()` for high precision

## Limitations

Current implementation:
- Supports addition (+), subtraction (-), multiplication (*), and division (/)
- Handles full PEMDAS order of operations for these operators
- Does not support parentheses
- Does not support exponentiation (^ or **)

Future enhancements could add:
- Support for parentheses
- Exponentiation operations
- More advanced mathematical functions

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
