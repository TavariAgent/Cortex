# Quick Start Guide: BasicArithmeticEngine

## Installation

Ensure dependencies are installed:
```bash
pip install sympy numpy mpmath
```

## Basic Usage

### 1. Simple Expression Computation

```python
from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure

# Setup
struct = Structure()
segment_mgr = SegmentManager(struct)
engine = BasicArithmeticEngine(segment_mgr)

# Compute addition
result = engine.compute('2+3')
print(result)  # Output: 5.0

# Compute multiplication
result = engine.compute('3*4')
print(result)  # Output: 12.0
```

### 2. Using Operators

```python
# Addition operator
engine._value = "10"
result = engine + 5
print(result._value)  # Output: 15.0

# Multiplication operator
engine._value = "7"
result = engine * 6
print(result._value)  # Output: 42.0
```

### 3. Viewing Debug Tracebacks

```python
engine = BasicArithmeticEngine(segment_mgr)
result = engine.compute('2+3')

# Print traceback
for trace in engine.traceback_info:
    print(f"{trace['step']}: {trace['info']}")
```

Output:
```
compute_start: Expression: 2+3
parse: Addition detected: ['2', '3']
conversion: Converted to mpmath: 2.0, 3.0
computation: Result: 5.0
```

## Running Tests

```bash
# Basic functionality tests
python test_basic_arithmetic.py

# Integration tests
python test_integration.py
```

## Key Features

✅ High precision using mpmath (no floats/Decimals)  
✅ Step-wise debug tracebacks  
✅ Segment manager integration  
✅ Error handling and validation  
✅ Async-compatible  

## Supported Operations

- Addition: `"2+3"` → `5.0`
- Multiplication: `"3*4"` → `12.0`

## Error Handling

```python
try:
    result = engine.compute('2+')  # Invalid expression
except ValueError as e:
    print(e)  # Empty operands in expression: 2+
```

## Documentation

- `IMPLEMENTATION.md` - Detailed implementation guide
- `SUMMARY.md` - Complete implementation summary
- `README.md` - Project overview

## Next Steps

See `IMPLEMENTATION.md` for advanced usage and architecture details.
