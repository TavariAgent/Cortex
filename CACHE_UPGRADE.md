# Segment Manager and Engine Cache Upgrade

## Summary of Changes

This document describes the enhancements made to the SegmentManager and Engine classes to support cache-driven ordering and improved segment management.

## Changes to SegmentManager (segment_manager.py)

### 1. Added segment_pools
- **Type**: `dict` (engine_name -> list of packed byte arrays)
- **Purpose**: Store packed byte arrays from engines for cache-driven processing
- **Location**: Initialized in `__init__`

### 2. Added completion_flags
- **Type**: `dict` (engine_name -> bool)
- **Purpose**: Track completion status of each engine
- **Location**: Initialized in `__init__`

### 3. Added receive_packed_segment Method
```python
def receive_packed_segment(self, engine_name, packed_bytes):
    """Accumulate packed byte arrays from engines."""
    if engine_name not in self.segment_pools:
        self.segment_pools[engine_name] = []
    self.segment_pools[engine_name].append(packed_bytes)
```
- **Purpose**: Accumulate packed byte segments from engines
- **Parameters**: 
  - `engine_name`: Name of the engine sending the segment
  - `packed_bytes`: Byte array to be stored

### 4. Enhanced _dropdown_assemble
- **Added**: Priority map for operator precedence ordering
- **Priority Levels**:
  - `^` (exponentiation): 4 (highest)
  - `*`, `/` (multiply/divide): 3
  - `+`, `-` (add/subtract): 2
  - `sum`, `limit`: 1
  - `other`: 0 (lowest)
- **Behavior**: Sorts operations by priority (descending) then left-to-right
- **Integration**: Uses existing `_determine_level` for structural nesting

### 5. Added all_complete Method
```python
def all_complete(self):
    """Check if all engines have completed their work."""
    if not self.completion_flags:
        return False
    return all(self.completion_flags.values())
```
- **Purpose**: Check if all registered engines have completed processing
- **Returns**: `True` if all flags are True, `False` otherwise

## Changes to MathEngine and BasicArithmeticEngine (abc_engines.py)

### 1. Added _cache to MathEngine.__init__
```python
def __init__(self, segment_manager):
    self.segment_manager = segment_manager
    self.parallel_tasks = []
    self._cache = []  # Cache for packed bytes before sending to segment_manager
```
- **Type**: `list` of byte arrays
- **Purpose**: Accumulate packed bytes before sending to segment_manager
- **Inheritance**: Available to all engine subclasses

### 2. Updated BasicArithmeticEngine._tokenize
- **Enhanced**: Now handles parentheses (both regular and function parens)
- **Supports**:
  - Regular parentheses for nesting: `(2+3)*4`
  - Function parentheses: `sin(3.14)`, `cos(0)`
  - Nested parentheses: `((2+3)*4)`
- **Delegation**: Operator ordering delegated to Structure/priority map via SegmentManager._dropdown_assemble

### 3. Updated __add__ Method
- **Added**: Cache accumulation before sending to segment_manager
- **Flow**:
  1. Perform addition using mpmath
  2. Convert result to packed bytes
  3. Append to `_cache`
  4. Send to `segment_manager.receive_packed_segment()`
  5. Send to `segment_manager.receive_part_order()` (for compatibility)
  6. Return new engine instance with propagated cache

### 4. Updated __mul__ Method
- **Added**: Cache accumulation before sending to segment_manager
- **Flow**: Same as `__add__` but for multiplication

### 5. Updated compute Method
- **Added**: Cache accumulation for computed results
- **Flow**:
  1. Parse and evaluate expression
  2. Convert result to packed bytes
  3. Append to `_cache`
  4. Send to `segment_manager.receive_packed_segment()`
  5. Send to `segment_manager.receive_part_order()` (for compatibility)

## Integration with Tracebacks

All changes maintain integration with the existing traceback system:
- Traceback entries recorded at each step
- Byte strings used throughout (avoiding floats as per guidelines)
- Cache propagation included in traceback context

## Testing

### New Test Files

1. **test_segment_cache.py**
   - Tests segment_pools functionality
   - Tests completion_flags functionality
   - Tests engine cache mechanism
   - Tests parentheses tokenization
   - Tests priority ordering

2. **test_integration_cache.py**
   - Integration tests for cache and segment_pools
   - Tests chained operations with cache propagation
   - Tests completion flag workflows

### Test Results
- All new functionality tests pass ✓
- All existing tests continue to pass ✓
- Backward compatibility maintained ✓

## Usage Example

```python
from abc_engines import BasicArithmeticEngine
from segment_manager import SegmentManager
from main import Structure

# Initialize
struct = Structure()
segment_mgr = SegmentManager(struct)
engine = BasicArithmeticEngine(segment_mgr)

# Perform computation
result = engine.compute('3*4-5+1+6/2+2')
# Result: 13.0

# Check cache
print(engine._cache)  # [b'13.0']

# Check segment_pools
print(segment_mgr.segment_pools['BasicArithmeticEngine'])
# [b'13.0']

# Set completion flag
segment_mgr.completion_flags['BasicArithmeticEngine'] = True

# Check if all complete
print(segment_mgr.all_complete())  # True
```

## Benefits

1. **Cache-Driven Ordering**: Engines accumulate results in cache before sending to segment manager
2. **Priority-Based Assembly**: Operations ordered by precedence (PEMDAS) with priority map
3. **Completion Tracking**: Track when engines finish processing
4. **Parentheses Support**: Enhanced tokenizer handles nested and function parentheses
5. **Byte String Integration**: All results stored as byte strings (avoiding floats)
6. **Backward Compatibility**: All existing functionality preserved

## Notes

- The `_cache` is propagated through chained operations (e.g., `(engine + 3) * 2`)
- Both `receive_packed_segment` and `receive_part_order` are called for compatibility
- Priority map in `_dropdown_assemble` matches the one in `MathEngine._set_part_order`
- All numeric operations use mpmath to avoid float precision issues
