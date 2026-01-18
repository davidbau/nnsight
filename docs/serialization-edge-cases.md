# Serialization Edge Case Tests

**Commit:** 9ac2de9
**Date:** 2026-01-18

## Goal

Verify that source-based serialization correctly handles various Python language constructs. These tests ensure that the serialization system is robust enough for real-world user code.

## Test Categories

### 1. Closures and Captured Variables

Functions that capture variables from enclosing scopes:

```python
def test_closure_with_captured_variable():
    multiplier = 3.0
    def scale_by_closure(x):
        return x * multiplier
    # Verify multiplier is captured and preserved
```

**Tested scenarios:**
- Single-level closures
- Nested closures (function inside function)
- Returned inner functions (closure factories)
- Triple-nested closures
- Mutable closure state (lists/dicts modified by closure)

### 2. Default Arguments

```python
def test_lambda_default_argument():
    def func_with_lambda_default(processor=lambda x: x * 2):
        return processor(5)
    # Lambda in default must be preserved
```

**Tested scenarios:**
- Lambda as default argument
- Complex default types (lists, dicts, None)
- Mix of positional and keyword defaults

### 3. Variable Scoping

```python
def test_comprehension_variable_shadowing():
    x = 42  # External
    def uses_external_and_comprehension():
        external_value = x  # Uses external x
        squares = [x * x for x in range(5)]  # Comprehension x is local
        return external_value, squares
```

**Tested scenarios:**
- Comprehension variables vs external variables
- Lambda parameters vs external references
- Nested function parameters shadowing outer variables
- Module variables overriding builtins

### 4. Special Function Types

```python
def test_async_function():
    async def async_double(x):
        await asyncio.sleep(0)
        return x * 2

def test_generator_function():
    def generate_numbers(n):
        for i in range(n):
            yield i * 2
```

**Tested scenarios:**
- Async functions (`async def`)
- Generator functions (`yield`)
- Functions with `*args` and `**kwargs`

### 5. Imports Inside Functions

```python
def test_import_inside_function():
    def func_with_import():
        import json
        return json.dumps([1, 2, 3])
```

**Tested scenarios:**
- `import` statements inside functions
- `from ... import` statements inside functions

### 6. Function Metadata

```python
def test_function_with_annotations():
    def annotated_func(x: int, y: float = 1.0) -> float:
        return x * y
    # Annotations must be preserved
```

**Tested scenarios:**
- Type annotations
- Docstrings
- Function `__name__`, `__qualname__`, `__module__`

### 7. Function References

```python
def test_function_referencing_other_function():
    def helper(x):
        return x + 1
    def main_func(x):
        return helper(x) * 2
    # Both functions must be captured
```

**Tested scenarios:**
- Functions calling other user-defined functions
- Multiple functions sharing the same closure variable

### 8. Integration with ML Libraries

```python
def test_torch_operations():
    from mymethods.stateful import normalize
    # User function that uses torch internally
```

**Tested scenarios:**
- User functions using torch operations
- Functions used inside model traces

## Known Limitations (at time of commit)

1. **Recursive functions**: Self-referencing functions caused infinite recursion during serialization. (Fixed in subsequent commit)

2. **Async in test context**: Async functions defined inside pytest test functions may have source extraction issues due to pytest's internal handling.

## Test File

`tests/test_serialization_edge_cases.py` - 580 lines, 28 test cases

## Running Tests

```bash
pytest tests/test_serialization_edge_cases.py -v
```
