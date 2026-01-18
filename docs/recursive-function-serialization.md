# Recursive Function Serialization Fix

**Commit:** (pending)
**Date:** 2026-01-18

## Goal

Enable serialization of recursive functions, including:
- **Direct recursion**: `factorial` calling itself
- **Mutual recursion**: `is_even` and `is_odd` calling each other (module-level)
- **Local recursion**: Nested functions that call themselves via closure
- **Local mutual recursion**: Two local functions that reference each other through closures

## The Problem

### Direct Recursion Failure

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # References itself in __globals__
```

When serializing `factorial`:
1. Pickle calls `_dynamic_function_reduce(factorial)`
2. We capture `factorial.__globals__`, which contains `{"factorial": <function factorial>}`
3. Pickle tries to serialize the globals dict
4. It encounters `factorial` again → infinite recursion

### Local Recursion Failure

```python
def outer():
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)  # References itself via closure
    return factorial
```

Here, `factorial` references itself through a **closure cell**, not globals. The same infinite recursion occurs when serializing closure values.

### Local Mutual Recursion Failure

```python
def outer():
    def is_even(n):
        if n == 0: return True
        return is_odd(n - 1)  # References is_odd via closure
    def is_odd(n):
        if n == 0: return False
        return is_even(n - 1)  # References is_even via closure
    return is_even, is_odd
```

Here, `is_even` references `is_odd` and vice versa through closure cells. This creates a cycle that causes infinite recursion during serialization.

## The Solution: Deferred State Application

We adopt pickle's **6-tuple reduce protocol** with a state setter, following cloudpickle's pattern for handling circular references.

### How Pickle's Memo Works

Pickle maintains a "memo" (cache) of already-serialized objects. When it encounters the same object again, it emits a reference to the memo entry instead of re-serializing.

The key insight: **state setters run AFTER memoization**. So:
1. Create object with minimal state
2. Pickle memoizes the object
3. State setter fills in full state (including self-references)
4. Self-references now resolve to the memoized object

### The 6-Tuple Reduce Protocol

```python
# Old: 5-tuple
(make_function, args, state, None, None)

# New: 6-tuple with state setter
(make_function, args, state, None, None, _source_function_setstate)
```

The 6th element is a callable that receives `(obj, state)` after the object is created and memoized.

## Implementation

### Phase 1: Create Function with Minimal State

`_dynamic_function_reduce` now returns:

```python
args = (
    source,           # Function source code
    name,             # __name__
    filename,         # co_filename
    qualname,         # __qualname__
    module_name,      # __module__
    doc,              # __doc__
    annotations,      # __annotations__
    defaults,         # __defaults__
    kwdefaults,       # __kwdefaults__
    base_globals,     # MINIMAL: just __package__, __name__, __path__, __file__
    closure_values,   # With ALL function refs replaced by None placeholders
    closure_names,    # co_freevars
)

state = (func_dict, {
    "__globals__": full_globals,      # INCLUDING self-references
    "__deferred_closure__": {idx: target_func, ...},  # Maps indices to target functions
})

return (make_function, args, state, None, None, _source_function_setstate)
```

### Phase 2: State Setter Fills in References

```python
def _source_function_setstate(func, state):
    func_dict, slotstate = state

    # Update func.__dict__
    func.__dict__.update(func_dict)

    # Fill in full globals (including self-references)
    # The function is already memoized, so self-refs resolve correctly!
    func.__globals__.update(slotstate["__globals__"])

    # Fill in deferred closure cells (for recursive/mutually recursive local functions)
    deferred = slotstate.get("__deferred_closure__", {})
    for idx, target_func in deferred.items():
        func.__closure__[idx].cell_contents = target_func  # Could be self or another function
```

### Handling Different Recursion Types

| Type | Reference Location | Solution |
|------|-------------------|----------|
| Module-level recursion | `__globals__["factorial"]` | Defer globals to state setter |
| Module-level mutual recursion | `__globals__["is_odd"]` | Both functions memoized before state applied |
| Local recursion | `__closure__[i]` (self) | Replace with None, fill in state setter |
| Local mutual recursion | `__closure__[i]` (other func) | Replace ALL functions with None, fill in state setter |

The key insight for local mutual recursion: we defer **ALL** function values in closures (not just self-references), allowing pickle's memo to break cycles. The deferred_closure dict maps closure indices to the actual target functions.

## Code Flow Example

For `factorial`:

```
1. _dynamic_function_reduce(factorial)
   - base_globals = {"__name__": "__main__", ...}  # No factorial!
   - state.__globals__ = {"factorial": <func>, "other": ...}

2. Pickle serializes args (no self-ref, no recursion)

3. Pickle creates function via make_function(args)
   - Function exists but __globals__ doesn't have "factorial" yet

4. Pickle MEMOIZES the function
   - memo[id] = factorial

5. Pickle serializes state
   - state.__globals__["factorial"] → found in memo! Emits reference.

6. Pickle calls _source_function_setstate(func, state)
   - func.__globals__.update(state.__globals__)
   - Now factorial.__globals__["factorial"] = factorial ✓
```

For local mutual recursion (`is_even` and `is_odd`):

```
1. _dynamic_function_reduce(is_even)
   - closure_values = [None]  # is_odd deferred
   - deferred_closure = {0: is_odd}
   - state.__globals__ = {...}

2. Pickle serializes args for is_even (is_odd deferred, no recursion yet)

3. Pickle creates is_even via make_function(args)

4. Pickle MEMOIZES is_even

5. Pickle serializes state for is_even
   - deferred_closure contains is_odd → pickle processes is_odd

6. _dynamic_function_reduce(is_odd) runs
   - closure_values = [None]  # is_even deferred
   - deferred_closure = {0: is_even}  # is_even already memoized!

7. Pickle creates is_odd, memoizes it

8. Pickle serializes state for is_odd
   - deferred_closure[0] = is_even → found in memo! Reference emitted.

9. _source_function_setstate(is_odd, state)
   - is_odd.__closure__[0].cell_contents = is_even ✓

10. _source_function_setstate(is_even, state)
    - is_even.__closure__[0].cell_contents = is_odd ✓
```

## Changes Summary

**`src/nnsight/intervention/serialization.py`**:

1. **`make_function`**: Now accepts minimal `base_globals` and `closure_values` (with placeholders for function refs). Uses factory pattern for closures.

2. **`_source_function_setstate`** (new): Applies full globals and fills deferred closure cells after memoization. Now stores actual target functions (not just True).

3. **`_dynamic_function_reduce`**: Returns 6-tuple. Defers ALL function values in closures (not just self-references). Stores actual target functions in deferred_closure dict.

## Limitations

1. **Source must be available**: The underlying source-based serialization requirement remains.

2. **Non-function closure values bound immediately**: Unlike functions, non-function closure values are bound at `make_function` time. This is necessary because the factory pattern requires values to create the closure.

## Testing

- `tests/test_serialization_edge_cases.py::test_recursive_function` - Direct recursion
- `tests/test_local_recursion.py` - Local/nested recursive functions (self-recursion)
- `tests/test_mutual_recursion.py` - Module-level mutual recursion (is_even/is_odd)
- `tests/test_local_mutual_recursion.py` - Local mutual recursion (now supported!)

```bash
pytest tests/test_serialization_edge_cases.py tests/test_local_recursion.py tests/test_mutual_recursion.py tests/test_local_mutual_recursion.py -v
```
