# Proposal: Source-Based Function Serialization for Cross-Version Compatibility

**Status:** Implemented in [davidbau/cloudpickle](https://github.com/davidbau/cloudpickle/tree/source-serial) fork

## Summary

Add an optional source-based serialization mode to cloudpickle that serializes functions by their source code instead of bytecode, enabling cross-Python-version compatibility.

## Motivation

### The Problem

Python bytecode is version-specific. Functions serialized with cloudpickle on Python 3.10 may fail to deserialize on Python 3.11 due to bytecode format changes. This is a significant limitation for distributed computing scenarios where:

- Client and server may run different Python versions
- Long-lived serialized functions may outlast Python version upgrades
- Debugging serialized functions is difficult (bytecode is not human-readable)

### Real-World Use Cases

1. **Distributed ML frameworks** (Ray, Dask, nnsight/NDIF): Users on Python 3.10 laptops submit jobs to servers running Python 3.11+

2. **Workflow persistence**: Serialized workflows stored in databases need to survive Python upgrades

3. **Debugging**: When serialization fails, developers need to understand what was captured

### Current Workaround

Users must subclass `cloudpickle.Pickler` and override `_dynamic_function_reduce`, reimplementing significant logic. This is fragile and duplicates cloudpickle internals.

## Proposed Solution

### New Pickler Parameters

```python
class Pickler(pickle.Pickler):
    def __init__(
        self,
        file,
        protocol=None,
        buffer_callback=None,
        # NEW PARAMETERS:
        by_source: bool = False,
        pickle_by_reference: Optional[Callable[[Any], bool]] = None,
    ):
        ...
```

### Parameter: `by_source`

Controls how dynamic functions are serialized:

- `False` (default): Current behavior - serialize code objects (bytecode)
- `True`: Serialize function source code via `inspect.getsource()`

When `True`:
- Source code is captured and stored as a string
- On deserialization, source is recompiled with `compile()` and `exec()`
- Function metadata (defaults, annotations, closure, globals) handled same as bytecode mode

### Parameter: `pickle_by_reference`

A callable that determines whether an object should be pickled by reference (import path) or by value:

```python
def pickle_by_reference(obj: Any) -> bool:
    """Return True to pickle by reference, False to pickle by value."""
    ...
```

If `None` (default), uses current behavior (`_should_pickle_by_reference`).

**Use case**: In distributed computing, you often know what packages exist on the remote end. A whitelist allows:

```python
REMOTE_PACKAGES = {"torch", "numpy", "transformers", ...}

def my_checker(obj):
    module = getattr(obj, "__module__", None)
    if module is None:
        return False
    top_level = module.split(".")[0]
    return top_level in REMOTE_PACKAGES

cloudpickle.dumps(obj, by_source=True, pickle_by_reference=my_checker)
```

## Implementation Details

The implementation (in the fork) handles several challenging cases:

### 1. Lambda Source Extraction

When multiple lambdas share a line, `inspect.getsource()` returns the entire line. The implementation uses `co_positions()` (Python 3.11+) and tokenization to extract just the target lambda:

```python
f, g = lambda x: x * 2, lambda x: x + 1  # Each serializes correctly
```

Handles:
- Multiple lambdas on same line
- Nested lambdas (`lambda x: lambda y: x + y`)
- Lambdas with lambda defaults (`lambda x=lambda: 1: x()`)
- Multi-line lambdas

### 2. Recursion Handling

Uses pickle's 6-tuple reduce protocol with a state setter that runs AFTER memoization:

```python
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)  # Self-reference in __globals__
```

Handles:
- Direct recursion (factorial)
- Local recursion (nested functions calling themselves)
- Module-level mutual recursion (is_even/is_odd)
- **Local mutual recursion** (two closures referencing each other)

The key insight: defer ALL function values in closures to the state setter, allowing pickle's memo to break cycles.

### 3. Closure Reconstruction

Uses a factory function pattern to properly bind closure variables:

```python
# For: lambda x: x * multiplier (where multiplier is captured)
def _cloudpickle_factory_(multiplier):
    return lambda x: x * multiplier
func = _cloudpickle_factory_(captured_multiplier_value)
```

### 4. Lambda Name Handling

Lambdas have `<lambda>` as their name, which isn't valid Python syntax. The implementation assigns to a temp variable:

```python
def _cloudpickle_factory_(closure_var):
    _lambda_result_ = lambda x: x * closure_var
    return _lambda_result_
```

## API Changes

### dumps/dump functions

```python
def dumps(obj, protocol=None, buffer_callback=None,
          by_source=False, pickle_by_reference=None):
    """
    Args:
        by_source: If True, serialize functions by source code instead of bytecode
        pickle_by_reference: Optional callback to control reference vs value serialization
    """
```

### Pickler class

```python
pickler = cloudpickle.Pickler(
    file,
    by_source=True,
    pickle_by_reference=my_whitelist_checker
)
```

## Compatibility

- **Backward compatible**: Default behavior unchanged (`by_source=False`)
- **Forward compatible**: Source-serialized functions can be deserialized by any cloudpickle version that supports the feature
- **Cross-version**: Source-serialized functions work across Python versions (as long as syntax is compatible)

## Limitations

1. **Source must be available**: Functions defined in interactive sessions or generated dynamically may not have source. Users can attach source manually via `func.__source__ = "..."`.

2. **Syntax compatibility**: Source using Python 3.12+ syntax won't work on 3.10. This is inherent to source-based approaches.

3. **Slight overhead**: Source strings are typically larger than bytecode, and recompilation adds deserialization time.

4. **Line numbers preserved**: The implementation prepends blank lines to align with original file, so tracebacks show correct line numbers.

5. **Python 3.11+ for precise lambda extraction**: Older versions fall back to full source, which works for simple cases but may fail for multiple lambdas on same line.

## Alternatives Considered

### 1. External Tool

A separate library could wrap cloudpickle. However:
- Duplicates cloudpickle internals (700+ lines in nnsight's workaround)
- Fragile to cloudpickle updates
- Users must choose between tools

### 2. Bytecode Translation

Translate bytecode between Python versions. However:
- Extremely complex
- Bytecode changes are not always mappable
- Maintenance burden across Python versions

### 3. AST-Based Serialization

Serialize the AST instead of source. However:
- AST format also changes between versions
- More complex than source strings
- Less human-readable

## Testing

The fork includes comprehensive tests:
- 54 edge case tests for various function patterns
- Lambda tests (simple, closures, multiple on line, nested, defaults)
- Recursion tests (direct, local, mutual, local mutual)
- Closure tests (single, nested, triple-nested, mutable state)
- Import tests (import inside function)
- Signature tests (defaults, *args, **kwargs, annotations)

## References

- [Fork implementation](https://github.com/davidbau/cloudpickle/tree/source-serial)
- [Python bytecode changes in 3.11](https://docs.python.org/3.11/whatsnew/3.11.html#cpython-bytecode-changes)
- [nnsight](https://github.com/ndif-team/nnsight) - Production use case with NDIF

## Open Questions

1. Should `by_source=True` be the default in some future version?

2. Should there be a hybrid mode that tries source first, falls back to bytecode?

3. How should we handle functions where `inspect.getsource()` fails but bytecode is available?

---

*This proposal is based on production experience with [nnsight](https://github.com/ndif-team/nnsight), which uses source-based serialization for cross-version compatibility with the NDIF distributed inference platform.*
