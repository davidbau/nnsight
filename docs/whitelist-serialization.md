# Whitelist-Based Source Serialization

**Commit:** 9c47d20
**Date:** 2026-01-18

## Goal

Automatically serialize user-defined functions by source code without requiring explicit registration. Previously, cloudpickle's `register_pickle_by_value(mymodule)` had to be called for user code to work with remote execution. The whitelist approach eliminates that requirement.

## Design

### The Problem

Standard pickle serializes functions by reference (module + qualified name). This works when the same module exists on both client and server, but fails for user-defined modules that don't exist on NDIF servers.

### The Solution

Use a **whitelist** of modules known to exist on the server. Functions from:
- **Whitelisted modules** (torch, numpy, transformers, etc.) → Serialize by reference
- **Non-whitelisted modules** (user code) → Serialize by source code

This leverages cloudpickle's existing source-based serialization, but changes the decision logic for when to use it.

## Implementation

### Server Modules Whitelist

```python
SERVER_MODULES_WHITELIST = frozenset({
    # Python standard library
    "abc", "ast", "asyncio", "collections", "functools", ...

    # Core ML/scientific packages
    "torch", "numpy", "scipy",

    # HuggingFace ecosystem
    "transformers", "tokenizers", "accelerate", "datasets", ...

    # nnsight itself
    "nnsight",

    # Common utilities
    "tqdm", "einops", "packaging",
})
```

### Patching cloudpickle

The key is patching `cloudpickle._should_pickle_by_reference`:

```python
def _patched_should_pickle_by_reference(obj, name=None):
    """Only pickle by reference for whitelisted modules."""
    if isinstance(obj, types.FunctionType):
        module_and_name = _lookup_module_and_qualname(obj, name=name)
        if module_and_name is None:
            return False
        module, _ = module_and_name
        module_name = getattr(module, "__name__", None)
        return _is_whitelisted_module(module_name)
    # ... similar for classes and modules

# Apply patch at module load time
_cloudpickle_internal._should_pickle_by_reference = _patched_should_pickle_by_reference
```

### How Whitelist Checking Works

A module is whitelisted if its **top-level package** is in the whitelist:
- `torch` → whitelisted (exact match)
- `torch.nn.functional` → whitelisted (`torch` is top-level)
- `myproject.utils` → NOT whitelisted

```python
def _is_whitelisted_module(module_name: str) -> bool:
    dot_idx = module_name.find(".")
    top_level = module_name[:dot_idx] if dot_idx != -1 else module_name
    return top_level in SERVER_MODULES_WHITELIST
```

## Usage

User-defined modules are automatically serialized by source - no registration needed:

```python
import mymethods

with model.trace("Hello", remote=True):
    hidden = mymethods.normalize(model.transformer.h[0].output[0])
    # Just works! mymethods is automatically serialized by source
```

Previously, cloudpickle's `register_pickle_by_value()` had to be called explicitly for each user module. The whitelist approach eliminates this requirement.

## Limitations

1. **Source code must be available**: Functions defined in interactive sessions (REPL, Jupyter without `%%file`) may not have extractable source.

2. **All dependencies must be capturable**: If a user function calls another user function, both must be serializable. Cloudpickle handles this by recursively capturing dependencies.

3. **No dynamic code generation**: Functions created with `exec()` or `eval()` without explicit `__source__` attributes will fail.

4. **Whitelist maintenance**: New server packages require whitelist updates.

5. **Global state**: Module-level mutable state is captured at serialization time, not shared with the original.

## Testing

See `tests/test_whitelist_serialization.py` for comprehensive tests including:
- Basic function serialization
- Classes with methods
- Closures and nested functions
- Module-level state
- Inheritance hierarchies
- Interaction with whitelisted modules (torch, numpy)
