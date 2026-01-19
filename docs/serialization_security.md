# Serialization Security: Module Blacklist

This document describes the security linting feature in nnsight's serialization system that catches dangerous code patterns before they're sent to NDIF servers.

## Overview

When you use nnsight to run interventions remotely on NDIF, your Python functions are serialized and sent to the server for execution. The serialization system includes a **lint-time security check** that raises `PicklingProhibitedError` when your code references modules or functions that could be dangerous on a shared server.

This is an early warning system—like a linter—not a sandbox. Real sandboxing happens server-side. The goal is to give you helpful feedback before your code is rejected by the server.

## What Gets Caught

The blacklist catches **direct references** to prohibited modules in your function's globals or closure:

```python
# CAUGHT: module in function globals
import os
def my_func():
    return os.getcwd()  # PicklingProhibitedError raised

# CAUGHT: function from prohibited module
from subprocess import run
def my_func():
    return run(["ls"])  # PicklingProhibitedError raised

# CAUGHT: module captured in closure
def outer():
    import socket
    def inner():
        return socket.gethostname()  # PicklingProhibitedError raised
    return inner

# NOT CAUGHT: import inside function body (caught server-side instead)
def my_func():
    import os  # Not caught at serialization time
    return os.getcwd()
```

## Prohibited Modules

The following categories of modules are prohibited:

| Category | Modules | Why Prohibited |
|----------|---------|----------------|
| Process execution | `os`, `subprocess`, `pty` | Can run arbitrary commands |
| File system | `shutil`, `pathlib`, `glob`, `io`, `tempfile` | Can read/write/delete files |
| Network | `socket`, `ssl`, `http`, `urllib`, `ftplib` | Can exfiltrate data or attack other systems |
| Code execution | `code`, `runpy`, `importlib`, `pickle` | Can execute arbitrary code |
| System manipulation | `sys`, `signal`, `threading`, `multiprocessing` | Can affect server state |
| Low-level access | `ctypes`, `gc`, `inspect` | Can bypass Python safety |

See `PROHIBITED_MODULES` in `serialization.py` for the complete list.

## Prohibited Builtins

Certain builtin functions are also prohibited when directly referenced:

- `eval`, `exec`, `compile` — arbitrary code execution
- `open` — file access
- `__import__` — dynamic imports
- `globals`, `locals`, `vars` — namespace access
- `getattr`, `setattr`, `delattr` — attribute manipulation

```python
# CAUGHT: builtin in function globals
fn = eval
def my_func(s):
    return fn(s)  # PicklingProhibitedError raised

# NOT CAUGHT: implicit builtin usage (caught server-side)
def my_func(s):
    return eval(s)  # eval resolved at runtime, not in globals
```

## Error Messages

When a prohibited reference is detected, you'll get a helpful error message:

```
PicklingProhibitedError: Function 'my_func' references prohibited module 'os'.
Code sent to NDIF servers cannot use modules that perform file system operations,
process execution, network access, or other potentially dangerous operations.
If you need this functionality, consider restructuring your code to perform
these operations locally before/after the trace.
```

## How to Fix

Restructure your code to perform prohibited operations **outside** the traced function:

```python
# BEFORE: prohibited
import os

with model.trace(input):
    def my_intervention(x):
        path = os.getcwd()  # Error!
        return x * 2
    output.save()

# AFTER: fixed
import os
path = os.getcwd()  # Do it locally, before the trace

with model.trace(input):
    def my_intervention(x):
        return x * 2  # Clean intervention
    output.save()
```

## Safe Modules

The following modules are explicitly allowed and won't trigger errors:

- **ML/Scientific**: `torch`, `numpy`, `scipy`, `transformers`, `accelerate`, `einops`
- **Standard library utilities**: `math`, `random`, `functools`, `itertools`, `operator`, `collections`, `dataclasses`, `typing`, `re`, `json`
- **nnsight itself**: `nnsight`

## Implementation Details

The security check happens in `CustomCloudPickler._dynamic_function_reduce()` during serialization:

1. **Globals check**: Iterates over the function's captured globals looking for prohibited modules or functions
2. **Closure check**: Iterates over closure cell values for nested functions that capture prohibited modules from enclosing scopes

Both checks run before any data is sent to the server, giving you immediate feedback.
