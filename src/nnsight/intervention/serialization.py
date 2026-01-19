"""Source-based function serialization for cross-version compatibility.

This module provides a custom serialization system built on top of cloudpickle
that serializes Python functions by their source code rather than bytecode.

Why source-based serialization?
    Standard pickle/cloudpickle serialize functions using Python bytecode, which
    is version-specific and can break when deserializing on a different Python
    version. By serializing the source code instead, we can reconstruct functions
    on any Python version that supports the syntax, enabling cross-version
    compatibility for remote execution (e.g., client on Python 3.10, server on 3.11).

Key components:
    - CustomCloudPickler: Serializes functions by capturing their source code,
      closure variables, and metadata instead of bytecode.
    - CustomCloudUnpickler: Deserializes data with support for persistent object
      references (objects that shouldn't be serialized but looked up by ID).
    - make_function: Reconstructs a function from its serialized components.
    - dumps/loads: High-level API for serializing and deserializing objects
      (named to match the standard pickle module API).

Persistent objects:
    Some objects (like model proxies or tensors) shouldn't be serialized directly
    but instead referenced by ID and resolved at deserialization time. Objects
    with a `_persistent_id` attribute in their __dict__ are handled this way.

Example:
    >>> import serialization
    >>> def my_func(x, y=10):
    ...     return x + y
    >>> data = serialization.dumps(my_func)
    >>> restored = serialization.loads(data)
    >>> restored(5)  # Returns 15
"""

import inspect
import io
import pickle
import textwrap
import tokenize
import types
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import cloudpickle
import cloudpickle.cloudpickle as _cloudpickle_internal
from cloudpickle.cloudpickle import _function_getstate, _get_cell_contents

# Default pickle protocol - protocol 4 is available in Python 3.4+ and supports large objects
DEFAULT_PROTOCOL = 4


class PicklingProhibitedError(pickle.PicklingError):
    """Raised when serialization of an object is explicitly prohibited.

    This exception is raised during serialization when code references
    modules or functions that are on the security blacklist. This serves
    as a lint-time check to catch dangerous patterns before code is sent
    to remote servers.
    """
    pass

# =============================================================================
# Module Whitelist Configuration
# =============================================================================
#
# Only modules in this whitelist are assumed to be available on the server.
# Functions and classes from non-whitelisted modules will be serialized by
# source code (pickle-by-value) rather than by reference.
#
# This ensures that user-defined modules are automatically captured without
# requiring explicit register() calls.

SERVER_MODULES_WHITELIST = frozenset({
    # Python standard library (commonly used)
    # Note: Some of these modules (os, sys, io, etc.) are also on the blacklist.
    # They are whitelisted here so cloudpickle doesn't try to serialize them by
    # source (which would fail), but the blacklist check in _dynamic_function_reduce
    # will catch user code that references them.
    "abc", "ast", "asyncio", "_asyncio", "base64", "builtins", "collections",
    "contextlib", "copy", "dataclasses", "datetime", "decimal", "enum",
    "functools", "hashlib", "heapq", "io", "itertools", "json", "logging",
    "math", "operator", "os", "pathlib", "pickle", "random", "re", "secrets",
    "statistics", "string", "struct", "sys", "threading", "time",
    "typing", "typing_extensions", "unittest", "uuid", "warnings",
    "weakref", "zlib",
    # Core ML/scientific packages
    "torch", "numpy", "np", "scipy",
    # HuggingFace ecosystem
    "transformers", "tokenizers", "accelerate", "datasets", "huggingface_hub",
    "safetensors", "peft",
    # nnsight itself
    "nnsight",
    # Common utilities
    "tqdm", "einops", "packaging",
    # Serialization (cloudpickle is used internally and must be whitelisted)
    "cloudpickle",
})

# =============================================================================
# Module Blacklist Configuration (Lint-time Safety Check)
# =============================================================================
#
# Modules in this blacklist will raise PicklingProhibitedError when code
# directly references them. This is an early warning system (like a linter)
# to catch common dangerous patterns before code is sent to the server.
#
# This is NOT a complete sandbox - real sandboxing happens server-side.
# This lint check only catches direct module references in function globals:
#   - `import os; os.getcwd()` - CAUGHT (os is in function globals)
#   - `from os import getcwd; getcwd()` - CAUGHT (getcwd is in function globals)
#   - `def f(): import os; ...` - NOT caught (import inside function)
#
# The goal is to give users helpful early feedback, not to prevent all
# possible dangerous operations.

PROHIBITED_MODULES = frozenset({
    # ==========================================================================
    # Process execution - can run arbitrary commands on the server
    # ==========================================================================
    "subprocess",       # subprocess.Popen, subprocess.call, etc.
    "os",               # os.system, os.popen, os.spawn*, os.exec*, os.fork
    "pty",              # pty.spawn - can spawn shells
    "commands",         # Deprecated but dangerous (Python 2)
    "popen2",           # Deprecated but dangerous (Python 2)

    # ==========================================================================
    # File system access - can read/write/delete files on server
    # ==========================================================================
    "shutil",           # File operations: copy, move, delete, rmtree
    "pathlib",          # Path manipulation and file operations
    "glob",             # File globbing
    "fnmatch",          # Filename matching
    "tempfile",         # Creates temporary files
    "fileinput",        # File input operations
    "filecmp",          # File comparisons
    "linecache",        # Can read arbitrary files
    "io",               # Low-level I/O operations
    "_io",              # C implementation of io (includes open())

    # ==========================================================================
    # Archive extraction - can write files anywhere (path traversal attacks)
    # ==========================================================================
    "tarfile",          # Tar archives (CVE path traversal risks)
    "zipfile",          # Zip archives (path traversal, zip bombs)
    "gzip",             # Gzip compression/decompression
    "bz2",              # Bzip2 compression/decompression
    "lzma",             # LZMA compression/decompression
    "zipimport",        # Import from zip files

    # ==========================================================================
    # Network access - can make connections, exfiltrate data
    # ==========================================================================
    "socket",           # Raw socket access
    "_socket",          # C implementation of socket
    "ssl",              # SSL/TLS (requires socket)
    "http",             # HTTP client/server
    "urllib",           # URL handling and fetching
    "ftplib",           # FTP client
    "smtplib",          # SMTP client (email)
    "poplib",           # POP3 client (email)
    "imaplib",          # IMAP client (email)
    "nntplib",          # NNTP client (news)
    "telnetlib",        # Telnet client
    "xmlrpc",           # XML-RPC

    # ==========================================================================
    # Code execution - can execute arbitrary code
    # ==========================================================================
    "code",             # Interactive interpreter
    "codeop",           # Code compilation
    "compileall",       # Compile Python files
    "py_compile",       # Compile Python files
    "runpy",            # Run Python modules
    "importlib",        # Dynamic imports
    "imp",              # Deprecated import machinery
    "pkgutil",          # Package utilities (can import)

    # ==========================================================================
    # Serialization - can execute arbitrary code on deserialize
    # ==========================================================================
    "pickle",           # Arbitrary code execution on load
    "shelve",           # Uses pickle internally
    "marshal",          # Similar risks to pickle
    "dill",             # Extended pickle

    # ==========================================================================
    # System manipulation - can affect server state
    # ==========================================================================
    "sys",              # sys.exit, sys.path manipulation, etc.
    "signal",           # Signal handling
    "atexit",           # Exit handlers
    "threading",        # Thread creation
    "multiprocessing",  # Process creation
    "concurrent",       # Concurrent execution
    "_thread",          # Low-level threading

    # ==========================================================================
    # Low-level access - can bypass Python safety
    # ==========================================================================
    "ctypes",           # C interface - can do anything
    "cffi",             # C FFI - similar to ctypes
    "_ctypes",          # ctypes internals
    "gc",               # Garbage collector manipulation
    "inspect",          # Code introspection (sandbox escape risk)
    "dis",              # Bytecode disassembly
    "traceback",        # Stack inspection

    # ==========================================================================
    # Platform/OS specific - system information and manipulation
    # ==========================================================================
    "platform",         # Platform information
    "resource",         # Resource limits
    "sysconfig",        # System configuration
    "posix",            # POSIX interface
    "nt",               # Windows interface
    "msvcrt",           # Windows C runtime
    "winreg",           # Windows registry
    "pwd",              # Password database
    "grp",              # Group database
    "spwd",             # Shadow password database

    # ==========================================================================
    # Debugging - can inspect/modify running code
    # ==========================================================================
    "pdb",              # Debugger
    "bdb",              # Debugger base
    "faulthandler",     # Fault handler
    "trace",            # Tracing
    "profile",          # Profiling
    "cProfile",         # C profiling
    "pstats",           # Profile statistics

    # ==========================================================================
    # Builtins module - prohibit the module object itself
    # ==========================================================================
    # The builtins module object gives access to all builtins including
    # dangerous ones like eval, exec, open. We prohibit the module object
    # but allow individual safe builtins (int, str, list, etc.) via
    # PROHIBITED_BUILTINS below.
    "builtins",

})

# =============================================================================
# Prohibited Builtins (Specific Dangerous Functions)
# =============================================================================
#
# These specific builtin functions are prohibited even though most builtins
# (int, str, list, dict, etc.) are safe. This catches direct references to
# dangerous builtins in function globals.
#
# Note: This only catches cases where the builtin is explicitly referenced,
# not implicit usage. For example:
#   - `fn = eval; fn(...)` - CAUGHT (eval is in function globals)
#   - `eval(...)` - NOT caught (eval resolved at runtime from builtins)
#
# Real sandboxing of builtin access happens server-side.

PROHIBITED_BUILTINS = frozenset({
    "eval",             # Execute arbitrary code from string
    "exec",             # Execute arbitrary code
    "compile",          # Compile source to code object
    "__import__",       # Dynamic imports
    "open",             # File access
    "input",            # Interactive input (can hang server)
    "breakpoint",       # Debugger
    "help",             # Interactive help (can hang server)
    "memoryview",       # Low-level memory access
    "globals",          # Access to global namespace
    "locals",           # Access to local namespace
    "vars",             # Access to object's __dict__
    "dir",              # Object introspection
    "getattr",          # Attribute access (can bypass restrictions)
    "setattr",          # Attribute modification
    "delattr",          # Attribute deletion
})


def _get_top_level_module(module_name: str) -> str:
    """Extract the top-level module name from a fully qualified module name.

    Args:
        module_name: The fully qualified module name (e.g., 'torch.nn.functional')

    Returns:
        The top-level module name (e.g., 'torch'), or empty string if invalid.
    """
    if module_name is None or not isinstance(module_name, str):
        return ""
    dot_idx = module_name.find(".")
    if dot_idx == -1:
        return module_name
    return module_name[:dot_idx]


def _is_whitelisted_module(module_name: str) -> bool:
    """Check if a module is in the server whitelist.

    A module is considered whitelisted if it or any of its parent packages
    are in the whitelist. For example, if 'torch' is whitelisted, then
    'torch.nn' and 'torch.nn.functional' are also whitelisted.

    Args:
        module_name: The fully qualified module name (e.g., 'torch.nn.functional')

    Returns:
        True if the module is whitelisted, False otherwise.
    """
    top_level = _get_top_level_module(module_name)
    return top_level in SERVER_MODULES_WHITELIST if top_level else False


def _is_prohibited_module(module_name: str) -> bool:
    """Check if a module is in the prohibited blacklist.

    A module is considered prohibited if its top-level package is in the
    blacklist. For example, if 'os' is prohibited, then 'os.path' is also
    prohibited.

    Args:
        module_name: The fully qualified module name (e.g., 'os.path')

    Returns:
        True if the module is prohibited, False otherwise.
    """
    top_level = _get_top_level_module(module_name)
    return top_level in PROHIBITED_MODULES if top_level else False


# Store the original function for potential restoration
_original_should_pickle_by_reference = _cloudpickle_internal._should_pickle_by_reference

# Thread-local recursion guard to prevent infinite recursion
import threading
_recursion_guard = threading.local()


def _patched_should_pickle_by_reference(obj, name=None):
    """Patched version that only pickles by reference for whitelisted modules.

    This overrides cloudpickle's default behavior which pickles by reference
    for any importable module. Instead, we only pickle by reference for modules
    known to be available on the server (the whitelist).

    Non-whitelisted modules are pickled by value (source code), which triggers
    the CustomCloudPickler._dynamic_function_reduce method for source-based
    serialization.

    Note: Blacklist checks are done in CustomCloudPickler.reducer_override,
    not here, to avoid recursion issues with cloudpickle internals.
    """
    # Recursion guard - if we're already checking, defer to original
    if getattr(_recursion_guard, 'checking', False):
        return _original_should_pickle_by_reference(obj, name)

    _recursion_guard.checking = True
    try:
        if isinstance(obj, types.FunctionType):
            # For functions, check the module they're defined in
            module_and_name = _cloudpickle_internal._lookup_module_and_qualname(obj, name=name)
            if module_and_name is None:
                return False
            module, _ = module_and_name
            module_name = getattr(module, "__name__", None)
            return _is_whitelisted_module(module_name)
        elif isinstance(obj, type):
            # For classes, check the module they're defined in
            module_and_name = _cloudpickle_internal._lookup_module_and_qualname(obj, name=name)
            if module_and_name is None:
                return False
            module, _ = module_and_name
            module_name = getattr(module, "__name__", None)
            return _is_whitelisted_module(module_name)
        elif isinstance(obj, types.ModuleType):
            # For module objects, check the module name directly
            return _is_whitelisted_module(obj.__name__)
        else:
            raise TypeError(
                f"_patched_should_pickle_by_reference cannot check "
                f"importability of {type(obj).__name__} instances"
            )
    finally:
        _recursion_guard.checking = False


# Apply the patch at module load time
_cloudpickle_internal._should_pickle_by_reference = _patched_should_pickle_by_reference


def _extract_lambda_source(source: str, code: types.CodeType) -> str:
    """Extract a specific lambda's source when multiple lambdas share a line.

    Uses co_positions() (Python 3.11+) to find column offset and tokenization
    to extract just the target lambda. Falls back to full source on older Python.

    This handles tricky cases like:
    - Multiple lambdas on same line: f, g = lambda x: x*2, lambda x: x+1
    - Nested lambdas: lambda x: lambda y: x + y
    - Lambdas with lambda defaults: lambda x=lambda: 1: x()
    - Multi-line lambdas (automatically wrapped in parentheses)

    Args:
        source: The full source code containing the lambda (from inspect.getsource).
        code: The code object of the lambda function.

    Returns:
        The extracted source for just this specific lambda, or the original
        source if extraction fails or isn't needed.
    """
    if code.co_name != "<lambda>" or not hasattr(code, "co_positions"):
        return source

    # Find first meaningful position (skip entries with zero columns)
    target_line = target_col = None
    for line, _, col, end_col in code.co_positions():
        if col or end_col:
            target_line, target_col = line, col
            break
    if target_line is None:
        return source

    # Adjust to source-relative coordinates
    source_line = target_line - code.co_firstlineno + 1

    # Tokenize source
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return source

    # Find each lambda and its body-start position (the colon).
    # co_positions points to the body, so we need the lambda whose colon
    # is closest to but before the target position.
    best_lambda_idx = None
    best_colon_col = -1

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string == "lambda":
            # Find this lambda's body colon, skipping nested lambdas and structures
            depth = 0
            lambda_depth = 0
            for j in range(i + 1, len(tokens)):
                t = tokens[j]
                if t.type == tokenize.NAME and t.string == "lambda":
                    lambda_depth += 1
                elif t.type == tokenize.OP:
                    if t.string in "([{":
                        depth += 1
                    elif t.string in ")]}":
                        depth -= 1
                    elif t.string == ":" and depth == 0:
                        if lambda_depth > 0:
                            # This colon belongs to a nested lambda
                            lambda_depth -= 1
                        else:
                            # This is our lambda's body colon
                            colon_line, colon_col = t.start
                            if colon_line < source_line or (
                                colon_line == source_line and colon_col < target_col
                            ):
                                if colon_col > best_colon_col:
                                    best_lambda_idx = i
                                    best_colon_col = colon_col
                            break
        i += 1

    if best_lambda_idx is None:
        return source

    # Find end of lambda: comma/colon at depth 0 in body, closing paren, or newline.
    # A colon after the body starts indicates an enclosing lambda's body separator.
    idx = best_lambda_idx
    depth = 0
    lambda_depth = 0
    past_colon = False
    end_idx = idx
    for j in range(idx + 1, len(tokens)):  # Start after our lambda keyword
        t = tokens[j]
        if t.type == tokenize.NAME and t.string == "lambda":
            lambda_depth += 1
        elif t.type == tokenize.OP:
            if t.string == ":" and depth == 0:
                if lambda_depth > 0:
                    lambda_depth -= 1
                elif past_colon:
                    # Second colon at depth 0 - enclosing lambda's body
                    end_idx = j - 1
                    break
                else:
                    past_colon = True
            elif t.string in "([{":
                depth += 1
            elif t.string in ")]}":
                if depth == 0:
                    end_idx = j - 1
                    break
                depth -= 1
            elif t.string == "," and depth == 0 and past_colon:
                end_idx = j - 1
                break
        elif t.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            end_idx = j - 1
            break
        end_idx = j

    # Extract source span
    start, end = tokens[idx], tokens[end_idx]
    lines = source.splitlines(keepends=True)
    if start.start[0] == end.end[0]:
        return lines[start.start[0] - 1][start.start[1]:end.end[1]]
    parts = []
    for n in range(start.start[0], end.end[0] + 1):
        line = lines[n - 1]
        if n == start.start[0]:
            parts.append(line[start.start[1]:])
        elif n == end.end[0]:
            parts.append(line[:end.end[1]])
        else:
            parts.append(line)
    result = "".join(parts)
    # Multi-line lambdas need parentheses to be syntactically valid
    if "\n" in result:
        result = "(" + result + ")"
    return result


def make_function(
    source: str,
    name: str,
    filename: Optional[str],
    qualname: str,
    module: str,
    doc: Optional[str],
    annotations: Optional[dict],
    defaults: Optional[tuple],
    kwdefaults: Optional[dict],
    base_globals: dict,
    closure_values: Optional[list],
    closure_names: Optional[list],
) -> types.FunctionType:
    """Reconstruct a function from its serialized source code and metadata.

    This is the deserialization counterpart to CustomCloudPickler's function
    serialization. It recompiles source code and reconstructs the function
    with all its original attributes (defaults, annotations, closure, etc.).

    This function creates the function with minimal globals. The full globals
    (including any self-references for recursive functions) are applied later
    by _source_function_setstate, which is called after pickle memoizes the
    function. This two-phase approach enables proper handling of circular
    references like recursive or mutually recursive functions.

    Args:
        source: The function's source code as a string. May be indented.
        name: The function's __name__ attribute.
        filename: Original filename where the function was defined. Used for
            tracebacks and debugging. Falls back to "<serialization>" if None.
        qualname: The function's __qualname__ (qualified name including class).
        module: The function's __module__ attribute.
        doc: The function's docstring (__doc__).
        annotations: Type annotations dict (__annotations__).
        defaults: Default values for positional arguments (__defaults__).
        kwdefaults: Default values for keyword-only arguments (__kwdefaults__).
        base_globals: Minimal global variables dict. Full globals including
            self-references are added later by _source_function_setstate.
        closure_values: List of closure variable values (passed immediately, not
            deferred, because closures need factory pattern to bind properly).
        closure_names: List of closure variable names (co_freevars).

    Returns:
        A newly constructed function object. Note: the globals are minimal
        at this point - they're filled in by _source_function_setstate.

    Raises:
        ValueError: If the function name cannot be found in the compiled source.
    """
    # Remove any leading indentation (e.g., if function was defined inside a class)
    source = textwrap.dedent(source)

    # Set up the global namespace for the reconstructed function.
    # This is a minimal globals dict - full globals are added by _source_function_setstate.
    func_globals = {"__builtins__": __builtins__, **base_globals}

    if closure_values and closure_names:
        # CLOSURE HANDLING: Functions with closures require special treatment.
        #
        # Python closures work by capturing variables from enclosing scopes.
        # We can't directly create closure cells, so we use a factory pattern:
        # wrap the function definition inside another function that takes the
        # closure values as parameters, then call it to create the real function.
        #
        # Note: closure values are bound here (not deferred) because the factory
        # pattern requires them at function creation time. However, the globals
        # (including any self-references) ARE deferred to the state setter.
        closure_params = ", ".join(closure_names)
        factory_source = f"def _seri_factory_({closure_params}):\n"

        # Lambdas have '<lambda>' as their name, which is not valid Python syntax.
        # We handle them by assigning the lambda expression to a temporary variable.
        if name == "<lambda>":
            indented_source = "    _lambda_result_ = " + source + "\n"
            factory_source += indented_source
            factory_source += "    return _lambda_result_\n"
        else:
            indented_source = textwrap.indent(source, "    ")
            factory_source += indented_source + "\n"
            factory_source += f"    return {name}\n"

        # Compile and execute the factory, then call it with closure values
        try:
            factory_code = compile(
                factory_source, filename or "<serialization>", "exec"
            )
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile source for function '{name}'. "
                f"This may indicate corrupted serialized data or version incompatibility."
            ) from e
        exec(factory_code, func_globals)
        factory = func_globals["_seri_factory_"]
        func = factory(*closure_values)

        # The factory path doesn't preserve defaults, so restore them
        if defaults:
            func.__defaults__ = tuple(defaults)
    else:
        # NO CLOSURE: Compile source directly and extract the code object.
        try:
            module_code = compile(source, filename or "<serialization>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile source for function '{name}'. "
                f"This may indicate corrupted serialized data or version incompatibility."
            ) from e

        # Search through the module's constants to find our function's code object.
        # For lambdas with nested lambdas as defaults, multiple code objects have
        # co_name=="<lambda>". The outermost lambda is last, so don't break early.
        func_code = None
        for const in module_code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == name:
                func_code = const
                # Don't break for lambdas - we want the last (outermost) one

        if func_code is None:
            raise ValueError(f"Could not find function '{name}' in compiled source")

        # Create the function directly from the code object
        func = types.FunctionType(func_code, func_globals, name, defaults, None)

    # Restore function metadata
    if kwdefaults:
        func.__kwdefaults__ = kwdefaults

    if annotations:
        func.__annotations__ = annotations

    # Restore identity attributes for proper introspection
    func.__module__ = module
    func.__doc__ = doc
    func.__qualname__ = qualname

    # Attach original source for re-serialization (inspect.getsource won't work
    # because line numbers don't match the original file)
    func.__source__ = source

    return func


def _source_function_setstate(func: types.FunctionType, state: tuple) -> None:
    """Update the state of a source-serialized function after memoization.

    This is called by pickle after the function has been created and memoized.
    It updates the function's globals with the full captured values, including
    any self-references (for recursive functions) or cross-references (for
    mutually recursive functions).

    This two-phase approach (create minimal function, then fill in state) is
    essential for handling circular references. By the time this function is
    called, the function object is already in pickle's memo, so any references
    to it in the state (like self-references in globals) resolve correctly.

    For local recursive functions (where the self-reference is in a closure
    variable rather than globals), we also fill in deferred closure cells here.

    Args:
        func: The function to update (already created by make_function).
        state: A tuple of (func_dict, slotstate) where:
            - func_dict: Custom attributes to add to func.__dict__
            - slotstate: Dict with keys:
                - "__globals__": Full captured globals to merge into func.__globals__
                - "__deferred_closure__": Dict mapping closure cell indices to the
                  function objects that should fill those cells (handles both
                  self-references and cross-references for mutual recursion)
    """
    func_dict, slotstate = state

    # Update func.__dict__ with any custom attributes
    func.__dict__.update(func_dict)

    # Get the full globals from state
    full_globals = slotstate.get("__globals__", {})

    # Update globals in place - this is the key to handling self-references!
    # The function is already memoized, so any self-reference in full_globals
    # now points to the existing (memoized) function object.
    func.__globals__.update(full_globals)

    # Fill in deferred closure cells (for local recursive/mutually recursive functions).
    # These are closure cells containing function references that were replaced
    # with None placeholders during serialization to avoid infinite recursion.
    # Now that all functions are memoized, we can safely fill them in.
    deferred_closure = slotstate.get("__deferred_closure__", {})
    if deferred_closure and func.__closure__ is not None:
        for idx_str, target_func in deferred_closure.items():
            # The index might be a string if it came through JSON-like serialization
            idx = int(idx_str) if isinstance(idx_str, str) else idx_str
            # Fill in the function reference (could be self or another function)
            func.__closure__[idx].cell_contents = target_func


class CustomCloudPickler(cloudpickle.Pickler):
    """A cloudpickle-based pickler that serializes functions by source code.

    This pickler extends cloudpickle.Pickler to override how dynamic functions
    are serialized. Instead of using bytecode (which is Python version-specific),
    it captures the function's source code, enabling cross-version compatibility.

    Key features:
        - Source-based function serialization via _dynamic_function_reduce
        - Persistent object references via persistent_id for objects that
          shouldn't be fully serialized

    Example:
        >>> import io
        >>> def my_func(x):
        ...     return x * 2
        >>> buffer = io.BytesIO()
        >>> CustomCloudPickler(buffer).dump(my_func)
        >>> # Function is now serialized with its source code
    """

    def _dynamic_function_reduce(self, func: types.FunctionType) -> tuple:
        """Serialize a function by capturing its source code and metadata.

        This method is called by cloudpickle when serializing dynamic functions
        (functions defined at runtime or in __main__). We override it to capture
        source code instead of bytecode.

        The serialization uses pickle's 6-tuple reduce protocol with a state setter
        to properly handle circular references (like recursive functions). The
        pattern is:
            1. make_function creates a function with minimal globals and empty closures
            2. Pickle memoizes the function object
            3. _source_function_setstate fills in the full globals and closure values

        This deferred state application is essential because the full globals may
        contain references to the function itself (for recursive functions) or to
        other functions that reference this one (for mutual recursion). By the time
        the state setter runs, the function is already memoized, so these references
        resolve correctly.

        Args:
            func: The function to serialize.

        Returns:
            A 6-tuple for pickle's reduce protocol:
                - Callable to reconstruct the function (make_function)
                - Args tuple with minimal state (source, metadata, empty closures)
                - State tuple (func_dict, slotstate with full globals/closure)
                - None (for list items, unused)
                - None (for dict items, unused)
                - State setter function (_source_function_setstate)

        Note:
            If the function has a __source__ attribute (manually attached),
            that is used instead of calling inspect.getsource(). This is useful
            for functions where source inspection might fail (e.g., dynamically
            generated functions with attached source).
        """
        # Get source code - prefer explicit __source__ attribute if present.
        # This allows users to attach source to dynamically generated functions
        # where inspect.getsource() would fail.
        if hasattr(func, "__source__"):
            source = func.__source__
        else:
            try:
                source = inspect.getsource(func)
            except OSError as e:
                raise pickle.PicklingError(
                    f"Cannot serialize function '{func.__name__}': source code unavailable. "
                    f"Attach source manually via func.__source__ = '...'. Original error: {e}"
                ) from e
            # For lambdas, extract just this lambda when multiple share a line
            source = _extract_lambda_source(source, func.__code__)

        # Use cloudpickle's internal helper to extract function state.
        # _function_getstate returns:
        #   - func_dict: func.__dict__ (custom attributes)
        #   - slotstate: dict of function slots (__globals__, __defaults__, etc.)
        func_dict, slotstate = _function_getstate(func)

        # Remove __source__ from func_dict since we handle it separately
        func_dict = {k: v for k, v in func_dict.items() if k != "__source__"}

        # Extract metadata for the args tuple
        name = slotstate["__name__"]
        defaults = slotstate["__defaults__"]
        kwdefaults = slotstate["__kwdefaults__"]
        annotations = slotstate["__annotations__"]
        filename = func.__code__.co_filename
        qualname = slotstate["__qualname__"]
        module_name = slotstate["__module__"]
        doc = slotstate["__doc__"]

        # BASE GLOBALS: Minimal globals for function creation.
        # Following cloudpickle's pattern, we only include module identity attrs.
        # The full globals (including any self-references) go in the state and
        # are applied AFTER memoization by _source_function_setstate.
        base_globals = {}
        func_globals = func.__globals__
        for k in ["__package__", "__name__", "__path__", "__file__"]:
            if k in func_globals:
                base_globals[k] = func_globals[k]

        # SECURITY CHECK: Check function globals for prohibited modules/functions.
        # This catches dangerous patterns like `import os; os.getcwd()` or
        # `from subprocess import run; run(...)` where the prohibited object
        # is captured in the function's globals.
        captured_globals = slotstate.get("__globals__", {})
        for global_name, global_value in captured_globals.items():
            # Skip special names
            if global_name.startswith("__") and global_name.endswith("__"):
                continue

            # Check for prohibited modules
            if isinstance(global_value, types.ModuleType):
                mod_name = getattr(global_value, "__name__", "")
                if _is_prohibited_module(mod_name):
                    raise PicklingProhibitedError(
                        f"Function '{name}' references prohibited module '{mod_name}'.\n"
                        f"Code sent to NDIF servers cannot use modules that perform "
                        f"file system operations, process execution, network access, "
                        f"or other potentially dangerous operations.\n"
                        f"If you need this functionality, consider restructuring your "
                        f"code to perform these operations locally before/after the trace."
                    )

            # Check for functions from prohibited modules
            if isinstance(global_value, (types.FunctionType, types.BuiltinFunctionType)):
                obj_module = getattr(global_value, "__module__", None) or ""
                obj_name = getattr(global_value, "__name__", global_name)

                # Special handling for builtins
                if obj_module == "builtins":
                    if obj_name in PROHIBITED_BUILTINS:
                        raise PicklingProhibitedError(
                            f"Function '{name}' references prohibited builtin '{obj_name}'.\n"
                            f"Code sent to NDIF servers cannot use dangerous builtins "
                            f"like eval, exec, open, compile, or __import__.\n"
                            f"If you need this functionality, consider restructuring your "
                            f"code to perform these operations locally before/after the trace."
                        )
                elif _is_prohibited_module(obj_module):
                    raise PicklingProhibitedError(
                        f"Function '{name}' references '{obj_name}' from prohibited module '{obj_module}'.\n"
                        f"Code sent to NDIF servers cannot use modules that perform "
                        f"file system operations, process execution, network access, "
                        f"or other potentially dangerous operations.\n"
                        f"If you need this functionality, consider restructuring your "
                        f"code to perform these operations locally before/after the trace."
                    )

        # CLOSURE: Extract closure values and names.
        # For recursive/mutually recursive local functions, the closure may contain
        # function references that create cycles. We defer ALL function values in
        # closures to the state setter, which runs after memoization. This allows
        # pickle's memo to break the cycle.
        deferred_closure = {}  # Maps index -> function to fill in after memoization
        if func.__closure__ is not None:
            closure_names = list(func.__code__.co_freevars)
            closure_values = []
            for i, cell in enumerate(func.__closure__):
                value = _get_cell_contents(cell)
                closure_name = closure_names[i] if i < len(closure_names) else f"cell_{i}"

                # SECURITY CHECK: Check closure values for prohibited modules/functions.
                # This catches nested functions that capture prohibited modules from
                # their enclosing scope (e.g., `import os; def inner(): os.getcwd()`).
                if isinstance(value, types.ModuleType):
                    mod_name = getattr(value, "__name__", "")
                    if _is_prohibited_module(mod_name):
                        raise PicklingProhibitedError(
                            f"Function '{name}' captures prohibited module '{mod_name}' "
                            f"in closure variable '{closure_name}'.\n"
                            f"Code sent to NDIF servers cannot use modules that perform "
                            f"file system operations, process execution, network access, "
                            f"or other potentially dangerous operations.\n"
                            f"If you need this functionality, consider restructuring your "
                            f"code to perform these operations locally before/after the trace."
                        )
                elif isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
                    obj_module = getattr(value, "__module__", None) or ""
                    obj_name = getattr(value, "__name__", closure_name)

                    # Special handling for builtins
                    if obj_module == "builtins":
                        if obj_name in PROHIBITED_BUILTINS:
                            raise PicklingProhibitedError(
                                f"Function '{name}' captures prohibited builtin '{obj_name}' "
                                f"in closure variable '{closure_name}'.\n"
                                f"Code sent to NDIF servers cannot use dangerous builtins "
                                f"like eval, exec, open, compile, or __import__.\n"
                                f"If you need this functionality, consider restructuring your "
                                f"code to perform these operations locally before/after the trace."
                            )
                    elif _is_prohibited_module(obj_module):
                        raise PicklingProhibitedError(
                            f"Function '{name}' captures '{obj_name}' from prohibited module "
                            f"'{obj_module}' in closure variable '{closure_name}'.\n"
                            f"Code sent to NDIF servers cannot use modules that perform "
                            f"file system operations, process execution, network access, "
                            f"or other potentially dangerous operations.\n"
                            f"If you need this functionality, consider restructuring your "
                            f"code to perform these operations locally before/after the trace."
                        )

                if isinstance(value, types.FunctionType):
                    # Defer ALL functions to state setter to handle cycles.
                    # This includes self-references and cross-references.
                    closure_values.append(None)  # Placeholder
                    deferred_closure[i] = value
                else:
                    closure_values.append(value)
        else:
            closure_values = None
            closure_names = None

        # STATE: Full globals and deferred closure info, applied after memoization.
        # This is where self-references and cross-references live - by the time
        # _source_function_setstate is called, the function is already memoized,
        # so these references resolve correctly.
        state_slotstate = {
            "__globals__": slotstate["__globals__"],
            "__deferred_closure__": deferred_closure,  # Maps indices to target functions
        }
        state = (func_dict, state_slotstate)

        # Args for make_function - creates function with minimal globals.
        # Closure values are passed here (not deferred) because the factory
        # pattern needs them at function creation time.
        args = (
            source,
            name,
            filename,
            qualname,
            module_name,
            doc,
            annotations,
            defaults,
            kwdefaults,
            base_globals,
            closure_values,
            closure_names,
        )

        # Return 6-tuple: (func, args, state, listitems, dictitems, state_setter)
        # The state_setter is called AFTER memoization to fill in full globals/closure
        return (make_function, args, state, None, None, _source_function_setstate)

    def persistent_id(self, obj: Any) -> Optional[Any]:
        """Return a persistent ID for objects that shouldn't be fully serialized.

        Pickle's persistent_id mechanism allows certain objects to be referenced
        by an ID rather than serialized. During deserialization, persistent_load
        resolves these IDs back to actual objects.

        This is critical for nnsight's remote execution where certain objects
        (like model proxies, intervention graph nodes, or large tensors) should
        not be serialized but instead looked up on the server side.

        Args:
            obj: The object being pickled.

        Returns:
            The persistent ID if obj has a `_persistent_id` in its __dict__,
            otherwise None (meaning pickle should serialize normally).

        Example:
            An object with obj.__dict__["_persistent_id"] = "node_42" will be
            serialized as just the reference "node_42", and during deserialization,
            persistent_load("node_42") will be called to resolve it.
        """
        try:
            return obj.__dict__["_persistent_id"]
        except (AttributeError, KeyError, TypeError):
            # Object doesn't have __dict__ or doesn't have _persistent_id
            pass

        return None



class CustomCloudUnpickler(pickle.Unpickler):
    """A custom unpickler that resolves persistent object references.

    Works in conjunction with CustomCloudPickler to handle objects that were
    serialized by reference (persistent_id) rather than by value. During
    deserialization, persistent IDs are looked up in the provided dictionary.

    This enables patterns where certain objects (like model proxies or graph
    nodes) are referenced by ID in the serialized data and resolved to actual
    objects on the server side.

    Args:
        file: File-like object to read pickle data from.
        persistent_objects: Dictionary mapping persistent IDs to actual objects.
            When a persistent ID is encountered during deserialization, it's
            looked up in this dictionary.

    Example:
        >>> # On the server side
        >>> model_proxy = get_model_proxy("gpt2")
        >>> persistent_objects = {"model_ref_1": model_proxy}
        >>> data = receive_from_client()
        >>> obj = CustomCloudUnpickler(io.BytesIO(data), persistent_objects).load()
        >>> # Any references to "model_ref_1" in the data are now resolved
    """

    def __init__(self, file: BinaryIO, persistent_objects: Optional[dict] = None):
        """Initialize the unpickler with a file and optional persistent objects.

        Args:
            file: Binary file-like object containing pickle data.
            persistent_objects: Optional dict mapping persistent IDs to objects.
                Defaults to empty dict if not provided.
        """
        super().__init__(file)
        self.persistent_objects = persistent_objects or {}

    def persistent_load(self, pid: Any) -> Any:
        """Resolve a persistent ID to its corresponding object.

        Called automatically by pickle when it encounters a persistent reference
        (created by persistent_id during serialization).

        Args:
            pid: The persistent ID to resolve.

        Returns:
            The object corresponding to the persistent ID.

        Raises:
            pickle.UnpicklingError: If the persistent ID is not found in
                the persistent_objects dictionary.
        """
        if pid in self.persistent_objects:
            return self.persistent_objects[pid]

        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


def dumps(
    obj: Any,
    path: Optional[Union[str, Path]] = None,
    protocol: int = DEFAULT_PROTOCOL,
) -> Optional[bytes]:
    """Serialize an object using source-based function serialization.

    This is the high-level API for serializing objects with CustomCloudPickler.
    Functions in the object graph will be serialized by source code rather than
    bytecode, enabling cross-Python-version compatibility.

    Args:
        obj: Any picklable object. Functions will be serialized by source code.
        path: Optional file path to write the serialized data to.
            Accepts both string paths and pathlib.Path objects.
            If None, returns the serialized bytes directly.
        protocol: Pickle protocol version to use. Defaults to DEFAULT_PROTOCOL (4).
            Protocol 4 is available in Python 3.4+ and supports large objects.

    Returns:
        If path is None: The serialized data as bytes.
        If path is provided: None (data is written to file).

    Example:
        >>> # Serialize to bytes (for network transmission)
        >>> data = dumps(my_function)
        >>> send_to_server(data)
        >>>
        >>> # Serialize to file (for persistence)
        >>> dumps(my_function, "/path/to/function.pkl")
        >>>
        >>> # Using pathlib.Path
        >>> from pathlib import Path
        >>> dumps(my_function, Path("./functions/my_func.pkl"))
    """
    if path is None:
        # In-memory serialization - return bytes directly
        buffer = io.BytesIO()
        CustomCloudPickler(buffer, protocol=protocol).dump(obj)
        buffer.seek(0)
        return buffer.read()

    # File-based serialization - write to disk
    # Normalize to Path for consistent handling
    path = Path(path)
    with path.open("wb") as file:
        CustomCloudPickler(file, protocol=protocol).dump(obj)


def loads(
    data: Union[str, bytes, Path],
    persistent_objects: Optional[dict] = None,
) -> Any:
    """Deserialize data that was serialized with dumps().

    This is the high-level API for deserializing objects with CustomCloudUnpickler.
    Functions serialized by source code will be reconstructed by recompiling
    their source on the current Python version.

    Args:
        data: One of:
            - bytes: Serialized data (e.g., received over network)
            - str: Path to a file containing serialized data
            - Path: pathlib.Path to a file containing serialized data
        persistent_objects: Optional dictionary mapping persistent IDs to objects.
            Used to resolve objects that were serialized by reference rather
            than by value. See CustomCloudUnpickler for details.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If a persistent ID is encountered that isn't
            in the persistent_objects dictionary.
        FileNotFoundError: If a file path is provided but the file doesn't exist.

    Example:
        >>> # Load from bytes (received over network)
        >>> data = receive_from_client()
        >>> obj = loads(data)
        >>>
        >>> # Load from file (string path)
        >>> obj = loads("/path/to/function.pkl")
        >>>
        >>> # Load from file (pathlib.Path)
        >>> from pathlib import Path
        >>> obj = loads(Path("./functions/my_func.pkl"))
        >>>
        >>> # Load with persistent object resolution
        >>> persistent = {"model_proxy": actual_model}
        >>> obj = loads(data, persistent_objects=persistent)
    """
    if isinstance(data, bytes):
        # In-memory deserialization from bytes
        return CustomCloudUnpickler(io.BytesIO(data), persistent_objects).load()

    # File-based deserialization - data is a file path
    # Normalize to Path for consistent handling
    path = Path(data)
    with path.open("rb") as file:
        return CustomCloudUnpickler(file, persistent_objects).load()


# Backward-compatible aliases for existing code that uses save/load
save = dumps
load = loads
