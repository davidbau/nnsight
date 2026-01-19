"""Source-based function serialization for cross-version compatibility.

This module provides serialization support for nnsight's remote execution,
using cloudpickle's source-based serialization mode for cross-Python-version
compatibility.

The key insight is that Python bytecode is version-specific, but source code
is portable. By serializing functions as source code, we can send user-defined
functions from a client on Python 3.10 to a server on Python 3.11+.

This module uses cloudpickle's native `by_source=True` parameter combined with
a `pickle_by_reference` callback to control which modules are serialized by
source vs. by reference:

- Whitelisted modules (torch, numpy, transformers, etc.) are assumed to exist
  on the server and are pickled by reference (just the import path)
- Non-whitelisted modules (user code) are pickled by source code

Example:
    >>> from nnsight.intervention.serialization import dumps, loads
    >>> def my_func(x):
    ...     return x * 2
    >>> data = dumps(my_func)  # Serialized by source code
    >>> restored = loads(data)
    >>> restored(5)  # Returns 10
"""

from collections import defaultdict
import io
import pickle
from pathlib import Path
from types import FrameType
from typing import Any, Optional, Union

import cloudpickle
from cloudpickle import PicklingProhibitedError

from ..util import Patcher, Patch
from .envoy import Envoy

# Default pickle protocol - protocol 4 is available in Python 3.4+ and supports large objects
DEFAULT_PROTOCOL = 4

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
    # Python standard library (commonly used, safe for remote execution)
    "abc", "ast", "base64", "collections", "contextlib", "copy",
    "dataclasses", "datetime", "decimal", "enum", "functools", "hashlib",
    "heapq", "itertools", "json", "logging", "math", "operator", "random",
    "re", "secrets", "statistics", "string", "struct", "time", "typing",
    "typing_extensions", "unittest", "uuid", "warnings", "weakref", "zlib",
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
#   - `open(...)`, `eval(...)` - NOT caught (builtins resolved at runtime)
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


def _get_top_level_module(module_name: str) -> Optional[str]:
    """Extract the top-level module name from a fully qualified module name.

    Args:
        module_name: The fully qualified module name (e.g., 'torch.nn.functional')

    Returns:
        The top-level module name (e.g., 'torch'), or None if invalid.
    """
    if module_name is None:
        return None
    if not isinstance(module_name, str):
        return None
    dot_idx = module_name.find(".")
    if dot_idx == -1:
        return module_name
    return module_name[:dot_idx]


def _is_whitelisted_module(module_name: str) -> bool:
    """Check if a module is in the server whitelist.

    A module is considered whitelisted if its top-level package is in the
    whitelist. For example, if 'torch' is whitelisted, then 'torch.nn' and
    'torch.nn.functional' are also whitelisted.

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


def _should_pickle_by_reference(obj: Any) -> bool:
    """Determine if an object should be pickled by reference or by value.

    This callback is passed to cloudpickle to control serialization behavior:
    - Objects from PROHIBITED modules raise PicklingProhibitedError
    - Objects from WHITELISTED modules are pickled by reference (import path)
    - Objects from other modules are pickled by value (source code)

    Args:
        obj: The object being pickled.

    Returns:
        True if the object should be pickled by reference, False for by value.

    Raises:
        PicklingProhibitedError: If the object is from a prohibited module.
    """
    module = getattr(obj, "__module__", None)
    obj_name = getattr(obj, "__name__", repr(obj))

    # Check blacklist first - prohibited modules raise an error
    if _is_prohibited_module(module):
        raise PicklingProhibitedError(
            f"Cannot serialize '{obj_name}' from module '{module}'. "
            f"This module is prohibited for security reasons. "
            f"Code sent to NDIF servers cannot use modules that perform "
            f"file system operations, process execution, network access, "
            f"or other potentially dangerous operations."
        )

    # Check whitelist - whitelisted modules are pickled by reference
    return _is_whitelisted_module(module)


# =============================================================================
# Pickler and Unpickler Classes
# =============================================================================

# Store original setstate for Envoy restoration
_original_envoy_setstate = Envoy.__setstate__


class CustomCloudPickler(cloudpickle.Pickler):
    """A cloudpickle-based pickler with source serialization and persistent IDs.

    This pickler extends cloudpickle.Pickler to:
    1. Use source-based serialization for cross-version compatibility
    2. Apply whitelist-based pickle-by-reference for known server modules
    3. Support persistent IDs for special objects (like FrameType)

    The source serialization and whitelist behavior is configured via the
    `by_source` and `pickle_by_reference` parameters passed to the parent.
    """

    def __init__(self, file, protocol=None):
        """Initialize the pickler with source serialization enabled.

        Args:
            file: File-like object to write serialized data to.
            protocol: Pickle protocol version. Defaults to DEFAULT_PROTOCOL.
        """
        super().__init__(
            file,
            protocol=protocol or DEFAULT_PROTOCOL,
            by_source=True,
            pickle_by_reference=_should_pickle_by_reference,
        )

    def reducer_override(self, obj):
        """Override reducer to check for prohibited modules.

        This intercepts all objects being pickled and checks if they are
        modules (or from modules) that are prohibited. This catches cases
        that the pickle_by_reference callback misses, such as module objects
        in function globals.

        Args:
            obj: The object being pickled.

        Returns:
            NotImplemented to use default handling, or a reduction tuple.

        Raises:
            PicklingProhibitedError: If the object is a prohibited module.
        """
        import types

        # Check if this is a module object from the prohibited list
        if isinstance(obj, types.ModuleType):
            module_name = getattr(obj, "__name__", "")
            if _is_prohibited_module(module_name):
                raise PicklingProhibitedError(
                    f"Cannot serialize module '{module_name}'. "
                    f"This module is prohibited for security reasons. "
                    f"Code sent to NDIF servers cannot use modules that perform "
                    f"file system operations, process execution, network access, "
                    f"or other potentially dangerous operations."
                )

        # Check if this is a function/method from a prohibited module
        # This catches things like os.getcwd, subprocess.run, etc.
        if callable(obj) and hasattr(obj, "__module__"):
            obj_module = getattr(obj, "__module__", None) or ""
            obj_name = getattr(obj, "__name__", repr(obj))

            # Special handling for builtins - only block specific dangerous ones
            if obj_module == "builtins":
                if obj_name in PROHIBITED_BUILTINS:
                    raise PicklingProhibitedError(
                        f"Cannot serialize builtin '{obj_name}'. "
                        f"This builtin function is prohibited for security reasons. "
                        f"Code sent to NDIF servers cannot use dangerous builtins "
                        f"like eval, exec, open, compile, or __import__."
                    )
                # Safe builtins (int, str, list, etc.) are allowed
            elif _is_prohibited_module(obj_module):
                raise PicklingProhibitedError(
                    f"Cannot serialize '{obj_name}' from module '{obj_module}'. "
                    f"This module is prohibited for security reasons. "
                    f"Code sent to NDIF servers cannot use modules that perform "
                    f"file system operations, process execution, network access, "
                    f"or other potentially dangerous operations."
                )

        # Delegate to parent's reducer_override for all other handling
        return super().reducer_override(obj)

    def persistent_id(self, obj: Any) -> Optional[str]:
        """Return a persistent ID for objects that shouldn't be fully serialized.

        Certain objects (like frame objects or model proxies) need special
        handling - they're referenced by ID rather than serialized.

        Args:
            obj: The object being pickled.

        Returns:
            A persistent ID string if the object should be referenced, None otherwise.
        """
        # Handle frame objects specially
        if isinstance(obj, FrameType):
            return f"FRAME{id(obj)}"

        # Handle objects with explicit persistent IDs (e.g., model proxies)
        try:
            return obj.__dict__["_persistent_id"]
        except (AttributeError, KeyError, TypeError):
            pass

        return None


class CustomCloudUnpickler(pickle.Unpickler):
    """A custom unpickler that resolves persistent object references.

    Works in conjunction with CustomCloudPickler to handle objects that were
    serialized by reference. Also handles Envoy state restoration.

    Args:
        file: File-like object to read pickle data from.
        root: Root Envoy for state restoration (optional).
        frame: Frame object for frame reference resolution (optional).
        persistent_objects: Dictionary mapping persistent IDs to objects (optional).
    """

    def __init__(
        self,
        file,
        root: Optional[Envoy] = None,
        frame: Optional[FrameType] = None,
        persistent_objects: Optional[dict] = None,
    ):
        super().__init__(file)
        self.root = root
        self.frame = frame
        self.persistent_objects = persistent_objects or {}
        self.proxy_frames = defaultdict(dict)

    def load(self):
        """Load and deserialize the pickled object.

        If a root Envoy is provided, injects state restoration logic.
        """
        if self.root is not None:
            # Envoy state restoration logic
            def inject(_self, state):
                _original_envoy_setstate(_self, state)

                envoy = self.root.get(_self.path.removeprefix("model"))

                _self._module = envoy._module
                _self._interleaver = envoy._interleaver

                for key, value in envoy.__dict__.items():
                    if key not in _self.__dict__:
                        _self.__dict__[key] = value

            with Patcher([Patch(Envoy, inject, "__setstate__")]):
                return super().load()
        else:
            return super().load()

    def persistent_load(self, pid: str) -> Any:
        """Resolve a persistent ID to its corresponding object.

        Args:
            pid: The persistent ID to resolve.

        Returns:
            The object corresponding to the persistent ID.

        Raises:
            pickle.UnpicklingError: If the persistent ID is not recognized.
        """
        # Handle frame references
        if pid.startswith("FRAME"):
            return self.proxy_frames[pid]

        # Handle custom persistent objects
        if pid in self.persistent_objects:
            return self.persistent_objects[pid]

        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


# =============================================================================
# High-Level API
# =============================================================================


def dumps(
    obj: Any,
    path: Optional[Union[str, Path]] = None,
    protocol: int = DEFAULT_PROTOCOL,
) -> Optional[bytes]:
    """Serialize an object using source-based function serialization.

    Functions from non-whitelisted modules are serialized by source code,
    enabling cross-Python-version compatibility for remote execution.

    Args:
        obj: Any picklable object. Functions will be serialized by source code
            if they're from non-whitelisted modules.
        path: Optional file path to write the serialized data to.
            If None, returns the serialized bytes directly.
        protocol: Pickle protocol version to use. Defaults to DEFAULT_PROTOCOL.

    Returns:
        If path is None: The serialized data as bytes.
        If path is provided: None (data is written to file).

    Example:
        >>> data = dumps(my_function)  # Serialize to bytes
        >>> dumps(my_function, "func.pkl")  # Serialize to file
    """
    if path is None:
        buffer = io.BytesIO()
        CustomCloudPickler(buffer, protocol=protocol).dump(obj)
        buffer.seek(0)
        return buffer.read()

    path = Path(path)
    with path.open("wb") as file:
        CustomCloudPickler(file, protocol=protocol).dump(obj)


def loads(
    data: Union[str, bytes, Path],
    model: Optional[Envoy] = None,
    frame: Optional[FrameType] = None,
    persistent_objects: Optional[dict] = None,
) -> Any:
    """Deserialize data that was serialized with dumps().

    Functions serialized by source code will be reconstructed by recompiling
    their source on the current Python version.

    Args:
        data: One of:
            - bytes: Serialized data
            - str: Path to a file containing serialized data
            - Path: pathlib.Path to a file
        model: Optional root Envoy for state restoration.
        frame: Optional frame for frame reference resolution.
        persistent_objects: Optional dict mapping persistent IDs to objects.

    Returns:
        The deserialized object.

    Example:
        >>> obj = loads(data)  # From bytes
        >>> obj = loads("func.pkl")  # From file
    """
    if isinstance(data, bytes):
        return CustomCloudUnpickler(
            io.BytesIO(data), model, frame, persistent_objects
        ).load()

    path = Path(data)
    with path.open("rb") as file:
        return CustomCloudUnpickler(file, model, frame, persistent_objects).load()


# Backward-compatible aliases
save = dumps
load = loads
