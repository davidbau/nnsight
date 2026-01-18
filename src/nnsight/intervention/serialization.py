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
    # Python standard library (commonly used)
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
    if module_name is None:
        return False
    if not isinstance(module_name, str):
        return False
    # Fast path: check if the top-level package is whitelisted
    dot_idx = module_name.find(".")
    if dot_idx == -1:
        top_level = module_name
    else:
        top_level = module_name[:dot_idx]
    return top_level in SERVER_MODULES_WHITELIST


def _should_pickle_by_reference(obj: Any) -> bool:
    """Determine if an object should be pickled by reference or by value.

    This callback is passed to cloudpickle to control serialization behavior.
    Objects from whitelisted modules are pickled by reference (just the import
    path), while objects from non-whitelisted modules are pickled by value
    (source code for functions, full state for classes).

    Args:
        obj: The object being pickled.

    Returns:
        True if the object should be pickled by reference, False for by value.
    """
    module = getattr(obj, "__module__", None)
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
