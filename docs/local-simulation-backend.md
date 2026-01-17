# Local Simulation Backend (`remote='local'`)

**Commit:** 981c313
**Date:** 2026-01-17

## Goal

Provide a way to test serialization locally without requiring an NDIF server connection. This enables developers to verify that their code will serialize correctly before submitting to remote execution.

## Design

The local simulation backend mimics what happens during remote execution:

1. **Serialize** the trace using the same `CustomCloudPickler` that would be used for NDIF
2. **Block user modules** during deserialization to simulate the server environment
3. **Deserialize** and execute the trace locally
4. **Return results** as if they came from a remote server

This catches serialization errors early, particularly:
- Missing dependencies that wouldn't exist on the server
- Functions that can't have their source code extracted
- Objects that aren't properly serializable

## Implementation

### New Files

- `src/nnsight/intervention/backends/local_simulation.py` - The `LocalSimulationBackend` class

### Key Components

```python
class LocalSimulationBackend:
    """Backend that simulates remote execution locally."""

    def execute(self, trace):
        # 1. Serialize the trace
        data = serialization.dumps(trace)

        # 2. Block user modules during deserialization
        with self._block_user_modules():
            restored_trace = serialization.loads(data)

        # 3. Execute locally
        return restored_trace.execute()
```

### Usage

```python
from nnsight import LanguageModel

model = LanguageModel("gpt2")

# Test serialization locally before remote execution
with model.trace("Hello", remote="local"):
    hidden = model.transformer.h[0].output[0]
    result = hidden.mean().save()

print(result)  # Works just like remote=True, but runs locally
```

## Limitations

1. **Not a true server simulation**: The local environment still has access to installed packages. Only explicit user module blocking is performed.

2. **No resource constraints**: Local execution doesn't simulate server memory/compute limits.

3. **Same Python version**: Doesn't test cross-version compatibility (client Python 3.10 → server Python 3.11).

## Testing

See `tests/test_local_simulation.py` for basic usage tests.
