"""
Edge case tests for source-based serialization in v0.5.16-bau.

These tests verify that the cloudpickle-based source serialization handles
various edge cases correctly. They are adapted from the source-serialization
branch's test_serialization_edge_cases.py.

Run with: pytest tests/test_serialization_edge_cases.py -v
"""

import sys
import pytest
import torch

sys.path.insert(0, "tests")

from nnsight.intervention.serialization import dumps, loads


# =============================================================================
# Test 1: Closures with captured variables
# =============================================================================


def test_closure_with_captured_variable():
    """Test that closures correctly capture outer scope variables."""
    multiplier = 3.0

    def scale_by_closure(x):
        return x * multiplier

    # Serialize
    data = dumps(scale_by_closure)

    # Should contain source
    assert b"def scale_by_closure" in data

    # Deserialize and verify
    restored = loads(data)
    result = restored(torch.tensor([1.0, 2.0, 3.0]))
    expected = torch.tensor([3.0, 6.0, 9.0])
    assert torch.allclose(result, expected)


def test_nested_closure():
    """Test nested closures with multiple levels of captured variables."""
    outer_val = 10

    def outer_func(x):
        inner_val = 5

        def inner_func(y):
            return y + outer_val + inner_val

        return inner_func(x)

    # Serialize
    data = dumps(outer_func)

    # Deserialize and verify
    restored = loads(data)
    assert restored(1) == 16  # 1 + 10 + 5


def test_returned_inner_function():
    """Test serializing a returned inner function (closure factory)."""

    def make_adder(n):
        def adder(x):
            return x + n

        return adder

    add_5 = make_adder(5)

    # Serialize the returned inner function
    data = dumps(add_5)

    # Deserialize and verify captured value is preserved
    restored = loads(data)
    assert restored(10) == 15


def test_triple_nested_closure():
    """Test three levels of function nesting."""

    def level1(a):
        def level2(b):
            def level3(c):
                return a + b + c

            return level3

        return level2

    # Create a deeply nested closure: level3 with a=1, b=2
    nested = level1(1)(2)

    # Serialize
    data = dumps(nested)

    # Deserialize and verify all captured values preserved
    restored = loads(data)
    assert restored(3) == 6  # 1 + 2 + 3


def test_mutable_closure_state():
    """Test that mutable closure state is preserved during serialization."""

    def make_counter():
        count = [0]  # Mutable container

        def increment():
            count[0] += 1
            return count[0]

        return increment

    counter = make_counter()
    counter()  # count = 1
    counter()  # count = 2

    # Serialize with current state
    data = dumps(counter)

    # Deserialize and verify state continues from where it left off
    restored = loads(data)
    assert restored() == 3  # Continues from 2
    assert restored() == 4  # And keeps incrementing


# =============================================================================
# Test 2: Lambda default arguments
# =============================================================================


def test_lambda_default_argument():
    """Test functions with lambda default arguments."""

    def func_with_lambda_default(processor=lambda x: x * 2):
        return processor(5)

    # Serialize
    data = dumps(func_with_lambda_default)

    # Source should include the lambda
    assert b"lambda" in data

    # Deserialize and verify
    restored = loads(data)
    assert restored() == 10
    assert restored(lambda x: x + 1) == 6


def test_complex_default_arguments():
    """Test functions with various default argument types."""

    def func_with_defaults(
        a,
        b=10,
        c="hello",
        d=None,
        e=[1, 2, 3],
        f={"key": "value"},
    ):
        return (a, b, c, d, e, f)

    # Serialize
    data = dumps(func_with_defaults)

    # Deserialize and verify defaults work
    restored = loads(data)
    result = restored(1)
    assert result[0] == 1
    assert result[1] == 10
    assert result[2] == "hello"
    assert result[3] is None
    assert result[4] == [1, 2, 3]
    assert result[5] == {"key": "value"}


# =============================================================================
# Test 3: Comprehension variable shadowing
# =============================================================================


def test_comprehension_variable_shadowing():
    """Test that comprehension variables don't incorrectly shadow external names."""
    x = 42  # External variable

    def uses_external_and_comprehension():
        # 'x' here refers to external x
        external_value = x

        # 'x' here is a comprehension-local variable (different scope)
        squares = [x * x for x in range(5)]

        return external_value, squares

    # Serialize
    data = dumps(uses_external_and_comprehension)

    # Deserialize and verify
    restored = loads(data)
    external, squares = restored()
    assert external == 42
    assert squares == [0, 1, 4, 9, 16]


# =============================================================================
# Test 4: Lambda parameters
# =============================================================================


def test_lambda_parameters_not_external():
    """Test that lambda parameters are correctly scoped."""

    def func_with_lambda():
        # 'x' is a lambda parameter, not external
        fn = lambda x: x * 2
        return fn(5)

    # Serialize and verify it works
    data = dumps(func_with_lambda)
    restored = loads(data)
    assert restored() == 10


def test_lambda_with_external_and_param():
    """Test lambda that uses both parameters and external references."""
    multiplier = 3

    def func_with_mixed():
        # 'x' is lambda param, 'multiplier' is external
        fn = lambda x: x * multiplier
        return fn(5)

    # Serialize and verify
    data = dumps(func_with_mixed)
    restored = loads(data)
    assert restored() == 15


# =============================================================================
# Test 5: Module variable overriding builtin
# =============================================================================


def test_constant_overriding_builtin():
    """Test that module variables overriding builtins are captured."""
    # Override 'len' with a constant
    CUSTOM_LENGTH = 42
    len = CUSTOM_LENGTH  # noqa: F841

    def uses_overridden_len():
        return len

    # Serialize
    data = dumps(uses_overridden_len)

    # Deserialize and verify
    restored = loads(data)
    assert restored() == 42


# =============================================================================
# Test 6: Nested function parameter shadowing
# =============================================================================


def test_nested_function_shadowing():
    """Test that inner function parameters don't shadow outer references."""
    multiplier = 3

    def outer_uses_multiplier():
        # Uses external 'multiplier'
        base = multiplier * 10

        # Inner function has its own 'multiplier' parameter
        def inner(multiplier):
            return multiplier * 2

        return base, inner(5)

    # Serialize and verify
    data = dumps(outer_uses_multiplier)
    restored = loads(data)
    base, inner_result = restored()
    assert base == 30  # 3 * 10
    assert inner_result == 10  # 5 * 2


# =============================================================================
# Test 7: Import inside function
# =============================================================================


def test_import_inside_function():
    """Test that imports inside functions work correctly."""

    def func_with_import():
        import json

        return json.dumps([1, 2, 3])

    # Serialize and verify
    data = dumps(func_with_import)
    restored = loads(data)
    assert restored() == "[1, 2, 3]"


def test_from_import_inside_function():
    """Test that from imports inside functions work correctly."""

    def func_with_from_import():
        from collections import Counter

        return Counter([1, 1, 2, 3]).most_common(1)

    # Serialize and verify
    data = dumps(func_with_from_import)
    restored = loads(data)
    assert restored() == [(1, 2)]


# =============================================================================
# Test 8: Async functions
# =============================================================================


def test_async_function():
    """Test that async functions can be serialized and reconstructed."""
    import asyncio

    async def async_double(x):
        await asyncio.sleep(0)  # Simulate async operation
        return x * 2

    # Serialize
    data = dumps(async_double)
    assert b"async def" in data

    # Deserialize
    restored = loads(data)
    assert asyncio.iscoroutinefunction(restored)

    # Actually run it
    result = asyncio.run(restored(5))
    assert result == 10


# =============================================================================
# Test 9: Generator functions
# =============================================================================


def test_generator_function():
    """Test that generator functions can be serialized and reconstructed."""

    def generate_numbers(n):
        for i in range(n):
            yield i * 2

    # Serialize
    data = dumps(generate_numbers)
    assert b"yield" in data

    # Deserialize and verify
    restored = loads(data)
    results = list(restored(5))
    assert results == [0, 2, 4, 6, 8]


# =============================================================================
# Test 10: Deeply nested state in closures
# =============================================================================


def test_deeply_nested_closure_state():
    """Test closures with deeply nested captured data."""
    config = {
        "level1": {
            "level2": {
                "level3": {
                    "value": 42,
                    "tensor": torch.tensor([1.0, 2.0]),
                }
            }
        }
    }

    def access_deep_config():
        return config["level1"]["level2"]["level3"]["value"]

    # Serialize and verify
    data = dumps(access_deep_config)
    restored = loads(data)
    assert restored() == 42


# =============================================================================
# Test 11: Function with *args and **kwargs
# =============================================================================


def test_args_kwargs():
    """Test functions with *args and **kwargs."""

    def variadic_func(*args, **kwargs):
        return sum(args) + sum(kwargs.values())

    # Serialize and verify
    data = dumps(variadic_func)
    restored = loads(data)
    assert restored(1, 2, 3, a=4, b=5) == 15


def test_mixed_positional_keyword_args():
    """Test functions with mixed argument types."""

    def mixed_args(a, b, *args, c=10, **kwargs):
        return a + b + sum(args) + c + sum(kwargs.values())

    # Serialize and verify
    data = dumps(mixed_args)
    restored = loads(data)
    assert restored(1, 2, 3, 4, c=5, d=6) == 21  # 1+2+3+4+5+6


# =============================================================================
# Test 12: Recursive functions
# =============================================================================


def test_recursive_function():
    """Test that recursive functions work correctly.

    Both module-level and local recursive functions are supported thanks to
    the 6-tuple reduce protocol with deferred state application.
    """
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    # Serialize and verify
    data = dumps(factorial)
    restored = loads(data)
    assert restored(5) == 120


# =============================================================================
# Test 13: Function with type annotations
# =============================================================================


def test_function_with_annotations():
    """Test functions with type annotations."""

    def annotated_func(x: int, y: float = 1.0) -> float:
        return x * y

    # Serialize
    data = dumps(annotated_func)

    # Deserialize and verify
    restored = loads(data)
    assert restored(5, 2.0) == 10.0
    assert restored.__annotations__ == {"x": int, "y": float, "return": float}


# =============================================================================
# Test 14: Function with docstring
# =============================================================================


def test_function_with_docstring():
    """Test that docstrings are preserved."""

    def documented_func(x):
        """This function doubles its input.

        Args:
            x: The value to double.

        Returns:
            The doubled value.
        """
        return x * 2

    # Serialize
    data = dumps(documented_func)

    # Deserialize and verify docstring is preserved
    restored = loads(data)
    assert restored.__doc__ is not None
    assert "doubles" in restored.__doc__


# =============================================================================
# Test 15: Function referencing other functions
# =============================================================================


def test_function_referencing_other_function():
    """Test functions that call other user-defined functions."""

    def helper(x):
        return x + 1

    def main_func(x):
        return helper(x) * 2

    # Serialize main_func - should capture helper too
    data = dumps(main_func)

    # Deserialize and verify
    restored = loads(data)
    assert restored(5) == 12  # (5+1)*2


# =============================================================================
# Test 16: Multiple functions with shared closure
# =============================================================================


def test_shared_closure():
    """Test multiple functions sharing the same closure variable."""
    shared_value = 10

    def func1():
        return shared_value * 2

    def func2():
        return shared_value + 5

    # Serialize both
    data1 = dumps(func1)
    data2 = dumps(func2)

    # Deserialize and verify they both work independently
    restored1 = loads(data1)
    restored2 = loads(data2)
    assert restored1() == 20
    assert restored2() == 15


# =============================================================================
# Test 17: Function with torch operations
# =============================================================================


def test_torch_operations():
    """Test functions that use torch operations via user module."""
    from mymethods.stateful import normalize

    # Use a user-defined function that uses torch
    data = dumps(normalize)
    restored = loads(data)

    # Create a tensor and normalize it
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = restored(x)

    # Verify the result has unit norm along the last dimension
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Test 18: Error line numbers (basic check)
# =============================================================================


def test_error_preserves_context():
    """Test that errors in restored functions have meaningful context."""

    def error_func(x):
        y = x + 1
        z = y / 0  # This will raise ZeroDivisionError
        return z

    # Serialize
    data = dumps(error_func)
    restored = loads(data)

    # Call and catch error
    with pytest.raises(ZeroDivisionError):
        restored(5)


# =============================================================================
# Test 19: Empty function
# =============================================================================


def test_empty_function():
    """Test functions with just pass."""

    def empty_func():
        pass

    # Serialize and verify
    data = dumps(empty_func)
    restored = loads(data)
    assert restored() is None


# =============================================================================
# Test 20: Imported function in model trace
# =============================================================================


@pytest.fixture(scope="module")
def tiny_model():
    """Create a tiny GPT-2 model for testing."""
    from nnsight import LanguageModel
    model = LanguageModel("hf-internal-testing/tiny-random-gpt2", dispatch=True)
    return model


@torch.no_grad()
def test_imported_function_in_trace(tiny_model):
    """Test that an imported function works with remote='local'."""
    from mymethods.stateful import normalize

    with tiny_model.trace("Hello", remote="local"):
        hidden = tiny_model.transformer.h[0].output[0]
        normed = normalize(hidden)
        result = normed.save()

    # Verify normalization worked (unit norm along last dim)
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
