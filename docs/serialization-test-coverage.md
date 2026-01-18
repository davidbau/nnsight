# Serialization Test Coverage

**Date:** 2026-01-18

## Overview

This document describes the comprehensive test suite for source-based function serialization. The tests cover edge cases discovered during development and ensure the serialization system handles real-world Python patterns correctly.

## Test Files

### `test_serialization_edge_cases.py`

Core edge case tests covering various function patterns:

| Test | Description |
|------|-------------|
| `test_closure_with_captured_variable` | Function capturing outer variable |
| `test_nested_closure` | Closure inside closure |
| `test_returned_inner_function` | Factory function returning inner function |
| `test_triple_nested_closure` | Three levels of nested closures |
| `test_mutable_closure_state` | Closure modifying nonlocal variable |
| `test_lambda_default_argument` | Lambda with default args |
| `test_complex_default_arguments` | Mutable defaults, None, nested |
| `test_comprehension_variable_shadowing` | Variables shadowed in comprehensions |
| `test_lambda_parameters_not_external` | Lambda params vs external captures |
| `test_lambda_with_external_and_param` | Mixed lambda params and closures |
| `test_constant_overriding_builtin` | Local `len = 5` overriding builtin |
| `test_nested_function_shadowing` | Inner function shadows outer var |
| `test_import_inside_function` | `import json` inside function body |
| `test_from_import_inside_function` | `from json import dumps` inside |
| `test_async_function` | `async def` functions |
| `test_generator_function` | Generator with `yield` |
| `test_deeply_nested_closure_state` | Deep closure state chains |
| `test_args_kwargs` | `*args, **kwargs` handling |
| `test_mixed_positional_keyword_args` | Complex signature patterns |
| `test_recursive_function` | Direct recursion (factorial) |
| `test_function_with_annotations` | Type annotations |
| `test_function_with_docstring` | Docstring preservation |
| `test_function_referencing_other_function` | Function calling another |
| `test_shared_closure` | Multiple functions sharing closure |
| `test_torch_operations` | PyTorch tensor operations |
| `test_error_preserves_context` | Error context preservation |
| `test_empty_function` | `def f(): pass` |
| `test_imported_function_in_trace` | Functions using imports |

### `test_local_recursion.py`

Tests for local/nested recursive functions:

| Test | Description |
|------|-------------|
| `test_local_recursive_function` | Nested function calling itself via closure |

### `test_mutual_recursion.py`

Tests for mutually recursive functions at module level:

| Test | Description |
|------|-------------|
| `test_mutual_recursion_single` | Serialize one function that calls another |
| `test_mutual_recursion_together` | Serialize both is_even/is_odd together |

### `test_local_mutual_recursion.py`

Tests for local mutually recursive functions (the hardest case):

| Test | Description |
|------|-------------|
| `test_local_mutual_recursion` | Two local functions referencing each other through closures |

### `test_lambda_serialization.py`

Tests for lambda function edge cases:

| Test | Description |
|------|-------------|
| `test_simple_lambda` | Basic lambda `lambda x: x * 2` |
| `test_lambda_with_closure` | Lambda capturing variable |
| `test_lambda_with_multiple_args` | `lambda x, y, z: x + y + z` |
| `test_lambda_with_default_args` | Defaults including dicts with colons |
| `test_nested_lambda` | Lambda returning lambda |
| `test_lambda_as_default_value` | `lambda x=lambda: 1: x()` |
| `test_multiple_lambdas_same_line` | `f, g = lambda x: x*2, lambda x: x+1` |
| `test_multiline_lambda_with_neighbors` | Multi-line lambdas sharing lines |
| `test_lambda_in_list` | Lambdas in collections |
| `test_lambda_in_dict` | Lambdas as dict values |

### `test_whitelist_serialization.py`

Tests for module whitelist behavior:

| Test | Description |
|------|-------------|
| `test_whitelisted_modules` | Known modules are whitelisted |
| `test_non_whitelisted_modules` | User modules not whitelisted |
| `test_whitelist_includes_submodules` | `torch.nn` whitelisted if `torch` is |
| `test_non_whitelisted_function_serialized_by_source` | Auto source serialization |
| `test_non_whitelisted_class_serialized_by_source` | Class source serialization |
| `test_non_whitelisted_instance_serialized_by_source` | Instance serialization |
| `test_user_function_in_trace` | User functions in traces |
| `test_user_class_in_trace` | User classes in traces |
| `test_module_state_isolation` | Module state doesn't leak |
| `test_serialization_includes_module_functions` | Dependencies captured |
| `test_no_register_needed` | No manual registration required |
| `test_whitelisted_function_not_source_serialized` | Whitelist uses references |

## Running Tests

```bash
# Run all serialization tests
pytest tests/test_serialization_edge_cases.py \
       tests/test_local_recursion.py \
       tests/test_mutual_recursion.py \
       tests/test_local_mutual_recursion.py \
       tests/test_lambda_serialization.py \
       tests/test_whitelist_serialization.py -v

# Run specific category
pytest tests/test_lambda_serialization.py -v  # Lambda tests only
pytest -k "recursion" -v  # All recursion tests
```

## Test Categories

### Closure Tests
Tests that closures (captured variables) are properly serialized and work after deserialization.

### Recursion Tests
Tests for self-referential functions: direct recursion, local recursion, mutual recursion (both module-level and local).

### Lambda Tests
Tests for lambda-specific challenges: source extraction, multiple on same line, nested lambdas, lambdas as defaults.

### Import Tests
Tests that imports inside functions work correctly after deserialization.

### Signature Tests
Tests for complex function signatures: defaults, *args, **kwargs, annotations.

### Integration Tests
Tests that verify the whitelist mechanism and automatic source serialization for user-defined code.
