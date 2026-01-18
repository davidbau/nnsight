# Lambda Function Serialization

**Commit:** (pending)
**Date:** 2026-01-18

## The Challenge

Lambda functions present unique challenges for source-based serialization:

1. **Multiple lambdas on the same line**: `f, g = lambda x: x*2, lambda x: x+1`
2. **Nested lambdas**: `lambda x: lambda y: x + y`
3. **Lambdas with lambda defaults**: `lambda x=lambda: 1: x()`
4. **Multi-line lambdas** spanning multiple lines
5. **Lambda name is `<lambda>`**: Not valid Python syntax for `return <lambda>`

When `inspect.getsource()` is called on a lambda, it returns the entire line or statement containing the lambda - not just the lambda itself. This causes problems when multiple lambdas share a line.

## The Solution

### 1. Lambda Source Extraction (`_extract_lambda_source`)

Uses Python 3.11+'s `co_positions()` to get precise column information, then tokenizes the source to extract just the target lambda.

```python
def _extract_lambda_source(source: str, code: types.CodeType) -> str:
    """Extract a specific lambda's source when multiple lambdas share a line."""
```

**Algorithm:**
1. Get the lambda's bytecode position via `code.co_positions()`
2. Tokenize the source with Python's `tokenize` module
3. Find all lambda tokens and their body colons
4. Match the lambda whose colon is closest to (but before) the target position
5. Find the end of the lambda (comma, closing paren, or newline at depth 0)
6. Extract just that source span

**Tricky cases handled:**
- Nested structures: `([{` tracking for proper depth
- Nested lambdas: Separate lambda_depth counter for inner lambdas
- Lambda as default: `lambda x=lambda: 1: x()` - the inner lambda's colon doesn't end the outer

### 2. Lambda Handling in `make_function`

Lambdas have `<lambda>` as their `__name__`, which isn't valid Python syntax. We can't do `return <lambda>`.

**For closures (factory pattern):**
```python
if name == "<lambda>":
    # Assign lambda to temp variable
    indented_source = "    _lambda_result_ = " + source + "\n"
    factory_source += indented_source
    factory_source += "    return _lambda_result_\n"
else:
    # Normal function - can return by name
    factory_source += f"    return {name}\n"
```

**For no-closure case:**
```python
# Don't break early - for nested lambda defaults,
# multiple code objects have co_name=="<lambda>"
# The outermost lambda is last
for const in module_code.co_consts:
    if isinstance(const, types.CodeType) and const.co_name == name:
        func_code = const
        # Don't break for lambdas - we want the last (outermost) one
```

## Graceful Fallback

On Python < 3.11 (no `co_positions`), or if tokenization fails, we fall back to the full source. This works for simple cases but may fail for multiple lambdas on the same line.

```python
if code.co_name != "<lambda>" or not hasattr(code, "co_positions"):
    return source  # Fallback to original
```

## Implementation Details

### Tokenization Parsing

The tokenizer gives us tokens like:
```
NAME 'lambda'
NAME 'x'
OP ':'
NAME 'x'
OP '*'
NUMBER '2'
OP ','
NAME 'lambda'
...
```

We track:
- `depth`: Parentheses/brackets depth for knowing when comma ends lambda
- `lambda_depth`: Nested lambda count for knowing whose colon is whose
- `past_colon`: Whether we've seen the body colon (to detect enclosing lambda's colon)

### Multi-line Lambda Handling

Multi-line lambdas like:
```python
lambda y: (
    y * 2 +
    y * 3
)
```

Need to be wrapped in parentheses to be valid as a standalone expression:
```python
if "\n" in result:
    result = "(" + result + ")"
```

## Testing

Tests in `tests/test_lambda_serialization.py`:

| Test | Description |
|------|-------------|
| `test_simple_lambda` | Basic lambda `lambda x: x * 2` |
| `test_lambda_with_closure` | Lambda capturing variable |
| `test_lambda_with_multiple_args` | Multiple parameters |
| `test_lambda_with_default_args` | Default values including dicts with colons |
| `test_nested_lambda` | Lambda returning lambda |
| `test_lambda_as_default_value` | `lambda x=lambda: 1: x()` |
| `test_multiple_lambdas_same_line` | `f, g = lambda x: x*2, lambda x: x+1` |
| `test_multiline_lambda_with_neighbors` | Multi-line lambdas with others on same lines |
| `test_lambda_in_list` | Lambdas in a list |
| `test_lambda_in_dict` | Lambdas as dict values |

## Limitations

1. **Python 3.11+ required for precise extraction**: Older versions fall back to full source
2. **Source must be available**: Same as regular functions
3. **Complex multi-statement lambdas**: Walrus operator etc. may confuse tokenizer

## Code Changes

**`src/nnsight/intervention/serialization.py`**:

1. Added `import tokenize`
2. Added `_extract_lambda_source()` function (~140 lines)
3. Updated `_dynamic_function_reduce()` to call `_extract_lambda_source()`
4. Updated `make_function()` with lambda-specific handling for closures and code extraction
