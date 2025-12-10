# Name Parameter Forwarding Implementation

## Overview

This document describes the implementation of `name` parameter forwarding tracking in the Constellate Starlark analyzer. This enhancement provides a stronger signal for detecting wrapper functions (traditional Bazel macros) by tracking when the `name` parameter is explicitly forwarded from a function to underlying rules or macros.

## Motivation

A common pattern in Bazel is for wrapper functions (macros) to forward both the `name` parameter and additional kwargs to underlying rules:

```python
def go_binary(name, **kwargs):
    """Wrapper around native.go_binary with additional preprocessing."""
    native.go_binary(name=name, **kwargs)
```

Previously, only `**kwargs` forwarding was tracked via the `forwards_kwargs_to` field. However, tracking explicit `name` parameter forwarding provides a **stronger assertion** that a function is truly a wrapper macro, not just a helper function that happens to use `**kwargs`.

## Implementation Details

### 1. Proto Schema Changes

**File:** `src/main/java/build/stack/starlark/v1beta1/starlark.proto`

Added new field to the `Function` message at line 122:

```protobuf
message Function {
    stardoc_output.StarlarkFunctionInfo info = 1;
    SymbolLocation location = 2;
    repeated FunctionParam param = 3;
    repeated string calls_rule_or_macro = 4;
    repeated string forwards_kwargs_to = 5;

    // NEW: Tracks rules/macros that receive the 'name' parameter
    repeated string forwards_name_to = 6;
}
```

### 2. Core Analysis Logic

**File:** `src/main/java/build/stack/devtools/build/constellate/Constellate.java`

#### Data Structure (lines 365-369)
```java
// Tracks rules/macros that receive the 'name' parameter from a function
ImmutableListMultimap.Builder<String, Collection<String>> calledWithName =
    ImmutableListMultimap.builder();
```

#### Enhanced `resolveFunctionKwargs()` (lines 806-935)
For functions with `**kwargs`, this method now:
- Checks if the function has a `name` parameter
- Detects `Argument.Keyword` with name "name"
- Validates the value references the name parameter (direct reference or expression)
- Tracks which rules/macros receive the name parameter

#### New `resolveFunctionNameForwarding()` (lines 937-1038)
For functions **without** `**kwargs`, this dedicated method:
- Checks for name parameter in function signature
- Analyzes call expressions for name forwarding
- Handles both keyword (`name=name`) and positional forwarding
- Detects name usage in expressions (`name + "_suffix"`)

#### Helper Method: `referencesNameParameter()`
Recursively checks if an expression references the `name` parameter:
- Direct identifier: `name`
- Binary expressions: `name + "_lib"`
- Extensible to other expression types

### 3. Forwarding Pattern Detection

The implementation detects these patterns:

| Pattern | Example | Detected |
|---------|---------|----------|
| Keyword forwarding | `my_rule(name=name, **kwargs)` | ✓ |
| Positional forwarding | `my_rule(name, srcs=...)` | ✓ |
| Expression-based | `my_rule(name=name + "_lib")` | ✓ |
| Hardcoded value | `my_rule(name="fixed")` | ✗ |
| No name param | `def func(**kwargs): my_rule(**kwargs)` | ✗ |

### 4. Bug Fix: `resolveFunctionMacros()`

**Lines:** 713-768

Fixed existing bug where `ImmutableMap.Builder.build()` was called inside a loop, then additional `put()` calls were made, violating the builder contract and causing duplicate key errors.

**Old approach:**
```java
for (String ruleName : rulesCalledWithKwargs.keys()) {
  ImmutableMap<String, RuleInfo> rules = ruleInfoMap.build(); // ❌ Build in loop
  // ... later ...
  ruleInfoMap.put(macroName, macroInfo.build()); // ❌ Put after build
}
```

**New approach:**
```java
// Pre-build lookup map
Map<String, RuleInfo> ruleNameToInfo = new HashMap<>();
for (RuleInfoWrapper wrapper : ruleInfoList) {
  String name = ((PostAssignHookAssignableIdentifier) wrapper.getIdentifierFunction()).getAssignedName();
  ruleNameToInfo.put(name, wrapper.getRuleInfo().build());
}

// Track added macros to avoid duplicates
Set<String> addedMacros = new HashSet<>();
for (String ruleName : rulesCalledWithKwargs.keys()) {
  if (addedMacros.contains(macroName)) continue;
  // ... safe to add ...
  ruleInfoMap.put(macroName, macroInfo.build());
  addedMacros.add(macroName);
}
```

## Test Coverage

### Test File: `name_forwarding_test.bzl`

**Location:** `src/test/java/build/stack/devtools/build/constellate/testdata/name_forwarding_test.bzl`

Comprehensive test cases covering:

1. **`explicit_name_macro`** - Standard pattern: `my_rule(name=name, **kwargs)`
2. **`positional_name_macro`** - Positional forwarding: `my_rule(name, srcs=...)`
3. **`transformed_name_macro`** - Expression: `my_rule(name=name + "_lib")`
4. **`hardcoded_name_macro`** - No forwarding: `my_rule(name="fixed")`
5. **`multiple_name_macro`** - Forwarding to multiple rules
6. **`no_name_param_macro`** - Function without name parameter
7. **`name_from_kwargs_macro`** - Name comes from `**kwargs`
8. **`name_without_kwargs_macro`** - Name forwarding without `**kwargs`

### Integration Test: `testNameForwarding()`

**Location:** `src/test/java/build/stack/devtools/build/constellate/GrpcIntegrationTest.java:836-902`

Verifies:
- `forwards_name_to` field is correctly populated
- Different forwarding patterns are detected
- Functions without name forwarding show empty list
- Integration with `forwards_kwargs_to` tracking

**Test Status:** ✅ All tests passing

## Usage Example

### Input Starlark Code:
```python
def my_library_macro(name, srcs = [], **kwargs):
    """Wrapper around my_library rule."""
    my_library(
        name = name,
        srcs = srcs,
        **kwargs
    )
```

### Generated Proto Output:
```protobuf
Function {
  info {
    function_name: "my_library_macro"
    doc_string: "Wrapper around my_library rule."
    # ... parameter info ...
  }
  calls_rule_or_macro: "my_library"
  forwards_kwargs_to: "my_library"
  forwards_name_to: "my_library"    # ← NEW!
}
```

## Benefits

1. **Stronger Wrapper Detection** - Explicit name forwarding is a stronger signal than just `**kwargs` forwarding
2. **Pattern Differentiation** - Distinguish true wrapper macros from helper functions
3. **Better Tooling** - IDE/LSP can provide better autocomplete and navigation for macros
4. **Documentation** - Generated docs can highlight wrapper relationships more clearly

## Implementation Notes

### AST Analysis Approach

The implementation uses Starlark's AST visitor pattern:
- `NodeVisitor` traverses the function body
- `CallExpression` nodes are inspected for forwarding patterns
- `Argument.Keyword` and `Argument.Positional` are checked
- Expression values are recursively analyzed

### Performance Considerations

- Name parameter checking is O(n) where n = number of parameters (typically small)
- AST traversal is already done for `**kwargs` detection, minimal overhead
- Duplicate tracking uses `HashSet` for O(1) lookups

### Edge Cases Handled

1. **Positional arguments** - First positional arg can be name
2. **Expressions** - Binary operations like `name + "_suffix"` detected
3. **Multiple targets** - Same name forwarded to multiple rules
4. **No name parameter** - Gracefully handles functions without name param
5. **Null else blocks** - Fixed NPE when if-statement has no else clause

## Files Modified

1. `src/main/java/build/stack/starlark/v1beta1/starlark.proto` - Proto schema
2. `src/main/java/build/stack/devtools/build/constellate/Constellate.java` - Core logic
3. `src/test/java/build/stack/devtools/build/constellate/testdata/name_forwarding_test.bzl` - Test data
4. `src/test/java/build/stack/devtools/build/constellate/GrpcIntegrationTest.java` - Integration test

## Future Enhancements

Potential improvements:
1. Track other common parameters (e.g., `visibility`, `tags`)
2. Detect parameter transformation patterns
3. Track parameter flow through multiple wrapper layers
4. Provide warnings for inconsistent forwarding patterns

## Related Work

This implementation builds on the existing `forwards_kwargs_to` tracking and complements:
- `calls_rule_or_macro` - Broad tracking of all rule/macro calls
- `forwards_kwargs_to` - Specific `**kwargs` forwarding tracking
- `forwards_name_to` - **NEW** explicit name parameter tracking

---

**Implementation Date:** December 8, 2025
**Author:** Claude (with guidance from user pcj)
**Status:** ✅ Complete and tested
