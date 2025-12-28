# Constellate Test Suite

Comprehensive test suite for the Bazel constellate tool, covering all supported entity types and enhancement features.

## Overview

This test suite validates:

1. **Basic Extraction**: All Starlark entity types (rules, providers, functions, aspects, macros, repository rules, module extensions)
2. **Phase 1 - OriginKey**: Extraction of origin keys (defining file + export name) for all entities
3. **Phase 2 - Advertised Providers**: Extraction of providers that rules declare they return
4. **Phase 3 - Provider Init & Schema**: Extraction of provider init callbacks and field documentation

## Files

### Test Data

- `testdata/simple_test.bzl` - Basic test with simple rule, provider, and function
- `testdata/comprehensive_test.bzl` - Comprehensive test covering all entity types and features:
  - Provider with init callback and schema
  - Provider without init
  - Rule with advertised providers
  - Aspect
  - Repository rule
  - Symbolic macro (requires `--experimental_enable_first_class_macros`)
  - Module extension with tag classes (requires Bzlmod)
  - Function with comprehensive docstring
  - Nested entities in structs

### Test Scripts

- `run_tests.sh` - Main test runner that:
  - Builds constellate
  - Runs constellate on test files
  - Validates output files are created
  - Reports pass/fail status

- `validate_output.py` - Python validation script that:
  - Parses ModuleInfo protobuf output
  - Validates OriginKey presence and correctness
  - Validates advertised providers
  - Validates provider init callbacks
  - Validates provider schema field documentation
  - Prints detailed summary and validation results

## Usage

### Run All Tests

```bash
./run_tests.sh
```

This will:
1. Build the constellate tool
2. Run it on all test files
3. Report pass/fail status

### Validate Output

To validate a specific output file:

```bash
./validate_output.py /tmp/constellate_test_output/comprehensive_test.pb
```

### Manual Test

To manually test constellate on a file:

```bash
# Build constellate
bazel build //src/main/java/build/stack/devtools/build/constellate:constellate

# Run on a test file
bazel-bin/src/main/java/build/stack/devtools/build/constellate/constellate \
  --input=src/test/java/build/stack/devtools/build/constellate/testdata/simple_test.bzl \
  --output=/tmp/output.pb

# Validate the output
./validate_output.py /tmp/output.pb
```

## Expected Results

### Simple Test (`simple_test.bzl`)

Should extract:
- 1 function (`simple_function`) with OriginKey
- 1 provider (`SimpleInfo`) with OriginKey and field schema
- 1 rule (`simple_rule`) with OriginKey

### Comprehensive Test (`comprehensive_test.bzl`)

Should extract:
- 2 providers (`MyInfoProvider`, `SimpleProvider`) with OriginKeys
  - `MyInfoProvider` should have init callback
  - Both should have field schemas
- 1 rule (`my_rule`) with OriginKey and advertised providers
- 1 aspect (`my_aspect`) with OriginKey
- 1 repository rule (`my_repo_rule`) with OriginKey and environ
- 1 macro (`my_macro`) with OriginKey (if --experimental_enable_first_class_macros enabled)
- 1 module extension (`my_extension`) with OriginKey and tag classes (if Bzlmod enabled)
- 1 function (`my_function`) with OriginKey and comprehensive docstring
- Nested entities in `test_struct` (if enhancement supports nested discovery)

## Validation Checks

The `validate_output.py` script performs these checks:

### OriginKey Validation
- ✓ Every entity has an `origin_key` field
- ✓ `origin_key.name` is set (export name)
- ✓ `origin_key.file` is set (defining file label)

### Advertised Providers (Rules)
- ✓ Rules with `provides` have `advertised_providers` field
- ✓ Provider names and origin keys match in count
- ✓ Each provider has correct origin key

### Provider Init Callbacks
- ✓ Providers with `init` parameter have `init` field in output
- ✓ Init function has `function_name` and `origin_key`

### Provider Schema
- ✓ Provider fields have documentation strings

## Adding New Tests

To add a new test case:

1. Create a new `.bzl` file in `testdata/`
2. Add the test to `run_tests.sh`:
   ```bash
   run_test "my_new_test" "$TESTDATA_DIR/my_new_test.bzl"
   ```
3. Run the test suite and validate output

## Troubleshooting

### ModuleInfo protobuf not found

If `validate_output.py` cannot import the protobuf:

```bash
# Generate Python protobuf bindings
bazel build //src/main/protobuf:stardoc_output_py_pb2

# Or use protoc directly
protoc --python_out=. src/main/protobuf/stardoc_output.proto
```

### Constellate execution errors

Check the constellate logs for detailed error messages. Common issues:
- Missing dependencies in BUILD file
- Starlark syntax errors in test files
- Unsupported Bazel features (macros, module extensions)

### Missing OriginKey warnings

Some entities may not have complete OriginKeys if:
- The entity was not exported as a global
- The entity is nested in a struct (limited support)
- The evaluation failed and enhancement couldn't run

These are expected in some cases due to the "best-effort" nature of the enhancement.

## Implementation Notes

### Hybrid Approach

Constellate uses a hybrid approach:

1. **Fake API Pattern**: Intercepts Starlark API calls during evaluation to capture entity definitions in a fault-tolerant way
2. **Real Object Enhancement**: After evaluation, matches real evaluated objects to captured definitions to extract rich metadata (OriginKey, advertised providers, provider init)

This approach combines:
- **Fault tolerance** from the fake API pattern (works even if evaluation fails)
- **Rich metadata** from real object inspection (OriginKey, type information)

### Enhancement Process

The `RealObjectEnhancer` class:
1. Builds lookup maps from module globals
2. Matches StarlarkRuleFunction → RuleInfoWrapper
3. Matches StarlarkProvider → ProviderInfoWrapper
4. Matches StarlarkDefinedAspect → AspectInfoWrapper
5. Matches MacroFunction → MacroInfoWrapper
6. Matches RepositoryRuleFunction → RepositoryRuleInfoWrapper
7. Matches ModuleExtension → ModuleExtensionInfoWrapper
8. Extracts OriginKey, advertised providers, and provider init for each

### Graceful Degradation

Enhancement failures are logged but don't fail the overall extraction. This ensures that even if enhancement fails, the basic fake API extraction still produces valid output.

## Related Files

- `src/main/java/build/stack/devtools/build/constellate/RealObjectEnhancer.java` - Enhancement implementation
- `src/main/java/com/google/devtools/build/lib/starlarkdocextract/` - Upstream extractors
- `src/main/protobuf/stardoc_output.proto` - Output format definition
- `CONSTELLATE_INTEGRATION_ANALYSIS.md` - Design documentation
