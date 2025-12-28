# Constellate Tool Test Suite

This directory contains tests for the constellate tool, which extracts Starlark documentation from `.bzl` files in a fault-tolerant manner.

## Documentation

- **[STARDOC_PROTO_COVERAGE.md](STARDOC_PROTO_COVERAGE.md)** - Comprehensive analysis of stardoc_output.proto field coverage, including architectural limitations
- **[TEST_COVERAGE_SUMMARY.md](TEST_COVERAGE_SUMMARY.md)** - Summary of current test coverage and recommendations for improvement
- **[CONSTELLATE_INTEGRATION_ANALYSIS.md](CONSTELLATE_INTEGRATION_ANALYSIS.md)** - Analysis of integrating starlarkdocextract tools (OriginKey implementation)

## Test Files

### ConstellateTest.java
Unit tests for the Constellate API.

**Tests**:
- `testSimpleFile()` - Basic extraction of functions, providers, and rules
- `testComprehensiveFile()` - Detailed validation of all proto fields including:
  - StarlarkFunctionInfo with all parameter types (ordinary, keyword-only, varargs, kwargs)
  - ProviderInfo with field schemas
  - RuleInfo with attributes and attribute metadata
  - AspectInfo with aspect_attributes and attributes
  - OriginKey extraction for all entity types
  - Module docstring

**Coverage**: 47 proto fields validated (55% of total, 81% of extractable fields)

**Known Limitations** (documented as TODOs):
- ProviderInfo.init - blocked by fake API
- RuleInfo.advertised_providers - blocked by fake API
- AttributeInfo.provider_name_group - blocked by fake API

### GrpcIntegrationTest.java
Integration tests for the full gRPC request/response pipeline.

**Tests**:
- `testSingleFileExtraction()` - Basic gRPC extraction with OriginKey validation
- `testMultiFileExtraction()` - Cross-file loading and OriginKey preservation
- `testComprehensiveFileExtraction()` - All entity types via gRPC
- `testBestEffortExtractionOnLoadFailure()` - Fault-tolerant extraction when loads fail
- `testSymbolFiltering()` - Symbol filtering functionality
- `testOriginKeyFileFormat()` - Label format validation

**Coverage**: Full end-to-end pipeline including repository rules and module extensions

### DebugTest.java
Debugging test for understanding module globals and fake API behavior.

**Purpose**: Development/debugging tool that intentionally throws WIP exception.

## Test Data

### testdata/simple_test.bzl
Basic test file with:
- One function (`simple_function`)
- One provider (`SimpleInfo`)
- One rule (`simple_rule`)

**Purpose**: Smoke test and basic OriginKey extraction

### testdata/comprehensive_test.bzl
Comprehensive test file with:
- **Providers**:
  - `MyInfoProvider` - with init callback and field schema
  - `SimpleProvider` - without init
  - `test_struct.nested_provider` - nested in struct
- **Rules**:
  - `my_rule` - with attributes and advertised providers
- **Aspects**:
  - `my_aspect` - with attr_aspects and attributes
- **Repository Rules**:
  - `my_repo_rule` - with environ variables
- **Module Extensions**:
  - `my_extension` - with tag classes
- **Functions**:
  - `my_function` - with all parameter types, return, and deprecation docs
  - `test_struct.nested_function` - nested in struct

**Purpose**: Comprehensive field validation

### testdata/load_test_lib.bzl & load_test_main.bzl
Library and main files for testing cross-file loading.

**Purpose**: OriginKey tracking across load statements

### testdata/load_failure_test.bzl
File with failing load statement.

**Purpose**: Best-effort extraction validation

## Running Tests

```bash
# Run all constellate tests
bazel test //src/test/java/build/stack/devtools/build/constellate:all

# Run unit tests only
bazel test //src/test/java/build/stack/devtools/build/constellate:ConstellateTest

# Run integration tests only
bazel test //src/test/java/build/stack/devtools/build/constellate:GrpcIntegrationTest

# Run with verbose output
bazel test //src/test/java/build/stack/devtools/build/constellate:ConstellateTest --test_output=all
```

## Coverage Summary

### Extractable Fields: 81% Coverage
- **Fully Extracted**: 69 fields (functions, providers, rules, aspects, repo rules, module extensions)
- **Blocked by Fake API**: 3 fields (advertised_providers, provider init, attribute provider requirements)
- **Out of Scope**: 13 fields (macros, other symbols)

### Tested Fields: 55% Coverage
- **Unit Tests**: 43 fields with detailed validation
- **Integration Tests**: 4 additional fields via gRPC pipeline
- **Not Explicitly Tested**: 11 extractable fields (test/executable flags, enum values, etc.)

See [TEST_COVERAGE_SUMMARY.md](TEST_COVERAGE_SUMMARY.md) for detailed field-by-field coverage.

## Architectural Notes

### Fake API Pattern
The constellate tool uses interceptor objects (FakeProviderApi, RuleDefinitionIdentifier, etc.) instead of real Bazel objects. This enables:
- ✅ Fault-tolerant extraction (continues on errors)
- ✅ Fast extraction without full Bazel analysis
- ✅ Most metadata extraction (81% of fields)

But prevents:
- ❌ Advertised providers (requires real StarlarkRuleFunction)
- ❌ Provider init callbacks (requires real StarlarkProvider)
- ❌ Attribute provider requirements (requires real Attribute objects)

See [STARDOC_PROTO_COVERAGE.md](STARDOC_PROTO_COVERAGE.md) for detailed explanation.

### OriginKey Implementation
OriginKey extraction required:
1. Adding `BazelModuleContext` to modules via `Module.withPredeclaredAndData()`
2. Using `StarlarkFunctionInfoExtractor` for functions (includes OriginKey natively)
3. Setting OriginKey manually in `resolveGlobals()` for rules/providers/aspects

See [CONSTELLATE_INTEGRATION_ANALYSIS.md](CONSTELLATE_INTEGRATION_ANALYSIS.md) for integration analysis.

## Contributing

When adding new test cases:

1. **Add to comprehensive_test.bzl** if testing a new entity type or proto field
2. **Add assertions to testComprehensiveFile()** to validate the new field
3. **Update STARDOC_PROTO_COVERAGE.md** if changing architectural coverage
4. **Update TEST_COVERAGE_SUMMARY.md** if adding/changing test coverage
5. **Document fake API limitations** as TODO comments referencing STARDOC_PROTO_COVERAGE.md

## References

- [Stardoc Output Proto](../../../../main/protobuf/stardoc_output.proto) - Protocol buffer definition
- [Starlarkdocextract Package](../../../../main/java/com/google/devtools/build/lib/starlarkdocextract/) - Upstream extraction tools
- [FakeApi Package](../../../main/java/build/stack/devtools/build/constellate/fakebuildapi/) - Fake API interceptors
