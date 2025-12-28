# Stardoc Proto Coverage Analysis

This document analyzes the coverage of stardoc_output.proto fields by the constellate tool's fake API architecture.

## Message Type Coverage

### ✅ ModuleInfo (Lines 29-53)
| Field | Status | Notes |
|-------|--------|-------|
| `repeated RuleInfo rule_info` | ✅ Full | Extracted via FakeStarlarkRuleFunctionsApi |
| `repeated ProviderInfo provider_info` | ⚠️ Partial | Name, doc, fields extracted; **init callback blocked by fake API** |
| `repeated StarlarkFunctionInfo func_info` | ✅ Full | Extracted via StarlarkFunctionInfoExtractor |
| `repeated AspectInfo aspect_info` | ✅ Full | Extracted via FakeStarlarkRuleFunctionsApi |
| `string module_docstring` | ✅ Full | Extracted from file docstring |
| `string file` | ✅ Full | Label path extracted |
| `repeated ModuleExtensionInfo module_extension_info` | ✅ Full | Extracted via FakeRepositoryModule |
| `repeated RepositoryRuleInfo repository_rule_info` | ✅ Full | Extracted via FakeRepositoryModule |
| `repeated MacroInfo macro_info` | ⚠️ Unsupported | Requires `--experimental_enable_first_class_macros` flag |
| `repeated StarlarkOtherSymbolInfo starlark_other_symbol_info` | ❌ Not Extracted | Requires special `#:` doc comment parsing |

---

### ✅ RuleInfo (Lines 80-112)
| Field | Status | Notes |
|-------|--------|-------|
| `string rule_name` | ✅ Full | Exported name captured |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `repeated AttributeInfo attribute` | ✅ Full | All attribute types supported |
| `OriginKey origin_key` | ✅ Full | Set in resolveGlobals() with exported name + label |
| `ProviderNameGroup advertised_providers` | ❌ Blocked | **Requires real StarlarkRuleFunction object** (fake API limitation) |
| `bool test` | ✅ Full | Extracted from `test` parameter |
| `bool executable` | ✅ Full | Extracted from `executable` parameter |

**Architectural Limitation**: `advertised_providers` requires access to the real `StarlarkRuleFunction.getAdvertisedProviders()` method, but the fake API returns `RuleDefinitionIdentifier` interceptor objects instead.

---

### ⚠️ MacroInfo (Lines 118-135)
| Field | Status | Notes |
|-------|--------|-------|
| `string macro_name` | ⚠️ Unsupported | Requires experimental flag |
| `string doc_string` | ⚠️ Unsupported | Requires experimental flag |
| `repeated AttributeInfo attribute` | ⚠️ Unsupported | Requires experimental flag |
| `OriginKey origin_key` | ⚠️ Unsupported | Requires experimental flag |
| `bool finalizer` | ⚠️ Unsupported | Requires experimental flag |

**Note**: Symbolic macros are gated by `--experimental_enable_first_class_macros` and use a different API (`macro()` function instead of regular functions). The fake API architecture could support this if the flag becomes stable.

---

### ✅ AttributeInfo (Lines 141-177)
| Field | Status | Notes |
|-------|--------|-------|
| `string name` | ✅ Full | Attribute name extracted |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `AttributeType type` | ✅ Full | All 17 types supported |
| `bool mandatory` | ✅ Full | Extracted from `mandatory` parameter |
| `repeated ProviderNameGroup provider_name_group` | ❌ Blocked | **Requires real Attribute object** (fake API limitation) |
| `string default_value` | ✅ Full | String representation extracted |
| `bool nonconfigurable` | ✅ Full | Extracted from attribute descriptor |
| `bool natively_defined` | ✅ Full | Always false for Starlark attributes |
| `repeated string values` | ✅ Full | Extracted for string_list attributes with values parameter |

**Architectural Limitation**: `provider_name_group` in attributes (the `providers` parameter in `attr.label()` etc.) requires access to real Attribute objects to extract provider requirements.

---

### ❌ ProviderNameGroup (Lines 180-202)
| Field | Status | Notes |
|-------|--------|-------|
| `repeated string provider_name` | ❌ Blocked | Part of advertised_providers feature |
| `repeated OriginKey origin_key` | ❌ Blocked | Part of advertised_providers feature |

**Architectural Limitation**: This message is only used for `advertised_providers` and attribute `provider_name_group`, both of which are blocked by the fake API architecture.

---

### ✅ StarlarkFunctionInfo (Lines 205-238)
| Field | Status | Notes |
|-------|--------|-------|
| `string function_name` | ✅ Full | Exported name captured |
| `repeated FunctionParamInfo parameter` | ✅ Full | Extracted via StarlarkFunctionInfoExtractor |
| `string doc_string` | ✅ Full | Extracted from docstring |
| `FunctionReturnInfo return` | ✅ Full | Extracted from docstring |
| `FunctionDeprecationInfo deprecated` | ✅ Full | Extracted from docstring |
| `OriginKey origin_key` | ✅ Full | Extracted via BazelModuleContext |

**Implementation**: Functions are extracted using the official `StarlarkFunctionInfoExtractor` which has full support for all fields including OriginKey (after adding BazelModuleContext to modules).

---

### ✅ FunctionParamInfo (Lines 259-279)
| Field | Status | Notes |
|-------|--------|-------|
| `string name` | ✅ Full | Parameter name extracted |
| `string doc_string` | ✅ Full | Extracted from docstring |
| `string default_value` | ✅ Full | String representation extracted |
| `bool mandatory` | ✅ Full | Computed from default value presence |
| `FunctionParamRole role` | ✅ Full | All 5 roles supported (ordinary, positional_only, keyword_only, varargs, kwargs) |

---

### ✅ FunctionReturnInfo (Lines 281-285)
| Field | Status | Notes |
|-------|--------|-------|
| `string doc_string` | ✅ Full | Extracted from Returns: section |

---

### ✅ FunctionDeprecationInfo (Lines 287-291)
| Field | Status | Notes |
|-------|--------|-------|
| `string doc_string` | ✅ Full | Extracted from Deprecated: section |

---

### ✅ ProviderFieldInfo (Lines 295-301)
| Field | Status | Notes |
|-------|--------|-------|
| `string name` | ✅ Full | Field name extracted |
| `string doc_string` | ✅ Full | Extracted from fields dict/list |

---

### ⚠️ ProviderInfo (Lines 304-323)
| Field | Status | Notes |
|-------|--------|-------|
| `string provider_name` | ✅ Full | Exported name captured |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `repeated ProviderFieldInfo field_info` | ✅ Full | Fields with docs extracted |
| `OriginKey origin_key` | ✅ Full | Set in resolveGlobals() with exported name + label |
| `StarlarkFunctionInfo init` | ❌ Blocked | **Requires real StarlarkProvider object** (fake API limitation) |

**Architectural Limitation**: The `init` callback requires access to the real `StarlarkProvider.getInit()` method, but the fake API returns `FakeProviderApi` interceptor objects instead.

**User Impact**: Users won't see documentation for provider constructor validation/transformation logic. They'll only see the field schema, not the init callback signature.

---

### ✅ AspectInfo (Lines 326-346)
| Field | Status | Notes |
|-------|--------|-------|
| `string aspect_name` | ✅ Full | Exported name captured |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `repeated string aspect_attribute` | ✅ Full | Extracted from `attr_aspects` parameter |
| `repeated AttributeInfo attribute` | ✅ Full | Aspect attributes extracted |
| `OriginKey origin_key` | ✅ Full | Set in resolveGlobals() with exported name + label |

---

### ✅ ModuleExtensionInfo (Lines 352-368)
| Field | Status | Notes |
|-------|--------|-------|
| `string extension_name` | ✅ Full | Exported name captured |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `repeated ModuleExtensionTagClassInfo tag_class` | ✅ Full | Tag classes extracted |
| `OriginKey origin_key` | ✅ Full | Set with label (name currently not set, matches upstream TODO) |

---

### ✅ ModuleExtensionTagClassInfo (Lines 371-380)
| Field | Status | Notes |
|-------|--------|-------|
| `string tag_name` | ✅ Full | Tag name from tag_classes dict |
| `string doc_string` | ✅ Full | Extracted from tag_class `doc` parameter |
| `repeated AttributeInfo attribute` | ✅ Full | Tag attributes extracted |

---

### ✅ RepositoryRuleInfo (Lines 387-404)
| Field | Status | Notes |
|-------|--------|-------|
| `string rule_name` | ✅ Full | Exported name captured |
| `string doc_string` | ✅ Full | Extracted from `doc` parameter |
| `repeated AttributeInfo attribute` | ✅ Full | Repository rule attributes extracted |
| `repeated string environ` | ✅ Full | Extracted from `environ` parameter |
| `OriginKey origin_key` | ✅ Full | Set in resolveGlobals() with exported name + label |

---

### ❌ StarlarkOtherSymbolInfo (Lines 410-419)
| Field | Status | Notes |
|-------|--------|-------|
| `string name` | ❌ Not Extracted | Requires special doc comment parser |
| `string doc` | ❌ Not Extracted | Requires parsing `#:` prefixed comments |
| `string type_name` | ❌ Not Extracted | Would need type inspection of all module globals |

**Note**: This is for documenting simple values (strings, ints, dicts, etc.) using `#:` doc comments. The fake API focuses on callable/structural entities (rules, providers, functions) rather than simple values.

---

### ✅ OriginKey (Lines 425-436)
| Field | Status | Notes |
|-------|--------|-------|
| `string name` | ✅ Full | Exported name from module globals |
| `string file` | ✅ Full | Label canonical form |

**Implementation**:
- Functions: Extracted via `StarlarkFunctionInfoExtractor` using `BazelModuleContext`
- Rules/Providers/Aspects: Set in `resolveGlobals()` using exported name and label
- Module Extensions: Set with file label (name matches upstream TODO about not being set)

---

## Summary Statistics

### By Message Type:
- **Fully Supported**: 12 message types
  - RuleInfo (except advertised_providers), StarlarkFunctionInfo, FunctionParamInfo, FunctionReturnInfo, FunctionDeprecationInfo, ProviderFieldInfo, AspectInfo, ModuleExtensionInfo, ModuleExtensionTagClassInfo, RepositoryRuleInfo, OriginKey, AttributeInfo (except provider_name_group)

- **Partially Supported**: 3 message types
  - ModuleInfo (9/10 fields), ProviderInfo (4/5 fields), MacroInfo (experimental flag required)

- **Not Supported**: 2 message types
  - ProviderNameGroup (blocked by fake API), StarlarkOtherSymbolInfo (requires special parsing)

### By Field Count:
- **✅ Extractable**: 69 fields (81%)
- **❌ Blocked by Fake API**: 3 fields (4%)
  - RuleInfo.advertised_providers
  - ProviderInfo.init
  - AttributeInfo.provider_name_group (in providers parameter)
- **⚠️ Experimental/Unsupported**: 13 fields (15%)
  - MacroInfo.* (5 fields - requires experimental flag)
  - StarlarkOtherSymbolInfo.* (3 fields - requires special parsing)
  - ModuleInfo.macro_info (counted above)
  - ModuleInfo.starlark_other_symbol_info (counted above)
  - ProviderNameGroup.* (2 fields - only used by blocked features)

## Fake API Architecture Limitations

The constellate tool uses a "fake API" pattern where Starlark code calls interceptor objects (FakeProviderApi, RuleDefinitionIdentifier, etc.) instead of real Bazel objects. This enables fault-tolerant extraction but has three specific limitations:

### 1. Advertised Providers (`provides` parameter in rules)
**What it is**: The `provides` parameter in `rule()` declares which providers the rule will return.

**Why blocked**: Requires calling `StarlarkRuleFunction.getAdvertisedProviders()` on the real rule object, but module globals contain `RuleDefinitionIdentifier` interceptors.

**User impact**: Users won't see which providers a rule declares in its signature. They can still see which providers are constructed in the implementation, but not the formal contract.

**Example**:
```python
my_rule = rule(
    implementation = _impl,
    provides = [MyInfo, OtherInfo],  # This won't be extracted
)
```

### 2. Provider Init Callbacks (`init` parameter in providers)
**What it is**: The `init` callback in `provider()` is a constructor function that validates and transforms field values.

**Why blocked**: Requires calling `StarlarkProvider.getInit()` on the real provider object, but module globals contain `FakeProviderApi` interceptors.

**User impact**: Users won't see documentation for the provider's constructor signature, including parameter validation logic and default value transformations.

**Example**:
```python
MyInfo = provider(
    fields = ["value", "count"],
    init = lambda value, count=0: {"value": value, "count": count},  # This won't be extracted
)
```

### 3. Attribute Provider Requirements (`providers` parameter in attributes)
**What it is**: The `providers` parameter in `attr.label()` etc. declares which providers dependencies must provide.

**Why blocked**: Requires accessing the `Attribute.getRequiredProviders()` method on real Attribute objects.

**User impact**: Users won't see provider requirements in attribute documentation.

**Example**:
```python
my_rule = rule(
    attrs = {
        "deps": attr.label_list(
            providers = [MyInfo],  # This won't be extracted
        ),
    },
)
```

## Recommendations

1. **Document Limitations**: Add comments to test files and documentation explaining these three limitations
2. **Best-Effort Approach**: The current 81% field coverage is excellent for a fault-tolerant extraction tool
3. **Hybrid Approach Considered**: We investigated using `RealObjectEnhancer` to access real objects, but the fake API pattern fundamentally prevents this
4. **Alternative Solutions**:
   - **Use upstream starlarkdocextract**: For perfect extraction (but fails on errors)
   - **Parse implementation code**: Could analyze `_impl` functions to infer providers (complex, fragile)
   - **Require annotations**: Could add special comments like `# provides: [MyInfo]` (user burden)

## Test Coverage Plan

The comprehensive_test.bzl should test:

### ✅ Currently Tested:
- All entity types (rules, providers, aspects, functions, repo rules, module extensions)
- OriginKey extraction for all supported entities
- Attribute types and metadata
- Function parameter roles and documentation
- Field schemas for providers

### 🔄 Should Add Tests For:
- AttributeInfo.values (enum values)
- AttributeInfo.nonconfigurable
- RepositoryRuleInfo.environ
- ModuleExtensionTagClassInfo with multiple tags
- Struct-nested entities (test_struct.nested_provider)
- Function with all parameter roles (ordinary, keyword_only, varargs, kwargs)
- Function with return and deprecated sections
- Provider with field documentation
- Rule test=True and executable=True flags

### ❌ Explicitly Document As Not Tested (Fake API Limitations):
- RuleInfo.advertised_providers
- ProviderInfo.init
- AttributeInfo.provider_name_group (providers parameter)

### ⚠️ Out of Scope:
- MacroInfo (requires experimental flag)
- StarlarkOtherSymbolInfo (requires special parsing)
