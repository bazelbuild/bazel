# Starlark Proto Wrapper Message Analysis

Analysis of starlark.proto wrapper messages compared to stardoc_output.proto to identify missing wrappers for new entity types.

## Wrapper Message Pattern

The starlark.proto defines wrapper messages that combine:
1. **Core Info** - The stardoc_output.proto message (e.g., `RuleInfo`)
2. **Location** - A `SymbolLocation` with start/end position and name
3. **Children** - Nested wrapper messages (e.g., `repeated Attribute` in `Rule`)

**Purpose**: Provide location information for IDE features like go-to-definition, hover, etc.

## Current Wrapper Coverage

### ✅ Existing Wrappers in starlark.proto

| Wrapper Message | Wraps | Children | Status |
|-----------------|-------|----------|--------|
| `Rule` | `RuleInfo` | `repeated Attribute` | ✅ Exists |
| `Aspect` | `AspectInfo` | `repeated Attribute` | ✅ Exists |
| `Attribute` | `AttributeInfo` | None | ✅ Exists |
| `Provider` | `ProviderInfo` | `repeated ProviderField` | ✅ Exists |
| `ProviderField` | `ProviderFieldInfo` | None | ✅ Exists |
| `Function` | `StarlarkFunctionInfo` | `repeated FunctionParam` | ✅ Exists |
| `FunctionParam` | `FunctionParamInfo` | None | ✅ Exists |

### ❌ Missing Wrappers (New in stardoc_output.proto)

| Entity Type | Core Proto Message | Missing Wrapper? | Priority |
|-------------|-------------------|------------------|----------|
| **Repository Rules** | `RepositoryRuleInfo` | ❌ **YES** | **HIGH** |
| **Module Extensions** | `ModuleExtensionInfo` | ❌ **YES** | **HIGH** |
| **Module Extension Tag Classes** | `ModuleExtensionTagClassInfo` | ❌ **YES** | **HIGH** |
| **Macros** | `MacroInfo` | ❌ **YES** | **MEDIUM** |
| **Other Symbols** | `StarlarkOtherSymbolInfo` | ❌ **YES** | **LOW** |

## Detailed Analysis

### 1. Repository Rules (HIGH PRIORITY)

**stardoc_output.proto**:
```protobuf
message RepositoryRuleInfo {
  string rule_name = 1;
  string doc_string = 2;
  repeated AttributeInfo attribute = 3;
  repeated string environ = 4;
  OriginKey origin_key = 5;
}
```

**Missing starlark.proto wrapper**:
```protobuf
message RepositoryRule {
  stardoc_output.RepositoryRuleInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;  // Reuse existing Attribute wrapper
}
```

**Why needed**: Repository rules are first-class entities that need location tracking for IDE features.

**Current extraction status**: ✅ Extracted by constellate (see comprehensive_test.bzl)

---

### 2. Module Extensions (HIGH PRIORITY)

**stardoc_output.proto**:
```protobuf
message ModuleExtensionInfo {
  string extension_name = 1;
  string doc_string = 2;
  repeated ModuleExtensionTagClassInfo tag_class = 3;
  OriginKey origin_key = 4;
}
```

**Missing starlark.proto wrapper**:
```protobuf
message ModuleExtension {
  stardoc_output.ModuleExtensionInfo info = 1;
  SymbolLocation location = 2;
  repeated ModuleExtensionTagClass tag_class = 3;
}
```

**Why needed**: Module extensions are critical for Bzlmod support.

**Current extraction status**: ✅ Extracted by constellate (see comprehensive_test.bzl)

---

### 3. Module Extension Tag Classes (HIGH PRIORITY)

**stardoc_output.proto**:
```protobuf
message ModuleExtensionTagClassInfo {
  string tag_name = 1;
  string doc_string = 2;
  repeated AttributeInfo attribute = 3;
}
```

**Missing starlark.proto wrapper**:
```protobuf
message ModuleExtensionTagClass {
  stardoc_output.ModuleExtensionTagClassInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;  // Reuse existing Attribute wrapper
}
```

**Why needed**: Tag classes are sub-entities of module extensions, need location tracking.

**Current extraction status**: ✅ Extracted by constellate (nested in ModuleExtensionInfo)

---

### 4. Macros (MEDIUM PRIORITY)

**stardoc_output.proto**:
```protobuf
message MacroInfo {
  string macro_name = 1;
  string doc_string = 2;
  repeated AttributeInfo attribute = 3;
  OriginKey origin_key = 4;
  bool finalizer = 5;
}
```

**Missing starlark.proto wrapper**:
```protobuf
message Macro {
  stardoc_output.MacroInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;  // Reuse existing Attribute wrapper
}
```

**Why needed**: Symbolic macros are a new first-class entity type.

**Current extraction status**: ⚠️ Requires `--experimental_enable_first_class_macros` flag

**Priority**: Medium (experimental feature, but will become stable)

---

### 5. Other Symbols (LOW PRIORITY)

**stardoc_output.proto**:
```protobuf
message StarlarkOtherSymbolInfo {
  string name = 1;
  string doc = 2;
  string type_name = 3;
}
```

**Wrapper analysis**:
- These are simple values (strings, ints, dicts, etc.) documented with `#:` comments
- Constellate focuses on callable/structural entities, not simple values
- Less critical for IDE features

**Priority**: Low (out of scope for current use case)

---

## Module Message Update Required

The `Module` message currently embeds `stardoc_output.ModuleInfo` directly:

```protobuf
message Module {
    string name = 1;
    ModuleCategory category = 2;
    stardoc_output.ModuleInfo info = 3;  // ← Contains all entity types
    string filename = 4;
    repeated SymbolLocation symbol_location = 5;
    map<string,ValueInfo> global = 6;
    repeated LoadStmt load = 7;
}
```

**Problem**: No way to provide SymbolLocation for new entity types!

**Two Design Options**:

### Option A: Add Wrapper Lists to Module (Recommended)

```protobuf
message Module {
    string name = 1;
    ModuleCategory category = 2;
    stardoc_output.ModuleInfo info = 3;
    string filename = 4;
    repeated SymbolLocation symbol_location = 5;
    map<string,ValueInfo> global = 6;
    repeated LoadStmt load = 7;

    // NEW: Wrapper messages with locations
    repeated RepositoryRule repository_rule = 8;
    repeated ModuleExtension module_extension = 9;
    repeated Macro macro = 10;
}
```

**Pros**:
- Consistent with how we might use it (location-aware)
- Enables IDE features for new entity types
- Backward compatible (new fields)

**Cons**:
- Duplicates data (info field also has these entities)
- Clients must check both places

### Option B: Keep Only in ModuleInfo (Current State)

```protobuf
message Module {
    string name = 1;
    ModuleCategory category = 2;
    stardoc_output.ModuleInfo info = 3;  // Contains all entities
    // ... no wrapper fields
}
```

**Pros**:
- Simple, no duplication
- Single source of truth

**Cons**:
- ❌ No SymbolLocation for repository rules, module extensions, macros
- ❌ Can't support go-to-definition for these entities
- ❌ Inconsistent with Rule/Provider/Function which DO have wrappers

### Option C: Hybrid - Use symbol_location Map

```protobuf
message Module {
    string name = 1;
    ModuleCategory category = 2;
    stardoc_output.ModuleInfo info = 3;
    string filename = 4;
    repeated SymbolLocation symbol_location = 5;  // ← Generic for all entities
    // ... no wrapper fields needed
}
```

**Approach**: Store locations for all entities in `symbol_location`, match by name.

**Pros**:
- No duplication
- Single location list for all entity types

**Cons**:
- Requires name-based lookup (fragile)
- No type-specific metadata
- Less structured than dedicated wrappers

---

## Recommendation

### Phase 1: Add Missing Wrapper Messages (HIGH PRIORITY)

Add to starlark.proto:

```protobuf
message RepositoryRule {
  stardoc_output.RepositoryRuleInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;
}

message ModuleExtension {
  stardoc_output.ModuleExtensionInfo info = 1;
  SymbolLocation location = 2;
  repeated ModuleExtensionTagClass tag_class = 3;
}

message ModuleExtensionTagClass {
  stardoc_output.ModuleExtensionTagClassInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;
}

message Macro {
  stardoc_output.MacroInfo info = 1;
  SymbolLocation location = 2;
  repeated Attribute attribute = 3;
}
```

### Phase 2: Update Module Message

Add wrapper fields to Module:

```protobuf
message Module {
    string name = 1;
    ModuleCategory category = 2;
    stardoc_output.ModuleInfo info = 3;
    string filename = 4;
    repeated SymbolLocation symbol_location = 5;
    map<string,ValueInfo> global = 6;
    repeated LoadStmt load = 7;

    // Wrapper messages with locations for new entity types
    repeated RepositoryRule repository_rule = 8;
    repeated ModuleExtension module_extension = 9;
    repeated Macro macro = 10;
}
```

### Phase 3: Populate Wrappers in StarlarkServer

Update `StarlarkServer.evalModuleInfo()` to:
1. Extract repository rules, module extensions, macros from ModuleInfo
2. Get SymbolLocations from moduleBuilder
3. Create wrapper messages combining info + location
4. Add to Module

---

## Testing Strategy

### Test Coverage Needed

1. **RepositoryRule Wrapper Test**
   - Extract `my_repo_rule` from comprehensive_test.bzl
   - Verify `RepositoryRule` message has:
     - `info` field populated with name, doc, attributes, environ
     - `location` field with start/end position
     - `attribute` list with nested Attribute wrappers

2. **ModuleExtension Wrapper Test**
   - Extract `my_extension` from comprehensive_test.bzl
   - Verify `ModuleExtension` message has:
     - `info` field populated with name, doc, tag_classes
     - `location` field with start/end position
     - `tag_class` list with nested ModuleExtensionTagClass wrappers

3. **ModuleExtensionTagClass Wrapper Test**
   - Extract tag class from `my_extension`
   - Verify `ModuleExtensionTagClass` message has:
     - `info` field populated with name, doc, attributes
     - `location` field with start/end position
     - `attribute` list with nested Attribute wrappers

4. **Macro Wrapper Test** (when experimental flag is stable)
   - Extract `my_macro` from comprehensive_test.bzl
   - Verify `Macro` message has:
     - `info` field populated with name, doc, attributes, finalizer
     - `location` field with start/end position
     - `attribute` list with nested Attribute wrappers

### Test File Structure

Create `GrpcWrapperTest.java`:

```java
@Test
public void testRepositoryRuleWrapper() throws Exception {
    String label = "//testdata:comprehensive_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);

    // Verify wrapper message
    assertTrue("Should have repository rules", response.getRepositoryRuleCount() > 0);
    RepositoryRule repoRule = response.getRepositoryRule(0);

    // Verify info field
    assertTrue("Should have info", repoRule.hasInfo());
    assertEquals("my_repo_rule", repoRule.getInfo().getRuleName());

    // Verify location field
    assertTrue("Should have location", repoRule.hasLocation());
    assertTrue("Location should have name", !repoRule.getLocation().getName().isEmpty());
    assertTrue("Location should have start position", repoRule.getLocation().hasStart());

    // Verify nested attributes
    assertTrue("Should have attributes", repoRule.getAttributeCount() > 0);
}
```

---

## Summary

### Missing Wrappers:
- ✅ **RepositoryRule** - HIGH PRIORITY (extracted, no wrapper)
- ✅ **ModuleExtension** - HIGH PRIORITY (extracted, no wrapper)
- ✅ **ModuleExtensionTagClass** - HIGH PRIORITY (extracted, no wrapper)
- ⚠️ **Macro** - MEDIUM PRIORITY (experimental, no wrapper)
- ❌ **StarlarkOtherSymbolInfo** - LOW PRIORITY (out of scope)

### Next Steps:
1. Add 4 missing wrapper message definitions to starlark.proto
2. Update Module message to include wrapper field lists
3. Update StarlarkServer to populate wrapper messages
4. Create GrpcWrapperTest.java with comprehensive assertions
5. Update existing tests to validate wrapper fields

### Benefits:
- ✅ IDE features for all entity types (go-to-definition, hover, etc.)
- ✅ Consistent API design (all entities have location wrappers)
- ✅ Future-proof for new stardoc_output.proto additions
- ✅ Better user experience in language server/IDE integrations
