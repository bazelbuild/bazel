# ModuleInfoRequest Workspace Field Analysis

Analysis of `workspace_cwd`, `workspace_output_base`, and `workspace_name` fields in the ModuleInfoRequest proto message to determine if they can be safely removed.

## Proto Definition

From `src/main/java/build/stack/starlark/v1beta1/starlark.proto`:

```protobuf
message ModuleInfoRequest {
    string target_file_label = 1;
    string workspace_name = 2;           // ← ANALYZE
    string rel = 3;
    repeated string symbol_names = 4;
    repeated string dep_roots = 5;
    string builtins_bzl_path = 6;
    string module_content = 7;
    string workspace_cwd = 8;            // ← ANALYZE
    string workspace_output_base = 9;    // ← ANALYZE
}
```

## Field Usage Analysis

### 1. `workspace_cwd` (Field 8)

**Status**: ❌ **COMPLETELY UNUSED - SAFE TO REMOVE**

**Evidence**:
- ✅ No getter calls: `getWorkspaceCwd()` never called in constellate code
- ✅ No setter calls: `setWorkspaceCwd()` never called in constellate code
- ✅ No field references: `workspace_cwd` only appears in proto definition
- ✅ Not passed to any constructor or method
- ✅ Not stored in any field

**Codebase Search Results**:
```bash
# Search entire constellate package
grep -r "getWorkspaceCwd\|setWorkspaceCwd\|workspace_cwd" src/main/java/build/stack/devtools/build/constellate/
# Result: Only found in proto files, never used in Java code
```

**Original Intent**: Likely intended for language server to resolve relative file paths, but never implemented.

**Recommendation**: **REMOVE** - This field provides zero value and clutters the API.

---

### 2. `workspace_output_base` (Field 9)

**Status**: ❌ **COMPLETELY UNUSED - SAFE TO REMOVE**

**Evidence**:
- ✅ No getter calls: `getWorkspaceOutputBase()` never called in constellate code
- ✅ No setter calls: `setWorkspaceOutputBase()` never called in constellate code
- ✅ No field references: `workspace_output_base` only appears in proto definition
- ✅ Not passed to any constructor or method
- ✅ Not stored in any field

**Codebase Search Results**:
```bash
# Search entire constellate package
grep -r "getWorkspaceOutputBase\|setWorkspaceOutputBase\|workspace_output_base" src/main/java/build/stack/devtools/build/constellate/
# Result: Only found in proto files, never used in Java code
```

**Original Intent**: Likely intended for language server to resolve Bazel output paths, but never implemented.

**Recommendation**: **REMOVE** - This field provides zero value and clutters the API.

---

### 3. `workspace_name` (Field 2)

**Status**: ⚠️ **MINIMALLY USED - CONSIDER REMOVING**

**Evidence**:

#### Used in 2 locations:

1. **StarlarkServer.java:160** - Passed to Constellate constructor
   ```java
   Constellate constellate = new Constellate(semantics, new FilesystemFileAccessor(),
       request.getWorkspaceName(), depRoots);
   ```

2. **Constellate.java:976** - Used in `pathOfLabel()` for workspace matching
   ```java
   if (label.getWorkspaceName().equals(workspaceName)) {
       logger.atInfo().log("2 pathOfLabel %s: workspaceRoot=%s", label, workspaceRoot);
       return Paths.get(label.toPathFragment().toString());
   }
   ```

#### Usage Analysis:

**What it does**:
- Stored as a private final field in Constellate
- Used in `pathOfLabel()` to determine if a label refers to the current workspace vs external workspace
- If label's workspace name matches the request's workspace name, the file is treated as local
- Otherwise, it's treated as an external dependency with `external/repo` prefix

**How it's currently used in tests**:
```java
// GrpcIntegrationTest.java - 5 tests use empty string
.setWorkspaceName("")

// GrpcIntegrationTest.java - 1 test uses "test_workspace" but doesn't validate it
.setWorkspaceName("test_workspace")  // testOriginKeyFileFormat
```

**Problem**: The logic at line 976 appears to have an issue:

```java
// Line 971: Get workspace root
String workspaceRoot = label.getWorkspaceRootForStarlarkOnly(semantics);

// Line 972-975: If no workspace root (local label), return path
if (workspaceRoot.isEmpty()) {
    return Paths.get(label.toPathFragment().toString());
}

// Line 976-978: If label's workspace == our workspace, return path
if (label.getWorkspaceName().equals(workspaceName)) {
    return Paths.get(label.toPathFragment().toString());
}

// Line 980-981: Otherwise, prepend external/repo
return Paths.get(workspaceRoot, label.toPathFragment().toString());
```

**Issue**: Lines 976-978 are **unreachable** if workspaceName is empty string (which it is in 5/6 tests):
- If `workspaceRoot.isEmpty()` is true (local workspace), we return at line 974
- If `workspaceRoot.isEmpty()` is false (external workspace), then `label.getWorkspaceName()` is non-empty
- But if `workspaceName` is empty string (which it always is in tests), then `label.getWorkspaceName().equals(workspaceName)` can only be true if `label.getWorkspaceName()` is also empty
- But if `label.getWorkspaceName()` is empty, then `workspaceRoot` would be empty, so we'd return at line 974

**Conclusion**: The check at line 976 is **dead code** when `workspaceName` is empty (which it is in 99% of usage).

#### Real-World Usage Scenarios:

**Scenario 1: Local workspace file (main repo)**
```
Label: //src/foo:bar.bzl
label.getWorkspaceRootForStarlarkOnly() → ""
label.getWorkspaceName() → ""
workspaceName from request → ""

Flow: Returns at line 974 (workspaceRoot.isEmpty())
Result: src/foo/bar.bzl
```

**Scenario 2: External workspace file**
```
Label: @rules_go//go:def.bzl
label.getWorkspaceRootForStarlarkOnly() → "external/rules_go"
label.getWorkspaceName() → "rules_go"
workspaceName from request → ""

Flow: Skips line 974, fails line 976 check ("rules_go" != ""), returns at line 981
Result: external/rules_go/go/def.bzl
```

**Scenario 3: External workspace file with matching workspace_name** (theoretical)
```
Label: @my_workspace//src/foo:bar.bzl
label.getWorkspaceRootForStarlarkOnly() → "external/my_workspace"
label.getWorkspaceName() → "my_workspace"
workspaceName from request → "my_workspace"

Flow: Skips line 974, passes line 976 check, returns at line 978
Result: src/foo/bar.bzl (WRONG! Should be external/my_workspace/src/foo/bar.bzl)
```

**BUG IDENTIFIED**: If workspace_name is set to match an external workspace, the code incorrectly treats it as a local file, stripping the "external/repo" prefix.

### Alternative: Always Use Empty String

**Option 1: Keep field, always pass empty string**
- Pro: Backward compatible
- Pro: Simple logic (always go to line 981 for external deps)
- Con: Dead code at line 976-978
- Con: Confusing API (why have a field that's always empty?)

**Option 2: Remove field entirely**
- Pro: Cleaner API
- Pro: Removes dead/buggy code
- Pro: Matches actual usage (empty string in 5/6 tests)
- Con: Requires code changes
- Con: May break external consumers (if any)

**Recommendation**: **REMOVE** - The field has buggy/dead code and provides no real value.

### Refactored Logic (After Removal):

```java
public Path pathOfLabel(Label label) throws EvalException {
    String workspaceRoot = label.getWorkspaceRootForStarlarkOnly(semantics);
    if (workspaceRoot.isEmpty()) {
        // Local workspace file
        return Paths.get(label.toPathFragment().toString());
    }
    // External workspace file
    return Paths.get(workspaceRoot, label.toPathFragment().toString());
}
```

Much simpler and correct!

---

## Summary Table

| Field | Used? | Purpose | Recommendation |
|-------|-------|---------|----------------|
| `workspace_cwd` | ❌ No | Language server feature (unimplemented) | **REMOVE** |
| `workspace_output_base` | ❌ No | Language server feature (unimplemented) | **REMOVE** |
| `workspace_name` | ⚠️ Barely | Buggy external workspace logic | **REMOVE** |

## Migration Plan

### Phase 1: Deprecate Fields (SAFE - No Breaking Changes)

1. **Update proto with deprecation**:
   ```protobuf
   message ModuleInfoRequest {
       string target_file_label = 1;
       string workspace_name = 2 [deprecated = true];
       string rel = 3;
       repeated string symbol_names = 4;
       repeated string dep_roots = 5;
       string builtins_bzl_path = 6;
       string module_content = 7;
       string workspace_cwd = 8 [deprecated = true];
       string workspace_output_base = 9 [deprecated = true];
   }
   ```

2. **Update StarlarkServer.java** to ignore workspace_name:
   ```java
   // Before
   Constellate constellate = new Constellate(semantics, new FilesystemFileAccessor(),
       request.getWorkspaceName(), depRoots);

   // After
   Constellate constellate = new Constellate(semantics, new FilesystemFileAccessor(),
       "", depRoots);  // Always use empty string
   ```

3. **Update Constellate.pathOfLabel()** to remove dead code:
   ```java
   public Path pathOfLabel(Label label) throws EvalException {
       String workspaceRoot = label.getWorkspaceRootForStarlarkOnly(semantics);
       if (workspaceRoot.isEmpty()) {
           return Paths.get(label.toPathFragment().toString());
       }
       // Workspace name check removed - was dead code
       return Paths.get(workspaceRoot, label.toPathFragment().toString());
   }
   ```

4. **Update tests** to not set workspace_name (or keep setting it to show it's ignored)

### Phase 2: Remove Fields (BREAKING - After Deprecation Period)

1. **Remove from proto**:
   ```protobuf
   message ModuleInfoRequest {
       string target_file_label = 1;
       // workspace_name removed (was field 2)
       string rel = 3;
       repeated string symbol_names = 4;
       repeated string dep_roots = 5;
       string builtins_bzl_path = 6;
       string module_content = 7;
       // workspace_cwd removed (was field 8)
       // workspace_output_base removed (was field 9)
   }
   ```

2. **Remove workspaceName from Constellate constructor**:
   ```java
   public Constellate(StarlarkSemantics semantics, StarlarkFileAccessor fileAccessor,
       List<String> depRoots) {  // workspaceName parameter removed
       this.semantics = semantics;
       this.fileAccessor = fileAccessor;
       // this.workspaceName = workspaceName; // REMOVED

       if (depRoots.isEmpty()) {
           this.depRoots = ImmutableList.of(".");
       } else {
           this.depRoots = depRoots;
       }
   }
   ```

3. **Remove workspaceName field**:
   ```java
   // private final String workspaceName; // REMOVED
   ```

## Testing

### Tests to Update:

1. **GrpcIntegrationTest.java** - Remove all `.setWorkspaceName()` calls (6 occurrences)
2. **Verify pathOfLabel() behavior** - Add test cases:
   - Local label: `//src/foo:bar.bzl` → `src/foo/bar.bzl`
   - External label: `@repo//pkg:file.bzl` → `external/repo/pkg/file.bzl`

## External Impact

**Who might be affected?**
- Any external consumers of the gRPC StarlarkServer
- Any code that constructs ModuleInfoRequest messages

**How to check**:
```bash
# Search for external repos using this proto
grep -r "ModuleInfoRequest" --include="*.go" --include="*.java" external_repos/
```

**Mitigation**:
- Phase 1 (deprecation) is backward compatible
- Phase 2 (removal) requires coordination with external consumers
- Consider versioning: Keep v1beta1 with deprecated fields, create v1 without them

## Conclusion

**Recommendation: REMOVE ALL THREE FIELDS**

1. **workspace_cwd**: Never used - safe to remove immediately
2. **workspace_output_base**: Never used - safe to remove immediately
3. **workspace_name**: Barely used, buggy logic, dead code - safe to remove after fixing pathOfLabel()

**Benefits**:
- Cleaner, simpler API
- Removes dead/buggy code
- Reduces confusion for API consumers
- Easier to maintain

**Risks**:
- Low risk (fields are unused or minimally used)
- Phase 1 (deprecation) has zero risk
- Phase 2 (removal) requires checking for external consumers

**Next Steps**:
1. Implement Phase 1 (deprecation + ignore values)
2. Wait for deprecation period (e.g., 1-2 releases)
3. Check for external usage
4. Implement Phase 2 (complete removal)
