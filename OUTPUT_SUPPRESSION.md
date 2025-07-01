# Output Suppression in Bazel

This document outlines a proposal for a new mechanism to manage and suppress output in Bazel.

## 1. State of the World

Bazel builds can often produce a large volume of warnings, debug messages, and other "spew". While this output can be useful for debugging, it can also be overwhelming and make it difficult to identify important issues. This is especially true when dealing with third-party dependencies that you don't control.

### Existing Mechanisms

There is no single, unified mechanism in Bazel for suppressing output. Instead, there are a variety of techniques that can be used, each with its own trade-offs:

*   **`--output_filter=<regex>`**: This flag can be used to filter build and compilation warnings by target. However, this is a blunt instrument, as it filters by target, not by the content of the warning.
*   **Compiler-Specific Flags (`copts`, etc.)**: These flags can be used to pass flags directly to the compiler to suppress specific warnings. However, this is specific to the toolchain and not a general solution.
*   **Rule-level Attributes**: Some rules may provide attributes to control their verbosity, such as a `quiet = True` option. However, this is implemented on a case-by-case basis by the rule author.
*   **Shell-level Filtering**: Piping Bazel's stderr to another tool like `grep -v` is a common but crude method that can accidentally hide real errors.
*   **Patching Dependencies**: Patching dependencies to remove the code that generates warnings is effective but creates a significant maintenance burden.

The ideal solution would be a sustainable, project-level, and scoped mechanism for managing spew.

## 2. Proposed User Experience

The proposed solution is to introduce a new repeatable flag, `--output_suppression`. This allows suppression rules to be defined directly in `.bazelrc` files, which can be checked into version control and shared by all developers.

### Example 1: Cleaning up Java Dependency Warnings

This example focuses on suppressing common warnings encountered when dealing with Java dependencies via `rules_jvm_external`.

**Before:**
```
% bazelisk build //src:bazel
Starting local Bazel server (8.3.0) and connecting to it...

INFO: Invocation ID: 42d12b3d-b768-4211-be76-a3dc8681ee51
DEBUG: /private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/rules_jvm_external+/private/extensions/maven.bzl:155:14: The maven repository 'maven' is used in two different bazel modules, originally in 'bazel' and now in 'grpc-java'
DEBUG: /private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/rules_jvm_external+/private/extensions/maven.bzl:155:14: The maven repository 'maven' is used in two different bazel modules, originally in 'bazel' and now in 'protobuf'
WARNING: /private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/+async_profiler_repos+async_profiler_macos/BUILD.bazel:4:10: in _copy_file rule @@+async_profiler_repos+async_profiler_macos//:libasyncProfiler: target '@@+async_profiler_repos+async_profiler_macos//:libasyncProfiler' depends on deprecated target '@@bazel_tools//src/conditions:host_windows_x64_constraint': No longer used by Bazel and will be removed in the future. Migrate to toolchains or define your own version of this setting.
DEBUG: /private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/rules_jvm_external+/private/rules/coursier.bzl:766:18: Found duplicate artifact versions
    com.google.auth:google-auth-library-credentials has multiple versions 1.6.0, 1.23.0
    com.google.auth:google-auth-library-oauth2-http has multiple versions 1.6.0, 1.23.0
...
```

**`.bazelrc` Suppressions:**
```
# Suppress known, non-actionable warnings from external dependencies.
# See https://our-issue-tracker/XYZ for details.

# This warning appears twice, and we want to be notified if that changes.
build --output_suppression="count:2 package://external/rules_jvm_external The maven repository 'maven' is used in two different bazel modules"

# Suppress all other known warnings from external packages.
build --output_suppression="package://external/.* depends on deprecated target"
build --output_suppression="package://external/rules_jvm_external Found duplicate artifact versions"
build --output_suppression="package://external/protobuf The py_proto_library macro is deprecated"
build --output_suppression="package://external/protobuf.* AccessController in java.security has been deprecated"
build --output_suppression="package://external/zstd-jni finalize() in Object has been deprecated"
build --output_suppression="package://external/protobuf.* deprecated item is not annotated with @Deprecated"
```

**Result:**
```
Starting local Bazel server (8.3.0) and connecting to it...

INFO: Invocation ID: 42d12b3d-b768-4211-be76-a3dc8681ee51
INFO: Analyzed target //src:bazel (546 packages loaded, 13860 targets configured).
INFO: Found 1 target...
INFO: Suppressed 34 warnings via --output_suppression; run with --no_output_suppression to see.
Target //src:bazel up-to-date:
  bazel-bin/src/bazel
INFO: Build completed successfully, 5153 total actions
```

### Example 2: Cleaning up protoc and Bzlmod warnings

This example from the Pigweed project shows how to suppress warnings related to Bzlmod version resolution and noisy output from `ProtoCompile` actions.

**Before:**
```
% bazelisk build  //pw_string/...
INFO: Invocation ID: 24667b28-f78e-4be1-a37d-e910b17ab082
WARNING: For repository 'com_google_protobuf', the root module requires module version protobuf@28.2, but got protobuf@29.0 in the resolved dependency graph. Please update the version in your MODULE.bazel or set --check_direct_dependencies=off
WARNING: For repository 'abseil-cpp', the root module requires module version abseil-cpp@20240116.1, but got abseil-cpp@20250127.0 in the resolved dependency graph. Please update the version in your MODULE.bazel or set --check_direct_dependencies=off
WARNING: WORKSPACE support will be removed in Bazel 9 (late 2025), please migrate to Bzlmod, see https://bazel.build/external/migration.
INFO: Analyzed 62 targets (383 packages loaded, 21553 targets configured).
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/source_context_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/descriptor_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/compiler/plugin_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/struct_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/timestamp_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/wrappers_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/field_mask_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/any_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/type_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/api_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/empty_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: From ProtoCompile external/protobuf+/python/google/protobuf/duration_pb2.py [for tool]:
external/protobuf+/.: warning: directory does not exist.
INFO: Found 62 targets...
INFO: Elapsed time: 163.052s, Critical Path: 19.64s
INFO: 790 processes: 972 action cache hit, 64 internal, 726 darwin-sandbox.
INFO: Build completed successfully, 790 total actions
⏳ Generating compile commands...
```

**`.bazelrc` Suppressions:**
```
# Suppress Bzlmod direct dependency warnings, which are not critical for this project.
build --output_suppression="*: For repository '.*', the root module requires module version .* but got .* in the resolved dependency graph"

# Suppress WORKSPACE deprecation warning during the migration period.
build --output_suppression="*: WORKSPACE support will be removed in Bazel 9"

# Suppress known protoc warnings about a missing directory from the external protobuf repository.
# We expect exactly 12 of these; if this number changes, we want the build to notify us.
build --output_suppression="count:12 package://external/protobuf.* warning: directory does not exist"
```

**After:**
```
% bazelisk build  //pw_string/...
INFO: Invocation ID: 24667b28-f78e-4be1-a37d-e910b17ab082
INFO: Found 62 targets...
INFO: Suppressed 15 messages via --output_suppression; pass --no_output_suppression to display.
INFO: Elapsed time: 163.052s, Critical Path: 19.64s
INFO: 790 processes: 972 action cache hit, 64 internal, 726 darwin-sandbox.
INFO: Build completed successfully, 790 total actions
⏳ Generating compile commands...
```

## 3. Design

The implemented solution introduces a new repeatable flag, `--output_suppression`, and a filtering mechanism that is performant and backward-compatible. It uses a new, generic `EventContext` object to provide rich, decoupled information for events that have it, and falls back to parsing raw file paths for events that don't (like those from Starlark).

### Decoupled Data Binding via `EventContext`

To avoid tightly coupling event generators with the suppression system, we introduced a generic context object. The code that creates an event should not need to know *why* context is needed, only that it should provide it. This design is focused on the standard `Event` stream that is processed by the `Reporter` and ultimately displayed to the user on the console; it does not affect the `BuildEvent` stream.

1.  **`EventContext.java`:** A simple POJO that acts as a generic container for information about an event's origin. It has fields for `targetLabel`, `package`, etc.
2.  **Instrument Event Creation:**
    *   **For Actions:** In `SkyframeActionExecutor`, when processing an action's result, we gather context from the `ActionOwner` and attach an `EventContext` object to any warning/info events.
    *   **For Core Bazel Warnings:** For warnings generated outside of an action context (e.g., during module resolution), we instrumented the specific call sites to add an `EventContext`.
    *   **For Starlark `print()` and other path-based events:** These events often lack a rich `EventContext`. The filter detects this and falls back to inspecting the `event.getLocation().file()` property, allowing filtering by the raw file path of the `.bzl` file that contains the `print` statement.

### Suppression Rule Format

Each instance of the `--output_suppression` flag defines a single rule. The rule string consists of one or more space-separated keywords followed by a regular expression.

**Syntax:** `[keyword:value]... <regular_expression>`

**Supported Keywords:**

*   `package`: Scopes the rule to a specific package (e.g., `package:@@foo//bar`). Best for events with a rich `EventContext`.
*   `path`: Scopes the rule to events where the file path in the `Location` matches the provided regex. This is the most reliable way to suppress warnings from Starlark files or other events that don't have a package context.
*   `target`: Scopes the rule to a specific target (e.g., `target://foo/bar:baz`).
*   `tag`: Scopes the rule to events with a specific tag.
*   `count`: (Optional) Specifies an exact number of expected matches for a rule. If the actual number of matches is different, a warning is issued.

### Relationship with Existing Flags

The new `--output_suppression` flag takes precedence over the existing `--output_filter` and `--auto_output_filter` flags. If any `--output_suppression` rules are active, the other two flags are ignored, and a warning is printed to the console to inform the user of the conflict.

### Detailed Implementation Plan

#### Phase 1: Standalone Filter Implementation
This phase will be developed outside of Bazel to perfect the logic.

1.  **`EventContext.java`:** Define the generic context POJO.
2.  **`SuppressionRule.java`:** Create a class to represent a single parsed suppression rule.
3.  **`OutputSuppressionFilter.java`:** Implement the `OutputFilter` interface. Its core logic will parse rule strings, manage match counts, and apply rules based on the `EventContext` found in an `Event`'s properties.
4.  **Post-Build Verification:** The filter will have a `verifyCounts(EventHandler reporter)` method to check for `count` mismatches.

#### Phase 2: Core Bazel Integration
1.  **Add Flag:** Add the repeatable `--output_suppression` flag to `OutputFilteringModule.java`.
2.  **Plumb Filter:** In `OutputFilteringModule`, create and install the `OutputSuppressionFilter` on the `Reporter`. Ensure `verifyCounts` is called in `afterCommand()`.

#### Phase 3: Enriching Events (Incremental)
1.  **Instrument Action Execution:** In `SkyframeActionExecutor`, construct and attach the `EventContext` to action-related warning/info events.
2.  **Instrument Core Warnings:** Incrementally find the source of other core Bazel warnings (like Bzlmod resolution) and enrich their corresponding `Event` creation sites with relevant context.

### Relevant Files

This section lists files that are relevant to the implementation of this feature and may need to be revisited.

*   **`src/main/java/com/google/devtools/build/lib/outputfilter/OutputFilteringModule.java`**: Where the `--output_suppression` flag will be defined and the filter will be instantiated.
*   **`src/main/java/com/google/devtools/build/lib/events/OutputFilter.java`**: Defines the core interface that the new filter will implement.
*   **`src/main/java/com/google/devtools/build/lib/events/Reporter.java`**: The central class for handling and dispatching events.
*   **`src/main/java/com/google/devtools/build/lib/events/Event.java`**: Defines the `Event` object and its property system.
*   **`src/main/java/com/google/devtools/build/lib/skyframe/SkyframeActionExecutor.java`**: A primary source for action-related events where context can be gathered.
*   **`src/main/java/com/google/devtools/build/lib/bazel/bzlmod/BazelModuleResolutionFunction.java`**: The source of the Bzlmod version resolution warnings.
*   **`src/test/java/com/google/devtools/build/lib/outputfilter/OutputFilterTest.java`**: An example of how to write unit tests for an `OutputFilter`.
*   **`src/test/java/com/google/devtools/build/lib/outputfilter/AutoOutputFilterTest.java`**: An example of how to write data-driven, parameterized tests for filtering logic.

## Task Tracker

- [X] Phase 1: Standalone Filter Implementation
  - [X] `EventContext.java`
  - [X] `SuppressionRule.java`
  - [X] `OutputSuppressionFilter.java`
  - [X] Post-Build Verification
- [X] Phase 2: Core Bazel Integration
  - [X] Add Flag
  - [X] Plumb Filter
- [X] Phase 3: Enriching Events (Incremental)
  - [X] Instrument Action Execution
  - [X] Instrument Core Warnings
- [X] Phase 4: Testing and Documentation
  - [X] Add unit tests for core logic and real-world cases.
  - [ ] Add integration tests that run Bazel with a custom `.bazelrc`.
  - [X] Update this design document.
- [ ] Phase 5: Finalization
  - [ ] Achieve 100% suppression of warnings in the upstream Bazel build.
  - [ ] Remove all temporary debugging logs.
  - [ ] Submit the final, clean pull request.
