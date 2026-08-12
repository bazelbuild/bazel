# Design Proposal: Toolchain-Driven Unused Java Dependencies Checking in Bazel

This document outlines the design for introducing and configuring unused Java dependencies checking in Bazel. The configuration is driven by the Java toolchain, with supporting metadata tracking and configuration resolution implemented in `rules_java` Starlark rules, and native API execution handled by core Bazel.

---

## 1. Preface: Driver for the Change

Bazel currently enforces strict Java dependencies (`strict_java_deps`), ensuring that a target cannot directly reference classes from a transitive dependency without declaring a direct dependency. While this guarantees dependency completeness, it does not prevent **unused dependencies**—libraries declared as direct dependencies but not referenced in code. Unused dependencies bloat deployment artifacts, increase build graph size, and lead to unnecessary cache invalidations.

The goal is to implement a robust unused dependencies check (supporting `off` and `error` modes) that is configured via the Java toolchain but integrated into target analysis.

### Why `warn` Mode is Unnecessary
By leveraging `java_package_configuration`, teams can enforce the check as a hard compile-time error restricted to specific packages, gradually ratcheting up compliance package-by-package. This prevents warning noise in developer builds while enabling a safe, structured path to full enforcement.

### Key Technical Challenges
- **Conflation of Exported Transitive Dependencies**: Javac and JavaBuilder receive classpath inputs as flat lists of jar files. When a direct dependency exports a transitive dependency (via `exports`), those transitive dependency jars are merged into the target's direct classpath (`directJars`). This conflation occurs inside the Starlark rules of `@rules_java` (see [java_info.bzl#L612](https://github.com/bazelbuild/rules_java/blob/35e8391a1c37dd5941ba594a01dcb08d3985cbd8/java/private/java_info.bzl#L612) where exported compile jars are accumulated into the target's direct compile jars). To check for unused dependencies correctly without throwing false positives, the check must account for this conflation (where classes are consumed via the exported transitives of a declared direct dependency).

---

## 2. Toolchain-Driven Configuration API

The unused dependencies check configuration is defined at the toolchain level using the existing `java_package_configuration` API. This allows operators to define check strictness globally or for specific packages.

### Package Configuration Definition
Using `java_package_configuration`, unused dependencies checking modes (`off`, `error`) can be mapped to specific package specs:

```starlark
java_package_configuration(
    name = "unused_deps_error_config",
    package_specs = [":my_package_spec"],
    unused_deps = "error",  # "off" or "error"
)
```

The matching `java_package_configuration` rules are registered on the `java_toolchain` rule:

```starlark
java_toolchain(
    name = "toolchain",
    package_configuration = [
        ":unused_deps_error_config",
    ],
    ...
)
```

---

## 3. Expected Behaviors under Various Scenarios

The table below outlines the compilation outcomes under different dependency configurations (assuming `unused_deps = "error"` is active):

| Dependency Graph & Code Usage | Outcome | Reason / Action |
| :--- | :--- | :--- |
| **Simple Unused**: A depends on B. A's code does NOT reference B. | **Error** | B is unused. A must remove B from `deps`. |
| **Simple Used**: A depends on B. A's code references B. | **Success** | B is directly used. |
| **Exported Only**: B exports C. A depends on B, but A's code only references C (does NOT reference B). | **Error** | B is unused. A must add a direct dependency on C and remove B from `deps`. |
| **Exported & Direct Used**: B exports C. A depends on B, and A's code references both B and C. | **Success** | B is directly used. C's usage is allowed because B exports it. |
| **Indirect Used (Strict Deps)**: A depends on B. B depends on C (no export). A's code references C. | **Error** | Strict Java Deps violation. A must declare a direct dependency on C. |
| **Non-Main Repository Target**: Target is in an external repository (e.g., `@some_repo//pkg:target`). | **Success** | Unused dependencies checking is disabled (`off`) for external repositories to avoid breaking builds on third-party code. |

---

## 4. Implied Changes to `rules_java`

To support this toolchain-driven configuration, the Starlark rules in `rules_java` must be updated to define the attribute on `java_package_configuration`, track dependency declarations on compilation rules, and resolve the checking levels.

### A. Java Package Configuration Rule Update
The `java_package_configuration` rule definition (in `java/common/rules:java_package_configuration.bzl`) will define a new Starlark attribute to capture the checking strictness:
```starlark
unused_deps = attr.string(default = "off", values = ["off", "error"])
```

### B. Formatting Declared Dependencies into an Args Object
The rules (e.g., `java_library`, `java_binary`) map each declared dependency to its direct compile jars (excluding transitively exported jars) directly into a Starlark `Args` object (`ctx.actions.args()`). This avoids expanding artifact paths to strings during analysis time:

```starlark
# Inside rules_java compilation helper:
unused_deps_args = ctx.actions.args()
for dep in ctx.attr.deps:
    # Collect only compile jars generated directly by this target (excluding exports)
    for output in dep[JavaInfo].java_outputs:
        compile_jar = output.compile_jar if output.compile_jar else output.class_jar
        if compile_jar:
            unused_deps_args.add("--declared_dep", compile_jar, format = "%s::" + str(dep.label))
```

### C. Resolving Unused Dependencies Check Levels
The Starlark rules query the toolchain's package configurations to resolve the unused dependencies check level (`off` or `error`) for the package being compiled.
- It parses the matching `java_package_configuration` from `ctx.attr._java_toolchain` for the current package.
- It extracts the resolved checking mode directly from the new `unused_deps` attribute of the configuration object.
- It verifies if the target belongs to the main repository (using `not ctx.label.workspace_name`). If the target belongs to an external repository (where `workspace_name` is non-empty), checking is overridden to `"off"` to prevent breaking third-party dependencies.

### D. Calling Compilation Helper
If the unused deps check is resolved to `error` for the package, the `Args` object is appended to `extra_args` and forwarded via `java_common.compile` / `create_compilation_action`:

```starlark
java_common.compile(
    ctx,
    ...
    extra_args = extra_args,  # list of Args objects
)
```

---

## 5. Changes Required in Core Bazel

Core Bazel's native APIs and action builders are updated to accept the generic `extra_args` parameter (a list of `CommandLineArgsApi` / `Args`) and attach their command lines and input manifests to the action. Core Bazel remains agnostic to the specific flags passed.

### A. Native API Updates (`JavaStarlarkCommon.java` and `JavaCommonApi.java`)
Extend `createCompilationAction` in `JavaStarlarkCommon` to accept `extra_args`:

```java
@Param(
    name = "extra_args",
    allowedTypes = {
        @ParamType(type = Sequence.class, generic1 = CommandLineArgsApi.class),
        @ParamType(type = NoneType.class),
    },
    defaultValue = "None",
    named = true,
    positional = false)
Object extraArgs
```

### B. Action Compilation Pipeline (`JavaCompileActionBuilder.java` & `JavaCompileAction.java`)
Forward each constructed `CommandLine` from `extra_args` to `JavaCompileAction`'s command lines (with `PARAM_FILE_INFO`) and add any directory inputs from each `Args` to mandatory action inputs.

### C. JavaBuilder Compiler Plugin (`StrictJavaDepsPlugin.java`)
Update the strict Java dependencies plugin in JavaBuilder to:
- Parse `--declared_dep` in the format `path::label` to build a map of `JarPath -> DeclaredLabel`.
- Keep track of which direct dependency jar paths are actually loaded and used during compilation (leveraging the compiler plugin's existing AST traversal).
- If a declared target label has all of its associated jars completely unused during compilation, emit a `[unused-deps]` error referencing the declared target label.


