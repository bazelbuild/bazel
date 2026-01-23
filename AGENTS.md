# Bazel Codebase Guide for Gemini

Welcome to the Bazel repository. This document provides essential context and
workflows for working with the Bazel codebase.

## 🏗️ Building & Running Bazel

To build Bazel from source, use a pre-installed `bazel` or `bazelisk` binary.

*   **Fast Iteration Build (Recommended):** `bazel build //src:bazel-dev`
*   **Standard Build:** `bazel build //src:bazel`

### ⚡ Performance Tips

*   **Remote Execution:** If you have access to a Remote Build Execution (RBE)
cluster, use `--config=remote` to significantly speed up your builds:
    ```bash
    bazel build --config=remote //src:bazel-dev
    ```

### 🔄 Iterative Development Workflow

To test changes safely without interfering with your primary workspace,
use a separate binary and output base:

1.  **Build and Copy:**
    ```bash
    bazel build //src:bazel-dev && cp bazel-bin/src/bazel-dev /tmp/bazel
    ```
2.  **Run commands:**
    ```bash
    /tmp/bazel --output_base=/tmp/ob-dev <command>
    ```

**Note:** Using a custom `--output_base` (e.g., `/tmp/ob-dev`) is crucial to
avoid locking your main workspace server and ensures your development
environment remains isolated.

## 🧪 Testing

Tests are located primarily in `src/test`.

### Finding Relevant Tests
If you modify a file, find tests that transitively depend on it:
```bash
bazel query "rdeps(//src/test/..., path/to/file.java)"
```

### Running Tests

*   **Unit Tests:** Typically `java_test` targets.
*   **Integration Tests:**
    *   Java-based: Subclasses of `BuildIntegrationTestCase`.
    *   Shell-based: Located in `src/test/shell`, using a bash test framework.

## 📜 Architecture & Codebase

Bazel uses a **Client/Server** architecture. The client (C++) is a lightweight
wrapper that starts and communicates with a long-lived Java server.

*   **Skyframe:** The incremental evaluation framework. Most core logic is
implemented as `SkyFunction`s evaluating `SkyKey`s into `SkyValue`s.
*   **Starlark:** The configuration language used for `BUILD` and `.bzl` files.
*   **Loading/Analysis/Execution:** The three phases of a Bazel command.

For architectural details or introspection capabilities,
see: `docs/contribute/codebase.mdx`

## 🧹 Linting & Formatting

*   **Java:** Follows Google Java Style.
*   **Starlark:** Use `buildifier` for `BUILD` and `.bzl` files.
