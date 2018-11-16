---
layout: documentation
title: Adapting Bazel Rules for Remote Execution
---

# Adapting Bazel Rules for Remote Execution

Remote execution allows Bazel to execute actions on a separate platform, such as
a datacenter. A [gRPC protocol](https://github.com/googleapis/googleapis/blob/master/google/devtools/remoteexecution/v1test/remote_execution.proto)
is currently in development. You can try remote execution with [bazel-buildfarm](https://github.com/bazelbuild/bazel-buildfarm),
an open-source project that aims to provide a distributed remote execution
platform. This document is intended for Bazel users writing custom build and
test rules who want to understand the requirements for Bazel rules in
the context of remote execution.

This document uses the following terminology when referring to different
environment types or *platforms*:

*   **Host platform** - where Bazel runs.
*   **Execution platform** - where Bazel actions run.
*   **Target platform** - where the build outputs (and some actions) run.

## Overview

When configuring a Bazel build for remote execution, you must follow the
guidelines described in this document to ensure the build executes remotely
error-free. This is due to the nature of remote execution, namely:

*   **Isolated build actions.** Build tools do not retain state and dependencies
    cannot leak between them.

*   **Diverse execution environments.** Local build configuration is not always
    suitable for remote execution environments.

This document describes the issues that can arise when implementing custom Bazel
build and test rules for remote execution and how to avoid them. It covers the
following topics:

*  [Invoking build tools through toolchain rules](#invoking-build-tools-through-toolchain-rules)
*  [Managing dependencies](#managing-dependencies)
*  [Managing platform-dependent binaries](#managing-platform-dependent-binaries)
*  [Managing configure-style WORKSPACE rules](#managing-configure-style-workspace-rules)

## Invoking build tools through toolchain rules

A Bazel toolchain rule is a configuration provider that tells a build rule what
build tools, such as compilers and linkers, to use and how to configure them
using parameters defined by the rule's creator. A toolchain rule allows build
and test rules to invoke build tools in a predictable, preconfigured manner
that's compatible with remote execution. For example, use a toolchain rule
instead of invoking build tools via the `PATH`, `JAVA_HOME`, or other local
variables that may not be set to equivalent values (or at all) in the remote
execution environment.

Toolchain rules currently exist for Bazel build and test rules for
[Scala](https://github.com/bazelbuild/rules_scala/blob/master/scala/scala_toolch
ain.bzl),
[Rust](https://github.com/bazelbuild/rules_rust/blob/master/rust/toolchain.bzl),
and [Go](https://github.com/bazelbuild/rules_go/blob/master/go/toolchains.rst),
and new toolchain rules are under way for other languages and tools such as
[bash](https://docs.google.com/document/d/e/2PACX-1vRCSB_n3vctL6bKiPkIa_RN_ybzoAccSe0ic8mxdFNZGNBJ3QGhcKjsL7YKf-ngVyjRZwCmhi_5KhcX/pub).
If a toolchain rule does not exist for the tool your rule uses, consider
[creating a toolchain rule](/toolchains.html#creating-a-toolchain-rule).

## Managing implicit dependencies

If a build tool can access dependencies across build actions, those actions will
fail when remotely executed because each remote build action is executed
separately from others. Some build tools retain state across build actions and
access dependencies that have not been explicitly included in the tool
invocation, which will cause remotely executed build actions to fail.

For example, when Bazel instructs a stateful compiler to locally build _foo_,
the compiler retains references to foo's build outputs. When Bazel then
instructs the compiler to build _bar_, which depends on _foo_, without
explicitly stating that dependency in the BUILD file for inclusion in the
compiler invocation, the action executes successfully as long as the same
compiler instance executes for both actions (as is typical for local execution).
However, since in a remote execution scenario each build action executes a
separate compiler instance, compiler state and _bar_'s implicit dependency on
_foo_ will be lost and the build will fail.

To help detect and eliminate these dependency problems, Bazel 0.14.1 offers the
local Docker sandbox, which has the same restrictions for dependencies as remote
execution. Use the sandbox to prepare your build for remote execution by
identifying and resolving dependency-related build errors. See [Troubleshooting Bazel Remote Execution with Docker Sandbox](/remote-execution-sandbox.html)
for more information.

## Managing platform-dependent binaries

Typically, a binary built on the host platform cannot safely execute on an
arbitrary remote execution platform due to potentially mismatched dependencies.
For example, the SingleJar binary supplied with Bazel targets the host platform.
However, for remote execution, SingleJar must be compiled as part of the process
of building your code so that it targets the remote execution platform. (See the
[target selection logic](https://github.com/bazelbuild/bazel/blob/130aeadfd660336572c3da397f1f107f0c89aa8d/tools/jdk/BUILD#L115).)

Do not ship binaries of build tools required by your build with your source code
unless you are sure they will safely run in your execution platform. Instead, do
one of the following:

*   Ship or externally reference the source code for the tool so that it can be
    built for the remote execution platform.

*   Pre-install the tool into the remote execution environment (for example, a
    toolchain container) if it's stable enough and use toolchain rules to run it
    in your build.

## Managing `configure`-style WORKSPACE rules

Bazel's `WORKSPACE` rules can be used for probing the host platform for tools
and libraries required by the build, which, for local builds, is also Bazel's
execution platform. If the build explicitly depends on local build tools and
artifacts, it will fail during remote execution if the remote execution platform
is not identical to the host platform.

The following actions performed by `WORKSPACE` rules are not compatible with
remote execution:

*   **Building binaries.** Executing compilation actions in `WORKSPACE` rules
    results in binaries that are incompatible with the remote execution platform
    if different from the host platform.

*   **Installing `pip` packages.** `pip` packages  installed via `WORKSPACE`
    rules require that their dependencies be pre-installed on the host platform.
    Such packages, built specifically for the host platform, will be
    incompatible with the remote execution platform if different from the host
    platform.

*   **Symlinking to local tools or artifacts.** Symlinks to tools or libraries
    installed on the host platform created via `WORKSPACE` rules will cause the
    build to fail on the remote execution platform as Bazel will not be able to
    locate them. Instead, create symlinks using standard build actions so that
    the symlinked tools and libraries are accessible from Bazel's `runfiles`
    tree. Do not use [`repository_ctx.symlink`](https://docs.bazel.build/versions/master/skylark/lib/repository_ctx.html#symlink)
    to symlink target files outside of the external repo directory.

*   **Mutating the host platform.** Avoid creating files outside of the Bazel
    `runfiles` tree, creating environment variables, and similar actions, as
     they may behave unexpectedly on the remote execution platform.

To help find potential non-hermetic behavior you can use [Workspace rules log](/workspace-log.md).

If an external dependency executes specific operations dependent on the host
platform, we recommend splitting those operations between `WORKSPACE` and build
rules as follows:

*   **Platform inspection and dependency enumeration.** These operations are
    safe to execute locally via `WORKSPACE` rules, which can check which
    libraries are installed, download packages that must be built, and prepare
    required artifacts for compilation. For remote execution, these rules must
    also support using pre-checked artifacts to provide the information that
    would normally be obtained during host platform inspection. Pre-checked
    artifacts allow Bazel to describe dependencies as if they were local. Use
    conditional statements or the `--override_repository` flag for this.

*   **Generating or compiling target-specific artifacts and platform mutation**.
    These operations must be executed via regular build rules. Actions that
    produce target-specific artifacts for external dependencies must execute
    during the build.

To more easily generate pre-checked artifacts for remote execution, you can use
`WORKSPACE` rules to emit generated files. You can run those rules on each new
execution environment, such as inside each toolchain container, and check the
outputs of your remote execution build in to your source repo to reference.

For example, for Tensorflow's rules for [`cuda`](https://github.com/tensorflow/tensorflow/blob/master/third_party/gpus/cuda_configure.bzl)
and [`python`](https://github.com/tensorflow/tensorflow/blob/master/third_party/py/python_configure.bzl),
the `WORKSPACE` rules produce the following [`BUILD files`](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/cpus/py).
For local execution, files produced by checking the host environment are used.
For remote execution, a [conditional statement](https://github.com/tensorflow/tensorflow/blob/master/third_party/py/python_configure.bzl#L304)
on an environment variable allows the rule to use files that are checked into
the repo. The `BUILD` files declare [`genrules`](https://github.com/tensorflow/tensorflow/blob/master/third_party/py/python_configure.bzl#L84\)
that can run both locally and remotely, and perform the necessary processing
that was previously done via `repository_ctx.symlink` as shown [here](https://github.com/tensorflow/tensorflow/blob/d1ba01f81d8fa1d0171ba9ce871599063d5c7eb9/third_party/gpus/cuda_configure.bzl#L730).
