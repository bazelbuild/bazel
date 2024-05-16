Project: /_project.yaml
Book: /_book.yaml
# Bazel roadmap

{% include "_buttons.html" %}

## Overview

As the Bazel project continues to evolve in response to your needs, we want to
share our 2024 update.

This roadmap describes current initiatives and predictions for the future of
Bazel development, giving you visibility into current priorities and ongoing
projects.

## Bazel 8.0 Release
We plan to bring Bazel 8.0 [long term support
(LTS)](https://bazel.build/release/versioning) to you in late 2024.
The following features are currently planned to be implemented.

### Bzlmod: external dependency management system

[Bzlmod](https://bazel.build/docs/bzlmod) automatically resolves transitive
dependencies, allowing projects to scale while staying fast and
resource-efficient.

With Bazel 8, we will disable WORKSPACE support by default (it will still be
possible to enable it via `--enable_workspace`), with Bazel 9 WORKSPACE support
will be removed. Starting with Bazel 7.1, you can set `--noenable_workspace` to
opt into the new behavior.

Bazel 8.0 will contain a number of enhancements to [Bazel's external dependency
management](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit#heading=h.lgyp7ubwxmjc)
functionality, including:

*   The new flag `--enable_workspace` can be set to `false` to completely
    disable WORKSPACE functionality.
*   New directory watching API (see
    [#21435](https://github.com/bazelbuild/bazel/pull/21435), shipped in Bazel
    7.1).
*   Improved scheme for generating canonical repository names for better
    cacheability of actions across dependency version updates.
    ([#21316](https://github.com/bazelbuild/bazel/pull/21316), shipped in Bazel
    7.1)
*   An improved shared repository cache (see
    [#12227](https://github.com/bazelbuild/bazel/issues/12227)).
*   Vendor/offline mode support — allows users to run builds with pre-downloaded
    dependencies (see
    [#19563](https://github.com/bazelbuild/bazel/issues/19563)).
*   Reduced merge conflicts in lock files
    ([#20396](https://github.com/bazelbuild/bazel/issues/20369)).
*   Segmented MODULE.bazel
    ([#17880](https://github.com/bazelbuild/bazel/issues/17880))
*   Allow overriding module extension generated repository
    ([#19301](https://github.com/bazelbuild/bazel/issues/19301))
*   Improved documentation (e.g.
    [#18030](https://github.com/bazelbuild/bazel/issues/18030),
    [#15821](https://github.com/bazelbuild/bazel/issues/15821)) and migration
    guide and migration tooling.


### Remote execution improvements

*   Add support for asynchronous execution, speeding up remote execution via
    increased parallelism with flag `--jobs`.
*   Make it easier to debug cache misses by a new compact execution log,
    reducing its size by 100x and its runtime overhead significantly (see
    [#18643](https://github.com/bazelbuild/bazel/issues/18643)).
*   Implement garbage collection for the disk cache (see
    [#5139](https://github.com/bazelbuild/bazel/issues/5139)).
*   Implement remote output service to allow lazy downloading of arbitrary build
    outputs (see
    [#20933](https://github.com/bazelbuild/bazel/discussions/20933)).


### Migration of Android, C++, Java, Python, and Proto rules

Complete migration of Android, C++, Java, and Python rulesets to dedicated
repositories and decoupling them from the Bazel releases. This effort allows
Bazel users and rule authors to

*   Update rules independently of Bazel.
*   Update and customize rules as needed.

The new location of the rulesets is going to be `bazelbuild/rules_android`,
`rules_cc`, `rules_java`, `rules_python` and `google/protobuf`. `rules_proto` is
going to be deprecated.

Bazel 8 will provide a temporary migration flag that will automatically use the
rulesets that were previously part of the binary from their repositories. All
the users of those rulesets are expected to eventually depend on their
repositories and load them similarly to other rulesets that were never part of
Bazel.

Bazel 8 will also improve on the existing extending rules and subrule APIs and
mark them as non-experimental.


### Starlark improvements

*   Symbolic Macros are a new way of writing macros that is friendlier to
    `BUILD` users, macro authors, and tooling. Compared to legacy macros, which
    Bazel has only limited insight into, symbolic macros help users avoid common
    pitfalls and enforce best practices.
*   Package finalizers are a proposed feature for adding first-class support for
    custom package validation logic. They are intended to help us deprecate
    `native.existing_rules()`.

### Configurability

*   Output path mapping continues to stabilize: promising better remote cache
    performance and build speed for rule designers who use transitions.
*   Automatically set build flags suitable for a given `--platforms`.
*   Define project-supported flag combinations and automatically build targets
    with default flags without having to set bazelrcs.
*   Don't redo build analysis every time build flags change.


### Project Skyfocus - minimize retained data structures

Bazel holds a lot of state in RAM for fast incremental builds. However,
developers often change a small subset of the source files (e.g. almost never
one of the external dependencies). With Skyfocus, Bazel will provide an
experimental way to drop unnecessary incremental state and reduce Bazel's memory
footprint, while still providing the same fast incremental build experience.

The initial scope aims to improve the retained heap metric only. Peak heap
reduction is a possibility, but not included in the initial scope.


### Misc

*   Mobile install v3, a simpler and better maintained approach to incrementally
    deploy Android applications.
*   Garbage collection for repository caches and Bazel’s `install_base`.
*   Reduced sandboxing overhead.


### Bazel-JetBrains* IntelliJ IDEA support
Incremental IntelliJ plugin updates to support the latest JetBrains plugin release.

*This roadmap snapshots targets, and should not be taken as guarantees. Priorities are subject to change in response to developer and customer feedback, or new market opportunities.*

*To be notified of new features — including updates to this roadmap — join the [Google Group](https://groups.google.com/g/bazel-discuss) community.*

*Copyright © 2022 JetBrains s.r.o. JetBrains and IntelliJ are registered trademarks of JetBrains s.r.o
