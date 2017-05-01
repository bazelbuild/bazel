---
layout: documentation
title: Best practices
---

# Best practices for Bazel

This document assumes that you are familiar with Bazel and provides advice on structuring your
projects to take full advantage of Bazel's features.

The overall goals are:

- To use fine-grained dependencies to allow parallelism and incrementality.
- To keep dependencies well-encapsulated.
- To make code well-structured and testable.
- To create a build configuration that is easy to understand and maintain.

These guidelines are not requirements: few projects will be able to adhere to all of them.  As the
man page for lint says, "A special reward will be presented to the first person to produce a real
program that produces no errors with strict checking." However, incorporating as many of these
principles as possible should make a project more readable, less error-prone, and faster to build.

This document uses the requirement levels described in
[this RFC](https://www.ietf.org/rfc/rfc2119.txt).

## Contents

- [General structure](#general-structure)
   - [Running builds and tests](#running-builds-and-tests)
   - [Third party dependencies](#third-party-dependencies)
   - [Depending on binaries](#depending-on-binaries)
   - [Versioning](#versioning)
   - [.bazelrc](#bazelrc)
   - [Packages](#packages)
- [BUILD files](#build-files)
   - [BUILD file style guide](#build-file-style-guide)
   - [Formatting](#formatting)
   - [References to targets in the current package](#references-to-targets-in-the-current-package)
   - [Target naming](#target-naming)
   - [Visibility](#visibility)
   - [Dependencies](#dependencies)
   - [Globs](#globs)
- [Skylark](#skylark)
   - [Skylark style guide](#skylark-style-guide)
   - [Packaging rules](#packaging-rules)
   - [Rule choice](#rule-choice)
- [WORKSPACE files](#workspace-files)
   - [Repository rules](#repository-rules)
   - [Custom BUILD files](#custom-build-files)
   - [Skylark repository rules](#skylark-repository-rules)
- [Java](#java)
   - [Directory structure](#directory-structure)
   - [BUILD files](#build-files)
- [C++](#c)
   - [BUILD files](#build-files)
   - [Include paths](#include-paths)
- [Protos](#protos)
   - [Recommended Code Organization](#recommended-code-organization)

# General structure

## Running builds and tests

A project should always be able to run `bazel build //...` and `bazel test //...` successfully on
its stable branch. Targets that are necessary but do not build under certain circumstances (e.g.,
require specific build flags, do not build on a certain platform, require license agreements)
should be tagged as specifically as possible (e.g., "`requires-osx`"). This tagging allows
targets to be filtered at a more fine-grained level than the "manual" tag and allows someone
inspecting the BUILD file to understand what a target's restrictions are.

## Third party dependencies

Prefer declaring third party dependencies as remote repositories in the WORKSPACE file. If it's
necessary to check third party dependencies into your repository, put them in a directory called
`third_party/` under your workspace directory.   Note that all BUILD files in `third_party/` must
include [license](https://bazel.build/versions/master/docs/be/functions.html#licenses)
declarations.

## Depending on binaries

Everything should be built from source whenever possible. Generally this means that, instead of
depending on a library `some-library.so`, you'd create a BUILD file and build `some-library.so`
from its sources, then depend on that target.

Building from source prevents a build from using an library that was build with incompatible flags
or a different architecture.  There are also some features like coverage, static analysis, or
dynamic analysis that will only work on the source.

## Versioning

Prefer building all code from head whenever possible.  When versions must be used, avoid including
the version in the target name (e.g., `//guava`, not `//guava-20.0`). This naming makes the library
easier to update (only one target needs to be updated).  It is also more resilient to diamond
dependency issues: if one library depends on `guava-19.0` and one depends on `guava-20.0`, you
could end up with a library that tries to depend on two different versions. If you created a
misleading alias to point both targets to one guava library, then the BUILD files are misleading.

## `.bazelrc`

For project-specific options, use the configuration file `_your-workspace_/tools/bazel.rc`.

For options that you **do not** want to check into source control, create the configuration file
`_your-workspace_/.bazelrc` and add `.bazelrc` to your `.gitignore`.  Note that this file has a
different name than the file above (`bazel.rc` vs `.bazelrc`).

## Packages

Every directory that contains buildable files should be a package. If a BUILD file refers to files
in subdirectories (e.g., `srcs = ["a/b/C.java"]`) it is a sign that a BUILD file should be added to
that subdirectory.  The longer this structure exists, the more likely circular dependencies will be
inadvertently created, a target's scope will creep, and an increasing number of reverse
dependencies will have to be updated.

# BUILD files

## BUILD file style guide

See the [BUILD file style
guide](https://bazel.build/versions/master/docs/skylark/build-style.html).

## Formatting

[Buildifier](https://github.com/bazelbuild/buildifier) should be used to achieve the correct
formatting for BUILD files.  Editors should be configured to automatically format BUILD files on
save.  Humans should not try to format BUILD files themselves.

If there is a question as to what the correct formatting is, the answer is "how buildifier formats
it."

## References to targets in the current package

Files should be referred to by their paths relative to the package directory (without ever using
up-references, such as `..`).  Generated files should be prefixed with "`:`" to indicate that they
are not sources.  Source files should not be prefixed with `:`. Rules should be prefixed with `:`.
For example, assuming `x.cc` is a source file:

```python
cc_library(
    name = "lib",
    srcs = ["x.cc"],
    hdrs = [":gen-header"],
)

genrule(
    name = "gen-header",
    srcs = [],
    outs = ["x.h"],
    cmd = "echo 'int x();' > $@",
)
```

## Target naming

Target names should be descriptive. If a target contains one source file, the target should
generally be named after that source (e.g., a `cc_library` for `chat.cc` should be named "`chat`").

The eponymous target for a package (the target with the same name as the containing directory)
should provide the functionality described by the directory name. If there is no such target, do
not create an eponymous target.

Prefer using the short name when referring to an eponymous target (`//x` instead of `//x:x`).  If
you are in the same package, prefer the local reference (`:x` instead of `//x`).

## Visibility

Do not set the default visibility of a package to `//visibility:public`.  `//visibility:public`
should be individually set for targets in the project's public API. These could be libraries which
are designed to be depended on by external projects or binaries that could be used by an external
project's build process.

Otherwise, visibility should be scoped as tightly as possible, while still allowing access by tests
and reverse dependencies. Prefer using `__pkg__` to `__subpackages__`.

## Dependencies

Dependencies should be restricted to direct dependencies (dependencies needed by the sources listed
in the rule). Do not list transitive dependencies.

Package-local dependencies should be listed first and referred to in a way compatible with the
[References to targets in the current package](#references-to-targets-in-the-current-package)
section above (not by their absolute package name).

## Globs

Do not use recursive globs (e.g., `glob(["**/*.java"])`). Recursive globs make BUILD files
difficult to read, as they skip subdirectories containing BUILD files. Non-recursive globs are
generally acceptable, see language-specific advice below for details.

Indicate "no targets" with `[]`. Do not use a glob that matches nothing: it is more error-prone and
less obvious than an empty list.

# Skylark

## Skylark style guide

See the [Style guide for .bzl
files](https://bazel.build/versions/master/docs/skylark/bzl-style.html) for Skylark rule guidelines.

## Packaging rules

See [Packaging rules](https://bazel.build/versions/master/docs/skylark/deploying.html) for advice
on how to structure and where to put new Skylark rules.

## Rule choice

When using a language for which Bazel has built-in rules (e.g., C++), prefer using these rules to
writing your own in Skylark. These rules are documented in the [build
encyclopedia](https://bazel.build/versions/master/docs/be/overview.html).

# WORKSPACE files

## Repository rules

Prefer `http_archive` and `new_http_archive` to `git_repository`, `new_git_repository`, and
`maven_jar`.

`git_repository` depends on jGit, which has several unpleasant bugs, and `maven_jar` uses Maven's
internal API, which generally works but is less optimized for Bazel than `http_archive`'s
downloader logic. Track the following issues filed to remediate these problems:

-  [Use `http_archive` as `git_repository`'s
   backend.](https://github.com/bazelbuild/bazel/issues/2147)
-  [Improve `maven_jar`'s backend.](https://github.com/bazelbuild/bazel/issues/1752)

Do not use `bind()`.  See "[Consider removing
bind](https://github.com/bazelbuild/bazel/issues/1952)" for a long discussion of its issues and
alternatives.

## Custom BUILD files

When using a `new_` repository rule, prefer to specify `build_file_content`, not `build_file`.

## Skylark repository rules

A Skylark repository rule should generally be responsible for:

-  Detecting system settings and writing them to files.
-  Finding resources elsewhere on the system.
-  Downloading resources from URLs.
-  Generating or symlinking BUILD files into the external repository directory.

Avoid using `repository_ctx.execute` when possible.  For example, when using a non-Bazel C++
library that has a build using Make, it is preferable to use `respository_ctx.download()` and then
write a BUILD file that builds it, instead of running `ctx.execute(["make"])`.

# Java

## Directory structure

Prefer Maven's standard directory layout (sources under `src/main/java`, tests under
`src/test/java`).

## BUILD files

Use one BUILD file per package containing Java sources. Every BUILD file should contain one
`java_library` rule that looks like this:

```python
java_library(
    name = "directory-name",
    srcs = glob(["*.java"]),
    deps = [...],
)
```

The name of the library should be the name of the directory containing the BUILD file.  The sources
should be a non-recursive glob of all Java files in the directory.

Tests should be in a matching directory under `src/test` and depend on this library.

# C++

## BUILD files

Each BUILD file should contain one `cc_library` rule target per compilation unit in the directory.
C++ libraries should be as fine-grained as possible to provide as much incrementality as possible.

If there is a single source file in `srcs`, the library should be named based on that C++ file's
name. This library should contain a C++ file(s), any matching header file(s), and the library's
direct dependencies.  For example,

```python
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = ["mylib.h"],
    deps = [":lower-level-lib"]
)
```

There should be one `cc_test` rule target per `cc_library` target in the file. The `cc_test`'s
source should be a file named `[libname]_test.cc`.  For example, a test for the target above might
look like:

```
cc_test(
    name = "mylib_test",
    srcs = ["mylib_test.cc"],
    deps = [":mylib"]
)
```

## Include paths

All include paths should be relative to the workspace directory. Use `includes` only if a public
header needs to be widely used at a non-workspace-relative path (for legacy or `third_party` code).
Otherwise, prefer to use the `copts` attribute, not the `includes` attribute.

Using `cc_inc_library` is discouraged, prefer `copts` or `includes`.
See [the design document](https://docs.google.com/document/d/18qUWh0uUiJBv6ZOySvp6DEV0NjVnBoEy-r-ZHa9cmhU/edit#heading=h.kmep1cl5ym9k)
on C++ include directories for reasoning.

# Protos

## Recommended Code Organization

-  One `proto_library` rule per `.proto` file.
-  A file named `foo.proto` will be in a rule named `foo_proto`, which is located in the same
   package.
-  A `[language]_proto_library` that wraps a `proto_library` named `foo_proto` should be called
   `foo_[language]_proto`, and be located in the same package.
