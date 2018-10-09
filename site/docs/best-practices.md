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
- [WORKSPACE files](#workspace-files)
   - [Repository rules](#repository-rules)
   - [Custom BUILD files](#custom-build-files)
   - [Repository rules](#repository-rules)
- [Protos and Bazel](#protos-and-bazel)

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
include [license](https://docs.bazel.build/be/functions.html#licenses)
declarations.

## Depending on binaries

Everything should be built from source whenever possible. Generally this means that, instead of
depending on a library `some-library.so`, you'd create a BUILD file and build `some-library.so`
from its sources, then depend on that target.

Always building from source ensures that a build is not using a library that was built with
incompatible flags or a different architecture. There are also some features like coverage,
static analysis, or dynamic analysis that will only work on the source.

## Versioning

Prefer building all code from head whenever possible.  When versions must be used, avoid including
the version in the target name (e.g., `//guava`, not `//guava-20.0`). This naming makes the library
easier to update (only one target needs to be updated).  It is also more resilient to diamond
dependency issues: if one library depends on `guava-19.0` and one depends on `guava-20.0`, you
could end up with a library that tries to depend on two different versions. If you created a
misleading alias to point both targets to one guava library, then the BUILD files are misleading.

## `.bazelrc`

For project-specific options, use the configuration file `_your-workspace_/.bazelrc` (see
[bazelrc format](https://docs.bazel.build/user-manual.html#bazelrc)).

If you want to support per-user options for your project that you **do not** want to check 
into source control, include the line
```
try-import user.bazelrc
```
(or any other file name) in your `_your-workspace_/.bazelrc` and 
add `user.bazelrc` to your `.gitignore`.  

## Packages

Every directory that contains buildable files should be a package. If a BUILD file refers to files
in subdirectories (e.g., `srcs = ["a/b/C.java"]`) it is a sign that a BUILD file should be added to
that subdirectory.  The longer this structure exists, the more likely circular dependencies will be
inadvertently created, a target's scope will creep, and an increasing number of reverse
dependencies will have to be updated.

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

## Repository rules

A repository rule should generally be responsible for:

-  Detecting system settings and writing them to files.
-  Finding resources elsewhere on the system.
-  Downloading resources from URLs.
-  Generating or symlinking BUILD files into the external repository directory.

Avoid using `repository_ctx.execute` when possible.  For example, when using a non-Bazel C++
library that has a build using Make, it is preferable to use `repository_ctx.download()` and then
write a BUILD file that builds it, instead of running `ctx.execute(["make"])`.


# Protos and Bazel

Recommended code organization:

-  One `proto_library` rule per `.proto` file.
-  A file named `foo.proto` will be in a rule named `foo_proto`, which is located in the same
   package.
-  A `[language]_proto_library` that wraps a `proto_library` named `foo_proto` should be called
   `foo_[language]_proto`, and be located in the same package.
