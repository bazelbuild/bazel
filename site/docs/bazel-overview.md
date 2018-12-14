---
layout: documentation
title: Bazel Overview
---

# What is Bazel?

Bazel is an open-source build and test tool similar to Make, Maven, and Gradle.
It uses a human-readable, high-level build language. Bazel supports projects in
multiple languages and builds outputs for multiple platforms. Bazel supports
large codebases across multiple repositories, and large numbers of users.


# Why should I use Bazel?

Bazel offers the following advantages:

*   **High-level build language.** Bazel uses an abstract, human-readable
    language to describe the build properties of your project at a high
    semantical level. Unlike other tools, Bazel operates on the *concepts*
    of libraries, binaries, scripts, and data sets, shielding you from the
    complexity of writing individual calls to tools such as compilers and
    linkers.

*   **Bazel is fast and reliable.** Bazel caches all previously done work and
    tracks changes to both file content and build commands. This way, Bazel
    knows when something needs to be rebuilt, and rebuilds only that. To further
    speed up your builds, you can set up your project to build in a  highly
    parallel and incremental fashion.

*   **Bazel is multi-platform.** Bazel runs on Linux, macOS, and Windows. Bazel
    can build binaries and deployable packages for multiple platforms, including
    desktop, server, and mobile, from the same project.

*   **Bazel scales.** Bazel maintains agility while handling builds with 100k+
    source files. It works with multiple repositories and user bases in the tens
    of thousands.

*   **Bazel is extensible.** Many [languages](be/overview.html#rules) are
    supported, and you can extend Bazel to support any other language or
    framework.


# How do I use Bazel?

To build or test a project with Bazel, you typically do the following:

1.  **Set up Bazel.** Download and [install Bazel](install.html).

2.  **Set up a project [workspace](build-ref.html#workspaces)**, which is a
    directory where Bazel looks for build inputs and `BUILD` files, and where it
    stores build outputs.

3.  **Write a `BUILD` file**, which tells Bazel what to build and how to
    build it.

    You write your `BUILD` file by declaring build targets using
    [Starlark](skylark/language.html), a domain-specific language. (See example
    [here](https://github.com/bazelbuild/bazel/blob/master/examples/cpp/BUILD).)

    A build target specifies a set of input artifacts that Bazel will build plus
    their dependencies, the build rule Bazel will use to build it, and options
    that configure the build rule.

    A build rule specifies the build tools Bazel will use, such as compilers and
    linkers, and their configurations. Bazel ships with a number of build rules
    covering the most common artifact types in the supported languages on
    supported platforms.

4. **Run Bazel** from the [command line](command-line-reference.html). Bazel
   places your outputs within the workspace.

In addition to building, you can also use Bazel to run
[tests](test-encyclopedia.html) and [query](master/query-how-to.html) the build
to trace dependencies in your code.


# How does Bazel work?

When running a build or a test, Bazel does the following:

1.  **Loads** the `BUILD` files relevant to the target.

2.  **Analyzes** the inputs and their
    [dependencies](build-ref.html#dependencies), applies the specified build
    rules, and produces an [action](skylark/concepts.html#evaluation-model)
    graph.

3.  **Executes** the build actions on the inputs until the final build outputs
    are produced.

Since all previous build work is cached, Bazel can identify and reuse cached
artifacts and only rebuild or retest what's changed. To further enforce
correctness, you can set up Bazel to run builds and tests
[hermetically](user-manual.html#sandboxing) through sandboxing, minimizing skew
and maximizing [reproducibility](user-manual.html#correctness).


## What is the action graph?

The action graph represents the build artifacts, the relationships between them,
and the build actions that Bazel will perform. Thanks to this graph, Bazel can
[track](user-manual.html#build-consistency-and-incremental-builds) changes to
file content as well as changes to actions, such as build or test commands, and
know what build work has previously been done. The graph also enables you to
easily [trace dependencies](query-how-to.html) in your code.


# How do I get started?

To get started with Bazel, see [Getting Started](getting-started.html) or jump
directly to the Bazel tutorials:

*   [Tutorial: Build a C++ Project](tutorial/cpp.html)
*   [Tutorial: Build a Java Project](tutorial/java.html)
*   [Tutorial: Build an Android Application](tutorial/android-app.html)
*   [Tutorial: Build an iOS Application](tutorial/ios-app.html)
