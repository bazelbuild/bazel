---
layout: documentation
title: Bazel Overview
---

# Bazel Overview

Bazel is a build tool which coordinates builds and runs tests. The extension
language allows it to work with source files written in any language, with
native support for Java, C, C++ and Python. Bazel produces builds and runs
tests for multiple platforms.

## BUILD files use a simple declarative language

Bazel’s BUILD files describe how Bazel should build your project. They have a
declarative structure and use a language similar to Python. BUILD files
allow you to work at a high level of the system by listing rules and their
attributes. The complexity of the build process is handled by these pre-existing
rules. You can modify rules to tweak the build process, or write new rules to
extend Bazel to work with any language or platform.

Below is the content of one of the BUILD files from a Hello World program. The
two rules used here are `cc_library` and `cc_binary`.

```
cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-time",
        "//lib:hello-greet",
    ],
)
```

## The dependency graph describes the entire system

Build dependencies are declared explicitly in the BUILD files, allowing Bazel
to create an accurate dependency graph of the entire source code. The graph is
maintained in memory, and incremental builds and parallel execution are possible
because of this accurate dependency graph.

Here’s the graph of the target ‘hello-world’ from the BUILD file above:

![Dependency graph of a hello-world target](/assets/graph_hello-world.svg)


Bazel’s query language allows you to produce images of the graph like the one
above. You can also use the query language to access information about build
dependencies and their relationships.

## Build and tests are fast, correct, and reproducible

Hermetic rules and sandboxing allow Bazel to produce correct, reproducible
artifacts and test results. Caching allows reuse of build artifacts and test
results.

Bazel’s builds are fast. Incremental builds allow Bazel to do the minimum
required work for a rebuild or retest. Correct and reproducible builds allow
Bazel to reuse cached artifacts for whatever is not changed. If you change a
library, Bazel will not rebuild your entire source.

Confidence in these correct results also means that you will never need to run
`bazel clean`. If you ever need to run `bazel clean`, there’s a bug in Bazel.
