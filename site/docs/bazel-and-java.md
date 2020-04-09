---
layout: documentation
title: Java and Bazel
---

# Java and Bazel

This page contains resources that help you use Bazel with Java projects. It
links to a tutorial, build rules, and other information specific to building
Java projects with Bazel.

## Contents

- [Working with Bazel](#working-with-bazel)
- [Migrating to Bazel](#migrating-to-bazel)
- [Best practices](#best-practices)
   - [Directory structure](#directory-structure)
   - [BUILD files](#build-files)
- [Java and new rules](#java-and-new-rules)

## Working with Bazel

The following resources will help you work with Bazel on Java projects:

*  [Tutorial: Building a Java Project](tutorial/java.html)
*  [Java rules](be/java.html)

## Migrating to Bazel

If you currently build your Java projects with Maven, follow the steps in the
migration guide to start building your Maven projects with Bazel:

*  [Migrating from Maven to Bazel](migrate-maven.html)

## Best practices

In addition to [general Bazel best practices](best-practices.html), below are
best practices specific to Java projects.

### Directory structure

Prefer Maven's standard directory layout (sources under `src/main/java`, tests
under `src/test/java`).

### BUILD files

Follow these guidelines when creating your BUILD files:

*  Use one BUILD file per package containing Java sources.

*  Every BUILD file should contain one `java_library` rule that looks like this:

   ```python
   java_library(
       name = "directory-name",
       srcs = glob(["*.java"]),
       deps = [...],
   )
   ```
*  The name of the library should be the name of the directory containing the
   BUILD file.

*  The sources should be a non-recursive [`glob`](be/functions.html#glob)
   of all Java files in the directory.

*  Tests should be in a matching directory under `src/test` and depend on this
   library.

## Java and new rules

**Note**: Creating new rules is for advanced build and test scenarios.
You do not need it when getting started with Bazel.

The following modules, configuration fragments, and providers will help you
[extend Bazel's capabilities](starlark/concepts.html)
when building your Java projects:

*  Modules:

   *  [`java_annotation_processing`](starlark/lib/java_annotation_processing.html)
   *  [`java_common`](starlark/lib/java_common.html)
   *  [`java_compilation_info`](starlark/lib/java_compilation_info.html)
   *  [`java_output`](starlark/lib/java_output.html)
   *  [`java_output_jars`](starlark/lib/java_output_jars.html)
   *  [`java_proto_common`](starlark/lib/java_proto_common.html)
   *  [`JavaRuntimeClasspathProvider`](starlark/lib/JavaRuntimeClasspathProvider.html)
   *  [`JavaRuntimeInfo`](starlark/lib/JavaRuntimeInfo.html)
   *  [`JavaToolchainSkylarkApiProvider`](starlark/lib/JavaToolchainSkylarkApiProvider.html)

*  Configuration fragments:

   *  [`java`](starlark/lib/java.html)

*  Providers:

   *  [`java`](starlark/lib/JavaSkylarkApiProvider.html)
   *  [`JavaInfo`](starlark/lib/JavaInfo.html)
