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
[extend Bazel's capabilities](skylark/concepts.html)
when building your Java projects:

*  Modules:

   *  [`java_annotation_processing`](skylark/lib/java_annotation_processing.html)
   *  [`java_common`](skylark/lib/java_common.html)
   *  [`java_compilation_info`](skylark/lib/java_compilation_info.html)
   *  [`java_output`](skylark/lib/java_output.html)
   *  [`java_output_jars`](skylark/lib/java_output_jars.html)
   *  [`java_proto_common`](skylark/lib/java_proto_common.html)
   *  [`JavaRuntimeClasspathProvider`](skylark/lib/JavaRuntimeClasspathProvider.html)
   *  [`JavaRuntimeInfo`](skylark/lib/JavaRuntimeInfo.html)
   *  [`JavaToolchainStarlarkApiProvider`](skylark/lib/JavaToolchainStarlarkApiProvider.html)

*  Configuration fragments:

   *  [`java`](skylark/lib/java.html)

*  Providers:

   *  [`java`](skylark/lib/JavaStarlarkApiProvider.html)
   *  [`JavaInfo`](skylark/lib/JavaInfo.html)

## Configuring the JDK

Bazel is configured to use a default OpenJDK 11 for building and testing
JVM-based projects. However, you can switch to another JDK using the
[`--java_toolchain`](command-line-reference.html#flag--java_toolchain) and
[`--javabase`](command-line-reference.html#flag--javabase) flags.

In short,

* `--java_toolchain`: A [`java_toolchain`](be/java.html#java_toolchain)
  target that defines the set of Java tools for building target binaries.
* `--javabase`: A [`java_runtime`](be/java.html#java_runtime) target defining
  the Java runtime for running target JVM binaries.

The
[`--host_java_toolchain`](command-line-reference.html#flag--host_java_toolchain)
and [`--host_javabase`](command-line-reference.html#flag--host_javabase)
variants are meant for building and running host binaries that Bazel
uses for building target binaries. These host binaries belong to
`--java_toolchain`, which includes `JavaBuilder` and `Turbine`.

Bazel's default flags essentially look like this:

```
$ bazel build \
      --host_javabase=@bazel_tools//tools/jdk:remote_jdk11 \
      --javabase=@bazel_tools//tools/jdk:remote_jdk11 \
      --host_java_toolchain=@bazel_tools//tools/jdk:toolchain_java11 \
      --java_toolchain=@bazel_tools//tools/jdk:toolchain_java11 \
      //my/java:target
```

`@bazel_tools` comes with a number of `java_toolchain` targets. Run the
following command to list them:

```
$ bazel query 'kind(java_toolchain, @bazel_tools//tools/jdk:all)'
```

Similarly for `java_runtime` targets:

```
$ bazel query 'kind(java_runtime, @bazel_tools//tools/jdk:all)'
```

For example, if you'd like to use a locally installed JDK installed at
`/usr/lib/jvm/java-13-openjdk`, use the `absolute_javabase` `java_runtime`
target and the `toolchain_vanilla` `java_toolchain` target, and define
`ABSOLUTE_JAVABASE` as the absolute path to the JDK.


```
bazel build \
    --define=ABSOLUTE_JAVABASE=/usr/lib/jvm/java-13-openjdk \
    --javabase=@bazel_tools//tools/jdk:absolute_javabase \
    --host_javabase=@bazel_tools//tools/jdk:absolute_javabase \
    --java_toolchain=@bazel_tools//tools/jdk:toolchain_vanilla \
    --host_java_toolchain=@bazel_tools//tools/jdk:toolchain_vanilla \
    //my/java_13:target
```

Optionally, you can add the flags into your project's `.bazelrc` file to
avoid having to specify them every time:

```
build --define=ABSOLUTE_JAVABASE=/usr/lib/jvm/java-13-openjdk
build --javabase=@bazel_tools//tools/jdk:absolute_javabase
build --host_javabase=@bazel_tools//tools/jdk:absolute_javabase
build --java_toolchain=@bazel_tools//tools/jdk:toolchain_vanilla
build --host_java_toolchain=@bazel_tools//tools/jdk:toolchain_vanilla
```

You can also write your own `java_runtime` and `java_toolchain` targets. As a
tip, use `bazel query --output=build @bazel_tools//tools/jdk:all` to see how
the built-in runtime and toolchain targets are defined.
