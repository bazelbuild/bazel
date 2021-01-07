--------------------------------------------------------------------------------

layout: documentation

## title: Java and Bazel

# Java and Bazel

This page contains resources that help you use Bazel with Java projects. It
links to a tutorial, build rules, and other information specific to building
Java projects with Bazel.

## Working with Bazel

The following resources will help you work with Bazel on Java projects:

*   [Tutorial: Building a Java Project](tutorial/java.html)
*   [Java rules](be/java.html)

## Migrating to Bazel

If you currently build your Java projects with Maven, follow the steps in the
migration guide to start building your Maven projects with Bazel:

*   [Migrating from Maven to Bazel](migrate-maven.html)

## Java versions

There are two relevant versions of Java that are set with configuration flags:
 - the version of the source files in the repository
 - the version of the Java runtime that is used to execute the code and to test it

Without an additional configuration, Bazel assumes all Java source files in the
repository are written in a single Java version. To specify the version of the
sources in the repository add `build --java_language_version={ver}` to
`.bazelrc` file, where `{ver}` is for example `11`. Bazel repository owners
should set this flag so that Bazel and its users can reference the source code's
Java version number. For more details, see
[Java language version flag](user-manual.html#flag--java_language_version).

Bazel uses one JDK for compilation and another JVM to execute and test the code.

By default Bazel compiles the code using a JDK it downloads and it executes and
tests the code with the JVM installed on the local machine. Bazel searches for
the JVM using `JAVA_HOME` or path.

The resulting binaries are compatible with locally installed JVM in system
libraries, which means the resulting binaries depend on what is installed on the
machine.

To create a hermetic compile, you can use command line flag
`--java_runtime_version=remotejdk_11`. The code is compiled for, executed, and
tested on the JVM downloaded from a remote repository. For more details, see
[Java runtime version flag](user-manual.html#flag--java_runtime_version).

There is a second pair of JDK and JVM used to build and execute tools, which are
used in the build process, but are not in the build results. That JDK and JVM
are controlled using `--tool_java_language_version` and
`--tool_java_runtime_version`. Default values are 11 and `remotejdk_11`,
respectively.

For more details, see
[configuring Java toolchains](#Configuring-the-Java-toolchains).

## Best practices

In addition to [general Bazel best practices](best-practices.html), below are
best practices specific to Java projects.

### Directory structure

Prefer Maven's standard directory layout (sources under `src/main/java`, tests
under `src/test/java`).

### BUILD files

Follow these guidelines when creating your BUILD files:

*   Use one BUILD file per directory containing Java sources, because this
    improves build performance.

*   Every BUILD file should contain one `java_library` rule that looks like
    this:

    ```python
    java_library(
        name = "directory-name",
        srcs = glob(["*.java"]),
        deps = [...],
    )
    ```

*   The name of the library should be the name of the directory containing the
    BUILD file. This makes the label of the library shorter, that is use
    `"//package"` instead of `"//package:package"`.

*   The sources should be a non-recursive [`glob`](be/functions.html#glob) of
    all Java files in the directory.

*   Tests should be in a matching directory under `src/test` and depend on this
    library.

## Creating new rules for advanced Java builds

**Note**: Creating new rules is for advanced build and test scenarios. You do
not need it when getting started with Bazel.

The following modules, configuration fragments, and providers will help you
[extend Bazel's capabilities](skylark/concepts.html) when building your Java
projects:

*   Main Java provider: [`java_common`](skylark/lib/java_common.html)
*   Main Java module: [`JavaInfo`](skylark/lib/JavaInfo.html)
*   Configuration fragment: [`java`](skylark/lib/java.html)
*   Other modules:

    *   [`java_annotation_processing`](skylark/lib/java_annotation_processing.html)
    *   [`java_compilation_info`](skylark/lib/java_compilation_info.html)
    *   [`java_output`](skylark/lib/java_output.html)
    *   [`java_output_jars`](skylark/lib/java_output_jars.html)
    *   [`java_proto_common`](skylark/lib/java_proto_common.html)
    *   [`JavaRuntimeInfo`](skylark/lib/JavaRuntimeInfo.html)
    *   [`JavaToolchainInfo`](skylark/lib/JavaToolchainInfo.html)

## Configuring the Java toolchains

Bazel uses two types of Java toolchains: - execution, used to execute and test
Java binaries, controlled with `--java_runtime_version` flag - compilation, used
to compile Java sources, controlled with `--java_language_version` flag

### Execution toolchains

Execution toolchain is the JVM, either local or from a repository, with some
additional information about its version, operating system, and CPU
architecture.

Java execution toolchains may added using `local_java_repository` or
`remote_java_repository` rules in the `WORKSPACE` file. Adding the rule makes
the JVM available using a flag. When multiple definitions for the same operating
system and CPU architecture are given, the first one is used.

Example configuration of local JVM: ```python
load("@bazel_tools//tools/jdk:local_java_repository.bzl",
"local_java_repository")

local_java_repository( name = "additionaljdk", # Can be used with
--java_runtime_version=additionaljdk or --java_runtime_version=11 version =
11, # Optional, if not set it is autodetected java_home = "/usr/lib/jdk-15/", #
Path to directory containing bin/java ) ```

Example configuration of remote JVM: ```python
load("@bazel_tools//tools/jdk:remote_java_repository.bzl",
"remote_java_repository")

remote_java_repository( name = "openjdk_canary_linux_arm", prefix =
"openjdk_canary", # Can be used with --java_runtime_version=openjdk_canary_11
version = "11", # or --java_runtime_version=11 exec_compatible_with = [ #
Specifies contraints this JVM is compatible with "@platforms//cpu:arm",
"@platforms//os:linux", ], urls = ... # Other parameters are from
http_repository rule. sha256 = ... strip_prefix = ... ) ```

### Compilation toolchains

Compilation toolchain is composed of JDK and multiple tools that Bazel uses
during the compilation and that provides additional features, such as: Error
Prone, strict Java dependenciess, header compilation, Android desugaring,
coverage instrumentation, and genclass handling for IDEs.

You can reconfigure the compilation by adding `default_java_toolchain` macro to
a `BUILD` file and registering it either by adding `register_toolchain` rule to
the `WORKSPACE` file or by using
[`--extra_toolchains`](user-manual.html#flag--extra_toolchains) flag.

Example toolchain configuration: ```python
load("@bazel_tools@bazel_tools//tools/jdk:default_java_toolchain.bzl",
"default_java_toolchain")

default_java_toolchain( name = "repository_default_toolchain", configuration =
DEFAULT_TOOLCHAIN_CONFIGURATION, # One of predefined configurations

\# Other parameters are from java_toolchain rule: java_runtime =
"//tools/jdk:remote_jdk11", # JDK to use for compilation jvm_opts =
JDK9_JVM_OPTS + ["--enable_preview"] # Additional JDK options misc =
DEFAULT_JAVACOPTS + ["--enable_preview"] # Additional javac options
source_version = "9", ) ```

Predefined configurations:

-   `DEFAULT_TOOLCHAIN_CONFIGURATION`: all features, supports JDK versions >= 9
-   `VANILLA_TOOLCHAIN_CONFIGURATION`: no additional features, supports all JDKs
-   `JVM8_TOOLCHAIN_CONFIGURATION`: all features, JDK version 8
-   `PREBUILT_TOOLCHAIN_CONFIGURATION`: same as default, but only use prebuilt
    tools (`ijar`, `singlejar`)
-   `NONPREBUILT_TOOLCHAIN_CONFIGURATION`: same as default, but all tools are
    built from sources (this may be useful on operating system with different
    libc)
