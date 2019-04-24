---
layout: documentation
title: Generate external dependencies for a Java project
---

> NOTE: `generate_workspace` is no longer maintained by the Bazel team. The
GitHub project has been archived. Instead, please use [`rules_jvm_external`](
https://github.com/bazelbuild/rules_jvm_external) to fetch and resolve
Maven artifacts transitively.

# Generate external dependencies from Maven projects

The tool `generate_workspace` helps automate the process of writing
the WORKSPACE file for a Java project. This tool is
helpful when the list of external dependencies is long, such as when working
with [external transitive dependencies](external.html#transitive-dependencies).

The `generate_workspace` tool will generate a `generate_workspace.bzl` file
which includes:

*   the `generated_maven_jars` macro that will contain the transitive
    dependencies, and
*   the `generated_java_libraries` macro that will contain a library for
    each maven_jar.

## Install `generate_workspace`

Bazel's binary installer does not include `generate_workspace`. To be able to
use this tool:

1.  Clone Bazel's migration tooling repo:

    ```
    git clone https://github.com/bazelbuild/migration-tooling.git
    ```

2.  Run the following to build the `generate_workspace` tool and see usage:

    ```
    bazel run //generate_workspace
    ```

## Generate a list of external dependencies

1.  Run the `generate_workspace` tool.

    When you run the tool, you can specify Maven projects (that is,
    directories containing a `pom.xml` file), or Maven artifact coordinates
    directly. For example:

    ```bash
    $ bazel run //generate_workspace -- \
    >    --maven_project=/path/to/my/project \
    >    --artifact=groupId:artifactId:version \
    >    --artifact=groupId:artifactId:version \
    >    --repositories=https://jcenter.bintray.com
    Wrote
    /usr/local/.../generate_workspace.runfiles/__main__/generate_workspace.bzl
    ```

    The tool creates one outputs, a `generate_workspace.bzl` file that contains
    two macros:

    1.  The `generated_maven_jars` macro that will contain the transitive
        dependencies of the given projects and artifacts.

    2.  The `generated_java_libraries` macro will contain a library
        for each maven_jar.

    If you specify multiple Maven projects or artifacts, they will all be
    combined into one `generate_workspace.bzl` file. For example, if an
    artifact depends on junit and the Maven project also depends on junit, then
    junit will only appear once as a dependency in the output.

2.  Copy the `generate_workspace.bzl` file to your workspace. The `.bzl`
    file's original location is listed in the commandline output.

    To access external dependencies:

    1.  Add the following to your WORKSPACE file:

        ```
        load("//:generate_workspace.bzl", "generated_maven_jars")
        generated_maven_jars()
        ```

        You can now access any of the jars in `generate_workspace.bzl`.

        This reference points to the jar, but not to any dependencies
        that the jar itself may have. To have a target depend on one of these
        jars, you must list the jar as well as each of that jar's dependencies.

        For example, to depend on the Guava jar from the
        [Guava project](https://github.com/google/guava/tree/master/guava),
        in the target definition you will need to list the jar and its
        transitive dependencies:

        ```bash
        deps = [
          "@com_google_guava_guava//jar",
          "@com_google_code_findbugs_jsr305//jar",
          "@com_google_errorprone_error_prone_annotations//jar",
          "@com_google_j2objc_j2objc_annotations//jar",
        ]
        ```

    2.  Optionally, you can also access the libraries. When you list a library
        as a dependency, the transitive dependencies are already included, and
        so you don't need to list them manually.

        To access the libraries, add the following to a BUILD file:

        ```
        load("//:generate_workspace.bzl", "generated_java_libraries")
        generated_java_libraries()
        ```

        The recommended location for this BUILD file is in a directory called
        `third_party`.

        You can now access any of the Java library targets in
        `generate_workspace.bzl`.

        For example, for a target to depend on Guava and its transitive
        dependencies, in the target definition you will need to list:

        ```bash
        deps = ["//third_party:com_google_guava_guava"]
        ```

3.  Ensure `generate_workspace.bzl` lists the correct version of each
    dependency.

    If several different versions of an artifact are requested (for example, by
    different libraries that depend on it), then `generate_workspace` chooses
    a version and annotates the `maven_jar` with the other versions requested.

    Here's an example of the contents of `generate_workspace.bzl`:

    ```python
    # org.springframework:spring:2.5.6
    # javax.mail:mail:1.4
    # httpunit:httpunit:1.6 wanted version 1.0.2
    # org.springframework:spring-support:2.0.2 wanted version 1.0.2
    # org.slf4j:nlog4j:1.2.24 wanted version 1.0.2
    native.maven_jar(
        name = "javax_activation_activation",
        artifact = "javax.activation:activation:1.1",
    )
    ```

    The example above indicates that `org.springframework:spring:2.5.6`,
    `javax.mail:mail:1.4`, `httpunit:httpunit:1.6`,
    `org.springframework:spring-support:2.0.2`, and `org.slf4j:nlog4j:1.2.24`
    all depend on `javax.activation`. However, two of these libraries wanted
    version 1.1 and three of them wanted 1.0.2. The WORKSPACE file is using
    version 1.1, but that might not be the right version to use.
