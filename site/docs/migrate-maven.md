---
layout: documentation
title: Migrating from Maven to Bazel
---

# Migrating from Maven to Bazel

When migrating from any build tool to Bazel, it’s best to have both build
tools running in parallel until you have fully migrated your development team,
CI system, and any other relevant systems. You can run Maven and Bazel in the
same repository.

## Table of contents


*  [Before you begin](#before-you-begin)
*  [Differences between Maven and Bazel](#differences-between-maven-and-bazel)
*  [Migrate from Maven to Bazel:](#migrate-from-maven-to-bazel)
   *  [1. Create the WORKSPACE file](#1-workspace)
      *  [Guava project example](#guava-1)
   *  [2. Create one BUILD file](#2-build)
      *  [Guava project example](#guava-2)
   *  [3. Create more BUILD files  (Optional)](#3-build)
   *  [4. Build using Bazel](#4-build)

## Before you begin

*  [Install Bazel](install.md) if it’s not yet installed.
*  If you’re new to Bazel, go through the tutorial
   [Introduction to Bazel: Build Java](tutorial/java.md) before you start
   migrating. The tutorial explains Bazel’s concepts, structure, and label
   syntax.

## Differences between Maven and Bazel

*  Maven uses top-level `pom.xml` file(s). Bazel supports multiple build
   files and multiple targets per BUILD file, allowing for builds that
   are more incremental than Maven's.
*  Maven takes charge of steps for the deployment process. Bazel does
   not automate deployment.
*  Bazel enables you to express dependencies between languages.
*  As you add new sections to the project, with Bazel you may need to add new
   BUILD files. Best practice is to add a BUILD file to each new Java package.

## Migrate from Maven to Bazel

The steps below describe how to migrate your project to Bazel:

1.  [Create the WORKSPACE file](#1-workspace)
2.  [Create one BUILD file](#2-build)
3.  [Create more BUILD files](#3-build)
4.  [Build using Bazel](#4-build)

Examples below come from a migration of the
[Guava project](https://github.com/google/guava) from Maven to Bazel. The Guava
project used is release 22.0. The examples using Guava do not walk through
each step in the migration, but they do show the files and contents that are
generated or added manually for the migration.

### <a name="1-workspace"></a>1. Create the WORKSPACE file

Create a file named `WORKSPACE` at the root of your project. If your project
has no external dependencies, the workspace file can be empty.

If your project depends on files or packages that are not in one of the
project’s directories, specify these external dependencies in the workspace
file. To automate the listing of external dependencies for the workspace file,
use the tool `generate_workspace`. For instructions about using this tool, see
[Generate a WORKSPACE file for a Java project](generate-workspace.md).

#### <a name="guava-1"></a>Guava project example: external dependencies

Below are the results of using the tool `generate_workspace` to list the
[Guava project's](https://github.com/google/guava) external dependencies.

1.  The new `WORKSPACE` file contains:

    ```bash
    load("//:generate_workspace.bzl", "generated_maven_jars")
    generated_maven_jars()
    ```

2.  The new `BUILD` file in the directory `third_party` enables access
    to external libraries. This BUILD file contains:

    ```bash
    load("//:generate_workspace.bzl", "generated_java_libraries")
    generated_java_libraries()
    ```

3.  The generated `generate_workspace.bzl` file contains:

    ```bash
    # The following dependencies were calculated from:
    #
    # generate_workspace --maven_project=/usr/local/.../guava


    def generated_maven_jars():
      # pom.xml got requested version
      # com.google.guava:guava-parent:pom:23.0-SNAPSHOT
      native.maven_jar(
          name = "com_google_code_findbugs_jsr305",
          artifact = "com.google.code.findbugs:jsr305:1.3.9",
          sha1 = "40719ea6961c0cb6afaeb6a921eaa1f6afd4cfdf",
      )


      # pom.xml got requested version
      # com.google.guava:guava-parent:pom:23.0-SNAPSHOT
      native.maven_jar(
          name = "com_google_errorprone_error_prone_annotations",
          artifact = "com.google.errorprone:error_prone_annotations:2.0.18",
          sha1 = "5f65affce1684999e2f4024983835efc3504012e",
      )


      # pom.xml got requested version
      # com.google.guava:guava-parent:pom:23.0-SNAPSHOT
      native.maven_jar(
          name = "com_google_j2objc_j2objc_annotations",
          artifact = "com.google.j2objc:j2objc-annotations:1.1",
          sha1 = "ed28ded51a8b1c6b112568def5f4b455e6809019",
      )




    def generated_java_libraries():
      native.java_library(
          name = "com_google_code_findbugs_jsr305",
          visibility = ["//visibility:public"],
          exports = ["@com_google_code_findbugs_jsr305//jar"],
      )


      native.java_library(
          name = "com_google_errorprone_error_prone_annotations",
          visibility = ["//visibility:public"],
          exports = ["@com_google_errorprone_error_prone_annotations//jar"],
      )


      native.java_library(
          name = "com_google_j2objc_j2objc_annotations",
          visibility = ["//visibility:public"],
          exports = ["@com_google_j2objc_j2objc_annotations//jar"],
      )
  ```

### <a name="2-build"></a>2. Create one BUILD file

Now that you have your workspace defined and external dependencies (if
applicable) listed, you need to create BUILD files to describe how your project
should be built. Unlike Maven with its one `pom.xml` file, Bazel can use many
BUILD files to build a project. These files specify multiple build targets,
which allow Bazel to produce incremental builds.

Add BUILD files in stages. Start with adding one BUILD file
at the root of your project and using it to do an initial build using Bazel.
Then, you refine your build by adding more BUILD files with more granular
targets.

1.  In the same directory as your `WORKSPACE` file, create a text file and
    name it `BUILD`.

2.  In this BUILD file, use the appropriate rule to create one target to
    build your project. Here are some tips:
    *  Use the appropriate rule:
       *  To build projects with a single Maven module, use the
          `java_library` rule as follows:

          ```bash
          java_library(
              name = "everything",
              srcs = glob(["src/main/java/**/*.java"]),
              resources = glob(["src/main/resources/**"]),
              deps = ["//:all-external-targets"],
          )
          ```
       *  To build projects with multiple Maven modules, use the
          `java_library` rule as follows:

          ```bash
          java_library(
              name = "everything",
              srcs = glob([
                  "Module1/src/main/java/**/*.java",
                  "Module2/src/main/java/**/*.java",
                  ...
              ]),
              resources = glob([
                  "Module1/src/main/resources/**",
                  "Module2/src/main/resources/**",
                  ...
              ]),
              deps = ["//:all-external-targets"],
          )
          ```
       *  To build binaries, use the `java_binary` rule:

          ```bash
          java_binary(
              name = "everything",
              srcs = glob(["src/main/java/**/*.java"]),
              resources = glob(["src/main/resources/**"]),
              deps = ["//:all-external-targets"],
              main_class = "com.example.Main"
          )
          ```
    *  Specify the attributes:
       *  `name`: Give the target a meaningful name. In the examples above
          we call the target “everything.”
       *  `srcs`: Use globbing to list all .java files in your project.
       *  `resources`: Use globbing to list all resources in your project.
       *  `deps`: You need to determine which external dependencies your
          project needs. For example, if you generated a list of external
          dependencies using the tool `generate_workspace`, the dependencies
          for `java_library` are the libraries listed in the
          `generated_java_libraries` macro.
    *  Take a look at the
       [example below of this top-level BUILD file](#guava-example-2) from
       the migration of the Guava project.

3.  Now that you have a BUILD file at the root of your project, build
    your project to ensure that it works. On the command line, from your
    workspace directory, use `bazel build //:everything` to build your
    project with Bazel.

    The project has now been successfully built with Bazel. You will need
    to add more BUILD files to allow incremental builds of the project.

#### <a name="guava-2"></a>Guava project example: start with one BUILD file

When migrating the Guava project to Bazel, initially one BUILD file is used
to build the entire project. Here are the contents of this initial `BUILD`
file in the workspace directory:

```bash
java_library(
    name = "everything",
    srcs = glob(["guava/src/**/*.java"]),
    deps = [
      "//third_party:com_google_code_findbugs_jsr305",
      "//third_party:com_google_errorprone_error_prone_annotations",
      "//third_party:com_google_j2objc_j2objc_annotations"
    ],
)
```

### <a name="3-build"></a>3. Create more BUILD files (Optional)

Bazel does work with just one BUILD file, as you saw after completing your first
build. You should still consider breaking the build into smaller chunks by
adding more BUILD files with granular targets.

Multiple BUILD files with multiple targets will give the build increased
granularity, allowing:

*  increased incremental builds of the project,
*  increased parallel execution of the build,
*  better maintainability of the build for future users, and
*  control over visibility of targets between packages, which can prevent
   issues such as libraries containing implementation details leaking into
   public APIs.

Tips for adding more BUILD files:

*  You can start by adding a BUILD file to each Java package. Start with
   Java packages that have the fewest dependencies and work you way up
   to packages with the most dependencies.
*  As you add BUILD files and specify targets, add these new targets to the
   `deps` sections of targets that depend on them. Note that the `glob()`
   function does not cross package boundaries, so as the number
   of packages grows the files matched by `glob()` will shrink.
*  Any time you add a BUILD file to a `main` directory, ensure that you add
   a BUILD file to the corresponding `test` directory.
*  Take care to limit visibility properly between packages.
*  To simplify troubleshooting errors in your setup of BUILD files, ensure
   that the project continues to build with Bazel as you add each build
   file. Run `bazel build //...` to ensure all of your targets still build.

### <a name="4-build"></a>4. Build using Bazel

You’ve been building using Bazel as you add BUILD files to validate the setup
of the build.

When you have BUILD files at the desired granularity, you can use Bazel
to produce all of your builds.
