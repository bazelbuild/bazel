---
layout: documentation
title: Build Tutorial - Java
---

Introduction to Bazel: Building a Java Project
==========

In this tutorial, you'll learn the basics of building Java applications with
Bazel. You will set up your workspace and build a simple Java project that
illustrates key Bazel concepts, such as targets and `BUILD` files.

Estimated completion time: 30 minutes.

## What you'll learn

In this tutorial you'll learn how to:

*  Build a target
*  Visualize the project's dependencies
*  Split the project into multiple targets and packages
*  Control target visibility across packages
*  Reference targets through labels
*  Deploy a target

## Contents

*  [Before you begin](#before-you-begin)
   *  [Install Bazel](#install-bazel)
   *  [Install the JDK](#install-the-jdk)
   *  [Get the sample project](#get-the-sample-project)
*  [Build with Bazel](#build-with-bazel)
   *  [Set up the workspace](#set-up-the-workspace)
   *  [Understand the BUILD file](#understand-the-build-file)
   *  [Build the project](#build-the-project)
   *  [Review the dependency graph](#review-the-dependency-graph)
*  [Refine your Bazel build](#refine-your-bazel-build)
   *  [Specify multiple build targets](#specify-multiple-build-targets)
   *  [Use multiple packages](#use-multiple-packages)
*  [Use labels to reference targets](#use-labels-to-reference-targets)
*  [Package a Java target for deployment](#package-a-java-target-for-deployment)
*  [Further reading](#further-reading)

## Before you begin

### Install Bazel

To prepare for the tutorial, first [Install Bazel](../install.md) if
you don't have it installed already.

### Install the JDK

1.  Install Java 8 JDK.

2.  Set the JAVA\_HOME environment variable to point to the JDK.
    *   On Linux/macOS:

            export JAVA_HOME="$(dirname $(dirname $(realpath $(which javac))))"
    *   On Windows:
        1.  Open Control Panel.
        2.  Go to "System&nbsp;and&nbsp;Security" &gt; "System" &gt; "Advanced&nbsp;System&nbsp;Settings" &gt; "Advanced"&nbsp;tab &gt; "Environment&nbsp;Variables..." .
        3.  Under the "User&nbsp;variables" list (the one on the top), click "New...".
        4.  In the "Variable&nbsp;name" field, enter `JAVA_HOME`.
        5.  Click "Browse&nbsp;Directory...".
        6.  Navigate to the JDK directory (for example `C:\Program Files\Java\jdk1.8.0_152`).
        7.  Click "OK" on all dialog windows.

### Get the sample project

Retrieve the sample project from Bazel's GitHub repository:

```sh
git clone https://github.com/bazelbuild/examples/
```

The sample project for this tutorial is in the `examples/java-tutorial`
directory and is structured as follows:

```
java-tutorial
├── BUILD
├── src
│   └── main
│       └── java
│           └── com
│               └── example
│                   ├── cmdline
│                   │   ├── BUILD
│                   │   └── Runner.java
│                   ├── Greeting.java
│                   └── ProjectRunner.java
└── WORKSPACE
```

## Build with Bazel

### Set up the workspace

Before you can build a project, you need to set up its workspace. A workspace is
a directory that holds your project's source files and Bazel's build outputs. It
also contains files that Bazel recognizes as special:

*  The `WORKSPACE` file, which identifies the directory and its contents as a
   Bazel workspace and lives at the root of the project's directory structure,

*  One or more `BUILD` files, which tell Bazel how to build different parts of
   the project. (A directory within the workspace that contains a `BUILD` file
   is a *package*. You will learn about packages later in this tutorial.)

To designate a directory as a Bazel workspace, create an empty file named
`WORKSPACE` in that directory.

When Bazel builds the project, all inputs and dependencies must be in the same
workspace. Files residing in different workspaces are independent of one
another unless linked, which is beyond the scope of this tutorial.

### Understand the BUILD file

A `BUILD` file contains several different types of instructions for Bazel.
The most important type is the *build rule*, which tells Bazel how to build the
desired outputs, such as executable binaries or libraries. Each instance
of a build rule in the `BUILD` file is called a *target* and points to a
specific set of source files and dependencies. A target can also point to other
targets.

Take a look at the `java-tutorial/BUILD` file:

```
java_binary(
    name = "ProjectRunner",
    srcs = glob(["src/main/java/com/example/*.java"]),
)
```

In our example, the `ProjectRunner` target instantiates Bazel's built-in
[`java_binary` rule](../be/java.html#java_binary). The rule tells Bazel to
build a `.jar` file and a wrapper shell script (both named after the target).

The attributes in the target explicitly state its dependencies and options.
While the `name` attribute is mandatory, many are optional. For example, in the
`ProjectRunner` rule target, `name` is the name of the target, `srcs` specifies
the source files that Bazel uses to build the target, and `main_class` specifies
the class that contains the main method. (You may have noticed that our example
uses [glob](../be/functions.html#glob) to pass a set of source files to Bazel
instead of listing them one by one.)

### Build the project

Let's build your sample project. Change into the `java-tutorial` directory
and run the following command:

```
bazel build //:ProjectRunner
```
Notice the target label - the `//` part is the location of our `BUILD` file
relative to the root of the workspace (in this case, the root itself), and
`ProjectRunner` is what we named that target in the `BUILD` file. (You will
learn about target labels in more detail at the end of this tutorial.)

Bazel produces output similar to the following:

```bash
   INFO: Found 1 target...
   Target //:ProjectRunner up-to-date:
      bazel-bin/ProjectRunner.jar
      bazel-bin/ProjectRunner
   INFO: Elapsed time: 1.021s, Critical Path: 0.83s
```

Congratulations, you just built your first Bazel target! Bazel places build
outputs in the `bazel-bin` directory at the root of the workspace. Browse
through its contents to get an idea for Bazel's output structure.

Now test your freshly built binary:

```sh
bazel-bin/ProjectRunner
```

### Review the dependency graph

Bazel requires build dependencies to be explicitly declared in BUILD files.
Bazel uses those statements to create the project's dependency graph, which
enables accurate incremental builds.

Let's visualize our sample project's dependencies. First, generate a text
representation of the dependency graph (run the command at the workspace root):

```
bazel query  --nohost_deps --noimplicit_deps "deps(//:ProjectRunner)" --output graph
```

The above command tells Bazel to look for all dependencies for the target
`//:ProjectRunner` (excluding host and implicit dependencies) and format the
output as a graph.

Then, paste the text into [GraphViz](http://www.webgraphviz.com/).


As you can see, the project has a single target that build two source files with
no additional dependencies:

![Dependency graph of the target 'ProjectRunner'](/assets/tutorial_java_01.svg)

Now that you have set up your workspace, built your project, and examined its
dependencies, let's add some complexity.

## Refine your Bazel build

While a single target is sufficient for small projects, you may want to split
larger projects into multiple targets and packages to allow for fast incremental
builds (that is, only rebuild what's changed) and to speed up your builds by
building multiple parts of a project at once.

### Specify multiple build targets

Let's split our sample project build into two targets. Replace the contents of
the `java-tutorial/BUILD` file with the following:

```
java_binary(
    name = "ProjectRunner",
    srcs = ["src/main/java/com/example/ProjectRunner.java"],
    main_class = "com.example.ProjectRunner",
    deps = [":greeter"],
)

java_library(
    name = "greeter",
    srcs = ["src/main/java/com/example/Greeting.java"],
)
```

With this configuration, Bazel first builds the `greeter` library, then the
`ProjectRunner` binary. The `deps` attribute in `java_binary` tells Bazel that
the `greeter` library is required to build the `ProjectRunner` binary.

Let's build this new version of our project. Run the following command:

```
bazel build //:ProjectRunner
```

Bazel produces output similar to the following:

```
INFO: Found 1 target...
Target //:ProjectRunner up-to-date:
  bazel-bin/ProjectRunner.jar
  bazel-bin/ProjectRunner
INFO: Elapsed time: 2.454s, Critical Path: 1.58s
```

Now test your freshly built binary:

```
bazel-bin/ProjectRunner
```

If you now modify `ProjectRunner.java` and rebuild the project, Bazel only
recompiles that file.

Looking at the dependency graph, you can see that `ProjectRunner` depends on the
same inputs as it did before, but the structure of the build is different:

![Dependency graph of the target 'ProjectRunner' after adding a dependency]
(/assets/tutorial_java_02.svg)

You've now built the project with two targets. The `ProjectRunner` target builds
two source files and depends on one other target (`:greeter`), which builds
one additional source file.

### Use multiple packages

Let’s now split the project into multiple packages. If you take a look at the
`src/main/java/com/example/cmdline` directory, you can see that it also contains
a `BUILD` file, plus some source files. Therefore, to Bazel, the workspace now
contains two packages, `//src/main/java/com/example/cmdline` and `//` (since
there is a `BUILD` file at the root of the workspace).

Take a look at the `src/main/java/com/example/cmdline/BUILD` file:

```
java_binary(
    name = "runner",
    srcs = ["Runner.java"],
    main_class = "com.example.cmdline.Runner",
    deps = ["//:greeter"]
)
```

The `runner` target depends on the `greeter` target in the `//` package (hence
the target label `//:greeter`) - Bazel knows this through the `deps` attribute.
Take a look at the dependency graph:

![Dependency graph of the target 'runner'](/assets/tutorial_java_03.svg)

However, for the build to succeed, you must explicitly give the `runner` target in
`//src/main/java/com/example/cmdline/BUILD` visibility to targets in
`//BUILD` using the `visibility` attribute. This is because by default targets
are only visible to other targets in the same `BUILD` file. (Bazel uses target
visibility to prevent issues such as libraries containing implementation details
leaking into public APIs.)

To do this, add the `visibility` attribute to the `greeter` target in
`java-tutorial/BUILD` as shown below:

```
java_library(
    name = "greeter",
    srcs = ["src/main/java/com/example/Greeting.java"],
    visibility = ["//src/main/java/com/example/cmdline:__pkg__"],
    )
```

Let's now build the new package. Run the following command at the root of the
workspace:

```
bazel build //src/main/java/com/example/cmdline:runner
```

Bazel produces output similar to the following:

```
INFO: Found 1 target...
Target //src/main/java/com/example/cmdline:runner up-to-date:
  bazel-bin/src/main/java/com/example/cmdline/runner.jar
  bazel-bin/src/main/java/com/example/cmdline/runner
  INFO: Elapsed time: 1.576s, Critical Path: 0.81s
```

Now test your freshly built binary:

```
./bazel-bin/src/main/java/com/example/cmdline/runner

```

You've now modified the project to build as two packages, each containing one
target, and understand the dependencies between them.


## Use labels to reference targets

In `BUILD` files and at the command line, Bazel uses target labels to reference
targets - for example, `//:ProjectRunner` or
`//src/main/java/com/example/cmdline:runner`. Their syntax is as follows:

```
//path/to/package:target-name
```

If the target is a rule target, then `path/to/package` is the path to the
directory containing the `BUILD` file, and `target-name` is what you named the
target in the `BUILD` file (the `name` attribute). If the target is a file
target, then `path/to/package` is the path to the root of the package, and
`target-name` is the name of the target file, including its full path.

When referencing targets within the same package, you can skip the package path
and just use `//:target-name`. When referencing targets within the same `BUILD`
file, you can even skip the `//` workspace root identifier and just use
`:target-name`.

For example, for targets in the `java-tutorial/BUILD` file, you did not have to
specify a package path, since the workspace root is itself a package (`//`), and
your two target labels were simply `//:ProjectRunner` and `//:greeter`.

However, for targets in the `//src/main/java/com/example/cmdline/BUILD` file you
had to specify the full package path of `//src/main/java/com/example/cmdline`
and your target label was `//src/main/java/com/example/cmdline:runner`.

## Package a Java target for deployment

Let’s now package a Java target for deployment by building the binary with all
of its runtime dependencies. This lets you run the binary outside of your
development environment.

As you remember, the [java_binary](../be/java.html#java_binary) build rule
produces a `.jar` and a wrapper shell script. Take a look at the contents of
`runner.jar` using this command:

```
jar tf bazel-bin/src/main/java/com/example/cmdline/runner.jar
```

The contents are:

```
META-INF/
META-INF/MANIFEST.MF
com/
com/example/
com/example/cmdline/
com/example/cmdline/Runner.class
```
As you can see, `runner.jar` contains `Runner.class`, but not its dependency,
`Greeting.class`. The `runner` script that Bazel generates adds `greeter.jar`
to the classpath, so if you leave it like this, it will run locally, but it
won't run standalone on another machine. Fortunately, the `java_binary` rule
allows you to build a self-contained, deployable binary. To build it, add the
`_deploy.jar` suffix to the file name when building `runner.jar`
(<target-name>_deploy.jar):

```
bazel build //src/main/java/com/example/cmdline:runner_deploy.jar
```

Bazel produces output similar to the following:

```
INFO: Found 1 target...
Target //src/main/java/com/example/cmdline:runner_deploy.jar up-to-date:
  bazel-bin/src/main/java/com/example/cmdline/runner_deploy.jar
INFO: Elapsed time: 1.700s, Critical Path: 0.23s
```
You have just built `runner_deploy.jar`, which you can run standalone away from
your development environment since it contains the required runtime
dependencies.

## Further reading

*  [External Dependencies](../external.html) to learn more about working with
   local and remote repositories.

*  The [Build Encyclopedia](../be/overview.html) to learn more about Bazel.

*  The [C++ build tutorial](../tutorial/cpp.md) to get started with building
   C++ projects with Bazel.

*  The [Android application tutorial](../tutorial/android-app.md) and
   [iOS application tutorial](../tutorial/ios-app.md) to get started with
   building mobile applications for Android and iOS with Bazel.

Happy building!

