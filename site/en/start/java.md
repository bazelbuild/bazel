Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Build a Java Project

{% include "_buttons.html" %}

This tutorial covers the basics of building Java applications with
Bazel. You will set up your workspace and build a simple Java project that
illustrates key Bazel concepts, such as targets and `BUILD` files.

Estimated completion time: 30 minutes.

## What you'll learn

In this tutorial you learn how to:

*  Build a target
*  Visualize the project's dependencies
*  Split the project into multiple targets and packages
*  Control target visibility across packages
*  Reference targets through labels
*  Deploy a target

## Before you begin

### Install Bazel

To prepare for the tutorial, first [Install Bazel](/install) if
you don't have it installed already.

### Install the JDK

1.  Install Java JDK (preferred version is 11, however versions between 8 and 15 are supported).

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

```posix-terminal
git clone https://github.com/bazelbuild/examples
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
└── MODULE.bazel
```

## Build with Bazel

### Set up the workspace

Before you can build a project, you need to set up its workspace. A workspace is
a directory that holds your project's source files and Bazel's build outputs. It
also contains files that Bazel recognizes as special:

*  The `MODULE.bazel` file, which identifies the directory and its contents as a
   Bazel workspace and lives at the root of the project's directory structure,

*  One or more `BUILD` files, which tell Bazel how to build different parts of
   the project. (A directory within the workspace that contains a `BUILD` file
   is a *package*. You will learn about packages later in this tutorial.)

To designate a directory as a Bazel workspace, create an empty file named
`MODULE.bazel` in that directory.

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

```python
java_binary(
    name = "ProjectRunner",
    srcs = glob(["src/main/java/com/example/*.java"]),
)
```

In our example, the `ProjectRunner` target instantiates Bazel's built-in
[`java_binary` rule](/reference/be/java#java_binary). The rule tells Bazel to
build a `.jar` file and a wrapper shell script (both named after the target).

The attributes in the target explicitly state its dependencies and options.
While the `name` attribute is mandatory, many are optional. For example, in the
`ProjectRunner` rule target, `name` is the name of the target, `srcs` specifies
the source files that Bazel uses to build the target, and `main_class` specifies
the class that contains the main method. (You may have noticed that our example
uses [glob](/reference/be/functions#glob) to pass a set of source files to Bazel
instead of listing them one by one.)

### Build the project

To build your sample project, navigate to the `java-tutorial` directory
and run:

```posix-terminal
bazel build //:ProjectRunner
```
In the target label, the `//` part is the location of the `BUILD` file
relative to the root of the workspace (in this case, the root itself),
and `ProjectRunner` is the target name in the `BUILD` file. (You will
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

```posix-terminal
bazel-bin/ProjectRunner
```

### Review the dependency graph

Bazel requires build dependencies to be explicitly declared in BUILD files.
Bazel uses those statements to create the project's dependency graph, which
enables accurate incremental builds.

To visualize the sample project's dependencies, you can generate a text
representation of the dependency graph by running this command at the
workspace root:

```posix-terminal
bazel query  --notool_deps --noimplicit_deps "deps(//:ProjectRunner)" --output graph
```

The above command tells Bazel to look for all dependencies for the target
`//:ProjectRunner` (excluding host and implicit dependencies) and format the
output as a graph.

Then, paste the text into [GraphViz](http://www.webgraphviz.com/).

As you can see, the project has a single target that build two source files with
no additional dependencies:

![Dependency graph of the target 'ProjectRunner'](/docs/images/tutorial_java_01.svg)

After you set up your workspace, build your project, and examine its
dependencies, then you can add some complexity.

## Refine your Bazel build

While a single target is sufficient for small projects, you may want to split
larger projects into multiple targets and packages to allow for fast incremental
builds (that is, only rebuild what's changed) and to speed up your builds by
building multiple parts of a project at once.

### Specify multiple build targets

You can split the sample project build into two targets. Replace the contents of
the `java-tutorial/BUILD` file with the following:

```python
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

To build this new version of the project, run the following command:

```posix-terminal
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

```posix-terminal
bazel-bin/ProjectRunner
```

If you now modify `ProjectRunner.java` and rebuild the project, Bazel only
recompiles that file.

Looking at the dependency graph, you can see that `ProjectRunner` depends on the
same inputs as it did before, but the structure of the build is different:

![Dependency graph of the target 'ProjectRunner' after adding a dependency](
/docs/images/tutorial_java_02.svg)

You've now built the project with two targets. The `ProjectRunner` target builds
one source files and depends on one other target (`:greeter`), which builds
one additional source file.

### Use multiple packages

Let’s now split the project into multiple packages. If you take a look at the
`src/main/java/com/example/cmdline` directory, you can see that it also contains
a `BUILD` file, plus some source files. Therefore, to Bazel, the workspace now
contains two packages, `//src/main/java/com/example/cmdline` and `//` (since
there is a `BUILD` file at the root of the workspace).

Take a look at the `src/main/java/com/example/cmdline/BUILD` file:

```python
java_binary(
    name = "runner",
    srcs = ["Runner.java"],
    main_class = "com.example.cmdline.Runner",
    deps = ["//:greeter"],
)
```

The `runner` target depends on the `greeter` target in the `//` package (hence
the target label `//:greeter`) - Bazel knows this through the `deps` attribute.
Take a look at the dependency graph:

![Dependency graph of the target 'runner'](/docs/images/tutorial_java_03.svg)

However, for the build to succeed, you must explicitly give the `runner` target
in `//src/main/java/com/example/cmdline/BUILD` visibility to targets in
`//BUILD` using the `visibility` attribute. This is because by default targets
are only visible to other targets in the same `BUILD` file. (Bazel uses target
visibility to prevent issues such as libraries containing implementation details
leaking into public APIs.)

To do this, add the `visibility` attribute to the `greeter` target in
`java-tutorial/BUILD` as shown below:

```python
java_library(
    name = "greeter",
    srcs = ["src/main/java/com/example/Greeting.java"],
    visibility = ["//src/main/java/com/example/cmdline:__pkg__"],
)
```

Now you can build the new package by running the following command at the root
of the workspace:

```posix-terminal
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

```posix-terminal
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

When referencing targets at the repository root, the package path is empty,
just use `//:target-name`. When referencing targets within the same `BUILD`
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

As you remember, the [java_binary](/reference/be/java#java_binary) build rule
produces a `.jar` and a wrapper shell script. Take a look at the contents of
`runner.jar` using this command:

```posix-terminal
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
allows you to build a self-contained, deployable binary. To build it, append
`_deploy.jar` to the target name:

```posix-terminal
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
dependencies. Take a look at the contents of this standalone JAR using the
same command as before:

```posix-terminal
jar tf bazel-bin/src/main/java/com/example/cmdline/runner_deploy.jar
```

The contents include all of the necessary classes to run:

```
META-INF/
META-INF/MANIFEST.MF
build-data.properties
com/
com/example/
com/example/cmdline/
com/example/cmdline/Runner.class
com/example/Greeting.class
```

## Further reading

For more details, see:

*  [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external) for
   rules to manage transitive Maven dependencies.

*  [External Dependencies](/docs/external) to learn more about working with
   local and remote repositories.

*  The [other rules](/rules) to learn more about Bazel.

*  The [C++ build tutorial](/start/cpp) to get started with building
   C++ projects with Bazel.

*  The [Android application tutorial](/start/android-app ) and
   [iOS application tutorial](/start/ios-app)) to get started with
   building mobile applications for Android and iOS with Bazel.

Happy building!
