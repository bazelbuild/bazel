---
layout: documentation
title: Introduction to Bazel
---

Introduction to Bazel: Build Java
==========

This tutorial is an introduction for anyone getting started with Bazel. It
focuses on the concepts, setup, and use of Bazel using a Java sample project.

Estimated time: 30 min

## What you will learn

In this tutorial you'll learn how to:

*  Build a target from source files
*  Produce a visual representation of the dependency graph
*  Break a monolithic binary into smaller libraries
*  Use multiple Bazel packages
*  Control the visibility of a target between packages
*  Use labels to reference a target
*  Deploy your target

## Before you begin

*  [Install Bazel](/docs/install.md)

## Create the sample Java project

The first step in this tutorial is to create a small Java project. Even though
the project is in Java, this tutorial will focus on concepts that are helpful
for using Bazel in any language.

1.  Create the directory `~/my-project/`

2.  Move to this directory:

    ```
    cd ~/my-project
    ```

3.  Create the following directories under `my-project`:

    ```
    mkdir -p src/main/java/com/example
    ```

    Note that path uses conventions specific to Java  programs. Programs written
    in other languages may have a different workspace path and directory
    structure.

4.  In the directory you created, add a file called `Greetings.java` with the
    following contents:

    ```java
    package com.example;

    public class Greeting {
        public static void sayHi() {
            System.out.println("Hi!");
        }
    }
    ```

5.  Add a second file `ProjectRunner.java` with the following contents:

    ```java
    package com.example;

    public class ProjectRunner {
        public static void main(String args[]) {
            Greeting.sayHi();
        }
    }
    ```

You’ve now created a small Java project. It contains one file that will be
compiled into a library, and another which will be an executable that uses the
library.

The rest of this tutorial focuses on setting up and using Bazel to build these
source files.

## Build with Bazel

### Set up the workspace

Workspaces are directories that contain the source files for one or more
software projects, as well as a WORKSPACE file and BUILD files that contain
the instructions that Bazel uses to build the software. The workspace may also
contain symbolic links to output directories.

To define the workspace, create an empty text file at the root of the project
and name it `WORKSPACE`. You now have: `~/my-project/WORKSPACE`.

This directory and its subdirectories are now part of the same workspace. When
Bazel builds an output, all inputs and dependencies must be in the same
workspace. Anything in different workspaces are independent of each other,
though there are ways to link workspaces that are beyond the scope of this
introduction tutorial.

If you also do the [C++ tutorial](/docs/tutorial/cpp.md), you’ll notice it uses
the same workspace. Bazel can understand multiple targets in multiple languages
in a single workspace.

### Create a BUILD file

Bazel looks for files named `BUILD` which describe how to build the project.

1.  In the `~/my-project` directory, create a file and name it BUILD. This BUILD
    file is a sibling of the WORKSPACE file.

    In the BUILD file, you use a declarative language similar to Python to
    create instances of Bazel rules. These instances are called *rule targets*.
    In Bazel, *targets* are either files or rule targets and they are the
    elements in a workspace that you can ask Bazel to build.

    For this project, you’ll use the built-in rule `java_binary`. Bazel's
    built-in rules are all documented in the
    [Build Encyclopedia](/docs/be/overview.html). You can also create your own
    rules using the [Bazel rule extension framework](/docs/skylark/concepts.md).

2.  Add this text to the BUILD file:

    ```
    java_binary(
        name = "my-runner",
        srcs = glob(["src/main/java/com/example/*.java"]),
        main_class = "com.example.ProjectRunner",
    )
    ```
As you can see, the text in the BUILD file doesn’t describe what Bazel does
when it executes this rule target. The rule’s implementation handles the
complexity of how it works (such as the compiler used).

You can treat the rule as a black box, focusing on what inputs it needs, and
the outputs it produces. This rule builds a Java archive ("jar file") as well
as a wrapper shell script with the same name as the rule target.

When you’re writing your own BUILD file, go to the
[Build Encyclopedia](/docs/be/overview.html) for a description of what a rule
does and for its list of possible attributes you can define. For example,
here’s the entry for the [java_binary](/docs/be/java.html#java_binary) rule in
the Build Encyclopedia. The Build Encyclopedia has information about all of the
rules that are compiled into Bazel.

Let’s take a look at the rule target that you added to the BUILD file.

Each rule instantiation in the BUILD file creates one rule target. Here, you’ve
instantiated the rule `java_binary`, creating the target `my-runner`.

Different rules will require different attributes, though all must include a
“name” attribute. You use these attributes to explicitly list all of the
target’s dependencies and options. In the target above:

*  `my-runner` is the name of the rule target created

*  `glob(["src/main/java/com/example/*.java"])` includes every file in that
   directory that ends with .java (see the Build Encyclopedia for more
   information about [globbing](/docs/be/functions.html#glob))

*  `"com.example.ProjectRunner"` specifies the class that contains the main
   method.

### Build with Bazel

Now you’re ready to build the Java binary. To do so, you’ll use the command
`bazel build` with the target label `//:my-runner`. You reference targets by
using their label. Label syntax is described later in this tutorial.

1.  Build my-runner by using this command:

    ```
    bazel build //:my-runner
    ```

    You’ll see output similar to:

    ```
    INFO: Found 1 target...
    Target //:my-runner up-to-date:
      bazel-bin/my-runner.jar
      bazel-bin/my-runner
    INFO: Elapsed time: 1.021s, Critical Path: 0.83s
    ```

2.  Now execute the file by using this command:

    ```
    bazel-bin/my-runner
    ```

Congratulations, you've built your first Bazel target!

Let’s take a look at what you built. In `~/my-project`, Bazel created the
directory `bazel-bin` as well as other directories to store information about
the build. Open this directory to look at the files created during the build
process. These output directories keep the outputs separate from your source
tree.

### Review the dependency graph

Bazel requires build dependencies to be explicitly declared in BUILD
files. The build will fail if dependencies are missing, so when a build works
the declared dependencies are accurate. With this explicit information about
dependencies, Bazel creates a build graph and uses it to accurately perform
incremental builds. Our small Java project isn’t too exciting, but let’s check
out its build graph.

The command `bazel query` retrieves information about the graph and the
relationships between targets. Let’s use it to produce a visual representation
of the build graph.

1.  From the root of the workspace (`my-project`), produce a text description
    of the graph by using the command:

    ```
    bazel query --noimplicit_deps 'deps(//:my-runner)' --output graph
    ```

2.  Then, paste the output into Graphviz
    ([http://www.webgraphviz.com/](http://www.webgraphviz.com/)) to see the
    visual representation.

    The graph for the target my-runner will look like this:

    ![Dependency graph of the target 'my-runner'](/assets/tutorial_java_01.svg)

You can see that `my-runner` depends on the two source files in your Java
project.

You have now set up the workspace and BUILD file, and used Bazel to build your
project. You have also created a visual representation of the build graph to
see the structure of your build.

## Refine your Bazel build

### Add dependencies

Creating one rule target to build your entire project may be sufficient for
small projects. As projects get larger it's important to break up the build
into self-contained libraries that can be assembled into a final product.
Self-contained libraries mean that everything doesn't need to be rebuilt after
small changes and that Bazel can parallelize more of the build steps. These
self-contained libraries also encourages good code hygiene.

To break up a project, create a separate rule target for the each subcomponent
and then add the subcomponents as dependencies. For the project in this
tutorial, create a rule target to compile the library, and make the executable
depend on it.

1.  Replace the text in the BUILD file with the text below:

    ```
    java_binary(
        name = "my-runner",
        srcs = ["src/main/java/com/example/ProjectRunner.java"],
        main_class = "com.example.ProjectRunner",
        deps = [":greeter"],
    )

    java_library(
        name = "greeter",
        srcs = ["src/main/java/com/example/Greeting.java"],
    )
    ```

The new `deps` attribute in `java_binary` tells Bazel that the `greeter` library
will be needed to compile the binary. Rules for many languages support the
`deps` attribute, though the exact semantics of the attribute will vary based
on the language and the type of target. The rule
[java_library](/docs/be/java.html#java_library) compiles sources into
a .jar file. Remember to go to the [Build Encyclopedia](/docs/be/overview.html)
for details about specific rules.

This BUILD file builds the same files as before, but in a different way: now
Bazel will first build the `greeter` library and then build `my-runner`.

2.  Try building //:my-runner using the command:

    ```
    bazel build //:my-runner
    ```

    You’ll see output similar to:

    ```
    INFO: Found 1 target...
    Target //:my-runner up-to-date:
      bazel-bin/my-runner.jar
      bazel-bin/my-runner
    INFO: Elapsed time: 2.454s, Critical Path: 1.58s
    ```

    3. Execute the file by using this command::

    ```
    bazel-bin/my-runner
    ```

If you now edit `ProjectRunner.java` and rebuild `my-runner`, the source file
`Greeting.java` will not be recompiled. When the BUILD file had only the one
target, both source files would be recompiled after any change.

Looking at the dependency graph, you can see that `my-runner` depends on the
same inputs as it did before, but the structure of the build is different.

The original dependency graph for `my-runner` looked link this:

![Original dependency graph of the target 'my-runner'](/assets/tutorial_java_01.svg)

The dependency graph for `my-runner` after adding a dependency looks like this:

![Dependency graph of the target 'my-runner' after adding a dependency](/assets/tutorial_java_02.svg)

### Use multiple packages

For larger projects, you will often be dealing with several directories in your
workspace. You can organize your build process by adding a BUILD file to the
top directory of source files that you want to organize together. A directory
containing a BUILD file is called a package.

Note that Bazel and Java both have the concept of a package. These are
unrelated to each other, though both are related to the structure of the
directories.

Let’s build the java project using multiple packages.

1.  First, let’s make the Java project a bit more complex.

    1.  Add the following directory and file:

        ```
        mkdir -p src/main/java/com/example/cmdline
        ```
    2. In the directory cmdline, add the file Runner.java with the following
       contents:

        ```java
        package com.example.cmdline;

        import com.example.Greeting;

        public class Runner {
            public static void main(String args[]) {
                Greeting.sayHi();
            }
        }
        ```

        Now you have a slightly larger Java project that you can organize with
        multiple packages.

2.  In the directory `src/main/java/com/example/cmdline`, add an empty text
    file and name it BUILD. The structure of the Java project is now:

    ```
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

    Each directory in the workspace can be part of only one package. The
    workspace now has two BUILD files, and so has two packages:

    1.  The directory `my-project` and its subdirectories (but not including
    subdirectories with their own BUILD file, such as `cmdline`), and

    2.  The directory `cmdline` and any subdirectories.

3.  In the new BUILD file, add the following text:

    ```
    java_binary(
        name = "runner",
        srcs = ["Runner.java"],
        main_class = "com.example.cmdline.Runner",
        deps = ["//:greeter"]
    )
    ```

    The file `Runner.java` depends on `com.example.Greeting`. In the BUILD file
    this dependency is shown by listing the rule target `greeter` (with the
    label `//:greeter`).

    Below is what the dependency graph for runner will look like. You can see
    how `//:greeter` gives the dependency on `Greeting.java`.

    ![Dependency graph of the target 'runner'](/assets/tutorial_java_03.svg)


4.  However, if you try to build runner right now you'll get a permissions
    error. You can see the permission error by trying to build the target using
    the command:

    ```
    bazel build //src/main/java/com/example/cmdline:runner
    ```

    By default, rule targets are private, which means that they can only be
    depended on by targets in the same BUILD file. This privacy prevents
    libraries that are implementation details from leaking into public APIs,
    but it also means that you must explicitly allow `runner` to depend on
    `//:greeter`.


5.  Make a rule target visible to rule targets in other BUILD files by adding
    a `visibility` attribute. To make the `greeter` rule target in
    `~/my-project/BUILD` visible to any rule target in the new package, add the
    following visibility attribute:

    ```
    java_library(
        name = "greeter",
        srcs = ["src/main/java/com/example/Greeting.java"],
        visibility = ["//src/main/java/com/example/cmdline:__pkg__"],
    )
    ```

    The target `//:greeter` is now visible to any target in the
    `//src/main/java/com/example/cmdline` package.

    See the Build Encyclopedia for more
    [visibility options](/docs/be/common-definitions.html#common.visibility).


6.  Now you can build the runner binary by using the command:

    ```
    bazel build //src/main/java/com/example/cmdline:runner
    ```

    You’ll see output similar to:

    ```
    INFO: Found 1 target...
    Target //src/main/java/com/example/cmdline:runner up-to-date:
      bazel-bin/src/main/java/com/example/cmdline/runner.jar
      bazel-bin/src/main/java/com/example/cmdline/runner
    INFO: Elapsed time: 1.576s, Critical Path: 0.81s
    ```


7.  Execute the file by using this command:

    ```
    bazel-bin/src/main/java/com/example/cmdline/runner
    ```

You’ve now refined your build so that it is broken down into smaller
self-contained libraries, and so that the explicit dependencies are more
granular. You’ve also built the Java project using multiple packages.

## Use labels to reference targets

In the BUILD files and in the command line, you have been using target labels
to reference targets. The label’s syntax is: `//path/to/package:target-name`,
where “`//`” is the workspace’s root, and “`:`” separates the package name and
the target name. If the target is a rule target and so defined in a BUILD file,
“`path/to/package`” would be the path of the BUILD file itself. “`Target-name`”
would be the same as the “`name`” attribute in the target in the BUILD file.

The first BUILD file you created in this tutorial is in the same directory as
the WORKSPACE file. When referencing rule targets defined in that file, nothing
is needed for the path to the package because the workspace root and the package
root are the same directory. Here are the labels of the two targets defined
in that first BUILD file:

```
//:my-runner
//:greeter
```

The second BUILD file has a longer path from the workspace root to the package
root. The label for the target in that BUILD file is:

```
//src/main/java/com/example/cmdline:runner
```

Target labels can be shortened in a variety of ways. Within a BUILD file, if
you’re referencing a target from the same package, you can write the label
starting at “`:`”. For example, the rule target `greeter` can always be written
as `//:greeter`, and in the BUILD file where it’s defined, it can also be
written as `:greeter`. This shortened label in a BUILD file makes it immediately
clear which targets are in the current package.

A rule target’s name will always be defined by its name attribute. A target’s
name is a bit more complex when it’s in a directory other than the root
of the package. In that case, the target’s label is:
`//path/to/package:path/to/file/file_name`.

## Package a Java target for deployment

To understand what you’ve built and what else can be built with Bazel, you need
to understand the capabilities of the rules used in your BUILD files. Always go
to the [Build Encyclopedia](/docs/be/overview.html) for this information.

Let’s look at packaging a Java target for deployment, which requires you to
know the capabilities of the rule `java_binary`.

You’re able to run the Java binaries you created in this tutorial, but you
can’t simply run it on a server, because it relies on the greeting library jar
to actually run. "Packaging an artifact so it can be run reliably outside the
development environment involves bundling it with all of its runtime
dependencies.  Let's see now what’s needed to package the binaries.

The rule [java_binary](/docs/be/java.html#java_binary) produces a Java archive
(“jar file”) and a wrapper shell script. The file `<target-name>_deploy.jar`
is suitable for deployment, but it’s only built by this rule if explicitly
requested. Let’s investigate.

1.  Look at the contents of the output `runner.jar` by using this command:

    ```
    jar tf bazel-bin/src/main/java/com/example/cmdline/runner.jar
    ```

    You’ll see output similar to:

    ```
    META-INF/
    META-INF/MANIFEST.MF
    com/
    com/example/
    com/example/cmdline/
    com/example/cmdline/Runner.class
    ```

    You can see that `runner.jar` contains `Runner.class`, but not its
    dependency `Greeting.class`. The `runner` script that Bazel generates adds
    the greeter jar to the classpath, so running this program works locally. It
    will not work if you want to copy `runner.jar` to another machine and use
    it as a standalone binary.


2.  The rule `java_binary` allows you to build a self-contained binary that can
    be deployed. To create this binary, build `runner_deploy.jar` (or, more
    generally, `<target-name>_deploy.jar`)  by using this command:

    ```
    bazel build //src/main/java/com/example/cmdline:runner_deploy.jar
    ```

    You’ll see output similar to:

    ```
    INFO: Found 1 target...
    Target //src/main/java/com/example/cmdline:runner_deploy.jar up-to-date:
      bazel-bin/src/main/java/com/example/cmdline/runner_deploy.jar
    INFO: Elapsed time: 1.700s, Critical Path: 0.23s
    ```

    The file runner_deploy.jar will contain all of its dependencies, and so can
    be used as a standalone binary.

You’ve now created a Java target that you can distribute and deploy. To do so,
you had to be aware of what outputs the Bazel Java rule `java_binary` is able to
produce.

## Further topics

*  Try the tutorial [Build C++](/docs/tutorial/cpp.md).
*  Try the tutorial [Build Mobile Application](/docs/tutorial/app.md).

