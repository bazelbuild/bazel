Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Build a C++ Project

{% include "_buttons.html" %}

## Introduction

New to Bazel? You’re in the right place. Follow this First Build tutorial for a
simplified introduction to using Bazel. This tutorial defines key terms as they
are used in Bazel’s context and walks you through the basics of the Bazel
workflow. Starting with the tools you need, you will build and run three
projects with increasing complexity and learn how and why they get more complex.

While Bazel is a [build system](https://bazel.build/basics/build-systems) that
supports multi-language builds, this tutorial uses a C++ project as an example
and provides the general guidelines and flow that apply to most languages.

Estimated completion time: 30 minutes.

### Prerequisites

Start by [installing Bazel](https://bazel.build/install), if you haven’t
already. This tutorial uses Git for source control, so for best results
[install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) as
well.

Next, retrieve the sample project from Bazel's GitHub repository by running the
following in your command-line tool of choice:

```posix-terminal
git clone https://github.com/bazelbuild/examples
```

The sample project for this tutorial is in the `examples/cpp-tutorial` directory.

Take a look below at how it’s structured:

```
examples
└── cpp-tutorial
    ├──stage1
    │  ├── main
    │  │   ├── BUILD
    │  │   └── hello-world.cc
    │  └── WORKSPACE
    ├──stage2
    │  ├── main
    │  │   ├── BUILD
    │  │   ├── hello-world.cc
    │  │   ├── hello-greet.cc
    │  │   └── hello-greet.h
    │  └── WORKSPACE
    └──stage3
       ├── main
       │   ├── BUILD
       │   ├── hello-world.cc
       │   ├── hello-greet.cc
       │   └── hello-greet.h
       ├── lib
       │   ├── BUILD
       │   ├── hello-time.cc
       │   └── hello-time.h
       └── WORKSPACE
```

There are three sets of files, each set representing a stage in this tutorial.
In the first stage, you will build a single [target]
(https://bazel.build/reference/glossary#target) residing in a single [package]
(https://bazel.build/reference/glossary#package). In the second stage, you will
build both a binary and a library from a single package. In
the third and final stage, you will build a project with multiple packages and
build it with multiple targets.

### Summary: Introduction

By installing Bazel (and Git) and cloning the repository for this tutorial, you
have laid the foundation for your first build with Bazel. Continue to the next
section to define some terms and set up your [workspace](https://bazel.build/reference/glossary#workspace).

## Getting started

### Set up the workspace

  Before you can build a project, you need to set up its workspace. A workspace is
a directory that holds your project's source files and Bazel's build outputs. It
also contains these significant files:

*   The <code>[`WORKSPACE` file](https://bazel.build/reference/glossary#workspace-file)
</code>, which identifies the directory and its contents as a Bazel workspace and
lives at the root of the project's directory structure.
*   One or more <code>[`BUILD` files](https://bazel.build/reference/glossary#build-file)
</code>, which tell Bazel how to build different parts of the project. A
directory within the workspace that contains a <code>BUILD</code> file is a
[package](https://bazel.build/reference/glossary#package). (More on packages
later in this tutorial.)

In future projects, to designate a directory as a Bazel workspace, create an
empty file named `WORKSPACE` in that directory. For the purposes of this tutorial,
a `WORKSPACE` file is already present in each stage.

**NOTE**: When Bazel builds the project, all inputs must be in
the same workspace. Files residing in different workspaces are independent of
one another unless linked. More detailed information about workspace rules can
be found in [this guide](https://bazel.build/reference/be/workspace).


### Understand the BUILD file


A `BUILD` file contains several different types of instructions for Bazel. Each
`BUILD` file requires at least one [rule](https://bazel.build/reference/glossary#rule)
as a set of instructions, which tells Bazel how to build the desired outputs,
such as executable binaries or libraries. Each instance of a build rule in the
`BUILD` file is called a [target](https://bazel.build/reference/glossary#target)
and points to a specific set of source files and [dependencies](https://bazel.build/reference/glossary#dependency).
A target can also point to other targets.

Take a look at the `BUILD` file in the `cpp-tutorial/stage1/main` directory:

```
cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)
```

In our example, the `hello-world` target instantiates Bazel's built-in
<code>[cc_binary rule](https://bazel.build/reference/be/c-cpp#cc_binary)</code>.
The rule tells Bazel to build a self-contained executable binary from the
<code>hello-world.cc</code> source file with no dependencies.

### Summary: getting started

Now you are familiar with some key terms, and what they mean in the context of
this project and Bazel in general. In the next section, you will build and test
Stage 1 of the project.


## Stage 1: single target, single package

It’s time to build the first part of the project. For a visual reference, the
structure of the Stage 1 section of the project is:

```
examples
└── cpp-tutorial
    └──stage1
       ├── main
       │   ├── BUILD
       │   └── hello-world.cc
       └── WORKSPACE
```

Run the following to move to the `cpp-tutorial/stage1` directory:

```posix-terminal
cd cpp-tutorial/stage1
```

Next, run:

```posix-terminal
bazel build //main:hello-world
```

In the target label, the `//main:` part is the location of the `BUILD` file
relative to the root of the workspace, and `hello-world` is the target name in
the `BUILD` file.

Bazel produces something that looks like this:

```
INFO: Found 1 target...
Target //main:hello-world up-to-date:
  bazel-bin/main/hello-world
INFO: Elapsed time: 2.267s, Critical Path: 0.25s
```

You just built your first Bazel target. Bazel places build outputs in the
`bazel-bin` directory at the root of the
workspace.

Now test your freshly built binary, which is:

```posix-terminal
bazel-bin/main/hello-world
```

This results in a printed “`Hello world`” message.

Here’s the dependency graph of Stage 1:

![Dependency graph for hello-world displays a single target with a single source file.](/docs/images/cpp-tutorial-stage1.png "Dependency graph for hello-world displays a single target with a single source file.")


### Summary: stage 1

Now that you have completed your first build, you have a basic idea of how a build
is structured. In the next stage, you will add complexity by adding another
target.

## Stage 2: multiple build targets

While a single target is sufficient for small projects, you may want to split
larger projects into multiple targets and packages. This allows for fast
incremental builds – that is, Bazel only rebuilds what's changed – and speeds up your
builds by building multiple parts of a project at once. This stage of the
tutorial adds a target, and the next adds a package.

This is the directory you are working with for Stage 2:

```
    ├──stage2
    │  ├── main
    │  │   ├── BUILD
    │  │   ├── hello-world.cc
    │  │   ├── hello-greet.cc
    │  │   └── hello-greet.h
    │  └── WORKSPACE
```

Take a look below at the `BUILD` file in the `cpp-tutorial/stage2/main` directory:

```
cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-greet",
    ],
)
```

With this `BUILD` file, Bazel first builds the `hello-greet` library
(using Bazel's built-in <code>[cc_library rule](https://bazel.build/reference/be/c-cpp#cc_library)</code>),
then the <code>hello-world</code> binary. The <code>deps</code> attribute in
the <code>hello-world</code> target tells Bazel that the <code>hello-greet</code>
library is required to build the <code>hello-world</code> binary.

Before you can build this new version of the project, you need to change
directories, switching to the `cpp-tutorial/stage2` directory by running:

```posix-terminal
cd ../stage2
```

Now you can build the new binary using the following familiar command:

```posix-terminal
bazel build //main:hello-world
```

Once again, Bazel produces something that looks like this:

```
INFO: Found 1 target...
Target //main:hello-world up-to-date:
  bazel-bin/main/hello-world
INFO: Elapsed time: 2.399s, Critical Path: 0.30s
```

Now you can test your freshly built binary, which returns another “`Hello world`”:

```posix-terminal
bazel-bin/main/hello-world
```

If you now modify `hello-greet.cc` and rebuild the project, Bazel only recompiles
that file.

Looking at the dependency graph, you can see that `hello-world` depends on an extra input
named `hello-greet`:

![Dependency graph for `hello-world` displays dependency changes after modification to the file.](/docs/images/cpp-tutorial-stage2.png "Dependency graph for `hello-world` displays dependency changes after modification to the file.")

### Summary: stage 2

You've now built the project with two targets. The `hello-world` target builds
one source file and depends on one other target (`//main:hello-greet`), which
builds two additional source files. In the next section, take it a step further
and add another package.

## Stage 3: multiple packages

This next stage adds another layer of complication and builds a project with
multiple packages. Take a look below at the structure and contents of the
`cpp-tutorial/stage3` directory:

```
└──stage3
   ├── main
   │   ├── BUILD
   │   ├── hello-world.cc
   │   ├── hello-greet.cc
   │   └── hello-greet.h
   ├── lib
   │   ├── BUILD
   │   ├── hello-time.cc
   │   └── hello-time.h
   └── WORKSPACE
```

You can see that now there are two sub-directories, and each contains a `BUILD`
file. Therefore, to Bazel, the workspace now contains two packages: `lib` and
`main`.

Take a look at the `lib/BUILD` file:

```
cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
    visibility = ["//main:__pkg__"],
)
```

And at the `main/BUILD` file:

```
cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-greet",
        "//lib:hello-time",
    ],
)
```

The `hello-world` target in the main package depends on the` hello-time` target
in the `lib` package (hence the target label `//lib:hello-time`) - Bazel knows
this through the `deps` attribute. You can see this reflected in the dependency
graph:

![Dependency graph for `hello-world` displays how the target in the main package depends on the target in the `lib` package.](/docs/images/cpp-tutorial-stage3.png "Dependency graph for `hello-world` displays how the target in the main package depends on the target in the `lib` package.")

For the build to succeed, you make the `//lib:hello-time` target in `lib/BUILD`
explicitly visible to targets in `main/BUILD` using the visibility attribute.
This is because by default targets are only visible to other targets in the same
`BUILD` file. Bazel uses target visibility to prevent issues such as libraries
containing implementation details leaking into public APIs.

Now build this final version of the project. Switch to the `cpp-tutorial/stage3`
directory by running:

```posix-terminal
cd  ../stage3
```

Once again, run the following command:

```posix-terminal
bazel build //main:hello-world
```

Bazel produces something that looks like this:

```
INFO: Found 1 target...
Target //main:hello-world up-to-date:
  bazel-bin/main/hello-world
INFO: Elapsed time: 0.167s, Critical Path: 0.00s
```

Now test the last binary of this tutorial for a final `Hello world` message:

```posix-terminal
bazel-bin/main/hello-world
```

### Summary: stage 3

You've now built the project as two packages with three targets and understand
the dependencies between them, which equips you to go forth and build future
projects with Bazel. In the next section, take a look at how to continue your
Bazel journey.

## Next steps

You’ve now completed your first basic build with Bazel, but this is just the
start. Here are some more resources to continue learning with Bazel:

*   To keep focusing on C++, read about common [C++ build use cases](https://bazel.build/tutorials/cpp-use-cases).
*   To get started with building other applications with Bazel, see the tutorials
for [Java](https://bazel.build/start/java), [Android application](https://bazel.build/start/android-app ),
or [iOS application](https://bazel.build/start/ios-app)).
*   To learn more about working with local and remote repositories, read about
[external dependencies](https://bazel.build/docs/external).
*   To learn more about Bazel’s other rules, see this [reference guide](https://bazel.build/rules).

Happy building!
