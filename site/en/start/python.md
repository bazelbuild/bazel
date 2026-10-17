Project: /_project.yaml
Book: /_book.yaml

# Bazel Tutorial: Build a Python Project

{% include "_buttons.html" %}

This tutorial covers the basics of building Python applications with Bazel.
You will build and run the Python sample in Bazel's `examples/py` package and
use it to learn core Bazel concepts such as targets, labels, and dependencies.

Estimated completion time: 30 minutes.

## What you'll learn

In this tutorial you learn how to:

*  Build a target
*  Visualize the project's dependencies
*  Understand how Python binaries and libraries are split into targets
*  Use labels to reference targets
*  Use data files from another package
*  Build a distributable Python executable archive

## Before you begin

### Install Bazel

To prepare for the tutorial, first [Install Bazel](/install) if
you don't have it installed already.

### Install Python

1.  Install Python 3.8 or later.

2.  Verify Python is available:

```posix-terminal
python3 --version
```

### Get the sample project

Retrieve Bazel's source repository (which includes the tutorial sample):

```posix-terminal
git clone https://github.com/bazelbuild/bazel
```

The sample project for this tutorial is in the `examples/py` directory and is
structured as follows:

```
examples/py
├── BUILD
├── bin.py
├── lib.py
└── runfile.py
```

## Build with Bazel

### Understand the BUILD file

A `BUILD` file contains several different types of instructions for Bazel.
The most important type is the *build rule*, which tells Bazel how to build the
desired outputs, such as executable binaries or libraries. Each instance
of a build rule in the `BUILD` file is called a *target* and points to a
specific set of source files and dependencies. A target can also point to other
targets.

Take a look at `examples/py/BUILD`:

```python
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
		name = "lib",
		srcs = ["lib.py"],
)

py_binary(
		name = "bin",
		srcs = ["bin.py"],
		deps = [":lib"],
)

py_binary(
		name = "runfile",
		srcs = ["runfile.py"],
		data = ["//examples:runfile.txt"],
		deps = ["@rules_python//python/runfiles"],
)
```

In this example, `bin` and `runfile` instantiate Bazel's
[`py_binary` rule](/reference/be/python#py_binary), and `lib` instantiates
the [`py_library` rule](/reference/be/python#py_library).

### Build the project

From the repository root, build the main sample target:

```posix-terminal
bazel build //examples/py:bin
```

In the target label, the `//examples/py` part is the package path and `bin` is
the target name in the `BUILD` file.

Bazel produces output similar to the following:

```bash
INFO: Found 1 target...
Target //examples/py:bin up-to-date:
	bazel-bin/examples/py/bin
INFO: Elapsed time: 23.754s, Critical Path: 0.24s
```

Now run the built binary:

```posix-terminal
./bazel-bin/examples/py/bin
```

Expected output:

```text
Fib(5)=8
```

### Review the dependency graph

Bazel requires dependencies to be explicitly declared in BUILD files and uses
those declarations to construct the dependency graph.

To visualize dependencies for `//examples/py:bin`, run:

```posix-terminal
bazel query --notool_deps --noimplicit_deps "deps(//examples/py:bin)" --output graph
```

You can paste the graph output into [GraphViz](http://www.webgraphviz.com/).

The output includes edges like:

```text
"//examples/py:bin" -> "//examples/py:lib"
"//examples/py:bin" -> "//examples/py:bin.py"
"//examples/py:lib" -> "//examples/py:lib.py"
```

## Refine your Bazel build

### Specify multiple build targets

The sample already demonstrates a common pattern:

*  `lib` (`py_library`) holds reusable logic (`lib.py`)
*  `bin` (`py_binary`) is an executable that depends on `:lib`

This split keeps application code modular and helps Bazel rebuild only what
changed.

### Use multiple packages

The `runfile` target uses a data file from a different package via the label
`//examples:runfile.txt`.

Build and run it:

```posix-terminal
bazel build //examples/py:runfile
./bazel-bin/examples/py/runfile
```

Expected output:

```text
The content of the runfile is:
This is a runfile.
```

This demonstrates cross-package references using labels in the `data`
attribute.

## Use labels to reference targets

In BUILD files and on the command line, Bazel uses labels such as
`//examples/py:bin` or `//examples:runfile.txt`.

Their syntax is:

```
//path/to/package:target-name
```

For rule targets, `target-name` is the rule's `name` attribute.
For file targets, `target-name` is the file path relative to the package root.

Within the same package, you can use short labels like `:lib`.

## Build a Python executable archive

To build a distributable archive for the Python binary, use
`--build_python_zip`:

```posix-terminal
bazel build --build_python_zip //examples/py:bin
```

Bazel produces output similar to:

```bash
INFO: Found 1 target...
Target //examples/py:bin up-to-date:
	bazel-bin/examples/py/bin.zip
	bazel-bin/examples/py/bin
INFO: Elapsed time: 9.016s, Critical Path: 8.22s
```

The `bin.zip` artifact is a single-file executable Python archive.

## Further reading

For more details, see:

*  [Build Encyclopedia: Python rules](/reference/be/python)

*  [rules_python](https://github.com/bazelbuild/rules_python) for Python-specific
	 Bazel rules and tooling.

*  [External Dependencies](/docs/external) to learn more about working with
	 local and remote repositories.

*  The [other rules](/rules) to learn more about Bazel.

*  The [Java build tutorial](/start/java) to get started with building Java
	 projects with Bazel.

Happy building.
