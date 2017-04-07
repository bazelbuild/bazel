---
layout: contribute
title: Building Python on Windows
---

# Design Document: Building Python on Windows

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: Implemented

**Author**: [Yun Peng](mailto:pcloudy@google.com)

**Design document published**: 05 September 2016

**Relevant changes**:

1. [zipper now can specify actual path a file is added to](https://bazel-review.googlesource.com/4243)
2. [Create Python executable zip file](https://bazel-review.googlesource.com/4244)
3. [Make python executable zip a real self-extracting binary](https://bazel-review.googlesource.com/4263)
4. [Using stub\_template.txt as \_\_main\_\_.py and zip header in python executable zip](https://bazel-review.googlesource.com/5310)
5. [Get rid of python executable zip file header](https://bazel-review.googlesource.com/5350)
6. [Put runfiles tree under 'runfiles' directory to avoid conflict](https://bazel-review.googlesource.com/5351)

As we keep finding new problems and coming up with new solutions, you
can see some of the implementation in the previous changes has been
deprecated in the latter ones. Here we only present the final solution.

## Motivation

After providing basic support for C++ and Java on Windows (although not
perfect for C++ due to the wrapper scripts we use), Python becomes the
last language we need to fix in order to make Bazel no longer
experimental on Windows.

## Problem

Currently, as described on [bazel.build](/docs/be/python.html#py_binary),
the way py\_binary works on Unix is:

_A py\_binary is an executable Python program consisting of a collection
of `.py` source files (possibly belonging to other py\_library rules),
a `*.runfiles` directory tree containing all the code and data needed by
the program at run-time, and a [stub script](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/bazel/rules/python/stub_template.txt)
that starts up the program with the correct initial environment and data._

This doesn’t work on Windows, because we don’t have runfiles tree on Windows.
A runfiles client library (which maps a runfile from runfiles path to
its real path) doesn’t solve the whole problem, because we also need
`__init__.py` file under every python source directory to make the
directory a recognizable python package. And it doesn’t make sense to
create `__init__.py` files outside of runfiles tree. Therefore, how to
find a way to run python binary properly on Windows is our main problem.

## Solution

Python has the ability to execute zip file as scripts since version 2.6.
When invoking the python interpreter with a zip file as the first argument,
it executes the `__main__.py` file in the root directory of the archive.

The idea of the solution is to create a self-extracting zip file which
packages everything supposed to be in the original runfiles tree and add
the stub script as the `__main__.py` file. We tell the stub script whether
it’s in a zip file or not. If it is, it first extracts the zip file to a
temporary directory as the runfiles tree, then set the correct environment
variables and runs the main python script. At the end of the execution,
it deletes the temporary directory.

We did three more things to achieve this goal:

#### 1. Implement new feature in zipper for packaging runfiles tree easily

Creating the zip file is not a trivial thing, since we don’t have runfiles
tree at all. We should not only archive every runfile into the right path,
but also adding `__init__.py` file to every directory. To make things
easier, we introduce a new feature in zipper (a custom zip tool of Bazel)
which makes users able to specify the actual path a file is archived into.
Zipper now supports the following semantics:

`zipper cC x.zip [<zip_path>=][<file>]`

Examples:

```bash
$ zipper cC x.zip a/b/lib.py                  # Add file a/b/lib.py
$ zipper cC x.zip a/b/__init__.py=            # Add an empty file at a/b/__init__.py
$ zipper cC x.zip a/b/main.py=foo/bar/bin.py  # Add file foo/bar/bin.py as a/b/main.py
```

With the help of this feature, we can easily control the directory
structure in the zip file.

#### 2. Build the final python binary

The final python binary is the zip file with a `#!/usr/bin/env python`
shebang. By doing this, we can run the binary in the following two ways:

```bash
$ ./bazel-bin/foo/bar/bin
$ python ./bazel-bin/foo/bar/bin
```

And we can use the second way to run the binary from Windows native
command line(cmd.exe). Since the zip file packages everything in runfiles
tree, it can be copied to anywhere as a self-contained executable binary.

#### 3. Add `--build_python_zip` flag

This flag can be used to tell Bazel whether or not it should build a zip file
as the final binary. By default, it’s enabled on Windows,
disabled on other platform.
