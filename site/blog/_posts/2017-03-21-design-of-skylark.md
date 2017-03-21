---
layout: posts
title: A glimpse of the design of Skylark
---

This blog post describes the design of Skylark, the language used to specify
builds in Bazel.

## A brief history

Many years ago, code at Google was built using Makefiles. As [other people
noticed](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/03/hadrian.pdf),
Makefiles don't scale well with a large code base. A temporary solution was to
generate Makefiles using Python scripts, where the description of the build was
stored in `BUILD` files containing calls to the Python functions. But this
solution was way too slow, and the bottleneck was Make.

The project Blaze (later open-sourced as Bazel) was started in 2006. It used a
simple parser to read the `BUILD` files (supporting only function calls, list
comprehensions and variable assignments). When Blaze could not directly parse a
`BUILD` file, it used a preprocessing step that ran the Python interpreter on
the user `BUILD` file to generate a simplified `BUILD` file. The output was used
by Blaze.

This approach was simple and allowed developers to create their own macros. But
again, this led to lots of problems in terms of maintenance, performance, and
safety. It also made any kind of tooling more complicated, as Blaze was not able
to parse the `BUILD` files itself.

In the current iteration of Bazel, we've made the system saner by removing the
Python preprocessing step. We kept the Python syntax, though, in order to
migrate our codebase. This seems to be a good idea anyway: Many people like the
syntax of our `BUILD` files and other build tools (e.g.
[Buck](https://buckbuild.com/concept/build_file.html),
[Pants](http://www.pantsbuild.org/build_files.html), and
[Please](https://please.build/language.html)) have adopted it.

## Design requirements

We decided to separate description of the build from the extensions (macros and
rules). The description of the build resides in `BUILD` files and the extensions
reside in `.bzl` files, although they are all evaluated with the same
interpreter. We want the code to be easy to read and maintain. We designed Bazel
to be used by thousands of engineers. Most of them are not familiar with build
systems internals and most of them don't want to spend time learning a new
language. `BUILD` files need to be simple and declarative, so that we can build
tools to manipulate them.

The language also needed to:

*   Run on the JVM. Bazel is written in Java. The data structures should be
    shared between Bazel and the language (due to memory requirements in large
    builds).

*   Use a Python syntax, to preserve our codebase.

*   Be deterministic and hermetic. We have to guarantee that the execution of
    the code will always yield the same results. For example, we forbid access
    to I/O and date and time, and ensure deterministic iteration order of
    dictionaries.

*   Be thread-safe. We need to evaluate a lot of `BUILD` files in parallel.
    Execution of the code needs to be thread-safe in order to guarantee
    determinism.

Finally, we have performance concerns. A typical `BUILD` file is simple and can
be executed quickly. In most cases, evaluating the code directly is faster than
compiling it first.

## Parallelism and imports

One special feature of Skylark is how it handles parallelism. In Bazel, a large
build require the evaluation of hundreds of `BUILD` files, so we have to load
them in parallel. Each `BUILD` file may use any number of extensions, and those
extensions might need other files as well. This means that we end up with a
graph of dependencies.

Bazel first evaluates the leaves of this graph (i.e. the files that have no
dependencies) in parallel. It will load the other files as soon as their
dependencies have been loaded, which means the evaluation of `BUILD` and `.bzl`
files is interleaved. This also means that the order of the `load` statements
doesn't matter at all.

Each file is loaded at most once. Once it has been evaluated, its definitions
(the global variables and functions) are cached. Any other file can access the
symbols through the cache.

Since multiple threads can access a variable at the same time, we need a
restriction on side-effects to guarantee thread-safety. The solution is simple:
when we cache the definitions of a file, we "freeze" them. We make them
read-only, i.e. you can iterate on an array, but not modify its elements. You
may create a copy and modify it, though.

In a future blog post, we'll take a look at the other features of the language.

_By [Laurent Le Brun](https://github.com/laurentlb)_
