---
layout: documentation
title: .bzl style guide
---

# .bzl style guide

[Starlark](language.md) is a language that defines how software is built, and as
such it is both a programming and a configuration language.

You will use Starlark to write BUILD files, macros, and build rules. Macros and
rules are essentially meta-languages - they define how BUILD files are written.
BUILD files are intended to be simple and repetitive.

All software is read more often than it is written. This is especially true for
Starlark, as engineers read BUILD files to understand dependencies of their
targets and details of their builds.This reading will often happen in passing,
in a hurry, or in parallel to accomplishing some other task. Consequently,
simplicity and readability are very important so that users can parse and
comprehend BUILD files quickly.

When a user opens a BUILD file, they quickly want to know the list of targets in
the file; or review the list of sources of that C++ library; or remove a
dependency from that Java binary. Each time you add a layer of abstraction, you
make it harder for a user to do these tasks.

BUILD files are also analyzed and updated by many different tools. Tools may not
be able to edit your BUILD file if it uses abstractions. Keeping your BUILD
files simple will allow you to get better tooling. As a code base grows, it
becomes more and more frequent to do changes across many BUILD files in order to
update a library or do a cleanup.

Do not create a macro just to avoid some amount of repetition in BUILD files.
The [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle
doesn't really apply here. The goal is not to make the file shorter; the goal is
to make your files easy to process, both by humans and tools.

## General advice

*   Use [Buildifier](https://github.com/bazelbuild/buildtools/tree/master/buildifier#linter)
    as a formatter and linter.
*   Follow [testing guidelines](testing.md).
*   Follow [performance guidelines](performance.md) in your rules.
