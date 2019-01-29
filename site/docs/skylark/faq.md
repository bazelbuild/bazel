---
layout: documentation
title: Extension FAQ
---

# Frequently Asked Questions

These are some common issues and questions with writing extensions.

* ToC
{:toc}


## Why is my file not produced / my action never executed?

Bazel only executes the actions needed to produce the *requested* output files.

* If the file you want has a label, you can request it directly:
  `bazel build //pkg:myfile.txt`

* If the file is in an output group of the target, you may need to specify that
  output group on the command line:
  `bazel build //pkg:mytarget --output_groups=foo`

* If you want the file to be built automatically whenever your target is
  mentioned on the command line, add it to your rule's default outputs by
  returning a [`DefaultInfo`](lib/globals.html#DefaultInfo) provider.

See the [Rules page](rules.md#requesting-output-files) for more information.

## Why is my implementation function not executed?

Bazel analyzes only the targets that are requested for the build. You should
either name the target on the command line, or something that depends on the
target.

## A file is missing when my action or binary is executed

Make sure that 1) the file has been registered as an input to the action or
binary, and 2) the script or tool being executed is accessing the file using the
correct path.

For actions, you declare inputs by passing them to the `ctx.actions.*` function
that creates the action. The proper path for the file can be obtained using
[`File.path`](lib/File.html#path).

For binaries (the executable outputs run by a `bazel run` or `bazel test`
command), you declare inputs by including them in the
[runfiles](rules.md#runfiles). Instead of using the `path` field, use
[`File.short_path`](lib/File.html#short_path), which is file's path relative to
the runfiles directory in which the binary executes.

## How can I control which files are built by `bazel build //pkg:mytarget`?

Use the [`DefaultInfo`](lib/globals.html#DefaultInfo) provider to
[set the default outputs](rules.md#requesting-output-files).

## How can I run a program or do file I/O as part of my build?

A tool can be declared as a target, just like any other part of your build, and
run during the execution phase to help build other targets. To create an action
that runs a tool, use [`ctx.actions.run`](lib/actions.html#run) and pass in the
tool as the `executable` parameter.

During the loading and analysis phases, a tool *cannot* run, nor can you perform
file I/O. This means that tools and file contents (except the contents of BUILD
and .bzl files) cannot affect how the target and action graphs get created.

## What if I need to access the same structured data both before and during the execution phase?

You can format the structured data as a .bzl file. You can `load()` the file to
access it during the loading and analysis phases. You can pass it as an input or
runfile to actions and executables that need it during the execution phase.

## How should I document Starlark code?

For rules and rule attributes, you can pass a docstring literal (possibly
triple-quoted) to the `doc` parameter of `rule` or `attr.*()`. For helper
functions and macros, use a triple-quoted docstring literal following the format
given [here](skylint.md#docstrings). Rule implementation functions generally do
not need their own docstring.

Using string literals in the expected places makes it easier for automated
tooling to extract documentation. Feel free to use standard non-string comments
wherever it may help the reader of your code.
