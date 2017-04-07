---
layout: contribute
title: Generating C++ crosstool with a Skylark Remote Repository
---

# Design Document: Generating C++ crosstool with a Skylark Remote Repository

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status:** implemented

**Author:** dmarting@google.com

**Reviewers:** lberki@google.com

**Design document published**: 16 February 2016

## Context

[Skylark](/docs/skylark/index.html) is the
extension language for Bazel and  lets Bazel users describe the
build for new languages easily. External users do not create
native rules and we want to avoid them doing so.

[Remote repositories](/docs/external.html)
are a convenient way to specify your third party dependencies
and to fetch them along with the build if you don’t want to
check them in your repository.

[Skylark remote
repositories](/designs/2015/07/02/skylark-remote-repositories.html) is
an ongoing effort to support specifying new remote repositories using
Skylark.

## Why?

Configurability issues are stopping users from compiling and using
Bazel on complex setup. In particular,
[TensorFlow](https://tensorflow.io)’s users runs on
various hardware where gcc is installed on non-standard directory that
needs to change the
[CROSSTOOL](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/CROSSTOOL)
file (see
[issue #531](https://github.com/bazelbuild/bazel/issues/531)).
This generally requires to change the list of include directories, the
path to gcc and sometimes also the linking option to find the correct
libraries at runtime. Some platform even requires
[special wrappers around gcc](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/osx_gcc_wrapper.sh).

Java solved the problem by setting a custom repository
([@local_jdk](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE#L3))
where the path is automatically detected using the [location of the JVM
running
Bazel](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/src/main/java/com/google/devtools/build/lib/packages/WorkspaceFactory.java#L414).
But this approach does not scale well with complex language setup like
C++.

We reduced the number of C++ issues the user had with removing all C++
compilation in the bootstrap of Bazel. However, to properly handle
those platform, Bazel needs some level of auto-configuration
([Kythe](https://github.com/google/kythe/blob/a29f0adc6fa11550f66bc2278f17b89b9e02de18/setup_bazel.sh)
and
[Tensorflow](https://github.com/tensorflow/tensorflow/blob/a81c4f9cd01563e97fc6f179e4d70960fc9b02ae/configure)
have their own auto-configuration scripts). This document discuss how
to use a skylark remote repository to implement a simple
auto-configuration for C++ crosstool (step 4 of the roadmap from the
[Skylark remote
repositories](/design/2015/07/02/skylark-remote-repositories.html)
document).

## C++ toolchain detection

Until now here the various issues user have faced using a custom C++
toolchain:

  1. C++ compiler is not at the expected location.
  2. C++ compiler is `clang` and not `gcc` or behaves differently than
     what Bazel C++ rules expect.
  3. Libraries are not in the default location.
  4. Headers are not in the default location.
  5. Path of libraries or headers are outside of the default mounted
     paths.

The current fix we propose to the user for the various issue are:

  1. Change the tool paths in
     [tools/cpp/CROSSTOOL#L87](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/CROSSTOOL#L87).
  2. Add a wrapper like
     [tools/cpp/osx\_gcc\_wrapper.sh](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/osx_gcc_wrapper.sh)
     and modify some options from the CROSSTOOL file.
  3. Add `-Wl,rpath,` option to the
     [linker\_flags](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/CROSSTOOL#L93).
  4. Add the directories to the
     [cxx\_builtin\_include\_dirs](https://github.com/bazelbuild/bazel/blob/6b6ff76945c80fb8b11b71d402b5146c85b86859/tools/cpp/CROSSTOOL#L100).
  5. Deactivate sandboxing

To address those issues, we propose to add the followings methods to
the repository context object:

  1. `which(cmd)` returns the path to the binary designed by `cmd`,
     looking for it in the path environment variable (or equivalent).
     This will help getting the path to the C++ compiler.
  2. `execute([arg0, arg1, ..., argn])` executes a command and returns an
     `exec_result` struct containing:
     * `stdout` the content of the standard output,
     * `stderr` the content of the standard error output, and
     * `return_code` the return code of the execution.
     Executing `$(CC)` will help detect whether we are using gcc or
     clang.
  3. An `os` object with an environ map containing the list of
     environment variable. The os object will be extended to
     contains all OS specific variables (platform name and much more).
  4. `execute([..])` from 2 will be used to run [`gcc ...
-v`](http://stackoverflow.com/questions/11946294/dump-include-paths-from-g)
     to list the built-in include directories.
To address the issue 5, we can add the list of paths to dependencies to the
[crosstool rule in the BUILD
file](https://github.com/bazelbuild/bazel/wiki/Building-with-a-custom-toolchain).

## Writing the cpp package

Once we have resolved all the information from the system, we need to
write two or three files:

  - The `BUILD` file that will contains the corresponding
    `cc_toolchain` rules
  - The `CROSSTOOL` file
  - Optionally, the wrapper script.

We should extends the context with a `file(path, content)` method, where
path is a path relative to the repository root and content the content
of the file to write.

To ease the writing of crosstool, we should also provide a
`template(path, label, variables)` method which will write the file
pointed by path using the file pointed by label (should be a
FileValue) in which variables have been replaced in the same way that
[template_action](http://bazel.build/docs/skylark/lib/ctx.html#template_action)
works.

## Rollout plan

The implementation plan would be:

  1. Implements `which`, `execute`, `os`, `file` and `template`
     [__DONE__]
  2. Write the `cc_configure` repository rule which does the work. Use
     GitHub bugs as inputs on which platform to support. [__DONE__]
  3. Advertise the existence of `cc_configure` [__DONE__]
