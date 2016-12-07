---
layout: posts
title: Bazel on Windows
---

We first announced experimental Windows support in 0.3.0. Since then, we've
implemented support for building, running and testing C++, Java and Python,
as well as improved performance and stability. Starting with
Bazel version 0.3.2, we are making prebuilt Bazel Windows binaries available
as part of our
[releases](https://github.com/bazelbuild/bazel/releases)
([installation instructions](/docs/windows.html#using-the-release-binary)).

In addition to bootstrapping Bazel itself, we're also able to build
significant parts of TensorFlow with Bazel on Windows
([pull request](https://github.com/tensorflow/tensorflow/pull/4796)).
Bazel on Windows currently requires [msys2](https://msys2.github.io/) and
still has a number of issues. Some of the more important ones are:

 * [Workspace of the project needs to be on C: drive](https://github.com/bazelbuild/bazel/issues/1463)
 * [Runfiles will require additional tweaking](https://github.com/bazelbuild/bazel/issues/1212)
 * We support [building C++ code with MSVC toolchain](/docs/windows.html#build-c),
   but it is not yet the default toolchain.

Our GitHub issue tracker has a [full list of known issues](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22).

Now, we need your help! Please try building your Bazel project on Windows,
and let us know what works or what doesn't work yet, and what we can do better.

We are looking forward to what you build (on Windows)!

