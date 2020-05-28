---
layout: documentation
title: Installing Bazel using Bazelisk
---

# Installing Bazel using Bazelisk

[Bazelisk](https://github.com/bazelbuild/bazelisk) is a launcher for Bazel which
automatically downloads and installs an appropriate version of Bazel. Use
Bazelisk if you need to switch between different versions of Bazel depending on
the current working directory, or to always keep Bazel updated to the latest
release.

You can install Bazelisk in multiple ways, including:

* `npm install -g @bazel/bazelisk`
* using [a binary release](https://github.com/bazelbuild/bazelisk/releases) for
  Linux, macOS, or Windows
* using Homebrew on macOS
* by compiling from source using Go: `go get github.com/bazelbuild/bazelisk`

For more details, see
[the official README](https://github.com/bazelbuild/bazelisk/blob/master/README.md).
