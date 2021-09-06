---
layout: documentation
title: Installing Bazel using Bazelisk
category: getting-started
---

# Installing Bazel using Bazelisk

This page describes how to install Bazel using Bazelisk.

[Bazelisk](https://github.com/bazelbuild/bazelisk) is a launcher for Bazel which
automatically downloads and installs an appropriate version of Bazel. Use
Bazelisk if you need to switch between different versions of Bazel depending on
the current working directory, or to always keep Bazel updated to the latest
release.

You can install Bazelisk in multiple ways, including:

* using [a binary release](https://github.com/bazelbuild/bazelisk/releases) for
  Linux, macOS, or Windows
* using npm: `npm install -g @bazel/bazelisk`
* using Homebrew on macOS: `brew install bazelisk`
* by compiling from source using Go: `go install github.com/bazelbuild/bazelisk@latest` (needs Go 1.17 or later)

For more details, see
[the official README](https://github.com/bazelbuild/bazelisk/blob/master/README.md).
