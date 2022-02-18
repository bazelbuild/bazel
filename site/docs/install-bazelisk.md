---
layout: documentation
title: Installing Bazel using Bazelisk
category: getting-started
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/install/bazelisk" style="color: #0000EE;">https://bazel.build/install/bazelisk</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


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
