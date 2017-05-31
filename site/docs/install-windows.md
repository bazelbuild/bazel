---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Install Bazel on Windows

Windows support is highly experimental. Known issues are [marked with
label "Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on GitHub issues.

We currently support only 64 bit Windows 7 or higher and we compile Bazel as a
MSYS2 binary.

Install Bazel on Windows using one of the following methods:

  * [Use Chocolatey](#install-on-windows-chocolatey)
  * [Use the binary distribution](#download-binary-windows)
  * [Compile Bazel from source](install-compile-source.md) -- make sure
    your machine meets the [requirements](windows.md#requirements)


## <a name="install-on-windows-chocolatey"></a>Install using Chocolatey

You can install the unofficial package using the
[chocolatey](https://chocolatey.org) package manager:

```sh
choco install bazel
```

This will install the latest available version of Bazel, and dependencies.

This package is experimental. Please provide feedback to `@petemounce` in GitHub
issue tracker. See the [Chocolatey installation and package
maintenance](windows-chocolatey-maintenance.md) guide for more information.


## <a name="download-binary-windows"></a>Install using the binary distribution

We provide binary versions on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

The installer contains only the Bazel binary. You'll need additional software
(e.g. msys2 shell of the right version) and some setup in your environment to
run Bazel. See these requirements on our
[Windows page](windows.md#requirements).
