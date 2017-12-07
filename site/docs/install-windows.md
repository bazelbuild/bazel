---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Install Bazel on Windows

Supported Windows platforms:

*   64 bit Windows 7 or higher

Before installing Bazel, make sure your system meets the
[Windows requirements](windows.html#requirements) to run Bazel.

After installing Bazel on Windows, see
[Using Bazel on Windows](windows.html#using).

Install Bazel on Windows using one of the following methods:

*   [Download the binary (recommended)](#download-the-binary-recommended)
*   [Install using Chocolatey](#install-using-chocolatey)
*   [Compile Bazel from source](install-compile-source.html)

## Download the binary (recommended)

Go to Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases)
and download the Windows binary. We recommend putting the binary in a directory
that's on your `%PATH%`. You can, however, put the binary anywhere on your
filesystem.

After you download the binary, you'll need additional
software and some setup in your environment to run Bazel. For details, see the
[Windows requirements](windows.html).

## Install using Chocolatey


You can install the Bazel package using the [Chocolatey](https://chocolatey.org)
package manager:

```sh
choco install bazel
```

This command will install the latest available version of Bazel and most of
its dependencies, such as the msys2 shell. This will not install Visual C++
though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.
