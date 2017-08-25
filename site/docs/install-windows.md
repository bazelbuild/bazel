---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Install Bazel on Windows

Bazel runs on 64 bit Windows 7 or higher.

Known issues are [marked with label
"Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on GitHub issues.

To run Bazel on Windows, make sure your system meets the
[requirements](windows.html#requirements).

You can get Bazel for Windows using one of the following methods:

*   Download the binary distribution (recommended).

    We provide binary versions on our [GitHub releases
    page](https://github.com/bazelbuild/bazel/releases).

    The installer contains only the Bazel binary. You'll need additional
    software (e.g. msys2 shell) and some setup in your environment to run Bazel.
    You can find these on the [Windows requirements page](windows.html).

*   Use Chocolatey.

    You can install the Bazel package using the
    [Chocolatey](https://chocolatey.org) package manager:

    ```sh
    choco install bazel
    ```

    This command will install the latest available version of Bazel and most of
    its dependencies, such as the msys2 shell. This will not install Visual C++
    though.

    See [Chocolatey installation and package maintenance
    guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
    information about the Chocolatey package.

*   [Compile Bazel from source](install-compile-source.html).

## Next: see [Using Bazel on Windows](windows.html#using)
